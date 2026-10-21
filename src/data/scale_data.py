"""
Leakage-Safe Numerical Feature Scaling Module (Step 3.8).

Fits StandardScaler instances on TRAINING unique hourly observations ONLY.
Transforms train, validation, calibration, and test partitions using the
fitted training scalers.

Key Rules:
  - Separate scalers per modality: EnergyScaler and WeatherScaler
  - Fitted ONLY on unique valid hourly observations in the training time region
  - Validation, calibration, and test data are NEVER used for fitting
  - Metadata columns (date, n_observations, coverage_ratio, is_valid_hour) are unscaled
  - Future-weather and secondary electrical variables are excluded
"""

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Feature sets
PRIMARY_ENERGY_FEATURES = ["hourly_active_power"]
PRIMARY_WEATHER_FEATURES = [
    "temp",
    "feels_like",
    "humidity",
    "pressure",
    "speed",
    "wind_sin",
    "wind_cos",
]
METADATA_COLUMNS = ["date", "n_observations", "coverage_ratio", "is_valid_hour"]
OUTPUT_COLUMNS = [
    "date",
    "hourly_active_power",
    "temp",
    "feels_like",
    "humidity",
    "pressure",
    "speed",
    "wind_sin",
    "wind_cos",
    "n_observations",
    "coverage_ratio",
    "is_valid_hour",
]

# Forbidden columns for safety
FORBIDDEN_COLUMNS = ["temp_t+1", "feels_like_t+1"]
EXCLUDED_SECONDARY = [
    "current",
    "voltage",
    "reactive_power",
    "apparent_power",
    "power_factor",
    "main",
    "description",
    "temp_min",
    "temp_max",
]


def load_split_boundaries(splits_dir: str) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Load temporal boundaries (start, end) for each split from split window CSVs.

    Parameters
    ----------
    splits_dir : str
        Path to directory containing train_windows.csv, validation_windows.csv, etc.

    Returns
    -------
    dict
        Mapping of split name ('train', 'validation', 'calibration', 'test')
        to (start_timestamp, end_timestamp).
    """
    splits = ["train", "validation", "calibration", "test"]
    boundaries = {}

    for s in splits:
        file_path = os.path.join(splits_dir, f"{s}_windows.csv")
        assert os.path.exists(file_path), f"Split file not found: {file_path}"
        w_df = pd.read_csv(file_path, parse_dates=["history_start", "target_end"])
        start_ts = w_df["history_start"].min()
        end_ts = w_df["target_end"].max()
        boundaries[s] = (start_ts, end_ts)

    return boundaries


def get_training_hourly_region(
    hourly_df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_only: bool = True,
) -> pd.DataFrame:
    """
    Extract unique hourly observations from the training temporal region.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Full forecast_ready_hourly DataFrame.
    train_start : pd.Timestamp
        Earliest history timestamp in train windows.
    train_end : pd.Timestamp
        Latest target timestamp in train windows.
    valid_only : bool
        If True, only keep rows where is_valid_hour is True.

    Returns
    -------
    pd.DataFrame
        Unique training hourly observations.
    """
    mask = (hourly_df["date"] >= train_start) & (hourly_df["date"] <= train_end)
    if valid_only:
        mask = mask & (hourly_df["is_valid_hour"])

    train_obs = hourly_df[mask].copy().reset_index(drop=True)
    return train_obs


def check_constant_features(df: pd.DataFrame, feature_columns: List[str]) -> None:
    """
    Verify no feature has zero variance before fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Training observations.
    feature_columns : list of str
        Features to inspect.
    """
    zero_var_features = []
    for col in feature_columns:
        var = df[col].var()
        if np.isclose(var, 0.0) or np.isnan(var):
            zero_var_features.append(col)

    if zero_var_features:
        raise ValueError(
            f"Zero variance detected in training features: {zero_var_features}. "
            "Cannot fit StandardScaler on constant feature!"
        )


def fit_energy_scaler(
    train_obs: pd.DataFrame,
    energy_features: List[str] = PRIMARY_ENERGY_FEATURES,
) -> StandardScaler:
    """
    Fit StandardScaler on unique training energy observations.

    Parameters
    ----------
    train_obs : pd.DataFrame
        Unique valid training observations.
    energy_features : list of str
        Energy feature columns (hourly_active_power).

    Returns
    -------
    StandardScaler
        Fitted energy scaler.
    """
    check_constant_features(train_obs, energy_features)
    scaler = StandardScaler()
    scaler.fit(train_obs[energy_features])
    return scaler


def fit_weather_scaler(
    train_obs: pd.DataFrame,
    weather_features: List[str] = PRIMARY_WEATHER_FEATURES,
) -> StandardScaler:
    """
    Fit StandardScaler on unique training weather observations.

    Parameters
    ----------
    train_obs : pd.DataFrame
        Unique valid training observations.
    weather_features : list of str
        Weather feature columns.

    Returns
    -------
    StandardScaler
        Fitted weather scaler.
    """
    check_constant_features(train_obs, weather_features)
    scaler = StandardScaler()
    scaler.fit(train_obs[weather_features])
    return scaler


def transform_partition(
    partition_df: pd.DataFrame,
    energy_scaler: StandardScaler,
    weather_scaler: StandardScaler,
    energy_features: List[str] = PRIMARY_ENERGY_FEATURES,
    weather_features: List[str] = PRIMARY_WEATHER_FEATURES,
) -> pd.DataFrame:
    """
    Apply fitted scalers to transform numerical features in a partition.

    Metadata columns are preserved unscaled.
    NaNs (if any in invalid hours) are preserved as NaNs.

    Parameters
    ----------
    partition_df : pd.DataFrame
        Partition hourly DataFrame.
    energy_scaler : StandardScaler
        Fitted energy scaler.
    weather_scaler : StandardScaler
        Fitted weather scaler.
    energy_features : list of str
        Energy features to transform.
    weather_features : list of str
        Weather features to transform.

    Returns
    -------
    pd.DataFrame
        Scaled DataFrame with OUTPUT_COLUMNS.
    """
    df = partition_df.copy()

    # Verify no forbidden columns
    for col in FORBIDDEN_COLUMNS:
        assert col not in df.columns, f"LEAKAGE: {col} in partition DataFrame!"

    # Transform energy features
    scaled_energy = energy_scaler.transform(df[energy_features])
    for i, col in enumerate(energy_features):
        df[col] = scaled_energy[:, i]

    # Transform weather features
    scaled_weather = weather_scaler.transform(df[weather_features])
    for i, col in enumerate(weather_features):
        df[col] = scaled_weather[:, i]

    return df[OUTPUT_COLUMNS].copy()


def save_scalers(
    energy_scaler: StandardScaler,
    weather_scaler: StandardScaler,
    output_dir: str,
) -> Tuple[str, str]:
    """
    Save fitted scaler objects to disk using joblib.

    Parameters
    ----------
    energy_scaler : StandardScaler
        Fitted energy scaler.
    weather_scaler : StandardScaler
        Fitted weather scaler.
    output_dir : str
        Directory to save scalers.

    Returns
    -------
    tuple of (energy_scaler_path, weather_scaler_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    energy_path = os.path.join(output_dir, "energy_scaler.pkl")
    weather_path = os.path.join(output_dir, "weather_scaler.pkl")

    joblib.dump(energy_scaler, energy_path)
    joblib.dump(weather_scaler, weather_path)

    print(f"  Saved energy scaler  -> {energy_path}")
    print(f"  Saved weather scaler -> {weather_path}")
    return energy_path, weather_path


def load_scalers(output_dir: str) -> Tuple[StandardScaler, StandardScaler]:
    """
    Load fitted scaler objects from disk.

    Parameters
    ----------
    output_dir : str
        Directory containing energy_scaler.pkl and weather_scaler.pkl.

    Returns
    -------
    tuple of (energy_scaler, weather_scaler)
    """
    energy_path = os.path.join(output_dir, "energy_scaler.pkl")
    weather_path = os.path.join(output_dir, "weather_scaler.pkl")

    assert os.path.exists(energy_path), f"Energy scaler not found: {energy_path}"
    assert os.path.exists(weather_path), f"Weather scaler not found: {weather_path}"

    energy_scaler = joblib.load(energy_path)
    weather_scaler = joblib.load(weather_path)
    return energy_scaler, weather_scaler


def generate_scaler_metadata(
    energy_scaler: StandardScaler,
    weather_scaler: StandardScaler,
    energy_features: List[str],
    weather_features: List[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    n_obs: int,
    output_path: str,
) -> Dict[str, Any]:
    """
    Create and save machine-readable scaler metadata JSON.

    Parameters
    ----------
    energy_scaler : StandardScaler
        Fitted energy scaler.
    weather_scaler : StandardScaler
        Fitted weather scaler.
    energy_features : list of str
        Names of energy features.
    weather_features : list of str
        Names of weather features.
    train_start : pd.Timestamp
        Earliest training timestamp used for fitting.
    train_end : pd.Timestamp
        Latest training timestamp used for fitting.
    n_obs : int
        Number of unique training observations.
    output_path : str
        Path to save metadata JSON.

    Returns
    -------
    dict
        Metadata dictionary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metadata = {
        "project_step": "3.8",
        "scaler_type": "StandardScaler",
        "fit_scope": "train_only",
        "training_fit_start": str(train_start),
        "training_fit_end": str(train_end),
        "n_unique_training_observations": int(n_obs),
        "energy_scaler": {
            "features": energy_features,
            "means": {feat: float(m) for feat, m in zip(energy_features, energy_scaler.mean_)},
            "standard_deviations": {
                feat: float(s) for feat, s in zip(energy_features, energy_scaler.scale_)
            },
        },
        "weather_scaler": {
            "features": weather_features,
            "means": {feat: float(m) for feat, m in zip(weather_features, weather_scaler.mean_)},
            "standard_deviations": {
                feat: float(s) for feat, s in zip(weather_features, weather_scaler.scale_)
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Scaler metadata saved -> {output_path}")
    return metadata


def validate_scaling_leakage(
    original_train: pd.DataFrame,
    scaled_train: pd.DataFrame,
    scaled_val: pd.DataFrame,
    scaled_cal: pd.DataFrame,
    scaled_test: pd.DataFrame,
    energy_scaler: StandardScaler,
    weather_scaler: StandardScaler,
    energy_features: List[str] = PRIMARY_ENERGY_FEATURES,
    weather_features: List[str] = PRIMARY_WEATHER_FEATURES,
) -> Dict[str, Any]:
    """
    Validate scaling integrity, numerical round-trip, and leakage prevention.

    Returns
    -------
    dict
        Audit results including round-trip difference and partition statistics.
    """
    all_features = energy_features + weather_features

    # 1. Round-trip numerical check on training data
    train_valid_orig = original_train[original_train["is_valid_hour"]]
    train_valid_scaled = scaled_train[scaled_train["is_valid_hour"]]

    energy_rec = energy_scaler.inverse_transform(train_valid_scaled[energy_features])
    weather_rec = weather_scaler.inverse_transform(train_valid_scaled[weather_features])

    energy_diff = np.max(np.abs(energy_rec - train_valid_orig[energy_features].values))
    weather_diff = np.max(np.abs(weather_rec - train_valid_orig[weather_features].values))
    max_roundtrip_diff = max(float(energy_diff), float(weather_diff))

    assert max_roundtrip_diff < 1e-10, (
        f"Round-trip reconstruction error too high: {max_roundtrip_diff}"
    )

    # 2. Partition statistics calculation (on valid hours)
    stats = {}
    partitions = {
        "train": scaled_train[scaled_train["is_valid_hour"]],
        "validation": scaled_val[scaled_val["is_valid_hour"]],
        "calibration": scaled_cal[scaled_cal["is_valid_hour"]],
        "test": scaled_test[scaled_test["is_valid_hour"]],
    }

    for name, p_df in partitions.items():
        p_stats = {}
        for feat in all_features:
            p_stats[feat] = {
                "mean": float(p_df[feat].mean()),
                "std": float(p_df[feat].std(ddof=0)),
                "min": float(p_df[feat].min()),
                "max": float(p_df[feat].max()),
            }
        stats[name] = p_stats

    # Check TRAIN valid statistics: mean ≈ 0, std ≈ 1
    for feat in all_features:
        t_m = stats["train"][feat]["mean"]
        t_s = stats["train"][feat]["std"]
        assert np.isclose(t_m, 0.0, atol=1e-5), f"Train mean for {feat} != 0: {t_m}"
        assert np.isclose(t_s, 1.0, atol=1e-5), f"Train std for {feat} != 1: {t_s}"

    return {
        "max_roundtrip_difference": max_roundtrip_diff,
        "roundtrip_passed": True,
        "partition_statistics": stats,
    }
