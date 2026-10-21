"""
Forecasting window construction for the household energy project (Step 3.6).

Creates and validates 192-hour forecasting windows:
  - 168 hours of historical context (energy + weather)
  - 24 hours of forecast horizon (target: hourly_active_power)

Key rules:
  - Windows are constructed only within continuous valid runs
  - A window must contain exactly 192 consecutive valid hours
  - No window may cross an invalid hour or a timestamp gap
  - No future-weather columns may appear in the feature set
  - No scaling is applied at this step
  - No random shuffling occurs; all data remains chronological
"""

import os

import numpy as np
import pandas as pd


# ---- Forecasting parameters ----
HISTORY_LENGTH = 168   # hours of historical context
FORECAST_HORIZON = 24  # hours to forecast
WINDOW_SIZE = HISTORY_LENGTH + FORECAST_HORIZON  # 192

# ---- Forbidden columns ----
FORBIDDEN_COLUMNS = ["temp_t+1", "feels_like_t+1"]

# ---- Primary target ----
TARGET_COLUMN = "hourly_active_power"


def count_valid_windows_in_run(run_length: int) -> int:
    """
    Count how many valid 192-hour windows fit in a continuous run.

    The first window needs 168 history hours + 24 target hours = 192 hours.
    Each subsequent window slides by 1 hour.

    Parameters
    ----------
    run_length : int
        Number of consecutive valid hours in the run.

    Returns
    -------
    int: number of valid forecasting windows.
    """
    if run_length < WINDOW_SIZE:
        return 0
    return run_length - WINDOW_SIZE + 1


def audit_forecasting_windows(df: pd.DataFrame,
                              timestamp_column: str = "date") -> dict:
    """
    Audit how many valid 192-hour forecasting windows exist
    without actually constructing them.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly dataset with run_id and is_valid_hour columns.
    timestamp_column : str
        Name of the datetime column.

    Returns
    -------
    dict with audit results.
    """
    total_rows = len(df)
    valid_rows = int(df["is_valid_hour"].sum())
    invalid_rows = total_rows - valid_rows

    # Run statistics
    valid_df = df[df["run_id"].notna()]
    if valid_df.empty:
        return {
            "total_rows": total_rows,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "num_runs": 0,
            "total_windows": 0,
        }

    run_lengths = valid_df.groupby("run_id").size()
    num_runs = len(run_lengths)
    longest_run = int(run_lengths.max())
    shortest_run = int(run_lengths.min())

    # Count windows per run
    windows_per_run = run_lengths.apply(count_valid_windows_in_run)
    total_windows = int(windows_per_run.sum())

    # Find earliest and latest forecast origins
    earliest_origin = None
    latest_origin = None

    if total_windows > 0:
        for run_id, run_len in run_lengths.items():
            n_windows = count_valid_windows_in_run(run_len)
            if n_windows > 0:
                run_data = valid_df[valid_df["run_id"] == run_id].sort_values(
                    timestamp_column
                )
                run_dates = run_data[timestamp_column].values

                # Forecast origin is the last hour of history
                # First window: origin at index HISTORY_LENGTH - 1
                first_origin = run_dates[HISTORY_LENGTH - 1]
                # Last window: origin at index run_len - FORECAST_HORIZON - 1
                last_origin = run_dates[run_len - FORECAST_HORIZON - 1]

                if earliest_origin is None or first_origin < earliest_origin:
                    earliest_origin = first_origin
                if latest_origin is None or last_origin > latest_origin:
                    latest_origin = last_origin

    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "valid_pct": round(100.0 * valid_rows / total_rows, 2),
        "num_runs": num_runs,
        "longest_run": longest_run,
        "shortest_run": shortest_run,
        "total_windows": total_windows,
        "earliest_forecast_origin": str(earliest_origin) if earliest_origin else None,
        "latest_forecast_origin": str(latest_origin) if latest_origin else None,
    }


def create_forecast_windows(df: pd.DataFrame,
                            timestamp_column: str = "date") -> pd.DataFrame:
    """
    Create a window index of all valid 192-hour forecasting windows.

    Each row in the output represents one valid window with metadata
    (no duplicated feature arrays are stored).

    Parameters
    ----------
    df : pd.DataFrame
        Hourly dataset with run_id, is_valid_hour, and timestamp columns.
    timestamp_column : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame with columns:
        window_id, forecast_origin, history_start, history_end,
        target_start, target_end
    """
    # Verify no forbidden columns
    for col in FORBIDDEN_COLUMNS:
        assert col not in df.columns, (
            f"LEAKAGE: Forbidden column '{col}' found!"
        )

    valid_df = df[df["run_id"].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame(columns=[
            "window_id", "forecast_origin",
            "history_start", "history_end",
            "target_start", "target_end",
        ])

    windows = []
    window_id = 0

    for run_id, run_group in valid_df.groupby("run_id"):
        run_data = run_group.sort_values(timestamp_column).reset_index(drop=True)
        run_dates = run_data[timestamp_column].values
        run_len = len(run_data)

        n_windows = count_valid_windows_in_run(run_len)
        if n_windows == 0:
            continue

        for i in range(n_windows):
            # History: indices i to i + HISTORY_LENGTH - 1
            history_start = run_dates[i]
            history_end = run_dates[i + HISTORY_LENGTH - 1]  # forecast origin

            # Target: indices i + HISTORY_LENGTH to i + WINDOW_SIZE - 1
            target_start = run_dates[i + HISTORY_LENGTH]
            target_end = run_dates[i + WINDOW_SIZE - 1]

            windows.append({
                "window_id": window_id,
                "forecast_origin": history_end,
                "history_start": history_start,
                "history_end": history_end,
                "target_start": target_start,
                "target_end": target_end,
            })
            window_id += 1

    window_df = pd.DataFrame(windows)
    return window_df


def validate_window_index(window_df: pd.DataFrame,
                          hourly_df: pd.DataFrame,
                          timestamp_column: str = "date") -> list[str]:
    """
    Validate the constructed window index against the hourly dataset.

    Checks a sample of windows (up to 100) to verify:
      - Exactly 168 historical timestamps
      - Exactly 24 target timestamps
      - Consecutive hourly timestamps in history and target
      - Target starts exactly 1 hour after history end
      - No invalid hours in the window span

    Parameters
    ----------
    window_df : pd.DataFrame
        Window index from create_forecast_windows().
    hourly_df : pd.DataFrame
        Full hourly dataset with is_valid_hour column.
    timestamp_column : str
        Name of the datetime column.

    Returns
    -------
    list of error strings (empty if all checks pass).
    """
    errors = []
    one_hour = pd.Timedelta(hours=1)

    # Sample windows for validation (check all if <= 100, else sample 100)
    if len(window_df) <= 100:
        sample = window_df
    else:
        # Deterministic sample: first 50 + last 50
        sample = pd.concat([window_df.head(50), window_df.tail(50)])

    hourly_indexed = hourly_df.set_index(timestamp_column)

    for _, row in sample.iterrows():
        wid = row["window_id"]
        h_start = pd.Timestamp(row["history_start"])
        h_end = pd.Timestamp(row["history_end"])
        t_start = pd.Timestamp(row["target_start"])
        t_end = pd.Timestamp(row["target_end"])

        # Check history length
        history_range = pd.date_range(start=h_start, end=h_end, freq="h")
        if len(history_range) != HISTORY_LENGTH:
            errors.append(
                f"Window {wid}: history has {len(history_range)} hours, "
                f"expected {HISTORY_LENGTH}"
            )

        # Check target length
        target_range = pd.date_range(start=t_start, end=t_end, freq="h")
        if len(target_range) != FORECAST_HORIZON:
            errors.append(
                f"Window {wid}: target has {len(target_range)} hours, "
                f"expected {FORECAST_HORIZON}"
            )

        # Check target starts 1 hour after history ends
        if t_start != h_end + one_hour:
            errors.append(
                f"Window {wid}: target_start {t_start} != "
                f"history_end + 1h ({h_end + one_hour})"
            )

        # Check all hours in the full span are valid
        full_range = pd.date_range(start=h_start, end=t_end, freq="h")
        for ts in full_range:
            if ts in hourly_indexed.index:
                if not hourly_indexed.loc[ts, "is_valid_hour"]:
                    errors.append(
                        f"Window {wid}: invalid hour at {ts}"
                    )
            else:
                errors.append(
                    f"Window {wid}: missing hour at {ts}"
                )

    return errors


def save_window_index(window_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the window index to CSV.

    Parameters
    ----------
    window_df : pd.DataFrame
        Window index from create_forecast_windows().
    output_path : str
        Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    window_df.to_csv(output_path, index=False)
    print(f"  Window index saved to: {output_path}")
    print(f"  Total windows: {len(window_df)}")


# =============================================================================
# Step 3.9: Model-Ready Sequence Extraction and Validation
# =============================================================================

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


def extract_partition_sequences(
    window_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    energy_features: list[str] = PRIMARY_ENERGY_FEATURES,
    weather_features: list[str] = PRIMARY_WEATHER_FEATURES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D model-ready sequences from scaled hourly data using window metadata.

    Parameters
    ----------
    window_df : pd.DataFrame
        Window metadata with history_start, history_end, target_start, target_end.
    scaled_df : pd.DataFrame
        Scaled hourly dataset with date column and scaled features.
    energy_features : list of str
        Energy feature column names (1 feature: hourly_active_power).
    weather_features : list of str
        Weather feature column names (7 features).

    Returns
    -------
    tuple of (X_energy, X_weather, y)
        X_energy: [N, 168, 1]
        X_weather: [N, 168, 7]
        y: [N, 24]
    """
    # Verify chronological ordering
    assert window_df["forecast_origin"].is_monotonic_increasing, (
        "Windows are not sorted chronologically!"
    )

    n_samples = len(window_df)
    n_energy_feats = len(energy_features)
    n_weather_feats = len(weather_features)

    # Preallocate NumPy arrays (float32 for ML/DL)
    X_energy = np.zeros((n_samples, HISTORY_LENGTH, n_energy_feats), dtype=np.float32)
    X_weather = np.zeros((n_samples, HISTORY_LENGTH, n_weather_feats), dtype=np.float32)
    y = np.zeros((n_samples, FORECAST_HORIZON), dtype=np.float32)

    # Ensure datetime on date column and build fast index lookup
    scaled_df = scaled_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(scaled_df["date"]):
        scaled_df["date"] = pd.to_datetime(scaled_df["date"])

    date_to_pos = {ts: i for i, ts in enumerate(scaled_df["date"])}
    energy_vals = scaled_df[energy_features].values.astype(np.float32)
    weather_vals = scaled_df[weather_features].values.astype(np.float32)
    target_vals = scaled_df[TARGET_COLUMN].values.astype(np.float32)
    is_valid_arr = scaled_df["is_valid_hour"].values

    for i, (_, row) in enumerate(window_df.iterrows()):
        h_start = pd.Timestamp(row["history_start"])
        h_end = pd.Timestamp(row["history_end"])
        t_start = pd.Timestamp(row["target_start"])
        t_end = pd.Timestamp(row["target_end"])
        f_origin = pd.Timestamp(row["forecast_origin"])

        # Timestamp boundary assertions
        assert h_start in date_to_pos, f"Missing timestamp {h_start}"
        assert h_end in date_to_pos, f"Missing timestamp {h_end}"
        assert t_start in date_to_pos, f"Missing timestamp {t_start}"
        assert t_end in date_to_pos, f"Missing timestamp {t_end}"

        p_h_start = date_to_pos[h_start]
        p_h_end = date_to_pos[h_end]
        p_t_start = date_to_pos[t_start]
        p_t_end = date_to_pos[t_end]

        # Span assertions
        assert p_h_end - p_h_start + 1 == HISTORY_LENGTH, (
            f"History span {p_h_end - p_h_start + 1} != {HISTORY_LENGTH}"
        )
        assert p_t_end - p_t_start + 1 == FORECAST_HORIZON, (
            f"Target span {p_t_end - p_t_start + 1} != {FORECAST_HORIZON}"
        )
        assert p_t_start == p_h_end + 1, "Target does not immediately follow history!"
        assert f_origin == h_end, "Forecast origin does not match history end!"

        # Continuity and validity: no invalid hours in the 192-hour window
        assert is_valid_arr[p_h_start : p_t_end + 1].all(), (
            f"Window {i} spans invalid hours!"
        )

        # Slice arrays
        X_energy[i] = energy_vals[p_h_start : p_h_end + 1]
        X_weather[i] = weather_vals[p_h_start : p_h_end + 1]
        y[i] = target_vals[p_t_start : p_t_end + 1]

    # Sanity checks on constructed arrays
    assert not np.isnan(X_energy).any(), "NaN in X_energy!"
    assert not np.isnan(X_weather).any(), "NaN in X_weather!"
    assert not np.isnan(y).any(), "NaN in y!"
    assert not np.isinf(X_energy).any(), "Inf in X_energy!"
    assert not np.isinf(X_weather).any(), "Inf in X_weather!"
    assert not np.isinf(y).any(), "Inf in y!"

    return X_energy, X_weather, y


def save_partition_sequences(
    output_dir: str,
    X_energy: np.ndarray,
    X_weather: np.ndarray,
    y: np.ndarray,
    metadata: dict,
) -> None:
    """
    Save partition sequences to compressed NumPy .npz files and metadata.json.

    Parameters
    ----------
    output_dir : str
        Directory to save partition files.
    X_energy : np.ndarray
        Energy history tensor [N, 168, 1].
    X_weather : np.ndarray
        Weather history tensor [N, 168, 7].
    y : np.ndarray
        Target tensor [N, 24].
    metadata : dict
        Partition metadata dictionary.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "X_energy.npz"), data=X_energy, arr_0=X_energy
    )
    np.savez_compressed(
        os.path.join(output_dir, "X_weather.npz"), data=X_weather, arr_0=X_weather
    )
    np.savez_compressed(os.path.join(output_dir, "y.npz"), data=y, arr_0=y)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved sequences & metadata -> {output_dir}")


def load_partition_sequences(
    partition_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load partition sequences and metadata.

    Parameters
    ----------
    partition_dir : str
        Directory containing X_energy.npz, X_weather.npz, y.npz, metadata.json.

    Returns
    -------
    tuple of (X_energy, X_weather, y, metadata)
    """
    import json

    x_e = np.load(os.path.join(partition_dir, "X_energy.npz"))["data"]
    x_w = np.load(os.path.join(partition_dir, "X_weather.npz"))["data"]
    y = np.load(os.path.join(partition_dir, "y.npz"))["data"]

    with open(os.path.join(partition_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return x_e, x_w, y, metadata


def audit_sample_alignment(
    X_energy: np.ndarray,
    X_weather: np.ndarray,
    y: np.ndarray,
    window_df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    partition_name: str,
) -> list[dict]:
    """
    Audit first, middle, and last sample values against source scaled hourly rows.

    Parameters
    ----------
    X_energy : np.ndarray
        [N, 168, 1]
    X_weather : np.ndarray
        [N, 168, 7]
    y : np.ndarray
        [N, 24]
    window_df : pd.DataFrame
        Window metadata.
    scaled_df : pd.DataFrame
        Source scaled hourly data.
    partition_name : str
        Name of partition.

    Returns
    -------
    list of dict
        Audit report for inspected samples.
    """
    scaled_indexed = scaled_df.set_index("date")
    n = len(window_df)
    sample_indices = [0, n // 2, n - 1]
    labels = ["first", "middle", "last"]
    audits = []

    for idx, label in zip(sample_indices, labels):
        row = window_df.iloc[idx]
        h_start = pd.Timestamp(row["history_start"])
        h_end = pd.Timestamp(row["history_end"])
        t_start = pd.Timestamp(row["target_start"])
        t_end = pd.Timestamp(row["target_end"])

        # Check energy values
        expected_e_first = scaled_indexed.loc[h_start, "hourly_active_power"]
        expected_e_last = scaled_indexed.loc[h_end, "hourly_active_power"]
        actual_e_first = X_energy[idx, 0, 0]
        actual_e_last = X_energy[idx, -1, 0]

        assert np.isclose(actual_e_first, expected_e_first, atol=1e-5)
        assert np.isclose(actual_e_last, expected_e_last, atol=1e-5)

        # Check weather values (temp)
        expected_temp_first = scaled_indexed.loc[h_start, "temp"]
        expected_temp_last = scaled_indexed.loc[h_end, "temp"]
        actual_temp_first = X_weather[idx, 0, 0]
        actual_temp_last = X_weather[idx, -1, 0]

        assert np.isclose(actual_temp_first, expected_temp_first, atol=1e-5)
        assert np.isclose(actual_temp_last, expected_temp_last, atol=1e-5)

        # Check target values
        expected_y_first = scaled_indexed.loc[t_start, "hourly_active_power"]
        expected_y_last = scaled_indexed.loc[t_end, "hourly_active_power"]
        actual_y_first = y[idx, 0]
        actual_y_last = y[idx, -1]

        assert np.isclose(actual_y_first, expected_y_first, atol=1e-5)
        assert np.isclose(actual_y_last, expected_y_last, atol=1e-5)

        audits.append({
            "partition": partition_name,
            "sample_label": label,
            "sample_index": int(idx),
            "forecast_origin": str(row["forecast_origin"]),
            "history_start": str(h_start),
            "history_end": str(h_end),
            "target_start": str(t_start),
            "target_end": str(t_end),
            "energy_h_start_val": float(actual_e_first),
            "energy_h_end_val": float(actual_e_last),
            "temp_h_start_val": float(actual_temp_first),
            "temp_h_end_val": float(actual_temp_last),
            "y_t_start_val": float(actual_y_first),
            "y_t_end_val": float(actual_y_last),
            "alignment_verified": True,
        })

    return audits

