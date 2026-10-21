"""
Hourly aggregation module for the household energy forecasting project.

Converts minute-level cleaned data into an hourly interim dataset.

Key design decisions:
  - Primary target: hourly_active_power = hourly mean of active_power
  - Wind direction uses circular (sin/cos) encoding, not arithmetic mean of deg
  - Future-weather columns (temp_t+1, feels_like_t+1) are excluded
  - Categorical weather columns (main, description) are excluded
  - temp_min, temp_max are excluded from the primary feature set
  - A complete hourly index is constructed so missing hours appear with
    n_observations=0 and coverage_ratio=0
  - No interpolation or filling is performed
"""

import os
from io import StringIO

import numpy as np
import pandas as pd


# ---- Columns excluded from the primary hourly dataset ----
FUTURE_WEATHER_COLUMNS = ["temp_t+1", "feels_like_t+1"]
EXCLUDED_WEATHER_COLUMNS = ["main", "description", "temp_min", "temp_max"]

# ---- Expected raw observations per hour (1-minute sampling) ----
EXPECTED_OBS_PER_HOUR = 60

# ---- Columns to aggregate with hourly mean ----
PRIMARY_TARGET_RAW = "active_power"
PRIMARY_TARGET_HOURLY = "hourly_active_power"

SECONDARY_ELECTRICAL = [
    "current", "voltage", "reactive_power", "apparent_power", "power_factor",
]

PRIMARY_WEATHER_NUMERIC = [
    "temp", "feels_like", "humidity", "pressure", "speed",
]


def add_wind_circular_features(df: pd.DataFrame,
                               deg_column: str = "deg") -> pd.DataFrame:
    """
    Convert wind direction (degrees) to circular sin/cos components.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a wind-direction column in degrees.
    deg_column : str
        Name of the column containing wind direction in degrees.

    Returns
    -------
    pd.DataFrame with added columns: wind_sin, wind_cos
    """
    rad = np.deg2rad(df[deg_column].astype(float))
    df = df.copy()
    df["wind_sin"] = np.sin(rad)
    df["wind_cos"] = np.cos(rad)
    return df


def aggregate_to_hourly(df: pd.DataFrame,
                        timestamp_column: str = "date") -> pd.DataFrame:
    """
    Aggregate minute-level data to hourly resolution using mean.

    The aggregation groups by the floored hour of each timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned minute-level DataFrame with wind_sin, wind_cos already added.
    timestamp_column : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame with one row per hour, containing:
        - hourly_active_power (mean of active_power)
        - secondary electrical means
        - primary weather means
        - wind_sin, wind_cos means
        - n_observations (count of raw records in that hour)
    """
    # Define columns to aggregate
    agg_columns = (
        [PRIMARY_TARGET_RAW]
        + SECONDARY_ELECTRICAL
        + PRIMARY_WEATHER_NUMERIC
        + ["wind_sin", "wind_cos"]
    )

    # Floor timestamps to the hour
    df = df.copy()
    df["hour"] = df[timestamp_column].dt.floor("h")

    # Build aggregation dictionary: mean for all feature columns
    agg_dict = {col: "mean" for col in agg_columns}

    hourly = df.groupby("hour").agg(**{
        **{col: pd.NamedAgg(column=col, aggfunc="mean") for col in agg_columns},
        "n_observations": pd.NamedAgg(column=timestamp_column, aggfunc="count"),
    })

    # Rename primary target
    hourly = hourly.rename(columns={PRIMARY_TARGET_RAW: PRIMARY_TARGET_HOURLY})

    # Reset index so hour becomes a regular column named 'date'
    hourly = hourly.reset_index().rename(columns={"hour": "date"})

    return hourly


def build_complete_hourly_index(hourly_df: pd.DataFrame,
                                timestamp_column: str = "date") -> pd.DataFrame:
    """
    Reindex the hourly DataFrame onto a complete hourly range so that
    hours with no observations appear with NaN values and n_observations=0.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Aggregated hourly data (may have gaps).
    timestamp_column : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame with a continuous hourly index.
    """
    min_hour = hourly_df[timestamp_column].min()
    max_hour = hourly_df[timestamp_column].max()
    complete_index = pd.date_range(start=min_hour, end=max_hour, freq="h")

    hourly_df = hourly_df.set_index(timestamp_column)
    hourly_df = hourly_df.reindex(complete_index)
    hourly_df.index.name = timestamp_column
    hourly_df = hourly_df.reset_index()

    # Fill n_observations with 0 for missing hours (no fabricated values)
    hourly_df["n_observations"] = hourly_df["n_observations"].fillna(0).astype(int)

    return hourly_df


def calculate_hourly_coverage(hourly_df: pd.DataFrame,
                              expected_per_hour: int = EXPECTED_OBS_PER_HOUR
                              ) -> pd.DataFrame:
    """
    Add coverage_ratio column: n_observations / expected_per_hour.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly DataFrame with n_observations column.
    expected_per_hour : int
        Expected number of raw observations per complete hour.

    Returns
    -------
    pd.DataFrame with added coverage_ratio column.
    """
    hourly_df = hourly_df.copy()
    hourly_df["coverage_ratio"] = hourly_df["n_observations"] / expected_per_hour
    return hourly_df


def save_hourly_dataset(hourly_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the hourly interim dataset to CSV.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Complete hourly DataFrame to save.
    output_path : str
        File path for the output CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hourly_df.to_csv(output_path, index=False)
    print(f"  Hourly dataset saved to: {output_path}")
    print(f"  Shape: {hourly_df.shape}")


def generate_coverage_report(hourly_df: pd.DataFrame,
                             report_path: str) -> str:
    """
    Generate and save a human-readable hourly coverage report.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Complete hourly DataFrame with n_observations and coverage_ratio.
    report_path : str
        Path to save the text report.

    Returns
    -------
    str containing the report text.
    """
    buf = StringIO()
    w = buf.write

    total = len(hourly_df)
    obs = hourly_df["n_observations"]
    cov = hourly_df["coverage_ratio"]

    w("HOURLY COVERAGE REPORT\n")
    w("=" * 70 + "\n\n")

    w("OVERVIEW\n")
    w("-" * 70 + "\n")
    w(f"  Total hourly slots          : {total:,}\n")
    w(f"  Date range                  : {hourly_df['date'].min()} to {hourly_df['date'].max()}\n\n")

    # Observation-count buckets
    w("OBSERVATION COUNT DISTRIBUTION\n")
    w("-" * 70 + "\n")
    n_60 = int((obs == 60).sum())
    n_54_59 = int(((obs >= 54) & (obs <= 59)).sum())
    n_30_53 = int(((obs >= 30) & (obs <= 53)).sum())
    n_1_29 = int(((obs >= 1) & (obs <= 29)).sum())
    n_0 = int((obs == 0).sum())
    n_gt60 = int((obs > 60).sum())

    w(f"  60 observations (complete)  : {n_60:,}  ({100.0*n_60/total:.2f}%)\n")
    w(f"  54-59 observations          : {n_54_59:,}  ({100.0*n_54_59/total:.2f}%)\n")
    w(f"  30-53 observations          : {n_30_53:,}  ({100.0*n_30_53/total:.2f}%)\n")
    w(f"  1-29 observations           : {n_1_29:,}  ({100.0*n_1_29/total:.2f}%)\n")
    w(f"  0 observations (missing)    : {n_0:,}  ({100.0*n_0/total:.2f}%)\n")
    w(f"  >60 observations (ANOMALY)  : {n_gt60:,}  ({100.0*n_gt60/total:.2f}%)\n\n")

    # Coverage ratio statistics
    w("COVERAGE RATIO STATISTICS\n")
    w("-" * 70 + "\n")
    w(f"  Minimum                     : {cov.min():.4f}\n")
    w(f"  Maximum                     : {cov.max():.4f}\n")
    w(f"  Mean                        : {cov.mean():.4f}\n")
    w(f"  Median                      : {cov.median():.4f}\n")
    w(f"  Std                         : {cov.std():.4f}\n\n")

    w("COVERAGE QUANTILES\n")
    w("-" * 70 + "\n")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        w(f"  {q*100:5.1f}th percentile         : {cov.quantile(q):.4f}\n")
    w("\n")

    # Candidate threshold analysis
    w("CANDIDATE THRESHOLD ANALYSIS\n")
    w("-" * 70 + "\n")
    for threshold in [0.80, 0.90, 0.95]:
        qualifying = int((cov >= threshold).sum())
        w(f"  >= {threshold*100:.0f}% coverage             : "
          f"{qualifying:,} hours qualify  "
          f"({100.0*qualifying/total:.2f}% of total)\n")
    w("\n")

    w("=" * 70 + "\n")
    w("END OF COVERAGE REPORT\n")

    report_text = buf.getvalue()

    # Save report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Coverage report saved to: {report_path}")

    return report_text
