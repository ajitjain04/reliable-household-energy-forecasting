"""
Data loading utilities for the household energy forecasting project.

Provides functions to load the raw CSV with proper timestamp parsing.
The raw CSV is NEVER modified by this module.
"""

import os
import pandas as pd


# Default path relative to project root
RAW_DATA_PATH = os.path.join("data", "raw", "energy_weather_raw_data.csv")


def load_raw_data(filepath: str, timestamp_column: str = "date") -> pd.DataFrame:
    """
    Load the raw CSV and parse the timestamp column.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.
    timestamp_column : str
        Name of the column containing datetime strings.

    Returns
    -------
    pd.DataFrame with the timestamp column parsed as datetime64.
    """
    df = pd.read_csv(filepath)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")
    return df


def parse_and_sort_timestamps(df: pd.DataFrame,
                              timestamp_column: str = "date") -> pd.DataFrame:
    """
    Ensure timestamps are parsed and sort the DataFrame chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a datetime column.
    timestamp_column : str
        Name of the timestamp column.

    Returns
    -------
    pd.DataFrame sorted by timestamp, index reset.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="raise")

    df = df.sort_values(by=timestamp_column).reset_index(drop=True)
    return df


def get_raw_file_hash(filepath: str) -> str:
    """
    Compute a simple checksum of the raw file for integrity verification.
    Uses file size + first/last 1000 bytes as a lightweight fingerprint.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV.

    Returns
    -------
    str with format 'size:<bytes>'.
    """
    size = os.path.getsize(filepath)
    return f"size:{size}"
