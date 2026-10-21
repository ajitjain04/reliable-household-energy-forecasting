"""
Data cleaning utilities for the household energy forecasting project.

Handles:
  - Exact duplicate row removal (keeping the first occurrence)
  - Identification and reporting of remaining duplicate timestamps

The raw CSV is NEVER modified by this module.
"""

import os
import pandas as pd


def remove_exact_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove completely duplicated rows, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (typically loaded from the raw CSV).

    Returns
    -------
    tuple of (cleaned DataFrame, number of duplicate rows removed)
    """
    n_before = len(df)
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_removed = n_before - len(df_clean)
    return df_clean, n_removed


def inspect_duplicate_timestamps(df: pd.DataFrame,
                                 timestamp_column: str = "date",
                                 save_path: str | None = None) -> pd.DataFrame:
    """
    After exact duplicate removal, find timestamps that still appear
    more than once (non-identical duplicate timestamps).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after exact duplicate removal.
    timestamp_column : str
        Name of the timestamp column.
    save_path : str or None
        If provided, save the conflicting records to this CSV path.

    Returns
    -------
    pd.DataFrame containing only the rows with duplicate timestamps.
        Empty DataFrame if no duplicate timestamps remain.
    """
    dup_mask = df[timestamp_column].duplicated(keep=False)
    dup_records = df[dup_mask].copy()

    if len(dup_records) > 0 and save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dup_records.to_csv(save_path, index=False)
        print(f"  Saved {len(dup_records)} duplicate-timestamp records to: {save_path}")

    return dup_records
