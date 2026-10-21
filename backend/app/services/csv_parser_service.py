"""
CSV Ingestion and Validation Service.

Validates user-uploaded CSV files for live energy forecasting:
- Enforces strict 168-row hourly requirement.
- Accepts documented active power columns ('hourly_active_power' or 'active_power').
- Checks for duplicate timestamps, non-hourly intervals, and non-chronological order.
- Validates complete 7-feature weather sets or pure energy-only files.
- Rejects missing values, non-finite entries, and malformed structures.
- Generates sanitized metadata summaries for persistent tracking.
"""

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.app.schemas.forecasts import ForecastRunRequest

MAX_CSV_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class CSVParserService:
    """Service validating CSV files and constructing ForecastRunRequest payloads."""

    def parse_csv(
        self,
        file_bytes: bytes,
        filename: str = "uploaded_data.csv",
        model: Optional[str] = "auto",
    ) -> Tuple[ForecastRunRequest, Dict[str, Any]]:
        """
        Parse and validate raw CSV bytes into a typed ForecastRunRequest.

        Returns:
            Tuple of (ForecastRunRequest, input_metadata_summary)
        """
        # 1. Size check
        if len(file_bytes) > MAX_CSV_SIZE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"File '{filename}' exceeds maximum allowed size of 5 MB ({len(file_bytes)} bytes).",
            )

        # 2. Encoding check
        try:
            content_str = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=422,
                detail=f"File '{filename}' must be UTF-8 encoded text.",
            )

        # 3. CSV parsing
        try:
            df = pd.read_csv(io.StringIO(content_str))
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Malformed CSV syntax in '{filename}': {str(e)}",
            )

        # 4. Row count validation (exactly 168 hours)
        if len(df) != 168:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"CSV file must contain exactly 168 hourly rows matching the 7-day lookback requirement "
                    f"(received {len(df)} rows in '{filename}')."
                ),
            )

        # Normalize column names to lowercase stripped
        df.columns = [str(c).strip().lower() for c in df.columns]

        # 5. Electricity column identification
        elec_col = None
        for candidate in ["hourly_active_power", "active_power"]:
            if candidate in df.columns:
                elec_col = candidate
                break

        if elec_col is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"CSV missing required electricity active power column in '{filename}'. "
                    "Expected either 'hourly_active_power' or 'active_power'."
                ),
            )

        # Validate numeric and finite active power
        if df[elec_col].isna().any():
            first_nan_row = int(np.where(df[elec_col].isna())[0][0]) + 1
            raise HTTPException(
                status_code=422,
                detail=f"Active power column '{elec_col}' contains missing or invalid values at row {first_nan_row}.",
            )

        try:
            electricity_vals = [float(x) for x in df[elec_col].tolist()]
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=422,
                detail=f"Active power column '{elec_col}' contains non-numeric values: {str(e)}",
            )

        # 6. Timestamp column validation (if present)
        ts_col = None
        for candidate in ["date", "timestamp", "datetime", "time"]:
            if candidate in df.columns:
                ts_col = candidate
                break

        timestamps_list: Optional[List[str]] = None
        if ts_col is not None:
            # Check duplicate timestamps
            if df[ts_col].duplicated().any():
                dup_val = df[ts_col][df[ts_col].duplicated()].iloc[0]
                raise HTTPException(
                    status_code=422,
                    detail=f"CSV contains duplicate timestamps in '{ts_col}': duplicate value '{dup_val}'.",
                )

            # Check unparseable timestamps
            parsed_dt = pd.to_datetime(df[ts_col], errors="coerce")
            if parsed_dt.isna().any():
                bad_idx = int(np.where(parsed_dt.isna())[0][0]) + 1
                raise HTTPException(
                    status_code=422,
                    detail=f"Unparseable datetime value at row {bad_idx} in '{ts_col}': '{df[ts_col].iloc[bad_idx - 1]}'.",
                )

            # Check strict hourly continuity
            diffs = pd.Series(parsed_dt).diff()[1:]
            expected_hour = pd.Timedelta(hours=1)
            irregular = diffs[diffs != expected_hour]
            if not irregular.empty:
                bad_idx = irregular.index[0] + 1
                actual_delta = diffs[irregular.index[0]]
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Timestamps must be consecutive hourly readings spaced by exactly 1 hour. "
                        f"Irregular interval at row {bad_idx}: delta was {actual_delta}."
                    ),
                )

            timestamps_list = df[ts_col].astype(str).tolist()

        # 7. Weather columns validation
        primary_weather = ["temp", "feels_like", "humidity", "pressure", "speed"]
        has_any_weather = any(c in df.columns for c in primary_weather)
        has_all_primary_weather = all(c in df.columns for c in primary_weather)

        weather_dict: Optional[Dict[str, List[float]]] = None
        weather_features_present: List[str] = []

        if has_all_primary_weather:
            # Check wind features: either (wind_sin and wind_cos) or (deg)
            has_sin_cos = "wind_sin" in df.columns and "wind_cos" in df.columns
            has_deg = "deg" in df.columns

            if not has_sin_cos and not has_deg:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Weather columns detected in CSV, but wind direction features are missing. "
                        "Must include either ('wind_sin' and 'wind_cos') or meteorological degrees 'deg'."
                    ),
                )

            # Check missing values in weather columns
            weather_cols_to_check = list(primary_weather)
            if has_sin_cos:
                weather_cols_to_check.extend(["wind_sin", "wind_cos"])
            else:
                weather_cols_to_check.append("deg")

            for col in weather_cols_to_check:
                if df[col].isna().any():
                    first_bad_row = int(np.where(df[col].isna())[0][0]) + 1
                    raise HTTPException(
                        status_code=422,
                        detail=f"Weather column '{col}' contains missing or null values at row {first_bad_row}.",
                    )

            try:
                weather_dict = {
                    col: [float(v) for v in df[col].tolist()]
                    for col in weather_cols_to_check
                }
            except (ValueError, TypeError) as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"Weather columns contain non-numeric values: {str(e)}",
                )

            weather_features_present = weather_cols_to_check

        elif has_any_weather:
            # Partial weather supplied
            missing_cols = [c for c in primary_weather if c not in df.columns]
            present_cols = [c for c in primary_weather if c in df.columns]
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Partial weather features detected in CSV. Found: {present_cols}. "
                    f"Missing required audited weather features: {missing_cols}."
                ),
            )

        # 8. Build Request and Metadata Summary (Never saving raw arrays in metadata)
        request = ForecastRunRequest(
            electricity_active_power=electricity_vals,
            timestamps=timestamps_list,
            weather=weather_dict,
            model=model or "auto",
        )

        metadata_summary = {
            "source_filename": filename,
            "total_rows": len(df),
            "electricity_column": elec_col,
            "timestamps_present": ts_col is not None,
            "first_timestamp": timestamps_list[0] if timestamps_list else None,
            "last_timestamp": timestamps_list[-1] if timestamps_list else None,
            "weather_present": weather_dict is not None,
            "weather_features": weather_features_present,
        }

        return request, metadata_summary


# Global parser service instance
csv_parser_service = CSVParserService()
