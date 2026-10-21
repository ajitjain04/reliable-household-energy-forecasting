"""
Weather Retrieval Service.

Integrates with Open-Meteo APIs to automatically retrieve historical weather observations:
- Resolves appropriate endpoint:
    * Forecast API (api.open-meteo.com) for recent past days (latency ~0, up to 92 days).
    * Historical Archive API (archive-api.open-meteo.com) for older dates (ERA5 reanalysis, >= 7 days old).
- Configures wind speed in metres per second (wind_speed_unit=ms).
- Strictly aligns returned weather records to the exact 168 electricity timestamps.
- Maps API fields to the 7 audited features:
    temp, feels_like, humidity, pressure, speed, wind_sin, wind_cos
- Converts wind direction in degrees to sine and cosine.
- Gracefully handles calm winds (speed == 0 m/s) with missing direction by assigning (sin=0.0, cos=1.0).
- Rejects incomplete or missing meteorological observations without silent imputation.
- Employs finite HTTP timeouts and robust error handling for 429 rate limits, 5xx server errors, and network failures.
"""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import numpy as np
import pandas as pd

from backend.app.schemas.forecasts import WeatherSeriesInput


class WeatherService:
    """Service retrieving and normalizing historical weather data from Open-Meteo."""

    FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
    ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"
    DEFAULT_TIMEOUT_SECONDS = 5.0

    def __init__(self, client: Optional[httpx.Client] = None, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self._client = client
        self.timeout = timeout

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self.timeout)

    @staticmethod
    def _normalize_key(dt: pd.Timestamp) -> str:
        """Produce canonical UTC YYYY-MM-DDTHH:00 key for exact timestamp matching."""
        if dt.tzinfo is not None:
            dt_utc = dt.tz_convert("UTC")
        else:
            dt_utc = dt.tz_localize("UTC")
        return dt_utc.strftime("%Y-%m-%dT%H:00")

    def determine_endpoint(self, max_dt: pd.Timestamp) -> Tuple[str, Optional[str]]:
        """
        Select primary and fallback Open-Meteo endpoints based on timestamp age.

        ERA5 reanalysis in the Archive API has an inherent publication delay of 5-7 days.
        For recent observations within the last 7 days, the Forecast API must be used.
        """
        now = datetime.now(timezone.utc)
        if max_dt.tzinfo is not None:
            dt_utc = max_dt.tz_convert("UTC") if hasattr(max_dt, "tz_convert") else max_dt.astimezone(timezone.utc)
            age_days = (now - dt_utc.to_pydatetime()).total_seconds() / 86400.0
        else:
            age_days = (datetime.now(timezone.utc).replace(tzinfo=None) - max_dt.to_pydatetime()).total_seconds() / 86400.0

        if age_days < 7.0:
            # Recent past: Forecast API has immediate operational availability
            return self.FORECAST_API_URL, self.ARCHIVE_API_URL
        elif age_days <= 92.0:
            # 7 to 92 days: Archive API (ERA5 reanalysis) preferred, Forecast API as fallback
            return self.ARCHIVE_API_URL, self.FORECAST_API_URL
        else:
            # Older than 92 days: Archive API is mandatory
            return self.ARCHIVE_API_URL, None

    def fetch_weather_for_timestamps(
        self,
        latitude: float,
        longitude: float,
        timestamps: List[str],
    ) -> Tuple[Optional[WeatherSeriesInput], Optional[str]]:
        """
        Fetch and strictly align 168 hours of weather matching the given electricity timestamps.

        Returns:
            Tuple of (WeatherSeriesInput, error_message).
            If successful, (weather_input, None).
            If unsuccessful, (None, error_message).
        """
        # 1. Validate coordinates
        if not (-90.0 <= latitude <= 90.0):
            return None, f"Invalid latitude {latitude}: must be between -90.0 and 90.0"
        if not (-180.0 <= longitude <= 180.0):
            return None, f"Invalid longitude {longitude}: must be between -180.0 and 180.0"

        # 2. Validate timestamps sequence
        if not timestamps or len(timestamps) != 168:
            return None, f"Weather retrieval requires exactly 168 timestamps (received {len(timestamps) if timestamps else 0})"

        try:
            parsed_dts = pd.to_datetime(timestamps)
            if parsed_dts.isna().any():
                return None, "Timestamps contain unparseable datetime entries"
        except Exception:
            try:
                parsed_dts = pd.to_datetime(timestamps, format="mixed")
                if parsed_dts.isna().any():
                    return None, "Timestamps contain unparseable datetime entries"
            except Exception as e:
                return None, f"Failed to parse electricity timestamps: {str(e)}"

        # Verify hourly continuity
        diffs = pd.Series(parsed_dts).diff()[1:]
        expected_hour = pd.Timedelta(hours=1)
        irregular = diffs[diffs != expected_hour]
        if not irregular.empty:
            bad_idx = irregular.index[0]
            return None, f"Timestamps must be consecutive hourly intervals (delta at index {bad_idx} was {diffs[bad_idx]})"

        # Preserve exact physical instant: convert timezone-aware timestamps to UTC,
        # or localize naive timestamps to UTC without shifting time.
        utc_dts = [
            dt.tz_convert("UTC") if dt.tzinfo is not None else dt.tz_localize("UTC")
            for dt in parsed_dts
        ]
        min_utc: pd.Timestamp = min(utc_dts)
        max_utc: pd.Timestamp = max(utc_dts)

        start_date = min_utc.strftime("%Y-%m-%d")
        end_date = max_utc.strftime("%Y-%m-%d")

        # Open-Meteo documented standard timezone format
        tz_param = "UTC"

        # Determine endpoints based on the UTC instant age
        primary_endpoint, fallback_endpoint = self.determine_endpoint(max_utc)

        params: Dict[str, Any] = {
            "latitude": round(float(latitude), 4),
            "longitude": round(float(longitude), 4),
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms",
            "timezone": tz_param,
        }

        # 3. Query Open-Meteo with fallback
        response_data: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None

        endpoints_to_try = [primary_endpoint]
        if fallback_endpoint and fallback_endpoint != primary_endpoint:
            endpoints_to_try.append(fallback_endpoint)

        for endpoint_url in endpoints_to_try:
            try:
                if self._client is not None:
                    resp = self._client.get(endpoint_url, params=params)
                else:
                    with httpx.Client(timeout=self.timeout) as client:
                        resp = client.get(endpoint_url, params=params)

                if resp.status_code == 200:
                    response_data = resp.json()
                    break
                elif resp.status_code == 429:
                    last_error = "Open-Meteo API rate limit exceeded (HTTP 429)"
                    break  # Do not retry another endpoint if rate-limited
                else:
                    last_error = f"Open-Meteo API returned HTTP {resp.status_code}: {resp.text[:200]}"
            except httpx.TimeoutException:
                last_error = f"Open-Meteo API connection timed out after {self.timeout}s"
            except httpx.RequestError as req_err:
                last_error = f"Open-Meteo network error: {str(req_err)}"
            except Exception as exc:
                last_error = f"Unexpected error querying Open-Meteo: {str(exc)}"

        if response_data is None:
            return None, last_error or "Open-Meteo weather data could not be retrieved"

        # 4. Parse and match hourly observations
        if "hourly" not in response_data or not isinstance(response_data["hourly"], dict):
            return None, "Open-Meteo response missing 'hourly' observation data"

        hourly = response_data["hourly"]
        time_series = hourly.get("time")
        if not time_series or not isinstance(time_series, list):
            return None, "Open-Meteo response missing 'time' index in hourly data"

        # Build lookup table indexed by canonical key
        weather_by_key: Dict[str, Dict[str, Any]] = {}
        for idx, t_str in enumerate(time_series):
            # Open-Meteo times are formatted like '2024-03-01T00:00'
            norm_key = str(t_str).strip()
            if " " in norm_key and "T" not in norm_key:
                norm_key = norm_key.replace(" ", "T")
            if len(norm_key) == 16:  # 'YYYY-MM-DDTHH:MM'
                pass
            elif len(norm_key) > 16:
                norm_key = norm_key[:16]

            weather_by_key[norm_key] = {
                "temp": hourly.get("temperature_2m", [None])[idx] if idx < len(hourly.get("temperature_2m", [])) else None,
                "feels_like": hourly.get("apparent_temperature", [None])[idx] if idx < len(hourly.get("apparent_temperature", [])) else None,
                "humidity": hourly.get("relative_humidity_2m", [None])[idx] if idx < len(hourly.get("relative_humidity_2m", [])) else None,
                "pressure": hourly.get("surface_pressure", [None])[idx] if idx < len(hourly.get("surface_pressure", [])) else None,
                "speed": hourly.get("wind_speed_10m", [None])[idx] if idx < len(hourly.get("wind_speed_10m", [])) else None,
                "deg": hourly.get("wind_direction_10m", [None])[idx] if idx < len(hourly.get("wind_direction_10m", [])) else None,
            }

        # 5. Extract and strictly align the 168 hourly steps
        out_temp: List[float] = []
        out_feels_like: List[float] = []
        out_humidity: List[float] = []
        out_pressure: List[float] = []
        out_speed: List[float] = []
        out_wind_sin: List[float] = []
        out_wind_cos: List[float] = []

        for orig_ts, dt in zip(timestamps, parsed_dts):
            norm_ts = self._normalize_key(dt)
            if norm_ts not in weather_by_key:
                return None, f"No weather observation found matching timestamp '{orig_ts}' (normalized key '{norm_ts}')"

            obs = weather_by_key[norm_ts]

            # Primary features check
            for feat in ["temp", "feels_like", "humidity", "pressure", "speed"]:
                val = obs[feat]
                if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                    return None, f"Invalid or missing '{feat}' observation at timestamp '{orig_ts}'"

            temp_val = float(obs["temp"])
            feels_like_val = float(obs["feels_like"])
            humidity_val = float(obs["humidity"])
            pressure_val = float(obs["pressure"])
            speed_val = float(obs["speed"])

            # Wind direction conversion with calm-wind safeguard
            deg_val = obs["deg"]
            if deg_val is None or not isinstance(deg_val, (int, float)) or not math.isfinite(deg_val):
                # Calm wind check: if wind speed is 0 m/s, wind direction is physically undefined
                if math.isclose(speed_val, 0.0, abs_tol=1e-5):
                    wind_sin_val = 0.0
                    wind_cos_val = 1.0
                else:
                    return None, f"Missing wind direction observation for non-calm wind ({speed_val:.2f} m/s) at timestamp '{orig_ts}'"
            else:
                rad = math.radians(float(deg_val))
                wind_sin_val = float(math.sin(rad))
                wind_cos_val = float(math.cos(rad))

            out_temp.append(temp_val)
            out_feels_like.append(feels_like_val)
            out_humidity.append(humidity_val)
            out_pressure.append(pressure_val)
            out_speed.append(speed_val)
            out_wind_sin.append(wind_sin_val)
            out_wind_cos.append(wind_cos_val)

        weather_series = WeatherSeriesInput(
            temp=out_temp,
            feels_like=out_feels_like,
            humidity=out_humidity,
            pressure=out_pressure,
            speed=out_speed,
            wind_sin=out_wind_sin,
            wind_cos=out_wind_cos,
        )

        return weather_series, None


# Singleton instance
weather_service = WeatherService()
