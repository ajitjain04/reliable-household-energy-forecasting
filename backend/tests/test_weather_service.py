"""
Comprehensive unit and integration test suite for the Weather Retrieval Service
and automatic historical weather integration.

Covers:
1. Coordinate validation (latitude [-90, 90], longitude [-180, 180]).
2. Hourly timestamp sequence validation (168 timestamps, hourly intervals).
3. Open-Meteo endpoint routing (Forecast API for recent past, Archive API for >= 7 days).
4. Exact 1-to-1 hourly timestamp alignment between electricity and weather observations.
5. Unit and feature mapping:
   - temperature_2m -> temp
   - apparent_temperature -> feels_like
   - relative_humidity_2m -> humidity
   - surface_pressure -> pressure
   - wind_speed_10m -> speed (requested in m/s via wind_speed_unit=ms)
   - wind_direction_10m -> wind_sin and wind_cos
6. Trigonometric wind direction conversions and calm-wind safeguard (speed=0 m/s with null direction -> sin=0, cos=1).
7. Rejection of missing/incomplete required meteorological observations.
8. Error handling: HTTP 429 rate limit, 5xx server errors, connection timeouts, network failures.
9. Precedence rule: user-supplied weather takes priority, never querying Open-Meteo.
10. Forecast pipeline integration in 'auto' mode:
    - Successful retrieval -> Early Fusion auto-selected with explanatory reason.
    - Retrieval failure -> Graceful fallback to Energy-Only with explanatory reason.
    - Missing timestamps -> Energy-Only preserved without querying API.
11. Explicit multimodal override policy:
    - Requested multimodal model with failed retrieval -> HTTP 502 Bad Gateway (never silent model switch).
    - Requested multimodal model with successful retrieval -> Executes requested model.
12. CSV upload integration with optional location form fields.
13. Safe testing practices: All tests use isolated disposable SQLite databases and mocked HTTP clients.
"""

import io
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import backend.app.db.database
from backend.app.core.config import settings
from backend.app.db.database import DatabaseManager
from backend.app.main import app
from backend.app.schemas.forecasts import (
    ForecastRunRequest,
    LocationInput,
    WeatherSeriesInput,
)
from backend.app.services.inference_service import APPROVED_MODELS, inference_service
from backend.app.services.record_service import record_service
from backend.app.services.user_service import user_service
from backend.app.services.weather_service import WeatherService, weather_service


def _build_synthetic_open_meteo_response(
    timestamps: List[str],
    temp_base: float = 20.0,
    calm_indices: List[int] = None,
) -> Dict[str, Any]:
    """Helper to generate a valid Open-Meteo hourly JSON response aligned to given timestamps."""
    calm_indices = calm_indices or []
    times = []
    temps = []
    feels = []
    humidities = []
    pressures = []
    speeds = []
    directions = []

    for idx, ts in enumerate(timestamps):
        dt = pd.to_datetime(ts)
        if dt.tzinfo is not None:
            dt_utc = dt.tz_convert("UTC")
        else:
            dt_utc = dt.tz_localize("UTC")
        norm_t = dt_utc.strftime("%Y-%m-%dT%H:00")
        times.append(norm_t)

        temps.append(temp_base + 3.0 * math.sin(idx / 12.0))
        feels.append(temp_base - 1.0 + 3.0 * math.sin(idx / 12.0))
        humidities.append(50.0 + 10.0 * math.cos(idx / 24.0))
        pressures.append(1013.25 + 2.0 * math.sin(idx / 48.0))

        if idx in calm_indices:
            speeds.append(0.0)
            directions.append(None)  # Open-Meteo returns null for calm wind direction
        else:
            speeds.append(3.5 + 1.0 * math.sin(idx / 6.0))
            # Vary direction cleanly across quadrants
            directions.append((idx * 15.0) % 360.0)

    return {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "generationtime_ms": 1.25,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "apparent_temperature": "°C",
            "relative_humidity_2m": "%",
            "surface_pressure": "hPa",
            "wind_speed_10m": "m/s",
            "wind_direction_10m": "°",
        },
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "apparent_temperature": feels,
            "relative_humidity_2m": humidities,
            "surface_pressure": pressures,
            "wind_speed_10m": speeds,
            "wind_direction_10m": directions,
        },
    }


class TestWeatherServiceUnit(unittest.TestCase):
    """Isolated unit tests for WeatherService logic and Open-Meteo API handling."""

    def setUp(self):
        self.service = WeatherService(timeout=2.0)
        base_time = datetime(2024, 6, 1, 0, 0, 0)
        self.valid_timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]

    # 1. Coordinate Validation
    def test_invalid_latitude_rejected(self):
        """Latitudes beyond [-90.0, 90.0] must be rejected."""
        res, err = self.service.fetch_weather_for_timestamps(95.0, 10.0, self.valid_timestamps)
        self.assertIsNone(res)
        self.assertIn("Invalid latitude", err)

        res, err = self.service.fetch_weather_for_timestamps(-90.1, 10.0, self.valid_timestamps)
        self.assertIsNone(res)
        self.assertIn("Invalid latitude", err)

    def test_invalid_longitude_rejected(self):
        """Longitudes beyond [-180.0, 180.0] must be rejected."""
        res, err = self.service.fetch_weather_for_timestamps(10.0, 185.0, self.valid_timestamps)
        self.assertIsNone(res)
        self.assertIn("Invalid longitude", err)

        res, err = self.service.fetch_weather_for_timestamps(10.0, -180.5, self.valid_timestamps)
        self.assertIsNone(res)
        self.assertIn("Invalid longitude", err)

    # 2. Timestamp Sequence Validation
    def test_invalid_timestamp_count_rejected(self):
        """Fewer or greater than 168 timestamps must return a clear error."""
        res, err = self.service.fetch_weather_for_timestamps(48.8, 2.3, self.valid_timestamps[:100])
        self.assertIsNone(res)
        self.assertIn("requires exactly 168 timestamps", err)

        res, err = self.service.fetch_weather_for_timestamps(48.8, 2.3, [])
        self.assertIsNone(res)
        self.assertIn("requires exactly 168 timestamps", err)

    def test_irregular_timestamp_gap_rejected(self):
        """Non-hourly gaps must be rejected without calling Open-Meteo."""
        bad_timestamps = list(self.valid_timestamps)
        # Introduce a 2-hour gap at index 20
        t19 = datetime.strptime(bad_timestamps[19], "%Y-%m-%d %H:%M:%S")
        bad_timestamps[20] = (t19 + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")

        res, err = self.service.fetch_weather_for_timestamps(48.8, 2.3, bad_timestamps)
        self.assertIsNone(res)
        self.assertIn("consecutive hourly intervals", err)

    # 3. Endpoint Routing Logic
    def test_endpoint_routing_recent_vs_archive(self):
        """Routing must select Forecast API for recent days and Archive API for older dates."""
        now = datetime.now(timezone.utc)

        # Recent timestamp (2 days ago)
        recent_dt = pd.Timestamp(now - timedelta(days=2))
        primary, fallback = self.service.determine_endpoint(recent_dt)
        self.assertEqual(primary, WeatherService.FORECAST_API_URL)
        self.assertEqual(fallback, WeatherService.ARCHIVE_API_URL)

        # Archive range (20 days ago)
        mid_dt = pd.Timestamp(now - timedelta(days=20))
        primary, fallback = self.service.determine_endpoint(mid_dt)
        self.assertEqual(primary, WeatherService.ARCHIVE_API_URL)
        self.assertEqual(fallback, WeatherService.FORECAST_API_URL)

        # Deep archive (150 days ago)
        old_dt = pd.Timestamp(now - timedelta(days=150))
        primary, fallback = self.service.determine_endpoint(old_dt)
        self.assertEqual(primary, WeatherService.ARCHIVE_API_URL)
        self.assertIsNone(fallback)

    # 4. Successful Retrieval, Alignment, and Field Mapping
    def test_successful_weather_retrieval_and_field_mapping(self):
        """Verify successful response parsing, field mapping, and wind conversion."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(err)
        self.assertIsNotNone(weather_out)
        self.assertIsInstance(weather_out, WeatherSeriesInput)

        # Verify all 7 feature arrays are present and have length 168
        for feat in ["temp", "feels_like", "humidity", "pressure", "speed", "wind_sin", "wind_cos"]:
            arr = getattr(weather_out, feat)
            self.assertEqual(len(arr), 168)
            self.assertTrue(all(math.isfinite(x) for x in arr))

        # Verify Open-Meteo call parameters
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args[1]
        params = call_kwargs["params"]
        self.assertEqual(params["wind_speed_unit"], "ms")
        self.assertEqual(params["latitude"], 48.8566)
        self.assertEqual(params["longitude"], 2.3522)
        self.assertIn("temperature_2m", params["hourly"])

    # 5. Trigonometric Wind Direction Conversion
    def test_wind_direction_trigonometric_accuracy(self):
        """Test specific wind cardinal angles (0°, 90°, 180°, 270°) convert correctly to sin and cos."""
        # Custom mock response with known wind angles for first 4 hours
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)
        mock_payload["hourly"]["wind_speed_10m"][:4] = [5.0, 5.0, 5.0, 5.0]
        mock_payload["hourly"]["wind_direction_10m"][:4] = [0.0, 90.0, 180.0, 270.0]

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, _ = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        # 0°: sin(0) = 0, cos(0) = 1
        self.assertAlmostEqual(weather_out.wind_sin[0], 0.0, places=5)
        self.assertAlmostEqual(weather_out.wind_cos[0], 1.0, places=5)

        # 90°: sin(90) = 1, cos(90) = 0
        self.assertAlmostEqual(weather_out.wind_sin[1], 1.0, places=5)
        self.assertAlmostEqual(weather_out.wind_cos[1], 0.0, places=5)

        # 180°: sin(180) = 0, cos(180) = -1
        self.assertAlmostEqual(weather_out.wind_sin[2], 0.0, places=5)
        self.assertAlmostEqual(weather_out.wind_cos[2], -1.0, places=5)

        # 270°: sin(270) = -1, cos(270) = 0
        self.assertAlmostEqual(weather_out.wind_sin[3], -1.0, places=5)
        self.assertAlmostEqual(weather_out.wind_cos[3], 0.0, places=5)

    # 6. Calm Wind Safeguard vs Non-Calm Wind Null Direction
    def test_calm_wind_with_null_direction_handled_safely(self):
        """When wind speed is 0.0 m/s and direction is null, map to sin=0.0, cos=1.0."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps, calm_indices=[10, 25])

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(err)
        self.assertIsNotNone(weather_out)
        self.assertEqual(weather_out.speed[10], 0.0)
        self.assertEqual(weather_out.wind_sin[10], 0.0)
        self.assertEqual(weather_out.wind_cos[10], 1.0)

    def test_non_calm_wind_missing_direction_rejected(self):
        """When wind speed is > 0.0 m/s and direction is null, request must be rejected."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)
        # Wind is 4.5 m/s but direction is None
        mock_payload["hourly"]["wind_speed_10m"][15] = 4.5
        mock_payload["hourly"]["wind_direction_10m"][15] = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(weather_out)
        self.assertIn("Missing wind direction observation for non-calm wind", err)

    # 7. Missing Meteorological Observations
    def test_missing_required_feature_rejected(self):
        """If any core observation (e.g. surface_pressure) contains null/NaN, reject."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)
        mock_payload["hourly"]["surface_pressure"][40] = None

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(weather_out)
        self.assertIn("Invalid or missing 'pressure' observation", err)

    # 8. Missing Timestamps in Weather Response
    def test_missing_timestamp_alignment_rejected(self):
        """If Open-Meteo response omits a required timestamp, reject without silent interpolation."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)
        # Remove the last hour from Open-Meteo response
        for k in mock_payload["hourly"]:
            mock_payload["hourly"][k] = mock_payload["hourly"][k][:-1]

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(weather_out)
        self.assertIn("No weather observation found matching timestamp", err)

    # 9. HTTP Errors, Rate Limits, and Timeouts
    def test_rate_limit_http_429(self):
        """HTTP 429 rate limit must return clear error and not retry another endpoint."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(weather_out)
        self.assertIn("rate limit exceeded (HTTP 429)", err)
        self.assertEqual(mock_client.get.call_count, 1)

    def test_api_server_error_with_fallback(self):
        """HTTP 500 on primary endpoint should trigger fallback endpoint."""
        recent_base = datetime.now(timezone.utc) - timedelta(days=20)
        recent_ts = [
            (recent_base + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]
        mock_payload = _build_synthetic_open_meteo_response(recent_ts)

        mock_client = MagicMock(spec=httpx.Client)
        resp_500 = MagicMock(spec=httpx.Response)
        resp_500.status_code = 500
        resp_500.text = "Internal Server Error"

        resp_200 = MagicMock(spec=httpx.Response)
        resp_200.status_code = 200
        resp_200.json.return_value = mock_payload

        mock_client.get.side_effect = [resp_500, resp_200]

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, recent_ts)

        self.assertIsNone(err)
        self.assertIsNotNone(weather_out)
        self.assertEqual(mock_client.get.call_count, 2)

    def test_http_timeout_handling(self):
        """TimeoutException must be caught and reported gracefully."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")

        service = WeatherService(client=mock_client, timeout=3.0)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(weather_out)
        self.assertIn("timed out after 3.0s", err)

    def test_network_request_error_handling(self):
        """httpx.RequestError must be caught and reported gracefully."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("Network unreachable")

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(48.8566, 2.3522, self.valid_timestamps)

        self.assertIsNone(weather_out)
        self.assertIn("Open-Meteo network error", err)

    # 9. Timezone Normalization and Instant Preservation
    def test_timezone_naive_timestamps_handling(self):
        """Naive timestamps must query Open-Meteo with timezone='UTC' and align all 168 hours."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(18.5204, 73.8567, self.valid_timestamps)

        self.assertIsNone(err)
        self.assertIsNotNone(weather_out)
        self.assertEqual(len(weather_out.temp), 168)

        # Verify Open-Meteo request parameters
        call_params = mock_client.get.call_args[1]["params"]
        self.assertEqual(call_params["timezone"], "UTC")
        self.assertEqual(call_params["start_date"], "2024-06-01")
        self.assertEqual(call_params["end_date"], "2024-06-07")

    def test_timezone_utc_timestamps_handling(self):
        """Explicit UTC timestamps (Z suffix) must query Open-Meteo with timezone='UTC' and align."""
        base_time = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        utc_timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(168)
        ]
        mock_payload = _build_synthetic_open_meteo_response(utc_timestamps)

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(51.5074, -0.1278, utc_timestamps)

        self.assertIsNone(err)
        self.assertIsNotNone(weather_out)
        self.assertEqual(len(weather_out.temp), 168)

        call_params = mock_client.get.call_args[1]["params"]
        self.assertEqual(call_params["timezone"], "UTC")
        self.assertEqual(call_params["start_date"], "2024-06-01")
        self.assertEqual(call_params["end_date"], "2024-06-07")

    def test_timezone_offset_plus_0530_preserves_actual_instant(self):
        """
        Timezone-aware timestamps with +05:30 offset (e.g. IST) must NOT be silently reinterpreted
        as local clock time in UTC. The physical instant (05:30 IST -> 00:00 UTC) must be preserved.
        """
        ist = timezone(timedelta(hours=5, minutes=30))
        base_time_ist = datetime(2024, 6, 1, 5, 30, 0, tzinfo=ist)
        ist_timestamps = [
            (base_time_ist + timedelta(hours=i)).isoformat()
            for i in range(168)
        ]

        # Construct synthetic Open-Meteo response where 00:00 UTC and 05:00 UTC have distinct temperatures
        mock_payload = _build_synthetic_open_meteo_response(ist_timestamps)
        # In mock_payload, time[0] is '2024-06-01T00:00' because ist_timestamps[0] is 00:00 UTC
        mock_payload["hourly"]["temperature_2m"][0] = 22.25
        mock_payload["hourly"]["apparent_temperature"][0] = 21.50
        mock_payload["hourly"]["surface_pressure"][0] = 1010.5

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(19.0760, 72.8777, ist_timestamps)

        self.assertIsNone(err)
        self.assertIsNotNone(weather_out)
        self.assertEqual(len(weather_out.temp), 168)

        # Confirm request params
        call_params = mock_client.get.call_args[1]["params"]
        self.assertEqual(call_params["timezone"], "UTC")
        self.assertEqual(call_params["start_date"], "2024-06-01")

        # The first observation (2024-06-01 05:30:00+05:30) must receive the 00:00 UTC observation
        self.assertEqual(weather_out.temp[0], 22.25)
        self.assertEqual(weather_out.feels_like[0], 21.50)
        self.assertEqual(weather_out.pressure[0], 1010.5)

    def test_timezone_offset_crossing_date_boundary(self):
        """
        When a +05:30 timestamp (e.g. 02:00 IST) crosses into the previous calendar day in UTC (20:30 UTC),
        the Open-Meteo start_date must reflect the UTC date (2024-05-31) to prevent clipping.
        """
        ist = timezone(timedelta(hours=5, minutes=30))
        # 02:00:00 IST on June 1 is 20:30:00 UTC on May 31
        base_time_ist = datetime(2024, 6, 1, 2, 0, 0, tzinfo=ist)
        ist_timestamps = [
            (base_time_ist + timedelta(hours=i)).isoformat()
            for i in range(168)
        ]

        mock_payload = _build_synthetic_open_meteo_response(ist_timestamps)

        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload
        mock_client.get.return_value = mock_resp

        service = WeatherService(client=mock_client)
        weather_out, err = service.fetch_weather_for_timestamps(19.0760, 72.8777, ist_timestamps)

        self.assertIsNone(err)
        self.assertIsNotNone(weather_out)
        self.assertEqual(len(weather_out.temp), 168)

        call_params = mock_client.get.call_args[1]["params"]
        self.assertEqual(call_params["timezone"], "UTC")
        # In UTC, 02:00 IST is 20:30 on 2024-05-31
        self.assertEqual(call_params["start_date"], "2024-05-31")


class TestLiveWeatherIntegrationAPI(unittest.TestCase):
    """
    API integration tests verifying end-to-end forecast execution with automatic weather retrieval.
    Operates against a safe, isolated disposable SQLite database.
    """

    @classmethod
    def setUpClass(cls):
        # 1. Establish isolated disposable database
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "test_weather_disposable.db"
        cls.db_mgr = DatabaseManager(cls.db_path)
        cls.db_mgr.init_db()

        # Save previous manager bindings
        cls._orig_db_mgr = backend.app.db.database.db_manager
        cls._orig_rec_db = record_service.db
        cls._orig_usr_db = user_service._db

        # Rebind to disposable database
        backend.app.db.database.db_manager = cls.db_mgr
        record_service.db = cls.db_mgr
        user_service._db = cls.db_mgr

        # 2. Test client setup
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Register and authenticate test user
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": "weather_test_household", "password": "Password123!"},
        )
        login_res = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": "weather_test_household", "password": "Password123!"},
        )
        cls.token = login_res.json()["access_token"]
        cls.auth_headers = {"Authorization": f"Bearer {cls.token}"}

        # 3. Base test payloads
        base_time = datetime(2024, 6, 1, 0, 0, 0)
        cls.valid_timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]
        cls.valid_electricity = [float(260.0 + 40.0 * math.sin(i / 12.0)) for i in range(168)]

        # Valid manual weather series
        cls.valid_weather_series = {
            "temp": [21.0 + 4.0 * math.cos(i / 12.0) for i in range(168)],
            "feels_like": [20.0 + 4.0 * math.cos(i / 12.0) for i in range(168)],
            "humidity": [50.0 + 10.0 * math.sin(i / 24.0) for i in range(168)],
            "pressure": [1012.0 + 3.0 * math.cos(i / 48.0) for i in range(168)],
            "speed": [2.8 + 1.0 * math.sin(i / 6.0) for i in range(168)],
            "wind_sin": [math.sin(i * 0.1) for i in range(168)],
            "wind_cos": [math.cos(i * 0.1) for i in range(168)],
        }

    @classmethod
    def tearDownClass(cls):
        import gc
        backend.app.db.database.db_manager = cls._orig_db_mgr
        record_service.db = cls._orig_rec_db
        user_service._db = cls._orig_usr_db
        cls.client.close()
        gc.collect()
        try:
            cls.temp_dir.cleanup()
        except Exception:
            pass

    # 1. Automatic Weather Retrieval in 'auto' Mode
    def test_auto_mode_with_location_retrieves_weather_and_selects_early_fusion(self):
        """When location is provided without weather in 'auto' mode, retrieve weather and select Early Fusion."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)

        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "location": {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "label": "Paris Household",
            },
            "model": "auto",
        }

        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_payload
            mock_get.return_value = mock_resp

            res = self.client.post(
                f"{self.prefix}/forecasts/run",
                json=payload,
                headers=self.auth_headers,
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_model"], "Early Fusion")
        self.assertEqual(data["model_key"], "early_fusion")
        self.assertEqual(data["selection_mode"], "auto")
        self.assertIn("Paris Household", data["selection_reason"])
        self.assertIn("Open-Meteo", data["selection_reason"])
        self.assertEqual(len(data["predictions_watts"]), 24)
        self.assertEqual(len(data["intervals"]), 24)

    # 2. Fallback to Energy-Only on Retrieval Failure in 'auto' Mode
    def test_auto_mode_fallback_to_energy_only_when_weather_fetch_fails(self):
        """When weather retrieval fails in 'auto' mode, fall back to Energy-Only with explanation."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "location": {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "label": "Paris Household",
            },
            "model": "auto",
        }

        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 429
            mock_get.return_value = mock_resp

            res = self.client.post(
                f"{self.prefix}/forecasts/run",
                json=payload,
                headers=self.auth_headers,
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_model"], "Energy-Only")
        self.assertEqual(data["model_key"], "energy_only")
        self.assertEqual(data["selection_mode"], "fallback")
        self.assertIn("rate limit exceeded (HTTP 429)", data["selection_reason"])
        self.assertIn("Energy-Only model", data["selection_reason"])
        self.assertEqual(len(data["predictions_watts"]), 24)

    # 3. Preserves Energy-Only When Timestamps Missing
    def test_auto_mode_preserves_energy_only_when_timestamps_missing(self):
        """When location is provided but timestamps are omitted, preserve Energy-Only without querying Open-Meteo."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": None,
            "location": {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "label": "Paris Household",
            },
            "model": "auto",
        }

        with patch("httpx.Client.get") as mock_get:
            res = self.client.post(
                f"{self.prefix}/forecasts/run",
                json=payload,
                headers=self.auth_headers,
            )
            mock_get.assert_not_called()

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_model"], "Energy-Only")
        self.assertEqual(data["model_key"], "energy_only")
        self.assertIn("Timestamps missing or incomplete", data["selection_reason"])

    # 4. Explicit Multimodal Override Behavior
    def test_explicit_multimodal_override_fails_502_when_weather_retrieval_fails(self):
        """When an explicit multimodal model is requested and retrieval fails, raise HTTP 502 Bad Gateway."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "location": {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "label": "Paris Household",
            },
            "model": "early_fusion",
        }

        with patch("httpx.Client.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Open-Meteo gateway timeout")

            res = self.client.post(
                f"{self.prefix}/forecasts/run",
                json=payload,
                headers=self.auth_headers,
            )

        self.assertEqual(res.status_code, 502)
        detail = res.json()["detail"]
        self.assertIn("Model 'Early Fusion' requires 168 hours of weather history", detail)
        self.assertIn("Automatic weather retrieval for household location failed", detail)

    def test_explicit_multimodal_override_succeeds_when_retrieval_succeeds(self):
        """When an explicit multimodal model is requested and retrieval succeeds, execute the requested model."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)

        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "location": {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "label": "Paris Household",
            },
            "model": "late_fusion",
        }

        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_payload
            mock_get.return_value = mock_resp

            res = self.client.post(
                f"{self.prefix}/forecasts/run",
                json=payload,
                headers=self.auth_headers,
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_model"], "Late Fusion")
        self.assertEqual(data["model_key"], "late_fusion")
        self.assertEqual(data["selection_mode"], "override")
        self.assertIn("Late Fusion with automatically retrieved weather", data["selection_reason"])

    # 5. User-Supplied Weather Takes Precedence
    def test_user_supplied_weather_takes_precedence_over_location(self):
        """When user provides both weather and location, supplied weather is used and Open-Meteo is never queried."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "weather": self.valid_weather_series,
            "location": {
                "latitude": 48.8566,
                "longitude": 2.3522,
                "label": "Paris Household",
            },
            "model": "auto",
        }

        with patch("httpx.Client.get") as mock_get:
            res = self.client.post(
                f"{self.prefix}/forecasts/run",
                json=payload,
                headers=self.auth_headers,
            )
            mock_get.assert_not_called()

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_model"], "Early Fusion")
        self.assertIn("Complete valid weather history provided", data["selection_reason"])

    # 6. CSV Upload with Location Form Parameters
    def test_upload_csv_with_location_form_fields_success(self):
        """CSV upload with active power and location form fields fetches weather and forecasts successfully."""
        mock_payload = _build_synthetic_open_meteo_response(self.valid_timestamps)

        # Build 168-line active-power CSV with timestamp
        csv_lines = ["timestamp,active_power"]
        for ts, p in zip(self.valid_timestamps, self.valid_electricity):
            csv_lines.append(f"{ts},{p:.2f}")
        csv_bytes = "\n".join(csv_lines).encode("utf-8")

        with patch("httpx.Client.get") as mock_get:
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_payload
            mock_get.return_value = mock_resp

            files = {"file": ("household_history.csv", csv_bytes, "text/csv")}
            data_form = {
                "model": "auto",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "location_label": "London Home",
            }
            res = self.client.post(
                f"{self.prefix}/forecasts/upload-csv",
                files=files,
                data=data_form,
                headers=self.auth_headers,
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["selected_model"], "Early Fusion")
        self.assertEqual(data["selection_mode"], "auto")
        self.assertIn("London Home", data["selection_reason"])

    def test_upload_csv_with_partial_coordinates_rejected_422(self):
        """CSV upload with latitude but missing longitude must be rejected with HTTP 422."""
        csv_lines = ["active_power"] + [f"{p:.2f}" for p in self.valid_electricity]
        csv_bytes = "\n".join(csv_lines).encode("utf-8")

        files = {"file": ("history.csv", csv_bytes, "text/csv")}
        data_form = {
            "model": "auto",
            "latitude": 51.5074,
            # longitude omitted!
        }
        res = self.client.post(
            f"{self.prefix}/forecasts/upload-csv",
            files=files,
            data=data_form,
            headers=self.auth_headers,
        )

        self.assertEqual(res.status_code, 422)
        self.assertIn("Both latitude and longitude must be provided", res.json()["detail"])


if __name__ == "__main__":
    unittest.main()
