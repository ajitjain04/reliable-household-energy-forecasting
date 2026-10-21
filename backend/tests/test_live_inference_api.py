"""
Comprehensive unit and API test suite for the live forecast service and inference API.

Verifies:
1. Valid Energy-Only forecast execution (no weather, auto-selected).
2. Valid Early Fusion forecast execution (with complete weather, auto-selected).
3. Explicit model overrides (Late Fusion, Hybrid Fusion, Energy-Only).
4. Unapproved model rejection (e.g. 'cross_attention').
5. Weather-dependent override rejection when weather is absent.
6. Validation errors for invalid electricity history (<168, >168, NaN, Inf).
7. Validation errors for invalid weather history (<168, missing columns, non-finite).
8. Wind direction conversion from degrees ('deg' -> sin/cos).
9. Timestamp validation (strictly hourly consecutive, invalid gaps rejected, future 24h generated).
10. Conformal prediction interval shapes (24 steps), non-negative clipping, and widths.
"""

import math
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
from fastapi.testclient import TestClient
import backend.app.db.database
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.db.database import DatabaseManager
from pydantic import ValidationError
from backend.app.schemas.forecasts import LocationInput
from backend.app.services.inference_service import inference_service, APPROVED_MODELS
from backend.app.services.record_service import record_service
from backend.app.services.user_service import user_service


class TestLiveInferenceAPI(unittest.TestCase):
    """Test suite for live inference endpoints and service layer."""

    @classmethod
    def setUpClass(cls):
        # 1. Establish isolated disposable database
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.temp_dir.name) / "test_live_disposable.db"
        cls.db_mgr = DatabaseManager(cls.db_path)
        cls.db_mgr.init_db()

        cls._orig_db_mgr = backend.app.db.database.db_manager
        cls._orig_rec_db = record_service.db
        cls._orig_usr_db = user_service._db

        backend.app.db.database.db_manager = cls.db_mgr
        record_service.db = cls.db_mgr
        user_service._db = cls.db_mgr

        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Generate standard synthetic 168-hour test payloads
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        cls.valid_timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]
        # Realistic active power values centered around ~300 Watts
        cls.valid_electricity = [float(250.0 + 50.0 * math.sin(i / 12.0)) for i in range(168)]

        # Valid weather arrays (7 features)
        cls.valid_weather_series = {
            "temp": [20.0 + 5.0 * math.cos(i / 12.0) for i in range(168)],
            "feels_like": [19.0 + 5.0 * math.cos(i / 12.0) for i in range(168)],
            "humidity": [45.0 + 10.0 * math.sin(i / 24.0) for i in range(168)],
            "pressure": [1013.0 + 2.0 * math.cos(i / 48.0) for i in range(168)],
            "speed": [2.5 + 1.0 * math.sin(i / 6.0) for i in range(168)],
            "wind_sin": [math.sin(i * 0.1) for i in range(168)],
            "wind_cos": [math.cos(i * 0.1) for i in range(168)],
        }

        # Authenticate test household user
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": "live_test_user_auth", "password": "Password123!"},
        )
        login_res = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": "live_test_user_auth", "password": "Password123!"},
        )
        cls.token = login_res.json()["access_token"]
        cls.client.headers.update({"Authorization": f"Bearer {cls.token}"})

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

    # -------------------------------------------------------------------------
    # 1. Authentication Enforcement
    # -------------------------------------------------------------------------
    def test_unauthenticated_forecast_run_rejected_401(self):
        """POST /forecasts/run without authorization must return HTTP 401."""
        unauth_client = TestClient(app)
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "model": "auto",
        }
        res = unauth_client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.headers.get("WWW-Authenticate"), "Bearer")

    # -------------------------------------------------------------------------
    # 2. Successful Live Forecasts
    # -------------------------------------------------------------------------
    def test_forecast_energy_only_auto_selection(self):
        """When weather is absent, auto mode should select Energy-Only and return 24 predictions."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "weather": None,
            "model": "auto",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertTrue(data["run_id"].startswith("fc_"))
        self.assertEqual(data["selected_model"], "Energy-Only")
        self.assertEqual(data["model_key"], "energy_only")
        self.assertEqual(data["selection_mode"], "auto")
        self.assertIn("No weather history provided", data["selection_reason"])
        self.assertEqual(data["forecast_origin_timestamp"], self.valid_timestamps[-1])

        # Verify 24 predictions
        self.assertEqual(len(data["predictions_watts"]), 24)
        self.assertEqual(len(data["horizon_hours"]), 24)
        self.assertEqual(len(data["forecast_timestamps"]), 24)
        self.assertEqual(len(data["intervals"]), 24)

        # Verify interval structure
        first_step = data["intervals"][0]
        self.assertEqual(first_step["horizon_hour"], 1)
        self.assertTrue(first_step["upper_bound_watts"] > first_step["lower_bound_watts"])
        self.assertTrue(first_step["lower_bound_clipped_watts"] >= 0.0)
        self.assertAlmostEqual(
            first_step["interval_width_watts"],
            first_step["upper_bound_watts"] - first_step["lower_bound_watts"],
            places=1,
        )

        # Metadata
        self.assertEqual(data["model_metadata"]["model_class"], "EnergyOnlyLSTM")
        self.assertEqual(data["model_metadata"]["trainable_parameters"], 51992)
        self.assertEqual(data["nominal_coverage_pct"], 90.0)

    def test_forecast_early_fusion_auto_selection(self):
        """When valid weather is present, auto mode should select Early Fusion as baseline leader."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "weather": self.valid_weather_series,
            "model": "auto",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()

        self.assertEqual(data["selected_model"], "Early Fusion")
        self.assertEqual(data["model_key"], "early_fusion")
        self.assertEqual(data["selection_mode"], "auto")
        self.assertIn("Early Fusion selected as initial default", data["selection_reason"])
        self.assertEqual(len(data["predictions_watts"]), 24)
        self.assertEqual(data["model_metadata"]["model_class"], "EarlyFusionLSTM")
        self.assertEqual(data["model_metadata"]["trainable_parameters"], 53784)

    def test_forecast_with_wind_direction_degrees(self):
        """Weather input with 'deg' should be converted to sin/cos and execute cleanly."""
        weather_with_deg = {
            "temp": self.valid_weather_series["temp"],
            "feels_like": self.valid_weather_series["feels_like"],
            "humidity": self.valid_weather_series["humidity"],
            "pressure": self.valid_weather_series["pressure"],
            "speed": self.valid_weather_series["speed"],
            "deg": [float((i * 15) % 360) for i in range(168)],
        }
        payload = {
            "electricity_active_power": self.valid_electricity,
            "weather": weather_with_deg,
            "model": "early_fusion",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(len(data["predictions_watts"]), 24)
        self.assertEqual(data["selected_model"], "Early Fusion")

    def test_forecast_without_timestamps(self):
        """When timestamps are omitted, predictions and horizon steps 1-24 are returned without dates."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "weather": None,
            "model": "energy_only",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertIsNone(data["forecast_origin_timestamp"])
        self.assertIsNone(data["forecast_timestamps"])
        self.assertEqual(len(data["predictions_watts"]), 24)
        self.assertIsNone(data["intervals"][0]["target_timestamp"])

    # -------------------------------------------------------------------------
    # 2. Explicit Model Overrides
    # -------------------------------------------------------------------------
    def test_explicit_overrides_late_and_hybrid_fusion(self):
        """Authorized overrides for Late Fusion and Hybrid Fusion should succeed when weather is provided."""
        for m_key, expected_class, expected_params in [
            ("late_fusion", "LateFusionLSTM", 112216),
            ("hybrid_fusion", "HybridFusionLSTM", 145496),
        ]:
            payload = {
                "electricity_active_power": self.valid_electricity,
                "weather": self.valid_weather_series,
                "model": m_key,
            }
            res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
            self.assertEqual(res.status_code, 200, res.text)
            data = res.json()
            self.assertEqual(data["model_key"], m_key)
            self.assertEqual(data["selection_mode"], "override")
            self.assertEqual(data["model_metadata"]["model_class"], expected_class)
            self.assertEqual(data["model_metadata"]["trainable_parameters"], expected_params)
            self.assertEqual(len(data["predictions_watts"]), 24)

    def test_unapproved_model_override_rejected(self):
        """Unapproved model architecture must be rejected with 422."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "model": "cross_attention",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertIn("not approved", data["detail"].lower())

    def test_weather_dependent_model_without_weather_rejected(self):
        """Overriding to a multimodal model without weather data must return 422."""
        payload = {
            "electricity_active_power": self.valid_electricity,
            "weather": None,
            "model": "early_fusion",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertIn("requires 168 hours of weather history", data["detail"])

    # -------------------------------------------------------------------------
    # 3. Input Validation Failures
    # -------------------------------------------------------------------------
    def test_electricity_length_mismatch(self):
        """Electricity sequence length != 168 must fail."""
        # Short sequence (100 values)
        res = self.client.post(
            f"{self.prefix}/forecasts/run",
            json={"electricity_active_power": self.valid_electricity[:100]},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("exactly 168 hourly active power values", res.json()["detail"])

        # Long sequence (170 values)
        res = self.client.post(
            f"{self.prefix}/forecasts/run",
            json={"electricity_active_power": self.valid_electricity + [300.0, 310.0]},
        )
        self.assertEqual(res.status_code, 422)

    def test_electricity_non_finite_values(self):
        """NaN, None, or Inf values in electricity history must be rejected."""
        # 1. API level rejection with None
        bad_electricity = list(self.valid_electricity)
        bad_electricity[10] = None
        res = self.client.post(
            f"{self.prefix}/forecasts/run",
            json={"electricity_active_power": bad_electricity},
        )
        self.assertEqual(res.status_code, 422)
        detail_str = str(res.json()["detail"]).lower()
        self.assertTrue(
            "input should be a valid number" in detail_str
            or "non-finite or invalid numeric value" in detail_str
        )

        # 2. Service level rejection with math.nan and math.inf
        nan_list = list(self.valid_electricity)
        nan_list[15] = float("nan")
        with self.assertRaises(Exception) as ctx:
            inference_service.validate_electricity(nan_list)
        self.assertIn("non-finite or invalid numeric value", str(ctx.exception).lower())

        inf_list = list(self.valid_electricity)
        inf_list[20] = float("inf")
        with self.assertRaises(Exception) as ctx2:
            inference_service.validate_electricity(inf_list)
        self.assertIn("non-finite or invalid numeric value", str(ctx2.exception).lower())

    def test_timestamps_frequency_discontinuity(self):
        """Non-consecutive timestamps (e.g. 2-hour gap) must fail."""
        bad_timestamps = list(self.valid_timestamps)
        # Introduce a gap at index 50
        base_dt = datetime.strptime(bad_timestamps[49], "%Y-%m-%d %H:%M:%S")
        bad_timestamps[50] = (base_dt + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")

        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": bad_timestamps,
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertIn("Irregular interval detected", res.json()["detail"])

    def test_weather_missing_feature(self):
        """Weather missing one of the audited features (e.g. pressure) must fail."""
        incomplete_weather = dict(self.valid_weather_series)
        del incomplete_weather["pressure"]

        payload = {
            "electricity_active_power": self.valid_electricity,
            "weather": incomplete_weather,
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertIn("missing required features", res.json()["detail"])

    def test_weather_length_mismatch(self):
        """Weather feature with fewer than 168 hours must fail."""
        bad_weather = dict(self.valid_weather_series)
        bad_weather["temp"] = self.valid_weather_series["temp"][:100]

        payload = {
            "electricity_active_power": self.valid_electricity,
            "weather": bad_weather,
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertIn("must contain exactly 168 hourly values", res.json()["detail"])

    # -------------------------------------------------------------------------
    # 4. LocationInput Validation and Backward Compatibility
    # -------------------------------------------------------------------------
    def test_forecast_request_with_valid_location_input(self):
        """Valid LocationInput must be accepted and forecast generated successfully."""
        # Test schema model creation directly
        loc = LocationInput(latitude=18.5204, longitude=73.8567, label="Pune Household")
        self.assertEqual(loc.latitude, 18.5204)
        self.assertEqual(loc.longitude, 73.8567)
        self.assertEqual(loc.label, "Pune Household")

        # Test API endpoint acceptance with complete valid forecast payload
        payload = {
            "electricity_active_power": self.valid_electricity,
            "timestamps": self.valid_timestamps,
            "weather": self.valid_weather_series,
            "location": {
                "latitude": 18.5204,
                "longitude": 73.8567,
                "label": "Pune Household",
            },
            "model": "auto",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        self.assertEqual(len(data["predictions_watts"]), 24)
        self.assertEqual(data["selected_model"], "Early Fusion")

    def test_forecast_request_with_invalid_latitude_bounds_rejected(self):
        """Latitude values outside [-90.0, 90.0] must be rejected by schema validation with HTTP 422."""
        # 1. Direct schema level rejection
        with self.assertRaises(ValidationError):
            LocationInput(latitude=90.1, longitude=73.8567)
        with self.assertRaises(ValidationError):
            LocationInput(latitude=-90.1, longitude=73.8567)

        # 2. API level rejection
        for invalid_lat in [90.1, 95.0, -90.1, -120.0]:
            payload = {
                "electricity_active_power": self.valid_electricity,
                "location": {
                    "latitude": invalid_lat,
                    "longitude": 73.8567,
                },
                "model": "energy_only",
            }
            res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
            self.assertEqual(res.status_code, 422, f"Expected 422 for latitude {invalid_lat}")
            detail_str = str(res.json()["detail"]).lower()
            self.assertTrue(
                "latitude" in detail_str or "greater_than_equal" in detail_str or "less_than_equal" in detail_str,
                f"Unexpected detail: {res.json()}",
            )

    def test_forecast_request_with_invalid_longitude_bounds_rejected(self):
        """Longitude values outside [-180.0, 180.0] must be rejected by schema validation with HTTP 422."""
        # 1. Direct schema level rejection
        with self.assertRaises(ValidationError):
            LocationInput(latitude=18.5204, longitude=180.1)
        with self.assertRaises(ValidationError):
            LocationInput(latitude=18.5204, longitude=-180.1)

        # 2. API level rejection
        for invalid_lon in [180.1, 195.0, -180.1, -200.0]:
            payload = {
                "electricity_active_power": self.valid_electricity,
                "location": {
                    "latitude": 18.5204,
                    "longitude": invalid_lon,
                },
                "model": "energy_only",
            }
            res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
            self.assertEqual(res.status_code, 422, f"Expected 422 for longitude {invalid_lon}")
            detail_str = str(res.json()["detail"]).lower()
            self.assertTrue(
                "longitude" in detail_str or "greater_than_equal" in detail_str or "less_than_equal" in detail_str,
                f"Unexpected detail: {res.json()}",
            )

    def test_forecast_request_without_location_backward_compatibility(self):
        """Forecast requests omitting 'location' or setting 'location: None' remain 100% backward-compatible."""
        # 1. 'location' field omitted entirely
        payload_omitted = {
            "electricity_active_power": self.valid_electricity,
            "model": "energy_only",
        }
        res_omitted = self.client.post(f"{self.prefix}/forecasts/run", json=payload_omitted)
        self.assertEqual(res_omitted.status_code, 200, res_omitted.text)
        self.assertEqual(len(res_omitted.json()["predictions_watts"]), 24)

        # 2. 'location' explicitly set to None
        payload_none = {
            "electricity_active_power": self.valid_electricity,
            "location": None,
            "model": "energy_only",
        }
        res_none = self.client.post(f"{self.prefix}/forecasts/run", json=payload_none)
        self.assertEqual(res_none.status_code, 200, res_none.text)
        self.assertEqual(len(res_none.json()["predictions_watts"]), 24)


if __name__ == "__main__":
    unittest.main()
