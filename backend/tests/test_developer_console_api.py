"""
Unit and integration test suite for Developer Console and Forecast Run Inspection (Step 5).

Verifies:
1. Retrieval of paginated forecast records from actual backend API.
2. Filter capabilities (status, model) and pagination response contracts.
3. Detailed retrieval of successful forecast runs including all 24 predictions,
   conformal intervals, model metadata, and sanitized input metadata.
4. Detailed retrieval of failed forecast runs verifying safe error reporting
   without fabricated predictions.
5. Error handling for non-existent run IDs (HTTP 404).
6. Strict distinction between run-specific data and population-level research benchmarks.
7. Verification that all frontend route registrations (household, developer, and 7 research routes)
   remain intact in code.
"""

import json
import math
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.schemas.records import (
    ForecastRecordListResponse,
    ForecastRecordDetail,
    ForecastStatusResponse,
)


class TestDeveloperConsoleApi(unittest.TestCase):
    """Test suite covering developer console backend APIs and inspection integrity."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Generate standard 168-hour sample electricity sequence
        cls.electricity = [
            float(270.0 + 50.0 * math.sin(i / 12.0)) for i in range(168)
        ]
        base_time = datetime(2024, 3, 1, 0, 0, 0)
        cls.timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]

        # Authenticate developer account for developer console inspection
        from backend.app.services.user_service import user_service
        dev_user = user_service.get_user_by_username("dev_tester_console")
        if not dev_user:
            user_service.create_user(
                username="dev_tester_console",
                password="DevPassword123!",
                role="developer",
            )
        login_res = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": "dev_tester_console", "password": "DevPassword123!"},
        )
        cls.token = login_res.json()["access_token"]
        cls.client.headers.update({"Authorization": f"Bearer {cls.token}"})

    def test_list_records_schema_and_pagination(self):
        """Verify GET /forecasts/records returns valid paginated list schema."""
        resp = self.client.get(f"{self.prefix}/forecasts/records?limit=5&offset=0")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        validated = ForecastRecordListResponse(**data)
        self.assertGreaterEqual(validated.total_count, 0)
        self.assertLessEqual(len(validated.records), 5)
        self.assertEqual(validated.limit, 5)
        self.assertEqual(validated.offset, 0)

        # Inspect fields for any returned records
        for rec in validated.records:
            self.assertTrue(rec.run_id.startswith("fc_") or len(rec.run_id) > 0)
            self.assertIn(rec.status, ["pending", "succeeded", "failed"])
            self.assertIn(rec.input_method, ["api", "csv"])

    def test_run_success_and_developer_inspection(self):
        """Execute a forecast run and inspect the full detailed record as a developer."""
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
            "user_id": "dev_test_user_inspector",
        }
        run_resp = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(run_resp.status_code, 200)
        run_data = run_resp.json()
        run_id = run_data["run_id"]

        # Developer inspection endpoint: GET /forecasts/records/{run_id}
        detail_resp = self.client.get(f"{self.prefix}/forecasts/records/{run_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail_data = detail_resp.json()

        # Validate with Pydantic schema
        validated = ForecastRecordDetail(**detail_data)
        self.assertEqual(validated.run_id, run_id)
        self.assertEqual(validated.status, "succeeded")
        self.assertEqual(validated.input_method, "api")
        self.assertEqual(validated.user_id, "dev_test_user_inspector")
        self.assertEqual(validated.selected_model, "Energy-Only")

        # Verify all 24 predictions in Watts
        self.assertIsNotNone(validated.predictions_watts)
        self.assertEqual(len(validated.predictions_watts), 24)
        for val in validated.predictions_watts:
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0.0)

        # Verify all 24 conformal intervals
        self.assertIsNotNone(validated.intervals)
        self.assertEqual(len(validated.intervals), 24)
        for step in validated.intervals:
            self.assertGreaterEqual(step.horizon_hour, 1)
            self.assertLessEqual(step.horizon_hour, 24)
            self.assertGreaterEqual(step.lower_bound_clipped_watts, 0.0)
            self.assertGreaterEqual(step.upper_bound_watts, step.lower_bound_clipped_watts)
            self.assertAlmostEqual(
                step.interval_width_watts,
                step.upper_bound_watts - step.lower_bound_watts,
                delta=0.05,
            )

        # Verify model metadata information
        self.assertIsNotNone(validated.model_metadata)
        self.assertIn("model_class", validated.model_metadata)
        self.assertIn("trainable_parameters", validated.model_metadata)
        self.assertGreater(validated.model_metadata["trainable_parameters"], 0)

        # Verify input metadata (no raw 168 time series stored!)
        self.assertIsNotNone(validated.input_metadata)
        self.assertNotIn("raw_values", validated.input_metadata)
        self.assertNotIn("electricity_active_power", validated.input_metadata)

    def test_run_failure_developer_inspection(self):
        """Verify failed runs persist with safe error codes without fabricated predictions."""
        # Intentionally invalid CSV payload (only 10 rows instead of 168)
        short_csv = "date,hourly_active_power\n" + "\n".join(
            f"2024-03-01 0{i}:00:00,250.0" for i in range(10)
        )
        files = {"file": ("invalid_short.csv", io_bytes := short_csv.encode("utf-8"), "text/csv")}
        csv_resp = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(csv_resp.status_code, 422)

        # List records to find the most recent failed run
        records_resp = self.client.get(f"{self.prefix}/forecasts/records?limit=5&status=failed")
        self.assertEqual(records_resp.status_code, 200)
        records = records_resp.json()["records"]
        self.assertGreater(len(records), 0)

        failed_run = records[0]
        self.assertEqual(failed_run["status"], "failed")
        self.assertIsNotNone(failed_run["error_code"])

        # Fetch detail of failed run
        detail_resp = self.client.get(f"{self.prefix}/forecasts/records/{failed_run['run_id']}")
        self.assertEqual(detail_resp.status_code, 200)
        failed_detail = detail_resp.json()

        # Predictions and intervals must be None for failed runs
        self.assertIsNone(failed_detail.get("predictions_watts"))
        self.assertIsNone(failed_detail.get("intervals"))
        self.assertIsNotNone(failed_detail.get("error_code"))
        self.assertIsNotNone(failed_detail.get("error_message"))

    def test_non_existent_run_id_returns_404(self):
        """Querying a non-existent run ID should return HTTP 404 with structured detail."""
        resp = self.client.get(f"{self.prefix}/forecasts/records/fc_non_existent_00000000")
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())

    def test_distinction_between_run_specific_data_and_research_benchmarks(self):
        """
        Verify scientific integrity:
        - Run-specific endpoints return single-horizon point forecasts and bounds.
        - Research benchmarks return population-level offline artifacts (587 windows, PICP, etc.).
        """
        # 1. Fetch aggregate research benchmarks
        bench_resp = self.client.get(f"{self.prefix}/benchmarks/clean")
        self.assertEqual(bench_resp.status_code, 200)
        bench_data = bench_resp.json()
        self.assertEqual(bench_data["benchmark_leader"], "Early Fusion")
        self.assertIn("records", bench_data)

        early_fusion_record = next(
            r for r in bench_data["records"] if r["model"] == "Early Fusion"
        )
        # Verified test MAE is 71.64 W across all 587 windows
        self.assertAlmostEqual(early_fusion_record["test_mae_watts"], 71.64, places=1)

        # 2. Conformal calibration summary is population-level
        conf_resp = self.client.get(f"{self.prefix}/conformal/summary")
        self.assertEqual(conf_resp.status_code, 200)
        conf_data = conf_resp.json()
        self.assertEqual(conf_data["calibration_samples"], 587)
        self.assertEqual(conf_data["confidence_level_pct"], 90.0)

        # 3. Individual run-specific record does NOT contain aggregate population metrics
        payload = {
            "electricity_active_power": self.electricity,
            "model": "early_fusion",
            "weather": {
                "temp": [20.0] * 168,
                "feels_like": [20.0] * 168,
                "humidity": [50.0] * 168,
                "pressure": [1013.0] * 168,
                "speed": [3.0] * 168,
                "deg": [180.0] * 168,
            },
        }
        run_resp = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(run_resp.status_code, 200)
        run_id = run_resp.json()["run_id"]

        detail_resp = self.client.get(f"{self.prefix}/forecasts/records/{run_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.json()

        # Run detail must contain exact 24-step forecast, not population test MAE or window count
        self.assertEqual(len(detail["predictions_watts"]), 24)
        self.assertNotIn("test_mae_watts", detail)
        self.assertNotIn("calibration_samples", detail)

    def test_frontend_route_registrations_preserved(self):
        """Verify that all household, developer, and research routes are registered in App.tsx."""
        app_tsx_path = PROJECT_ROOT / "frontend" / "src" / "App.tsx"
        self.assertTrue(app_tsx_path.exists(), "App.tsx must exist")
        content = app_tsx_path.read_text(encoding="utf-8")

        # Household routes
        self.assertIn('path="forecast/new"', content)
        self.assertIn('path="forecast/history"', content)
        self.assertIn('path="forecast/:runId"', content)

        # Developer Console routes
        self.assertIn('path="developer/forecast-records"', content)
        self.assertIn('path="developer/forecast-records/:runId"', content)

        # All 7 Research routes
        self.assertIn('path="overview"', content)
        self.assertIn('path="benchmarks"', content)
        self.assertIn('path="reliability"', content)
        self.assertIn('path="statistical-tests"', content)
        self.assertIn('path="xai"', content)
        self.assertIn('path="uncertainty"', content)
        self.assertIn('path="explorer"', content)


if __name__ == "__main__":
    unittest.main()
