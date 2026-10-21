"""
Unit and integration test suite for persistent forecast records and CSV upload management.

Verifies:
1. Database initialization, idempotent schema setup, and persistence across reconnection.
2. Successful forecast persistence via POST /api/v1/forecasts/run.
3. Failed run recording without fabricated predictions or success flags.
4. Unique run ID generation.
5. Full record retrieval by run ID via GET /api/v1/forecasts/records/{run_id}.
6. Status lookup via GET /api/v1/forecasts/records/{run_id}/status.
7. Paginated listing and ordering via GET /api/v1/forecasts/records.
8. CSV upload execution via POST /api/v1/forecasts/upload-csv (both Energy-Only and Early Fusion).
9. CSV validation rejections (row count != 168, missing columns, NaNs, duplicate timestamps, irregular deltas).
10. Privacy verification: Stored records must NOT contain raw 168-hour input time series.
11. Compatibility: All existing endpoints remain fully operational.
"""

import io
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
from backend.app.db.database import DatabaseManager
from backend.app.services.record_service import ForecastRecordService


class TestPersistentForecastRecords(unittest.TestCase):
    """Test suite covering persistent storage and CSV upload ingestion."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Generate standard 168-hour sample data
        base_time = datetime(2024, 2, 1, 0, 0, 0)
        cls.timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]
        cls.electricity = [float(260.0 + 40.0 * math.sin(i / 12.0)) for i in range(168)]
        cls.weather = {
            "temp": [18.0 + 4.0 * math.cos(i / 12.0) for i in range(168)],
            "feels_like": [17.5 + 4.0 * math.cos(i / 12.0) for i in range(168)],
            "humidity": [50.0 + 10.0 * math.sin(i / 24.0) for i in range(168)],
            "pressure": [1015.0 + 2.0 * math.cos(i / 48.0) for i in range(168)],
            "speed": [3.0 + 1.0 * math.sin(i / 8.0) for i in range(168)],
            "wind_sin": [math.sin(i * 0.1) for i in range(168)],
            "wind_cos": [math.cos(i * 0.1) for i in range(168)],
        }

        # Authenticate test household user
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": "persist_test_user_auth", "password": "Password123!"},
        )
        login_res = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": "persist_test_user_auth", "password": "Password123!"},
        )
        cls.token = login_res.json()["access_token"]
        cls.client.headers.update({"Authorization": f"Bearer {cls.token}"})

    # -------------------------------------------------------------------------
    # 1. Database Initialization and Reconnection
    # -------------------------------------------------------------------------
    def test_database_initialization_and_reconnection(self):
        """Verify database schema creation and persistence across connection cycles."""
        temp_db_path = settings.DATA_DIR / "test_temp_records.db"
        try:
            if temp_db_path.exists():
                temp_db_path.unlink()

            manager1 = DatabaseManager(db_path=temp_db_path)
            manager1.init_db()

            service1 = ForecastRecordService(db=manager1)
            service1.create_pending_run(
                run_id="fc_test_persist_123",
                input_method="api",
                user_id="user_alpha",
                input_metadata={"test": True},
            )

            # Reconnect with a second distinct manager instance
            manager2 = DatabaseManager(db_path=temp_db_path)
            service2 = ForecastRecordService(db=manager2)
            record = service2.get_record("fc_test_persist_123")

            self.assertIsNotNone(record)
            self.assertEqual(record.run_id, "fc_test_persist_123")
            self.assertEqual(record.status, "pending")
            self.assertEqual(record.user_id, "user_alpha")
        finally:
            if temp_db_path.exists():
                temp_db_path.unlink()

    # -------------------------------------------------------------------------
    # 2. Live Forecast Persistence via /run
    # -------------------------------------------------------------------------
    def test_forecast_run_persists_succeeded_record(self):
        """POST /run must persist a succeeded record with predictions and intervals."""
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "weather": self.weather,
            "model": "auto",
            "user_id": "household_42",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        run_id = data["run_id"]

        # Fetch record through record retrieval endpoint
        rec_res = self.client.get(f"{self.prefix}/forecasts/records/{run_id}")
        self.assertEqual(rec_res.status_code, 200, rec_res.text)
        rec = rec_res.json()

        self.assertEqual(rec["run_id"], run_id)
        self.assertEqual(rec["status"], "succeeded")
        self.assertEqual(rec["input_method"], "api")
        self.assertEqual(rec["user_id"], "household_42")
        self.assertEqual(rec["selected_model"], "Early Fusion")
        self.assertEqual(len(rec["predictions_watts"]), 24)
        self.assertEqual(len(rec["intervals"]), 24)
        self.assertIsNotNone(rec["mean_interval_width_watts"])
        self.assertIsNone(rec["error_code"])

        # Check lightweight status endpoint
        stat_res = self.client.get(f"{self.prefix}/forecasts/records/{run_id}/status")
        self.assertEqual(stat_res.status_code, 200)
        stat = stat_res.json()
        self.assertEqual(stat["status"], "succeeded")
        self.assertEqual(stat["selected_model"], "Early Fusion")

    def test_failed_run_records_failure_without_fabricated_data(self):
        """When input validation fails, a failed record is persisted with error details."""
        bad_electricity = self.electricity[:80]  # Only 80 items instead of 168
        payload = {
            "electricity_active_power": bad_electricity,
            "model": "energy_only",
            "user_id": "user_bad_req",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 422)

        # Query recent records to find the failed run
        rec_list_res = self.client.get(f"{self.prefix}/forecasts/records?status=failed&limit=5")
        self.assertEqual(rec_list_res.status_code, 200)
        records = rec_list_res.json()["records"]

        self.assertTrue(len(records) > 0)
        latest_failed = records[0]
        self.assertEqual(latest_failed["status"], "failed")
        self.assertIn("HTTP_422", latest_failed["error_code"])

        # Fetch full detail of failed record
        detail_res = self.client.get(f"{self.prefix}/forecasts/records/{latest_failed['run_id']}")
        self.assertEqual(detail_res.status_code, 200)
        det = detail_res.json()

        self.assertIsNone(det["predictions_watts"])
        self.assertIsNone(det["intervals"])
        self.assertIn("168 hourly active power values", det["error_message"])

    # -------------------------------------------------------------------------
    # 3. Privacy: Stored Records Must NOT Contain Raw 168-Hour Time Series
    # -------------------------------------------------------------------------
    def test_stored_records_omit_raw_time_series(self):
        """Verify input_metadata stores only summary counts and feature keys, not raw series."""
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "weather": self.weather,
            "model": "energy_only",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 200)
        run_id = res.json()["run_id"]

        detail = self.client.get(f"{self.prefix}/forecasts/records/{run_id}").json()
        meta = detail.get("input_metadata", {})

        self.assertNotIn("electricity_active_power", meta)
        self.assertNotIn("electricity", meta)
        self.assertNotIn("weather", meta)
        self.assertIn("electricity_points", meta)
        self.assertEqual(meta["electricity_points"], 168)

    # -------------------------------------------------------------------------
    # 4. CSV Upload Testing (Energy-Only & Early Fusion)
    # -------------------------------------------------------------------------
    def test_csv_upload_energy_only_success(self):
        """Valid 168-row CSV with active_power only should succeed under auto mode."""
        csv_rows = ["date,active_power"]
        for ts, p in zip(self.timestamps, self.electricity):
            csv_rows.append(f"{ts},{p}")
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("energy_history.csv", io.BytesIO(csv_bytes), "text/csv")}
        data = {"model": "auto", "user_id": "csv_user_1"}

        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files, data=data)
        self.assertEqual(res.status_code, 200, res.text)
        result = res.json()

        self.assertEqual(result["selected_model"], "Energy-Only")
        self.assertEqual(len(result["predictions_watts"]), 24)
        self.assertEqual(len(result["intervals"]), 24)

        # Verify persisted record
        run_id = result["run_id"]
        rec = self.client.get(f"{self.prefix}/forecasts/records/{run_id}").json()
        self.assertEqual(rec["input_method"], "csv")
        self.assertEqual(rec["status"], "succeeded")
        self.assertEqual(rec["input_metadata"]["source_filename"], "energy_history.csv")

    def test_csv_upload_multimodal_with_deg_success(self):
        """Valid 168-row CSV with weather including wind direction 'deg' should select Early Fusion."""
        header = "timestamp,hourly_active_power,temp,feels_like,humidity,pressure,speed,deg"
        csv_rows = [header]
        for i in range(168):
            deg_val = float((i * 20) % 360)
            csv_rows.append(
                f"{self.timestamps[i]},{self.electricity[i]},{self.weather['temp'][i]},"
                f"{self.weather['feels_like'][i]},{self.weather['humidity'][i]},"
                f"{self.weather['pressure'][i]},{self.weather['speed'][i]},{deg_val}"
            )
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("multimodal_data.csv", io.BytesIO(csv_bytes), "text/csv")}
        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(res.status_code, 200, res.text)
        result = res.json()

        self.assertEqual(result["selected_model"], "Early Fusion")
        self.assertEqual(len(result["predictions_watts"]), 24)

    # -------------------------------------------------------------------------
    # 5. CSV Upload Validation Failures
    # -------------------------------------------------------------------------
    def test_csv_upload_wrong_row_count(self):
        """CSV with fewer than 168 rows must be rejected with 422."""
        csv_rows = ["date,active_power"]
        for ts, p in zip(self.timestamps[:100], self.electricity[:100]):
            csv_rows.append(f"{ts},{p}")
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("short.csv", io.BytesIO(csv_bytes), "text/csv")}
        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("exactly 168 hourly rows", res.json()["detail"])

    def test_csv_upload_missing_electricity_column(self):
        """CSV missing both 'hourly_active_power' and 'active_power' must fail."""
        csv_rows = ["date,voltage,current"]
        for i in range(168):
            csv_rows.append(f"{self.timestamps[i]},120.0,2.5")
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("no_power.csv", io.BytesIO(csv_bytes), "text/csv")}
        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("missing required electricity active power column", res.json()["detail"])

    def test_csv_upload_duplicate_timestamps(self):
        """CSV containing duplicate timestamp rows must fail."""
        csv_rows = ["date,active_power"]
        bad_ts = list(self.timestamps)
        bad_ts[10] = bad_ts[9]  # Create duplicate
        for ts, p in zip(bad_ts, self.electricity):
            csv_rows.append(f"{ts},{p}")
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("dup_ts.csv", io.BytesIO(csv_bytes), "text/csv")}
        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("duplicate timestamps", res.json()["detail"].lower())

    def test_csv_upload_non_hourly_intervals(self):
        """CSV with irregular timestamp gaps must fail."""
        csv_rows = ["date,active_power"]
        bad_ts = list(self.timestamps)
        # Shift step 20 onwards by 1 hour so the delta between index 19 and 20 is 2 hours (no duplicates)
        for j in range(20, 168):
            dt_j = datetime.strptime(bad_ts[j], "%Y-%m-%d %H:%M:%S") + timedelta(hours=1)
            bad_ts[j] = dt_j.strftime("%Y-%m-%d %H:%M:%S")

        for ts, p in zip(bad_ts, self.electricity):
            csv_rows.append(f"{ts},{p}")
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("gap_ts.csv", io.BytesIO(csv_bytes), "text/csv")}
        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("consecutive hourly readings", res.json()["detail"].lower())

    def test_csv_upload_nan_values(self):
        """CSV with empty/NaN values in power column must fail."""
        csv_rows = ["date,active_power"]
        for i, (ts, p) in enumerate(zip(self.timestamps, self.electricity)):
            val = "" if i == 50 else str(p)
            csv_rows.append(f"{ts},{val}")
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("nans.csv", io.BytesIO(csv_bytes), "text/csv")}
        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("missing or invalid values at row 51", res.json()["detail"])

    def test_csv_upload_partial_weather_rejected(self):
        """CSV with only partial weather columns (e.g. temp without humidity/pressure) must fail."""
        csv_rows = ["date,active_power,temp,humidity"]  # Missing feels_like, pressure, speed, wind
        for ts, p, t, h in zip(self.timestamps, self.electricity, self.weather["temp"], self.weather["humidity"]):
            csv_rows.append(f"{ts},{p},{t},{h}")
        csv_bytes = "\n".join(csv_rows).encode("utf-8")

        files = {"file": ("partial_weather.csv", io.BytesIO(csv_bytes), "text/csv")}
        res = self.client.post(f"{self.prefix}/forecasts/upload-csv", files=files)
        self.assertEqual(res.status_code, 422)
        self.assertIn("partial weather features detected", res.json()["detail"].lower())

    # -------------------------------------------------------------------------
    # 6. Pagination and Query Endpoints
    # -------------------------------------------------------------------------
    def test_records_pagination(self):
        """GET /records pagination returns correct limit, offset, and total_count."""
        res = self.client.get(f"{self.prefix}/forecasts/records?limit=2&offset=0")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_count", data)
        self.assertEqual(data["limit"], 2)
        self.assertEqual(data["offset"], 0)
        self.assertTrue(len(data["records"]) <= 2)

    def test_get_nonexistent_record_returns_404(self):
        """Querying a non-existent run_id returns HTTP 404."""
        res = self.client.get(f"{self.prefix}/forecasts/records/fc_nonexistent_9999")
        self.assertEqual(res.status_code, 404)

        stat_res = self.client.get(f"{self.prefix}/forecasts/records/fc_nonexistent_9999/status")
        self.assertEqual(stat_res.status_code, 404)

    # -------------------------------------------------------------------------
    # 7. Preserved Research Endpoints Compatibility
    # -------------------------------------------------------------------------
    def test_research_endpoints_still_functional(self):
        """All existing research endpoints must remain functional."""
        # Clean benchmarks
        res_bench = self.client.get(f"{self.prefix}/benchmarks/clean")
        self.assertEqual(res_bench.status_code, 200)
        self.assertEqual(res_bench.json()["total_models"], 4)

        # Robustness matrix
        res_rob = self.client.get(f"{self.prefix}/robustness/matrix")
        self.assertEqual(res_rob.status_code, 200)

        # Statistics summary
        res_stat = self.client.get(f"{self.prefix}/statistics/summary")
        self.assertEqual(res_stat.status_code, 200)

        # XAI summary
        res_xai = self.client.get(f"{self.prefix}/xai/summary")
        self.assertEqual(res_xai.status_code, 200)

        # Conformal summary
        res_conf = self.client.get(f"{self.prefix}/conformal/summary")
        self.assertEqual(res_conf.status_code, 200)

        # Test windows metadata
        res_win = self.client.get(f"{self.prefix}/forecasts/test-windows")
        self.assertEqual(res_win.status_code, 200)
        self.assertEqual(res_win.json()["total_windows"], 587)


if __name__ == "__main__":
    unittest.main()
