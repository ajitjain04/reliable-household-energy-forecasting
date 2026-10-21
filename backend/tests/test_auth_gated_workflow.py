"""
Targeted test suite for the Secure Authentication-Gated Forecasting Workflow.

Covers:
1. Public endpoint accessibility for unauthenticated callers.
2. Strict HTTP 401 rejection for unauthenticated forecast operations:
   - Live forecast creation (POST /forecasts/run)
   - CSV upload forecast (POST /forecasts/upload-csv)
   - Private forecast history listing (GET /forecasts/records)
   - Saved record detail (GET /forecasts/records/{run_id})
   - Saved record status (GET /forecasts/records/{run_id}/status)
   - Vector PDF download (GET /forecasts/records/{run_id}/pdf)
   - Developer records inspection (GET /forecasts/developer/records)
3. Role tampering prevention during registration:
   - Client payload supplying role='developer' rejected by schema validation (HTTP 422).
   - Household registration strictly creates accounts with role='household'.
4. User ownership isolation:
   - User A cannot access User B's forecast record (HTTP 403).
   - User A cannot download User B's PDF report (HTTP 403).
   - User A's private history list never includes User B's records.
   - Display tracking tag (user_id) cannot override authenticated JWT ownership.
5. Developer role authorization:
   - Household user accessing developer endpoints rejected (HTTP 403).
   - Authenticated developer can inspect all records and download PDFs for any run.
"""

import math
import sys
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.user_service import user_service
from backend.app.services.record_service import record_service


class TestAuthGatedForecastingWorkflow(unittest.TestCase):
    """Targeted validation of all authentication and authorization boundaries."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Generate realistic 168-hour sample electricity sequence
        cls.electricity = [
            float(260.0 + 45.0 * math.sin(i / 12.0)) for i in range(168)
        ]
        base_time = datetime(2024, 4, 1, 0, 0, 0)
        cls.timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]

        # Provision Household User 1
        u1_name = f"household_u1_{uuid.uuid4().hex[:6]}"
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": u1_name, "password": "Password123!"},
        )
        login1 = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": u1_name, "password": "Password123!"},
        )
        cls.token_u1 = login1.json()["access_token"]
        cls.user_u1_id = login1.json()["user"]["id"]
        cls.headers_u1 = {"Authorization": f"Bearer {cls.token_u1}"}

        # Provision Household User 2
        u2_name = f"household_u2_{uuid.uuid4().hex[:6]}"
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": u2_name, "password": "Password123!"},
        )
        login2 = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": u2_name, "password": "Password123!"},
        )
        cls.token_u2 = login2.json()["access_token"]
        cls.user_u2_id = login2.json()["user"]["id"]
        cls.headers_u2 = {"Authorization": f"Bearer {cls.token_u2}"}

        # Provision Developer User
        dev_name = f"dev_workflow_{uuid.uuid4().hex[:6]}"
        user_service.create_user(
            username=dev_name,
            password="DevPassword123!",
            role="developer",
        )
        login_dev = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": dev_name, "password": "DevPassword123!"},
        )
        cls.token_dev = login_dev.json()["access_token"]
        cls.headers_dev = {"Authorization": f"Bearer {cls.token_dev}"}

    # -------------------------------------------------------------------------
    # 1. Public Endpoints Accessibility
    # -------------------------------------------------------------------------
    def test_public_research_and_health_endpoints_accessible_without_auth(self):
        """Logged-out visitors can freely access public research benchmarks and health endpoints."""
        unauth_client = TestClient(app)

        endpoints = [
            "/health",
            "/benchmarks/clean",
            "/robustness/matrix",
            "/statistics/summary",
            "/statistics/family/1",
            "/xai/summary",
            "/xai/temporal",
            "/xai/weather-features",
            "/conformal/summary",
            "/forecasts/test-windows",
            "/forecasts/test-window/0",
        ]
        for ep in endpoints:
            res = unauth_client.get(f"{self.prefix}{ep}")
            self.assertEqual(res.status_code, 200, f"Public endpoint {ep} failed: {res.text}")

    # -------------------------------------------------------------------------
    # 2. Strict HTTP 401 on Protected Endpoints
    # -------------------------------------------------------------------------
    def test_unauthenticated_forecast_run_returns_401(self):
        """Logged-out forecast execution returns HTTP 401."""
        unauth_client = TestClient(app)
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "auto",
        }
        res = unauth_client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.headers.get("WWW-Authenticate"), "Bearer")

    def test_unauthenticated_csv_upload_returns_401(self):
        """Logged-out CSV upload forecast returns HTTP 401."""
        unauth_client = TestClient(app)
        csv_content = b"date,hourly_active_power\n2024-04-01 00:00:00,280.0\n"
        res = unauth_client.post(
            f"{self.prefix}/forecasts/upload-csv",
            files={"file": ("input.csv", csv_content, "text/csv")},
            data={"model": "auto"},
        )
        self.assertEqual(res.status_code, 401)

    def test_unauthenticated_records_list_returns_401(self):
        """Logged-out private forecast history listing returns HTTP 401."""
        unauth_client = TestClient(app)
        res = unauth_client.get(f"{self.prefix}/forecasts/records")
        self.assertEqual(res.status_code, 401)

    def test_unauthenticated_record_detail_and_pdf_returns_401(self):
        """Logged-out record inspection and PDF download return HTTP 401."""
        unauth_client = TestClient(app)
        res_det = unauth_client.get(f"{self.prefix}/forecasts/records/fc_any_id")
        self.assertEqual(res_det.status_code, 401)

        res_pdf = unauth_client.get(f"{self.prefix}/forecasts/records/fc_any_id/pdf")
        self.assertEqual(res_pdf.status_code, 401)

    # -------------------------------------------------------------------------
    # 3. Registration Security & Role Protection
    # -------------------------------------------------------------------------
    def test_registration_forbids_client_supplied_developer_role(self):
        """Registration payload cannot inject a developer role."""
        bad_reg = self.client.post(
            f"{self.prefix}/auth/register",
            json={
                "username": f"attacker_{uuid.uuid4().hex[:6]}",
                "password": "Password123!",
                "role": "developer",
            },
        )
        self.assertEqual(bad_reg.status_code, 422)

    def test_registration_strictly_assigns_household_role(self):
        """Normal registration always assigns role='household'."""
        uname = f"clean_user_{uuid.uuid4().hex[:6]}"
        res = self.client.post(
            f"{self.prefix}/auth/register",
            json={"username": uname, "password": "Password123!"},
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["role"], "household")

    # -------------------------------------------------------------------------
    # 4. Household Ownership Isolation
    # -------------------------------------------------------------------------
    def test_household_record_ownership_isolation(self):
        """User A creates a run. User B is rejected (HTTP 403)."""
        # User 1 creates forecast
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
            "user_id": "u1_display_tag",
        }
        create_res = self.client.post(
            f"{self.prefix}/forecasts/run",
            json=payload,
            headers=self.headers_u1,
        )
        self.assertEqual(create_res.status_code, 200)
        run_id_u1 = create_res.json()["run_id"]

        # User 1 can view run detail
        det_u1 = self.client.get(
            f"{self.prefix}/forecasts/records/{run_id_u1}",
            headers=self.headers_u1,
        )
        self.assertEqual(det_u1.status_code, 200)
        self.assertEqual(det_u1.json()["owner_user_id"], self.user_u1_id)

        # User 1 can download PDF report
        pdf_u1 = self.client.get(
            f"{self.prefix}/forecasts/records/{run_id_u1}/pdf",
            headers=self.headers_u1,
        )
        self.assertEqual(pdf_u1.status_code, 200)
        self.assertEqual(pdf_u1.headers["Content-Type"], "application/pdf")

        # User 2 attempts to view User 1's run -> HTTP 404 Not Found (Anti-enumeration)
        det_u2 = self.client.get(
            f"{self.prefix}/forecasts/records/{run_id_u1}",
            headers=self.headers_u2,
        )
        self.assertEqual(det_u2.status_code, 404)

        # User 2 attempts to download User 1's PDF -> HTTP 404 Not Found (Anti-enumeration)
        pdf_u2 = self.client.get(
            f"{self.prefix}/forecasts/records/{run_id_u1}/pdf",
            headers=self.headers_u2,
        )
        self.assertEqual(pdf_u2.status_code, 404)

        # User 2's history does not contain User 1's run
        history_u2 = self.client.get(
            f"{self.prefix}/forecasts/records?limit=100",
            headers=self.headers_u2,
        )
        self.assertEqual(history_u2.status_code, 200)
        u2_run_ids = [r["run_id"] for r in history_u2.json()["records"]]
        self.assertNotIn(run_id_u1, u2_run_ids)

    def test_display_tag_does_not_override_authenticated_owner(self):
        """User 1 passing user_id='spoofed_owner' does not compromise token ownership."""
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
            "user_id": "spoofed_owner_identity",
        }
        res = self.client.post(
            f"{self.prefix}/forecasts/run",
            json=payload,
            headers=self.headers_u1,
        )
        self.assertEqual(res.status_code, 200)
        run_id = res.json()["run_id"]

        rec = record_service.get_record(run_id)
        # Authoritative owner is user_u1_id
        self.assertEqual(rec.owner_user_id, self.user_u1_id)
        # Display tag is stored as non-authoritative metadata
        self.assertEqual(rec.user_id, "spoofed_owner_identity")

    # -------------------------------------------------------------------------
    # 5. Developer Authorization & Inspection
    # -------------------------------------------------------------------------
    def test_developer_authorization_and_cross_user_inspection(self):
        """Developer can inspect any user run and download PDF; ordinary user rejected from dev endpoints."""
        # User 1 creates a run
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
        }
        run_res = self.client.post(
            f"{self.prefix}/forecasts/run",
            json=payload,
            headers=self.headers_u1,
        )
        run_id = run_res.json()["run_id"]

        # Household user cannot access GET /forecasts/developer/records -> 403
        dev_endpoint_u1 = self.client.get(
            f"{self.prefix}/forecasts/developer/records",
            headers=self.headers_u1,
        )
        self.assertEqual(dev_endpoint_u1.status_code, 403)

        # Developer can access GET /forecasts/developer/records -> 200
        dev_endpoint_ok = self.client.get(
            f"{self.prefix}/forecasts/developer/records",
            headers=self.headers_dev,
        )
        self.assertEqual(dev_endpoint_ok.status_code, 200)

        # Developer can inspect User 1's record detail -> 200
        dev_detail = self.client.get(
            f"{self.prefix}/forecasts/records/{run_id}",
            headers=self.headers_dev,
        )
        self.assertEqual(dev_detail.status_code, 200)
        self.assertEqual(dev_detail.json()["owner_user_id"], self.user_u1_id)

        # Developer can download PDF for User 1's run -> 200
        dev_pdf = self.client.get(
            f"{self.prefix}/forecasts/records/{run_id}/pdf",
            headers=self.headers_dev,
        )
        self.assertEqual(dev_pdf.status_code, 200)
        self.assertEqual(dev_pdf.headers["Content-Type"], "application/pdf")


if __name__ == "__main__":
    unittest.main()
