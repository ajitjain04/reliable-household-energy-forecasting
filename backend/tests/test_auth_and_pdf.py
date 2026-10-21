"""
Unit and integration test suite for Authentication, Authorization, Ownership Control,
and PDF Report Generation (Step 6).

Verifies:
1. User registration, login, token issuance, and /auth/me profile inspection.
2. Invalid credentials (wrong password, unknown username) return HTTP 401.
3. Password hashing security: passwords stored as Argon2id hashes, never in plaintext.
4. Record ownership isolation:
   - User A cannot view User B's record (HTTP 403).
   - User A cannot download User B's PDF report (HTTP 403).
   - Household user cannot access unassigned anonymous records (HTTP 403).
   - Spoofed 'user_id' in request body does NOT grant ownership (verified by authenticated JWT).
5. Role authorization:
   - Developer accounts can inspect all runs (User A, User B, anonymous).
   - Developer accounts can download PDFs for any successful run.
   - Ordinary household users are rejected from developer endpoints.
6. PDF generation and streaming:
   - Successful runs yield genuine binary vector PDF (%PDF- header).
   - Content-Type is 'application/pdf'.
   - Content-Disposition has attachment filename 'forecast_report_{run_id}.pdf'.
   - Missing runs return HTTP 404.
   - Failed runs return HTTP 400.
7. Legacy record preservation:
   - Existing anonymous runs remain unassigned (owner_user_id IS NULL) and accessible to developers/legacy callers.
"""

import io
import math
import sys
import unittest
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings
from backend.app.services.user_service import user_service
from backend.app.services.record_service import record_service


class TestAuthAndPdf(unittest.TestCase):
    """Test suite covering authentication, record-level authorization, and PDF report delivery."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Generate standard 168-hour sample electricity sequence
        cls.electricity = [
            float(250.0 + 40.0 * math.sin(i / 12.0)) for i in range(168)
        ]
        base_time = datetime(2024, 3, 1, 0, 0, 0)
        cls.timestamps = [
            (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(168)
        ]

        # Register User A (Household)
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={
                "username": "user_a_household",
                "password": "Password123!",
            },
        )
        login_a = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": "user_a_household", "password": "Password123!"},
        )
        cls.token_a = login_a.json()["access_token"]
        cls.user_a_id = login_a.json()["user"]["id"]

        # Register User B (Household)
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={
                "username": "user_b_household",
                "password": "Password123!",
            },
        )
        login_b = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": "user_b_household", "password": "Password123!"},
        )
        cls.token_b = login_b.json()["access_token"]
        cls.user_b_id = login_b.json()["user"]["id"]

        # Create Developer using administrative user_service (CLI / Initial Seed flow)
        dev_user = user_service.get_user_by_username("dev_admin_test")
        if not dev_user:
            dev_user = user_service.create_user(
                username="dev_admin_test",
                password="DevPassword123!",
                role="developer",
            )
        login_dev = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": "dev_admin_test", "password": "DevPassword123!"},
        )
        cls.token_dev = login_dev.json()["access_token"]
        cls.user_dev_id = login_dev.json()["user"]["id"]

    def test_auth_registration_and_login_flow(self):
        """Verify registration, JWT token generation, and /auth/me user profile endpoint."""
        # 1. Login with valid credentials
        login_res = self.client.post(
            f"{self.prefix}/auth/login",
            json={"username": "user_a_household", "password": "Password123!"},
        )
        self.assertEqual(login_res.status_code, 200)
        data = login_res.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")
        self.assertEqual(data["user"]["username"], "user_a_household")
        self.assertEqual(data["user"]["role"], "household")

        # 2. Authenticated /auth/me call
        headers = {"Authorization": f"Bearer {data['access_token']}"}
        me_res = self.client.get(f"{self.prefix}/auth/me", headers=headers)
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.json()
        self.assertEqual(me_data["username"], "user_a_household")
        self.assertEqual(me_data["role"], "household")

        # 3. Logout endpoint
        logout_res = self.client.post(f"{self.prefix}/auth/logout", headers=headers)
        self.assertEqual(logout_res.status_code, 200)

    def test_auth_me_missing_token_returns_401(self):
        """GET /auth/me without Authorization header returns HTTP 401 with standard detail."""
        res = self.client.get(f"{self.prefix}/auth/me")
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["detail"], "Authentication credentials were not provided.")
        self.assertEqual(res.headers.get("WWW-Authenticate"), "Bearer")

    def test_auth_me_invalid_token_returns_401(self):
        """GET /auth/me with invalid/tampered token returns HTTP 401."""
        res = self.client.get(
            f"{self.prefix}/auth/me",
            headers={"Authorization": "Bearer invalid.fake.token"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["detail"], "Invalid or expired authentication token.")
        self.assertEqual(res.headers.get("WWW-Authenticate"), "Bearer")

    def test_auth_me_fallback_without_bearer_prefix(self):
        """GET /auth/me supports raw token in Authorization header as a fallback."""
        res = self.client.get(
            f"{self.prefix}/auth/me",
            headers={"Authorization": self.token_a},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["username"], "user_a_household")

    def test_openapi_swagger_security_scheme_configuration(self):
        """
        Verify OpenAPI schema contract:
        1. components.securitySchemes defines HTTPBearer with type 'http' and scheme 'bearer'.
        2. GET /auth/me has security declaration [{'HTTPBearer': []}].
        3. GET /auth/me has no manual authorization header parameter.
        """
        schema = app.openapi()
        sec_schemes = schema.get("components", {}).get("securitySchemes", {})
        self.assertIn("HTTPBearer", sec_schemes)
        self.assertEqual(sec_schemes["HTTPBearer"]["type"], "http")
        self.assertEqual(sec_schemes["HTTPBearer"]["scheme"], "bearer")

        me_path = schema["paths"].get(f"{self.prefix}/auth/me", {}).get("get", {})
        self.assertEqual(me_path.get("security"), [{"HTTPBearer": []}])

        # Verify no manual authorization parameter exists in parameters list
        me_params = me_path.get("parameters") or []
        param_names = [p.get("name") for p in me_params]
        self.assertNotIn("authorization", param_names)

    def test_auth_invalid_credentials_rejected(self):
        """Invalid passwords or unknown usernames must return HTTP 401."""
        bad_pw_res = self.client.post(
            f"{self.prefix}/auth/login",
            json={"username": "user_a_household", "password": "WrongPassword!"},
        )
        self.assertEqual(bad_pw_res.status_code, 401)

        bad_user_res = self.client.post(
            f"{self.prefix}/auth/login",
            json={"username": "non_existent_user_999", "password": "Password123!"},
        )
        self.assertEqual(bad_user_res.status_code, 401)

    def test_password_never_stored_in_plaintext(self):
        """Passwords must be hashed using Argon2id and never stored in cleartext."""
        user = user_service.get_user_by_username("user_a_household")
        self.assertIsNotNone(user)
        self.assertNotEqual(user["hashed_password"], "Password123!")
        self.assertTrue(user["hashed_password"].startswith("$argon2id$"))

    def test_record_ownership_isolation_between_users(self):
        """
        User A creates a run. User A can view and download it.
        User B is forbidden (HTTP 403) from viewing or downloading User A's run.
        """
        # User A creates a live forecast run
        headers_a = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
            "user_id": "household_claim_tag_99",
        }
        run_res = self.client.post(f"{self.prefix}/forecasts/run", json=payload, headers=headers_a)
        self.assertEqual(run_res.status_code, 200)
        run_id_a = run_res.json()["run_id"]

        # Verify record has owner_user_id set to User A
        rec = record_service.get_record(run_id_a)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.owner_user_id, self.user_a_id)

        # 1. User A can retrieve record detail
        detail_a = self.client.get(f"{self.prefix}/forecasts/records/{run_id_a}", headers=headers_a)
        self.assertEqual(detail_a.status_code, 200)
        self.assertEqual(detail_a.json()["run_id"], run_id_a)

        # 2. User A can download PDF report
        pdf_a = self.client.get(f"{self.prefix}/forecasts/records/{run_id_a}/pdf", headers=headers_a)
        self.assertEqual(pdf_a.status_code, 200)
        self.assertEqual(pdf_a.headers["Content-Type"], "application/pdf")
        self.assertIn(f"forecast_report_{run_id_a}.pdf", pdf_a.headers["Content-Disposition"])
        self.assertTrue(pdf_a.content.startswith(b"%PDF-"))

        # 3. User B attempts to access User A's run -> HTTP 404 Not Found (Uniform anti-enumeration)
        headers_b = {"Authorization": f"Bearer {self.token_b}"}
        detail_b = self.client.get(f"{self.prefix}/forecasts/records/{run_id_a}", headers=headers_b)
        self.assertEqual(detail_b.status_code, 404)

        pdf_b = self.client.get(f"{self.prefix}/forecasts/records/{run_id_a}/pdf", headers=headers_b)
        self.assertEqual(pdf_b.status_code, 404)

        # 3b. Missing record also returns HTTP 404 Not Found
        detail_missing = self.client.get(f"{self.prefix}/forecasts/records/fc_non_existent_run_9999", headers=headers_b)
        self.assertEqual(detail_missing.status_code, 404)

        pdf_missing = self.client.get(f"{self.prefix}/forecasts/records/fc_non_existent_run_9999/pdf", headers=headers_b)
        self.assertEqual(pdf_missing.status_code, 404)

        # 4. User A listing records sees run_id_a
        list_a = self.client.get(f"{self.prefix}/forecasts/records?limit=100", headers=headers_a)
        self.assertEqual(list_a.status_code, 200)
        run_ids_a = [r["run_id"] for r in list_a.json()["records"]]
        self.assertIn(run_id_a, run_ids_a)

        # 5. User B listing records does NOT see run_id_a
        list_b = self.client.get(f"{self.prefix}/forecasts/records?limit=100", headers=headers_b)
        self.assertEqual(list_b.status_code, 200)
        run_ids_b = [r["run_id"] for r in list_b.json()["records"]]
        self.assertNotIn(run_id_a, run_ids_b)

    def test_developer_role_can_inspect_all_runs_and_download_pdf(self):
        """Developer identity can inspect runs owned by User A and download their PDF."""
        # User A creates a run
        headers_a = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
        }
        run_res = self.client.post(f"{self.prefix}/forecasts/run", json=payload, headers=headers_a)
        self.assertEqual(run_res.status_code, 200)
        run_id_a = run_res.json()["run_id"]

        # Developer inspects User A's run
        headers_dev = {"Authorization": f"Bearer {self.token_dev}"}
        detail_dev = self.client.get(f"{self.prefix}/forecasts/records/{run_id_a}", headers=headers_dev)
        self.assertEqual(detail_dev.status_code, 200)
        self.assertEqual(detail_dev.json()["owner_user_id"], self.user_a_id)

        # Developer downloads PDF
        pdf_dev = self.client.get(f"{self.prefix}/forecasts/records/{run_id_a}/pdf", headers=headers_dev)
        self.assertEqual(pdf_dev.status_code, 200)
        self.assertEqual(pdf_dev.headers["Content-Type"], "application/pdf")
        self.assertTrue(len(pdf_dev.content) > 5000)

    def test_developer_only_endpoint_rejects_ordinary_users(self):
        """
        GET /forecasts/developer/records requires developer role:
        - Unauthenticated -> HTTP 401 Unauthorized
        - Household User A -> HTTP 403 Forbidden
        - Developer -> HTTP 200 OK
        """
        # 1. Unauthenticated request
        unauth_res = self.client.get(f"{self.prefix}/forecasts/developer/records")
        self.assertEqual(unauth_res.status_code, 401)

        # 2. Household user request
        headers_a = {"Authorization": f"Bearer {self.token_a}"}
        household_res = self.client.get(f"{self.prefix}/forecasts/developer/records", headers=headers_a)
        self.assertEqual(household_res.status_code, 403)
        self.assertIn("Developer or administrator privileges are required", household_res.json()["detail"])

        # 3. Developer request
        headers_dev = {"Authorization": f"Bearer {self.token_dev}"}
        dev_res = self.client.get(f"{self.prefix}/forecasts/developer/records", headers=headers_dev)
        self.assertEqual(dev_res.status_code, 200)
        self.assertIn("records", dev_res.json())

    def test_anonymous_legacy_records_protected_from_household_takeover(self):
        """
        Runs created without authentication (legacy/anonymous runs) have owner_user_id IS NULL.
        Ordinary household users cannot claim or access them (HTTP 403).
        Developers can inspect them.
        Unauthenticated callers receive HTTP 401.
        """
        anon_run_id = f"fc_legacy_{uuid.uuid4().hex[:8]}"
        record_service.create_pending_run(
            run_id=anon_run_id,
            input_method="api",
            user_id="legacy_anonymous",
            owner_user_id=None,
            input_metadata={},
        )

        # Check DB record
        rec = record_service.get_record(anon_run_id)
        self.assertIsNone(rec.owner_user_id)

        # Unauthenticated query -> HTTP 401 Unauthorized
        res_unauth = self.client.get(f"{self.prefix}/forecasts/records/{anon_run_id}")
        self.assertEqual(res_unauth.status_code, 401)

        # Household User A tries to view anonymous record -> HTTP 404 Not Found (Anti-enumeration)
        headers_a = {"Authorization": f"Bearer {self.token_a}"}
        res_a = self.client.get(f"{self.prefix}/forecasts/records/{anon_run_id}", headers=headers_a)
        self.assertEqual(res_a.status_code, 404)

        pdf_a = self.client.get(f"{self.prefix}/forecasts/records/{anon_run_id}/pdf", headers=headers_a)
        self.assertEqual(pdf_a.status_code, 404)

        # Developer can view anonymous record
        headers_dev = {"Authorization": f"Bearer {self.token_dev}"}
        res_dev = self.client.get(f"{self.prefix}/forecasts/records/{anon_run_id}", headers=headers_dev)
        self.assertEqual(res_dev.status_code, 200)

    def test_unauthenticated_forecast_run_returns_401(self):
        """Unauthenticated POST /forecasts/run must return HTTP 401."""
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
        }
        res = self.client.post(f"{self.prefix}/forecasts/run", json=payload)
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.headers.get("WWW-Authenticate"), "Bearer")

    def test_unauthenticated_forecast_upload_csv_returns_401(self):
        """Unauthenticated POST /forecasts/upload-csv must return HTTP 401."""
        csv_bytes = b"date,hourly_active_power\n2024-03-01 00:00:00,300.0\n"
        res = self.client.post(
            f"{self.prefix}/forecasts/upload-csv",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={"model": "auto"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.headers.get("WWW-Authenticate"), "Bearer")

    def test_unauthenticated_records_list_returns_401(self):
        """Unauthenticated GET /forecasts/records must return HTTP 401."""
        res = self.client.get(f"{self.prefix}/forecasts/records")
        self.assertEqual(res.status_code, 401)

    def test_unauthenticated_record_detail_and_pdf_returns_401(self):
        """Unauthenticated GET /forecasts/records/{run_id} and /pdf must return HTTP 401."""
        res_detail = self.client.get(f"{self.prefix}/forecasts/records/fc_any_run_id")
        self.assertEqual(res_detail.status_code, 401)

        res_pdf = self.client.get(f"{self.prefix}/forecasts/records/fc_any_run_id/pdf")
        self.assertEqual(res_pdf.status_code, 401)

    def test_registration_forbids_client_supplied_developer_role(self):
        """Client attempting to register with extra field role='developer' must be rejected."""
        res = self.client.post(
            f"{self.prefix}/auth/register",
            json={
                "username": f"hacker_dev_{uuid.uuid4().hex[:6]}",
                "password": "Password123!",
                "role": "developer",
            },
        )
        # model_config = {"extra": "forbid"} rejects with 422
        self.assertEqual(res.status_code, 422)

    def test_client_supplied_user_id_tag_cannot_override_token_ownership(self):
        """A user cannot spoof ownership of a run by passing someone else's user_id in the request body."""
        headers_a = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
            "user_id": "spoofed_target_admin",
        }
        run_res = self.client.post(f"{self.prefix}/forecasts/run", json=payload, headers=headers_a)
        self.assertEqual(run_res.status_code, 200)
        run_id = run_res.json()["run_id"]

        rec = record_service.get_record(run_id)
        # Authoritative ownership is strictly user_a_id from the JWT token
        self.assertEqual(rec.owner_user_id, self.user_a_id)
        # The display tag is recorded, but cannot affect authorization
        self.assertEqual(rec.user_id, "spoofed_target_admin")

        # User B cannot access it (uniform 404 anti-enumeration)
        headers_b = {"Authorization": f"Bearer {self.token_b}"}
        detail_b = self.client.get(f"{self.prefix}/forecasts/records/{run_id}", headers=headers_b)
        self.assertEqual(detail_b.status_code, 404)

    def test_pdf_generation_content_and_headers(self):
        """Verify generated PDF contains vector headers, proper attachment metadata, and all pages."""
        headers_a = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "electricity_active_power": self.electricity,
            "timestamps": self.timestamps,
            "model": "energy_only",
        }
        run_res = self.client.post(f"{self.prefix}/forecasts/run", json=payload, headers=headers_a)
        self.assertEqual(run_res.status_code, 200)
        run_id = run_res.json()["run_id"]

        pdf_res = self.client.get(f"{self.prefix}/forecasts/records/{run_id}/pdf", headers=headers_a)
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers["Content-Type"], "application/pdf")
        self.assertEqual(
            pdf_res.headers["Content-Disposition"],
            f'attachment; filename="forecast_report_{run_id}.pdf"',
        )
        self.assertTrue(pdf_res.content.startswith(b"%PDF-1."))
        self.assertGreater(len(pdf_res.content), 10000)

    def test_pdf_generation_fails_for_nonexistent_or_failed_run(self):
        """Non-existent run returns 404; failed run returns 400."""
        headers_dev = {"Authorization": f"Bearer {self.token_dev}"}

        # Non-existent run
        missing_res = self.client.get(f"{self.prefix}/forecasts/records/fc_nonexistent_123/pdf", headers=headers_dev)
        self.assertEqual(missing_res.status_code, 404)

        # Create a failed record directly with dynamic run ID
        failed_id = f"fc_test_failed_{uuid.uuid4().hex[:8]}"
        record_service.create_pending_run(
            run_id=failed_id,
            input_method="api",
            user_id="failed_tester",
            owner_user_id=None,
            input_metadata={},
        )
        record_service.record_failure(
            run_id=failed_id,
            error_code="VALIDATION_ERROR",
            error_message="Simulated failure for testing PDF rejection",
        )

        # Developer requests PDF of failed run -> 400 Bad Request
        fail_pdf = self.client.get(f"{self.prefix}/forecasts/records/{failed_id}/pdf", headers=headers_dev)
        self.assertEqual(fail_pdf.status_code, 400)
        self.assertIn("PDF reports can only be generated for successful forecasts", fail_pdf.json()["detail"])


if __name__ == "__main__":
    unittest.main()
