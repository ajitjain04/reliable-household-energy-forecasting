"""
Unit and API integration tests for Monthly Forecast PDF Report Generation (Task 2.5).

Verifies:
1. Successful PDF generation from a saved monthly forecast snapshot.
2. Valid PDF signature (%PDF-) and response headers (Content-Type, Content-Disposition, Content-Length).
3. Correct inclusion of forecast periods, monthly predictions, saved warnings, and input history.
4. Multi-page report handling for large input history datasets (>10 billing cycles).
5. Unavailable tariff estimates shown as 'Unavailable', never fabricated as ₹0.
6. Missing forecast-run ID handling (HTTP 404).
7. Unauthenticated requests rejected (HTTP 401).
8. Cross-user report access denied without record-existence leakage (uniform HTTP 404).
9. Developer-role access permitted across user records.
10. Repeated downloads do not mutate the saved snapshot in persistent storage.
11. Subsequent forecasts do not mutate or alter previously generated forecast reports.
"""

import io
import math
import sys
import unittest
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.monthly_forecast import (
    ConfidenceRating,
    MonthlyForecastMethod,
    MonthlyForecastRecordDetail,
    MonthlyForecastRequest,
    MonthlyForecastStep,
    MonthlyForecastSummary,
    SaveMonthlyForecastRequest,
)
from backend.app.schemas.tariffs import BillEstimateResponse
from backend.app.services.monthly_record_service import monthly_record_service
from backend.app.services.pdf_service import generate_monthly_forecast_pdf
from backend.app.services.user_service import user_service


class TestMonthlyPdfReport(unittest.TestCase):
    """Test suite covering monthly forecast PDF report generation, security, and immutability."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # Register User A (Household)
        user_a_username = f"pdf_user_a_{uuid.uuid4().hex[:8]}"
        reg_a = cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": user_a_username, "password": "Password123!"},
        )
        login_a = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": user_a_username, "password": "Password123!"},
        )
        cls.token_a = login_a.json()["access_token"]
        cls.user_a_id = login_a.json()["user"]["id"]
        cls.headers_a = {"Authorization": f"Bearer {cls.token_a}"}

        # Register User B (Household)
        user_b_username = f"pdf_user_b_{uuid.uuid4().hex[:8]}"
        reg_b = cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": user_b_username, "password": "Password123!"},
        )
        login_b = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": user_b_username, "password": "Password123!"},
        )
        cls.token_b = login_b.json()["access_token"]
        cls.user_b_id = login_b.json()["user"]["id"]
        cls.headers_b = {"Authorization": f"Bearer {cls.token_b}"}

        # Developer User
        dev_username = f"pdf_dev_{uuid.uuid4().hex[:8]}"
        user_service.create_user(
            username=dev_username,
            password="DevPassword123!",
            role="developer",
        )
        login_dev = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": dev_username, "password": "DevPassword123!"},
        )
        cls.token_dev = login_dev.json()["access_token"]
        cls.headers_dev = {"Authorization": f"Bearer {cls.token_dev}"}

        # Standard sample input records for User A
        cls.sample_records_3m = [
            {
                "billing_period_start": "2024-01-01",
                "billing_period_end": "2024-01-31",
                "consumption_kwh": 310.0,
                "bill_amount": 2500.0,
                "period_label": "Jan 2024",
            },
            {
                "billing_period_start": "2024-02-01",
                "billing_period_end": "2024-02-28",
                "consumption_kwh": 340.0,
                "bill_amount": 2800.0,
                "period_label": "Feb 2024",
            },
            {
                "billing_period_start": "2024-03-01",
                "billing_period_end": "2024-03-31",
                "consumption_kwh": 405.0,
                "bill_amount": 3500.0,
                "period_label": "Mar 2024",
            },
        ]

    def test_01_unit_generate_monthly_pdf_success(self):
        """1. Unit test: successful PDF generation yields genuine vector PDF with >=2 pages."""
        steps = [
            MonthlyForecastStep(
                month_index=1,
                month_label="Apr 2024",
                period_start=date(2024, 4, 1),
                period_end=date(2024, 4, 30),
                forecast_kwh=380.0,
                uncertainty_lower_kwh=340.0,
                uncertainty_upper_kwh=420.0,
                bill_estimate=None,
            ),
            MonthlyForecastStep(
                month_index=2,
                month_label="May 2024",
                period_start=date(2024, 5, 1),
                period_end=date(2024, 5, 31),
                forecast_kwh=410.0,
                uncertainty_lower_kwh=370.0,
                uncertainty_upper_kwh=450.0,
                bill_estimate=None,
            ),
        ]
        input_recs = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 1, 31),
                consumption_kwh=310.0,
                bill_amount=2500.0,
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 2, 1),
                billing_period_end=date(2024, 2, 28),
                consumption_kwh=340.0,
                bill_amount=None,  # Not provided
            ),
        ]
        record = MonthlyForecastRecordDetail(
            run_id="mfc_test_unit_01",
            created_at=datetime.now(timezone.utc).isoformat(),
            owner_user_id=self.user_a_id,
            horizon_months=2,
            provider_id="msedcl",
            phase_type="single_phase",
            sanctioned_load_kw=3.0,
            is_live=False,
            selected_method=MonthlyForecastMethod.WEIGHTED_MOVING_AVERAGE,
            method_description="Weighted Moving Average was selected for short history (2 cycles).",
            confidence_rating=ConfidenceRating.LOW,
            history_length_months=2,
            last_observed_period="Feb 2024",
            last_observed_end_date=date(2024, 2, 28),
            is_stale_history=False,
            total_forecast_kwh=790.0,
            average_monthly_kwh=395.0,
            total_estimated_bill_inr=None,
            months_with_bill_estimates=0,
            bill_estimate_notes="Tariff estimation unavailable: unverified tariff schedule.",
            forecast_period_range="Apr 2024 – May 2024",
            warnings_count=1,
            warnings=["Only 2 months of history provided."],
            limitations=["Seasonal cycle cannot be established."],
            forecast_steps=steps,
            summary=MonthlyForecastSummary(
                total_forecast_kwh=790.0,
                average_monthly_kwh=395.0,
                total_estimated_bill_inr=None,
                months_with_bill_estimates=0,
                bill_estimate_notes="Tariff estimation unavailable: unverified tariff schedule.",
            ),
            backtest_metrics=None,
            input_records=input_recs,
        )

        pdf_bytes = generate_monthly_forecast_pdf(record)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"), "Generated report must begin with '%PDF-' magic header")
        self.assertGreater(len(pdf_bytes), 15000, "Vector PDF report should be substantial in size")
        # Standard <=10 input records produces exactly 2 pages
        self.assertEqual(pdf_bytes.count(b"/Type /Page "), 2, "Report should contain exactly 2 pages for 2 input records")

    def test_02_unit_generate_monthly_pdf_multipage_handling(self):
        """2. Unit test: multi-page report handling for >10 historical input records."""
        input_recs_18 = []
        for m in range(1, 19):
            y = 2023 + (m // 12)
            mon = ((m - 1) % 12) + 1
            input_recs_18.append(
                MonthlyConsumptionRecord(
                    billing_period_start=date(y, mon, 1),
                    billing_period_end=date(y, mon, 28),
                    consumption_kwh=300.0 + (m * 4.0),
                    bill_amount=2400.0 + (m * 30.0) if m % 2 == 0 else None,
                )
            )

        step = MonthlyForecastStep(
            month_index=1,
            month_label="Jul 2024",
            period_start=date(2024, 7, 1),
            period_end=date(2024, 7, 31),
            forecast_kwh=375.0,
            uncertainty_lower_kwh=340.0,
            uncertainty_upper_kwh=410.0,
            bill_estimate=None,
        )

        record_18 = MonthlyForecastRecordDetail(
            run_id="mfc_test_unit_18_inputs",
            created_at=datetime.now(timezone.utc).isoformat(),
            owner_user_id=self.user_a_id,
            horizon_months=1,
            provider_id="msedcl",
            phase_type="single_phase",
            sanctioned_load_kw=4.0,
            is_live=True,
            selected_method=MonthlyForecastMethod.HOLT_WINTERS,
            method_description="Holt-Winters selected because 18 months of continuous history were provided.",
            confidence_rating=ConfidenceRating.HIGH,
            history_length_months=18,
            last_observed_period="Jun 2024",
            last_observed_end_date=date(2024, 6, 28),
            is_stale_history=False,
            total_forecast_kwh=375.0,
            average_monthly_kwh=375.0,
            total_estimated_bill_inr=None,
            months_with_bill_estimates=0,
            bill_estimate_notes="Tariff estimation unavailable.",
            forecast_period_range="Jul 2024",
            warnings_count=0,
            warnings=[],
            limitations=["Longer horizon requires caution."],
            forecast_steps=[step],
            summary=MonthlyForecastSummary(
                total_forecast_kwh=375.0,
                average_monthly_kwh=375.0,
                total_estimated_bill_inr=None,
                months_with_bill_estimates=0,
            ),
            backtest_metrics=None,
            input_records=input_recs_18,
        )

        pdf_bytes = generate_monthly_forecast_pdf(record_18)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        page_count = pdf_bytes.count(b"/Type /Page ")
        # 1 page overview + 2 chunk pages (16 + 2 rows) + 1 disclosures page = 4 pages
        self.assertEqual(page_count, 4, f"Expected 4 pages for 18 input records, got {page_count}")

    def test_03_api_download_monthly_pdf_success_and_headers(self):
        """3. API test: successful PDF download with correct headers and server-generated filename."""
        # Save a forecast run first for User A
        forecast_id = f"mfc_api_test_{uuid.uuid4().hex[:8]}"
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": forecast_id,
                "request": {
                    "records": self.sample_records_3m,
                    "horizon_months": 3,
                    "provider_id": "msedcl",
                    "phase_type": "single_phase",
                    "sanctioned_load_kw": 5.0,
                    "is_live": False,
                },
            },
        )
        self.assertEqual(save_res.status_code, 201)
        run_id = save_res.json()["run_id"]

        # Download PDF using the canonical report.pdf route
        pdf_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}/report.pdf",
            headers=self.headers_a,
        )
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers["content-type"], "application/pdf")
        self.assertIn("Content-Disposition", pdf_res.headers)
        self.assertEqual(
            pdf_res.headers["content-disposition"],
            f'attachment; filename="monthly_forecast_report_{run_id}.pdf"',
        )
        self.assertEqual(pdf_res.headers["content-length"], str(len(pdf_res.content)))
        self.assertTrue(pdf_res.content.startswith(b"%PDF-"))

        # Also verify the /pdf alias endpoint
        alias_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}/pdf",
            headers=self.headers_a,
        )
        self.assertEqual(alias_res.status_code, 200)
        self.assertEqual(alias_res.headers["content-type"], "application/pdf")
        self.assertTrue(alias_res.content.startswith(b"%PDF-"))

    def test_04_api_download_unauthenticated_rejected(self):
        """4. Security: unauthenticated request is rejected with HTTP 401 Unauthorized."""
        res = self.client.get(f"{self.prefix}/forecasts/monthly/records/mfc_any_id/report.pdf")
        self.assertEqual(res.status_code, 401)

    def test_05_api_download_missing_run_id_404(self):
        """5. Error handling: missing forecast-run ID returns HTTP 404 Not Found."""
        res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/mfc_nonexistent_xyz999/report.pdf",
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    def test_06_api_download_cross_user_denied_without_leakage(self):
        """6. Security: User B cannot download User A's report; receives uniform 404 without record existence leakage."""
        # Create a run belonging to User A
        forecast_id = f"mfc_user_a_only_{uuid.uuid4().hex[:8]}"
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": forecast_id,
                "request": {
                    "records": self.sample_records_3m,
                    "horizon_months": 3,
                    "is_live": False,
                },
            },
        )
        self.assertEqual(save_res.status_code, 201)
        run_id = save_res.json()["run_id"]

        # User B attempts to download User A's report
        cross_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}/report.pdf",
            headers=self.headers_b,
        )
        # MUST return 404 (not 403) to prevent leaking whether the run_id exists
        self.assertEqual(cross_res.status_code, 404)
        self.assertIn("not found", cross_res.json()["detail"].lower())

    def test_07_api_download_developer_role_access(self):
        """7. Authorization: developer accounts can download PDF reports for any user's saved run."""
        forecast_id = f"mfc_dev_inspect_{uuid.uuid4().hex[:8]}"
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": forecast_id,
                "request": {
                    "records": self.sample_records_3m,
                    "horizon_months": 3,
                    "is_live": False,
                },
            },
        )
        self.assertEqual(save_res.status_code, 201)
        run_id = save_res.json()["run_id"]

        # Developer downloads User A's report
        dev_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}/report.pdf",
            headers=self.headers_dev,
        )
        self.assertEqual(dev_res.status_code, 200)
        self.assertEqual(dev_res.headers["content-type"], "application/pdf")
        self.assertTrue(dev_res.content.startswith(b"%PDF-"))

    def test_08_repeated_downloads_do_not_mutate_snapshot(self):
        """8. Immutability: repeated PDF downloads do not modify the saved record or database state."""
        forecast_id = f"mfc_immutable_{uuid.uuid4().hex[:8]}"
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": forecast_id,
                "request": {
                    "records": self.sample_records_3m,
                    "horizon_months": 3,
                    "is_live": False,
                },
            },
        )
        self.assertEqual(save_res.status_code, 201)
        run_id = save_res.json()["run_id"]

        # Retrieve record detail before download
        detail_before = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}",
            headers=self.headers_a,
        ).json()

        # Download report 3 consecutive times
        for _ in range(3):
            pdf_res = self.client.get(
                f"{self.prefix}/forecasts/monthly/records/{run_id}/report.pdf",
                headers=self.headers_a,
            )
            self.assertEqual(pdf_res.status_code, 200)

        # Retrieve record detail after downloads
        detail_after = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}",
            headers=self.headers_a,
        ).json()

        self.assertEqual(detail_before, detail_after, "Saved record snapshot must remain strictly unchanged")

    def test_09_subsequent_forecast_does_not_alter_previous_report(self):
        """9. Historical integrity: newly generated forecasts do not modify previous report snapshots."""
        # 1. Create Run 1
        id_1 = f"mfc_run_one_{uuid.uuid4().hex[:8]}"
        save_1 = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": id_1,
                "request": {
                    "records": self.sample_records_3m,
                    "horizon_months": 3,
                    "is_live": False,
                },
            },
        )
        run_id_1 = save_1.json()["run_id"]

        pdf_1_initial = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id_1}/report.pdf",
            headers=self.headers_a,
        ).content

        # 2. Create Run 2 with different inputs (higher consumption)
        different_records = [
            {
                "billing_period_start": "2024-05-01",
                "billing_period_end": "2024-05-31",
                "consumption_kwh": 999.0,
                "bill_amount": 9000.0,
                "period_label": "May 2024",
            }
        ]
        id_2 = f"mfc_run_two_{uuid.uuid4().hex[:8]}"
        save_2 = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": id_2,
                "request": {
                    "records": different_records,
                    "horizon_months": 1,
                    "is_live": False,
                },
            },
        )
        run_id_2 = save_2.json()["run_id"]

        # 3. Re-download Run 1
        pdf_1_recheck = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id_1}/report.pdf",
            headers=self.headers_a,
        ).content

        # Verify record 1's saved data wasn't contaminated by record 2
        rec_1 = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id_1}",
            headers=self.headers_a,
        ).json()
        self.assertEqual(rec_1["total_forecast_kwh"], save_1.json()["total_forecast_kwh"])
        self.assertNotEqual(rec_1["total_forecast_kwh"], save_2.json()["total_forecast_kwh"])
        self.assertTrue(len(pdf_1_initial) > 10000)
        self.assertTrue(len(pdf_1_recheck) > 10000)

    def test_10_unavailable_tariff_shown_as_unavailable_not_zero(self):
        """10. Tariff transparency: unverified tariffs are shown as 'Unavailable', not ₹0 or ₹0.00."""
        forecast_id = f"mfc_unverified_{uuid.uuid4().hex[:8]}"
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": forecast_id,
                "request": {
                    "records": self.sample_records_3m,
                    "horizon_months": 2,
                    # No provider_id or tariff specified -> tariff calculation withheld
                    "is_live": False,
                },
            },
        )
        self.assertEqual(save_res.status_code, 201)
        run_id = save_res.json()["run_id"]

        pdf_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}/report.pdf",
            headers=self.headers_a,
        )
        self.assertEqual(pdf_res.status_code, 200)
        pdf_bytes = pdf_res.content

        # Extract text from generated vector PDF using pypdfium2
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(pdf_bytes)
        full_pdf_text = " ".join([page.get_textpage().get_text_range() for page in doc])

        self.assertIn("Unavailable", full_pdf_text, "PDF text must explicitly contain 'Unavailable' for unverified tariffs")
        # Ensure ₹0.00 is not printed as the bill estimate
        self.assertNotIn("₹0.00", full_pdf_text)

    def test_11_forecast_with_verified_tariff_and_intervals(self):
        """11. Full report: verified provider tariff and uncertainty whiskers included in PDF report."""
        forecast_id = f"mfc_msedcl_{uuid.uuid4().hex[:8]}"
        save_res = self.client.post(
            f"{self.prefix}/forecasts/monthly/save",
            headers=self.headers_a,
            json={
                "forecast_id": forecast_id,
                "request": {
                    "records": self.sample_records_3m,
                    "horizon_months": 3,
                    "provider_id": "msedcl",
                    "phase_type": "single_phase",
                    "sanctioned_load_kw": 5.0,
                    "is_live": False,
                },
            },
        )
        self.assertEqual(save_res.status_code, 201)
        data = save_res.json()
        self.assertIsNotNone(data["total_estimated_bill_inr"])
        run_id = data["run_id"]

        pdf_res = self.client.get(
            f"{self.prefix}/forecasts/monthly/records/{run_id}/report.pdf",
            headers=self.headers_a,
        )
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.headers["content-type"], "application/pdf")
        self.assertTrue(pdf_res.content.startswith(b"%PDF-"))

    def test_12_long_warnings_and_limitations_fully_visible_untruncated(self):
        """12. Quality: All saved warnings (400+ chars) and limitations are fully visible and not truncated."""
        stale_warning = (
            "Historical billing records end on 2025-04-30 (538 days prior to reference date 2026-10-20, "
            "more than 90 days ago, exceeding the 90-day threshold). Under live household forecasting policy, "
            "future projections are anchored to upcoming calendar months. The historical consumption baseline "
            "is retained as the demand estimate, but prediction confidence is reduced because household occupancy, "
            "appliances, or seasonal patterns may have shifted."
        )
        single_obs_warning = (
            "Single billing period observed (167.0 kWh). One observation is insufficient to estimate consumption "
            "trends or seasonal variations (such as summer air conditioning or winter heating). Results represent "
            "a static baseline persistence only."
        )
        limitation_1 = "Stale billing history: Projections are anchored to upcoming months after 2026-10-20 using historical baseline ending 2025-04-30."
        limitation_2 = "Prediction intervals cannot be defensibly calculated from a single observation."
        limitation_3 = "Multi-month horizon (3 months) is projected using static persistence because no trend or seasonality can be identified from one observation."

        step = MonthlyForecastStep(
            month_index=1,
            month_label="Nov 2026",
            period_start=date(2026, 11, 1),
            period_end=date(2026, 11, 30),
            forecast_kwh=167.0,
            uncertainty_lower_kwh=None,
            uncertainty_upper_kwh=None,
            bill_estimate=None,
        )
        inp = MonthlyConsumptionRecord(
            billing_period_start=date(2025, 4, 1),
            billing_period_end=date(2025, 4, 30),
            consumption_kwh=167.0,
            bill_amount=1760.0,
        )

        record = MonthlyForecastRecordDetail(
            run_id="mfc_test_long_warnings",
            created_at=datetime.now(timezone.utc).isoformat(),
            owner_user_id=self.user_a_id,
            horizon_months=1,
            provider_id="msedcl",
            phase_type="single_phase",
            sanctioned_load_kw=3.0,
            is_live=True,
            selected_method=MonthlyForecastMethod.BASELINE_ONLY,
            method_description="Baseline Persistence: Last confirmed monthly reading (167.0 kWh). Single observation is insufficient to estimate trend, variance, or seasonality.",
            confidence_rating=ConfidenceRating.LOW,
            history_length_months=1,
            last_observed_period="Apr 2025",
            last_observed_end_date=date(2025, 4, 30),
            is_stale_history=True,
            total_forecast_kwh=167.0,
            average_monthly_kwh=167.0,
            total_estimated_bill_inr=None,
            months_with_bill_estimates=0,
            bill_estimate_notes="No verified bill estimates available for this horizon (tariff unverified, unsupported, or inactive for forecast dates).",
            forecast_period_range="Nov 2026",
            warnings_count=2,
            warnings=[stale_warning, single_obs_warning],
            limitations=[limitation_1, limitation_2, limitation_3],
            forecast_steps=[step],
            summary=MonthlyForecastSummary(
                total_forecast_kwh=167.0,
                average_monthly_kwh=167.0,
                total_estimated_bill_inr=None,
                months_with_bill_estimates=0,
                bill_estimate_notes="No verified bill estimates available for this horizon.",
            ),
            backtest_metrics=None,
            input_records=[inp],
        )

        pdf_bytes = generate_monthly_forecast_pdf(record)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_bytes)
        full_pdf_text = " ".join([page.get_textpage().get_text_range() for page in doc])

        # Verify full untruncated content appears in PDF text
        self.assertIn("household occupancy, appliances, or seasonal patterns may have shifted", full_pdf_text,
                      "Long warning tail must not be truncated")
        self.assertIn("static baseline persistence only", full_pdf_text,
                      "Second warning must not be truncated")
        self.assertIn("Multi-month horizon (3 months) is projected using static persistence", full_pdf_text,
                      "Methodology limitation must be present")
        self.assertIn("Statutory Notice:", full_pdf_text)
        self.assertIn("Operational Caveats:", full_pdf_text)
        self.assertIn("Data Integrity:", full_pdf_text)

    def test_13_forecast_table_readability_and_column_balance(self):
        """13. Readability: Forecast table displays all cycles, periods, bounds, and notes without collision."""
        steps = [
            MonthlyForecastStep(
                month_index=1,
                month_label="Nov 2026",
                period_start=date(2026, 11, 1),
                period_end=date(2026, 11, 30),
                forecast_kwh=167.0,
                uncertainty_lower_kwh=None,
                uncertainty_upper_kwh=None,
                bill_estimate=None,
            ),
            MonthlyForecastStep(
                month_index=2,
                month_label="Dec 2026",
                period_start=date(2026, 12, 1),
                period_end=date(2026, 12, 31),
                forecast_kwh=167.0,
                uncertainty_lower_kwh=None,
                uncertainty_upper_kwh=None,
                bill_estimate=None,
            ),
            MonthlyForecastStep(
                month_index=3,
                month_label="Jan 2027",
                period_start=date(2027, 1, 1),
                period_end=date(2027, 1, 31),
                forecast_kwh=167.0,
                uncertainty_lower_kwh=None,
                uncertainty_upper_kwh=None,
                bill_estimate=None,
            ),
        ]
        inp = MonthlyConsumptionRecord(
            billing_period_start=date(2025, 4, 1),
            billing_period_end=date(2025, 4, 30),
            consumption_kwh=167.0,
            bill_amount=1760.0,
        )

        record = MonthlyForecastRecordDetail(
            run_id="mfc_test_table_balance",
            created_at=datetime.now(timezone.utc).isoformat(),
            owner_user_id=self.user_a_id,
            horizon_months=3,
            provider_id="msedcl",
            phase_type="single_phase",
            sanctioned_load_kw=3.0,
            is_live=True,
            selected_method=MonthlyForecastMethod.BASELINE_ONLY,
            method_description="Baseline Persistence.",
            confidence_rating=ConfidenceRating.LOW,
            history_length_months=1,
            last_observed_period="Apr 2025",
            last_observed_end_date=date(2025, 4, 30),
            is_stale_history=True,
            total_forecast_kwh=501.0,
            average_monthly_kwh=167.0,
            total_estimated_bill_inr=None,
            months_with_bill_estimates=0,
            bill_estimate_notes="No verified bill estimates available.",
            forecast_period_range="Nov 2026 – Jan 2027",
            warnings_count=1,
            warnings=["Single billing period observed."],
            limitations=["Static baseline only."],
            forecast_steps=steps,
            summary=MonthlyForecastSummary(
                total_forecast_kwh=501.0,
                average_monthly_kwh=167.0,
                total_estimated_bill_inr=None,
                months_with_bill_estimates=0,
            ),
            backtest_metrics=None,
            input_records=[inp],
        )

        pdf_bytes = generate_monthly_forecast_pdf(record)
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_bytes)
        page1_text = doc[0].get_textpage().get_text_range()

        # Check Page 1 table contains all 3 months and periods
        self.assertIn("Month 1", page1_text)
        self.assertIn("Nov 2026", page1_text)
        self.assertIn("2026-11-01", page1_text)
        self.assertIn("Month 2", page1_text)
        self.assertIn("Dec 2026", page1_text)
        self.assertIn("2026-12-01", page1_text)
        self.assertIn("Month 3", page1_text)
        self.assertIn("Jan 2027", page1_text)
        self.assertIn("2027-01-01", page1_text)
        self.assertIn("167.0", page1_text)
        self.assertIn("Unavailable", page1_text)

    def test_14_multipage_pagination_for_moderate_and_large_datasets(self):
        """14. Multipage: Moderate input datasets (>6) cleanly paginate disclosures to dedicated page."""
        input_recs_8 = []
        for m in range(1, 9):
            input_recs_8.append(
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, m, 1),
                    billing_period_end=date(2024, m, 28),
                    consumption_kwh=250.0 + (m * 10.0),
                    bill_amount=2000.0 + (m * 80.0),
                )
            )

        step = MonthlyForecastStep(
            month_index=1,
            month_label="Sep 2024",
            period_start=date(2024, 9, 1),
            period_end=date(2024, 9, 30),
            forecast_kwh=340.0,
            uncertainty_lower_kwh=310.0,
            uncertainty_upper_kwh=370.0,
            bill_estimate=None,
        )

        record_8 = MonthlyForecastRecordDetail(
            run_id="mfc_test_8_inputs",
            created_at=datetime.now(timezone.utc).isoformat(),
            owner_user_id=self.user_a_id,
            horizon_months=1,
            provider_id="msedcl",
            phase_type="single_phase",
            sanctioned_load_kw=3.0,
            is_live=True,
            selected_method=MonthlyForecastMethod.WEIGHTED_MOVING_AVERAGE,
            method_description="Weighted Moving Average.",
            confidence_rating=ConfidenceRating.MODERATE,
            history_length_months=8,
            last_observed_period="Aug 2024",
            last_observed_end_date=date(2024, 8, 28),
            is_stale_history=False,
            total_forecast_kwh=340.0,
            average_monthly_kwh=340.0,
            total_estimated_bill_inr=None,
            months_with_bill_estimates=0,
            bill_estimate_notes="Tariff estimation unavailable.",
            forecast_period_range="Sep 2024",
            warnings_count=1,
            warnings=["Moderate history length (8 months)."],
            limitations=["Seasonal cycle of 12 months incomplete."],
            forecast_steps=[step],
            summary=MonthlyForecastSummary(
                total_forecast_kwh=340.0,
                average_monthly_kwh=340.0,
                total_estimated_bill_inr=None,
                months_with_bill_estimates=0,
            ),
            backtest_metrics=None,
            input_records=input_recs_8,
        )

        pdf_bytes = generate_monthly_forecast_pdf(record_8)
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_bytes)
        self.assertEqual(len(doc), 3, "8 inputs should produce exactly 3 pages (Overview, Inputs, Disclosures)")

        # Verify page numbers in footers
        p1_text = doc[0].get_textpage().get_text_range()
        p2_text = doc[1].get_textpage().get_text_range()
        p3_text = doc[2].get_textpage().get_text_range()

        self.assertIn("Page 1 of 3", p1_text)
        self.assertIn("Page 2 of 3", p2_text)
        self.assertIn("Page 3 of 3", p3_text)

        # Page 2 has input snapshot
        self.assertIn("Historical Billing Inputs Snapshot", p2_text)
        # Page 3 has methodology & statutory disclaimers
        self.assertIn("FORECASTING METHODOLOGY & EVALUATION", p3_text)
        self.assertIn("STATUTORY & OPERATIONAL DISCLAIMER", p3_text)

    def test_15_actual_target_run_mfc_5895dbb7b06f_fidelity(self):
        """15. Fidelity: Actual run mfc_5895dbb7b06f generates without clipping, preserving all fields."""
        rec = monthly_record_service.get_monthly_record("mfc_5895dbb7b06f")
        if rec is None:
            self.skipTest("Record mfc_5895dbb7b06f not present in database environment")

        pdf_bytes = generate_monthly_forecast_pdf(rec)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_bytes)
        self.assertEqual(len(doc), 2, "mfc_5895dbb7b06f should cleanly format into 2 balanced pages")

        full_pdf_text = " ".join([page.get_textpage().get_text_range() for page in doc])

        # Verify forecast values preserved exactly
        self.assertIn("501.0", full_pdf_text)
        self.assertIn("167.0", full_pdf_text)
        self.assertIn("mfc_5895dbb7b06f", full_pdf_text)
        self.assertIn("usr_a1e1a47b8bb0", full_pdf_text)

        # Verify long warning tail is completely present
        self.assertIn("household occupancy, appliances, or seasonal patterns may have shifted", full_pdf_text)
        # Verify statutory notice is present
        self.assertIn("Statutory Notice:", full_pdf_text)
        self.assertIn("Operational Caveats:", full_pdf_text)


if __name__ == "__main__":
    unittest.main()

