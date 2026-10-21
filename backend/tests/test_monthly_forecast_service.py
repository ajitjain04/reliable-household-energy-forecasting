"""
Unit and Integration Tests for Monthly Household Electricity Forecasting Pipeline.

Covers all 12 specified scenarios:
1. One historical month (N=1): baseline-only, limitation warning, no fabricated intervals.
2. Several consecutive months (N=3, 5): valid chronological forecast months.
3. Missing months: preserves gaps and reports them; does not synthesize observations.
4. Duplicate periods: rejected clearly (ValueError / HTTP 422).
5. Overlapping periods: rejected clearly (ValueError / HTTP 422).
6. Negative or invalid kWh: rejected by Pydantic validation.
7. Unsorted input: sorted chronologically according to schema conventions.
8. Forecast horizons: verified for 1, 3, and 6 months.
9. Longer synthetic history (N=12): chronological backtesting evaluation without future-data leakage.
10. Old historical test records: accepted for offline testing with stale-history warning.
11. Unknown or unverified tariff: consumption forecasting succeeds without fabricated bill estimate.
12. Authentication and invalid-input behavior for the API (POST /api/v1/forecasts/monthly/run).
"""

import calendar
import sys
import unittest
import uuid
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.monthly_forecast import (
    ConfidenceRating,
    MonthlyForecastMethod,
    MonthlyForecastRequest,
    MonthlyForecastResponse,
)
from backend.app.services.monthly_forecast_service import monthly_forecast_service


class TestMonthlyForecastService(unittest.TestCase):
    """Unit test suite for MonthlyForecastService business logic."""

    def test_01_single_month_baseline_only(self):
        """1. One historical month: returns a baseline-only forecast and a clear limitation warning."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                    period_label="Apr 2025",
                )
            ],
            horizon_months=3,
        )
        res = monthly_forecast_service.generate_forecast(req)

        self.assertEqual(res.history_length_months, 1)
        self.assertEqual(res.selected_method, MonthlyForecastMethod.BASELINE_ONLY)
        self.assertEqual(res.confidence_rating, ConfidenceRating.LOW)
        self.assertEqual(len(res.forecast_steps), 3)

        # All projected steps must equal the single confirmed baseline
        for step in res.forecast_steps:
            self.assertEqual(step.forecast_kwh, 167.0)
            self.assertIsNone(step.uncertainty_lower_kwh, "Single observation must not fabricate lower bound")
            self.assertIsNone(step.uncertainty_upper_kwh, "Single observation must not fabricate upper bound")

        # Must report clear limitation warnings
        self.assertTrue(any("single billing period" in w.lower() for w in res.warnings))
        self.assertTrue(any("cannot be defensibly calculated" in lim.lower() for lim in res.limitations))
        self.assertIsNone(res.backtest_metrics, "Cannot backtest on a single point")

    def test_02_consecutive_months_chronological_projection(self):
        """2. Several consecutive months: returns valid chronological forecast months."""
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 1),
                billing_period_end=date(2025, 1, 31),
                consumption_kwh=150.0,
                period_label="Jan 2025",
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 2, 1),
                billing_period_end=date(2025, 2, 28),
                consumption_kwh=160.0,
                period_label="Feb 2025",
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 3, 1),
                billing_period_end=date(2025, 3, 31),
                consumption_kwh=170.0,
                period_label="Mar 2025",
            ),
        ]
        req = MonthlyForecastRequest(records=records, horizon_months=3, is_live=False)
        res = monthly_forecast_service.generate_forecast(req)

        self.assertEqual(res.history_length_months, 3)
        self.assertEqual(res.selected_method, MonthlyForecastMethod.WEIGHTED_MOVING_AVERAGE)
        self.assertEqual(len(res.forecast_steps), 3)

        # Verify chronological dates
        self.assertEqual(res.forecast_steps[0].period_start, date(2025, 4, 1))
        self.assertEqual(res.forecast_steps[0].period_end, date(2025, 4, 30))
        self.assertEqual(res.forecast_steps[0].month_label, "Apr 2025")

        self.assertEqual(res.forecast_steps[1].period_start, date(2025, 5, 1))
        self.assertEqual(res.forecast_steps[1].period_end, date(2025, 5, 31))
        self.assertEqual(res.forecast_steps[1].month_label, "May 2025")

        self.assertEqual(res.forecast_steps[2].period_start, date(2025, 6, 1))
        self.assertEqual(res.forecast_steps[2].period_end, date(2025, 6, 30))
        self.assertEqual(res.forecast_steps[2].month_label, "Jun 2025")

        # For N=3, empirical uncertainty bounds must exist and be non-negative
        self.assertIsNotNone(res.forecast_steps[0].uncertainty_lower_kwh)
        self.assertIsNotNone(res.forecast_steps[0].uncertainty_upper_kwh)
        self.assertGreaterEqual(res.forecast_steps[0].uncertainty_lower_kwh, 0.0)
        self.assertGreater(res.forecast_steps[0].uncertainty_upper_kwh, res.forecast_steps[0].forecast_kwh)

    def test_03_missing_months_preserved_and_reported(self):
        """3. Missing months: preserves gaps and reports them; does not synthesize observations."""
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 1, 31),
                consumption_kwh=140.0,
            ),
            # Missing February & March 2024 (gap of ~60 days)
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 4, 1),
                billing_period_end=date(2024, 4, 30),
                consumption_kwh=160.0,
            ),
        ]
        req = MonthlyForecastRequest(records=records, horizon_months=1)
        res = monthly_forecast_service.generate_forecast(req)

        self.assertEqual(res.history_length_months, 2, "Must not inject synthetic records into history length")
        self.assertTrue(
            any("gap of" in w.lower() and "no synthetic observations" in w.lower() for w in res.warnings),
            "Must report gap warning without synthesizing observations",
        )

    def test_04_duplicate_periods_rejected(self):
        """4. Duplicate periods: rejected clearly."""
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 1),
                billing_period_end=date(2025, 1, 31),
                consumption_kwh=150.0,
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 1),
                billing_period_end=date(2025, 1, 31),
                consumption_kwh=155.0,
            ),
        ]
        with self.assertRaises(ValueError) as ctx:
            MonthlyForecastRequest(records=records, horizon_months=1)
            # Or direct service call
            monthly_forecast_service.generate_forecast(
                MonthlyForecastRequest.model_construct(records=records, horizon_months=1)
            )
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_05_overlapping_periods_rejected(self):
        """5. Overlapping periods: rejected clearly."""
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 1),
                billing_period_end=date(2025, 1, 31),
                consumption_kwh=150.0,
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 25),  # Overlaps with Jan 1 - Jan 31
                billing_period_end=date(2025, 2, 25),
                consumption_kwh=160.0,
            ),
        ]
        with self.assertRaises(ValueError) as ctx:
            monthly_forecast_service.generate_forecast(
                MonthlyForecastRequest.model_construct(records=records, horizon_months=1)
            )
        self.assertIn("overlap", str(ctx.exception).lower())

    def test_06_negative_or_invalid_kwh_rejected(self):
        """6. Negative or invalid kWh: rejected by Pydantic validation."""
        with self.assertRaises(ValidationError):
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 1),
                billing_period_end=date(2025, 1, 31),
                consumption_kwh=-25.0,  # Negative consumption
            )

    def test_07_unsorted_input_automatically_sorted(self):
        """7. Unsorted input: sorted chronologically according to existing schema conventions."""
        # Provide March, then January, then February
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 3, 1),
                billing_period_end=date(2025, 3, 31),
                consumption_kwh=170.0,
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 1),
                billing_period_end=date(2025, 1, 31),
                consumption_kwh=150.0,
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 2, 1),
                billing_period_end=date(2025, 2, 28),
                consumption_kwh=160.0,
            ),
        ]
        req = MonthlyForecastRequest(records=records, horizon_months=2, is_live=False)
        res = monthly_forecast_service.generate_forecast(req)

        # First forecast step must be April (following sorted March record)
        self.assertEqual(res.forecast_steps[0].period_start, date(2025, 4, 1))
        self.assertEqual(res.forecast_steps[0].month_label, "Apr 2025")

    def test_08_forecast_horizons_1_3_and_6_months(self):
        """8. Forecast horizons of 1, 3, and 6 months."""
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 1, 1),
                billing_period_end=date(2025, 1, 31),
                consumption_kwh=150.0,
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 2, 1),
                billing_period_end=date(2025, 2, 28),
                consumption_kwh=160.0,
            ),
        ]
        for h in [1, 3, 6]:
            req = MonthlyForecastRequest(records=records, horizon_months=h)
            res = monthly_forecast_service.generate_forecast(req)
            self.assertEqual(len(res.forecast_steps), h)
            self.assertEqual(res.forecast_steps[-1].month_index, h)

        # Single-month history (N=1) must also project exactly 1, 3, or 6 steps
        single_record = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2025, 4, 1),
                billing_period_end=date(2025, 4, 30),
                consumption_kwh=167.0,
                bill_amount=1760.0,
                period_label="Apr 2025",
            )
        ]
        for h in [1, 3, 6]:
            req_single = MonthlyForecastRequest(records=single_record, horizon_months=h)
            res_single = monthly_forecast_service.generate_forecast(req_single)
            self.assertEqual(len(res_single.forecast_steps), h)
            self.assertEqual(res_single.forecast_steps[-1].month_index, h)
            self.assertEqual(res_single.summary.total_forecast_kwh, round(167.0 * h, 2))
            self.assertEqual(res_single.summary.average_monthly_kwh, 167.0)
            for step in res_single.forecast_steps:
                self.assertEqual(step.forecast_kwh, 167.0)


    def test_09_longer_synthetic_history_chronological_evaluation_no_leakage(self):
        """9. Longer synthetic history (N>=13): chronological evaluation with no future-data leakage."""
        # 14 months with synthetic seasonal summer peak (Apr-Jun) and winter dip
        records = []
        monthly_pattern = [120, 130, 150, 210, 240, 220, 180, 160, 150, 140, 130, 125, 122, 133]
        for idx, kwh in enumerate(monthly_pattern):
            tot_m = 2024 * 12 + idx
            y = tot_m // 12
            m = (tot_m % 12) + 1
            s_date = date(y, m, 1)
            _, days = calendar.monthrange(y, m)
            e_date = date(y, m, days)
            records.append(
                MonthlyConsumptionRecord(
                    billing_period_start=s_date,
                    billing_period_end=e_date,
                    consumption_kwh=float(kwh),
                    period_label=s_date.strftime("%b %Y"),
                )
            )

        req = MonthlyForecastRequest(records=records, horizon_months=3, is_live=False)
        res = monthly_forecast_service.generate_forecast(req)

        self.assertEqual(res.history_length_months, 14)
        self.assertEqual(res.selected_method, MonthlyForecastMethod.SEASONAL_NAIVE)
        self.assertEqual(res.confidence_rating, ConfidenceRating.HIGH)

        # Backtest metrics must be computed out-of-sample (for N=14, k_eval=2)
        self.assertIsNotNone(res.backtest_metrics)
        self.assertEqual(res.backtest_metrics.backtest_periods, 2)
        self.assertGreaterEqual(res.backtest_metrics.mae_kwh, 0.0)
        self.assertGreaterEqual(res.backtest_metrics.rmse_kwh, 0.0)

        # Next forecast step after Feb 2025 must be Mar 2025, matching Mar 2024 (150.0 kWh)
        self.assertEqual(res.forecast_steps[0].month_label, "Mar 2025")
        self.assertEqual(res.forecast_steps[0].forecast_kwh, 150.0)
        self.assertEqual(res.forecast_steps[1].month_label, "Apr 2025")
        self.assertEqual(res.forecast_steps[1].forecast_kwh, 210.0)

    def test_10_old_historical_test_records_accepted_with_staleness_warning(self):
        """10. Old historical test records: accepted for offline testing with stale-history warning."""
        old_record = MonthlyConsumptionRecord(
            billing_period_start=date(2020, 1, 1),
            billing_period_end=date(2020, 1, 31),
            consumption_kwh=180.0,
        )
        req = MonthlyForecastRequest(records=[old_record], horizon_months=1)
        res = monthly_forecast_service.generate_forecast(req)

        self.assertEqual(res.history_length_months, 1)
        self.assertEqual(res.forecast_steps[0].forecast_kwh, 180.0)
        self.assertTrue(res.is_stale_history)
        self.assertTrue(any("more than 90 days ago" in w for w in res.warnings))

    def test_11_unknown_or_unverified_tariff_preserves_consumption_forecast(self):
        """11. Unknown or unverified tariff: consumption forecasting works without a fabricated bill estimate."""
        req_unverified = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=1,
            provider_id="bescom",  # BESCOM is unverified
        )
        res_unverified = monthly_forecast_service.generate_forecast(req_unverified)

        # Physical consumption forecast is completely preserved
        self.assertEqual(res_unverified.forecast_steps[0].forecast_kwh, 167.0)
        # Bill estimate must be None with transparent disclosure
        step_bill = res_unverified.forecast_steps[0].bill_estimate
        self.assertIsNotNone(step_bill)
        self.assertFalse(step_bill.is_tariff_verified)
        self.assertIsNone(step_bill.estimated_bill_amount)
        self.assertTrue(any("unverified" in d.lower() for d in step_bill.disclosures))

        # Verified provider (MSEDCL) computes official estimate within its verified window (FY 2024-25)
        req_verified = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=1,
            is_live=False,  # Projects May 2024 (covered by verified FY 2024-25 schedule)
            provider_id="msedcl",
            phase_type="single_phase",
        )
        res_verified = monthly_forecast_service.generate_forecast(req_verified)
        step_verified_bill = res_verified.forecast_steps[0].bill_estimate
        self.assertIsNotNone(step_verified_bill)
        self.assertTrue(step_verified_bill.is_tariff_verified)
        self.assertIsNotNone(step_verified_bill.estimated_bill_amount)
        self.assertEqual(step_verified_bill.estimated_bill_amount, 1635.91)
        self.assertGreater(step_verified_bill.estimated_bill_amount, 0.0)

    def test_13_live_forecast_anchoring_to_current_date(self):
        """13. Live forecasts anchor future months to the current date and never return May-July 2025 in Oct 2026."""
        # Scenario: User uploaded April 2025 bill. Current reference date is October 19, 2026.
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                    period_label="Apr 2025",
                )
            ],
            horizon_months=3,
            is_live=True,
            reference_date=date(2026, 10, 19),
            provider_id="msedcl",
            phase_type="single_phase",
        )
        res = monthly_forecast_service.generate_forecast(req)

        # In October 2026, future live forecast months must start in November 2026
        labels = [step.month_label for step in res.forecast_steps]
        self.assertEqual(labels, ["Nov 2026", "Dec 2026", "Jan 2027"])

        self.assertEqual(res.forecast_steps[0].period_start, date(2026, 11, 1))
        self.assertEqual(res.forecast_steps[0].period_end, date(2026, 11, 30))
        self.assertEqual(res.forecast_steps[1].period_start, date(2026, 12, 1))
        self.assertEqual(res.forecast_steps[1].period_end, date(2026, 12, 31))
        self.assertEqual(res.forecast_steps[2].period_start, date(2027, 1, 1))
        self.assertEqual(res.forecast_steps[2].period_end, date(2027, 1, 31))

        # MUST NEVER return May-July 2025 as upcoming live forecasts in October 2026
        self.assertNotIn("May 2025", labels)
        self.assertNotIn("Jun 2025", labels)
        self.assertNotIn("Jul 2025", labels)

        # Historical record dates must NOT be mutated
        self.assertEqual(res.last_observed_end_date, date(2025, 4, 30))
        self.assertEqual(res.last_observed_period, "Apr 2025")

        # Documented policy for stale history (>90 days old)
        self.assertTrue(res.is_stale_history)
        self.assertTrue(any("live household forecasting policy" in w.lower() for w in res.warnings))
        self.assertTrue(any("stale billing history" in lim.lower() for lim in res.limitations))

    def test_14_historical_offline_fixture_preserves_next_cycle(self):
        """14. Offline historical tests preserve immediate cycle progression without modifying record dates."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                    period_label="Apr 2025",
                )
            ],
            horizon_months=3,
            is_live=False,  # Offline backtest / fixture mode
        )
        res = monthly_forecast_service.generate_forecast(req)

        # Offline projection steps immediately forward from April 2025
        labels = [step.month_label for step in res.forecast_steps]
        self.assertEqual(labels, ["May 2025", "Jun 2025", "Jul 2025"])
        self.assertEqual(res.forecast_steps[0].period_start, date(2025, 5, 1))
        self.assertEqual(res.forecast_steps[0].period_end, date(2025, 5, 31))

        # Historical record dates are preserved
        self.assertEqual(res.last_observed_end_date, date(2025, 4, 30))

    def test_15_tariff_estimates_effective_dated_schedule_check(self):
        """15. Tariff estimates use effective-dated schedule and do NOT assume historical bill amounts recur."""
        # Historical fixture from 2020: MSEDCL verified schedule is effective from 2024-04-01.
        req_hist = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2020, 1, 1),
                    billing_period_end=date(2020, 1, 31),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                )
            ],
            horizon_months=1,
            is_live=False,
            provider_id="msedcl",
            phase_type="single_phase",
        )
        res_hist = monthly_forecast_service.generate_forecast(req_hist)
        step_hist = res_hist.forecast_steps[0]

        # Physical consumption forecast is preserved
        self.assertEqual(step_hist.forecast_kwh, 167.0)
        # Because no active verified schedule covers 2020, bill estimate is withheld
        self.assertIsNone(step_hist.estimated_bill_amount)
        self.assertFalse(step_hist.bill_estimate.is_tariff_verified)
        self.assertIsNone(step_hist.bill_estimate.estimated_bill_amount)
        self.assertTrue(any("no verified tariff schedule is active" in d.lower() for d in step_hist.bill_estimate.disclosures))

        # Historical bill amount of 1760 is NEVER used as a fake estimate
        self.assertNotEqual(step_hist.estimated_bill_amount, 1760.0)
        self.assertEqual(res_hist.summary.months_with_bill_estimates, 0)
        self.assertIsNone(res_hist.summary.total_estimated_bill_inr)

        # Active schedule within documented FY 2024-25 window: May 2024
        req_verified = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2024, 4, 1),
                    billing_period_end=date(2024, 4, 30),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                )
            ],
            horizon_months=1,
            is_live=False,  # Projects May 2024
            provider_id="msedcl",
            phase_type="single_phase",
        )
        res_verified = monthly_forecast_service.generate_forecast(req_verified)
        step_verified = res_verified.forecast_steps[0]

        self.assertEqual(step_verified.forecast_kwh, 167.0)
        self.assertEqual(step_verified.month_label, "May 2024")
        self.assertTrue(step_verified.bill_estimate.is_tariff_verified)
        self.assertIsNotNone(step_verified.estimated_bill_amount)
        self.assertEqual(step_verified.estimated_bill_amount, 1635.91)
        self.assertEqual(res_verified.summary.months_with_bill_estimates, 1)
        self.assertEqual(res_verified.summary.total_estimated_bill_inr, 1635.91)

        # Future live forecast in Oct 2026: Nov 2026 falls into FY 2026-27 (beyond verified FY 2024-25)
        # Physical forecast is preserved; monetary estimate is withheld with disclosure
        req_live = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=167.0,
                    bill_amount=1760.0,
                )
            ],
            horizon_months=1,
            is_live=True,
            reference_date=date(2026, 10, 19),
            provider_id="msedcl",
            phase_type="single_phase",
        )
        res_live = monthly_forecast_service.generate_forecast(req_live)
        step_live = res_live.forecast_steps[0]

        self.assertEqual(step_live.forecast_kwh, 167.0)
        self.assertEqual(step_live.month_label, "Nov 2026")
        self.assertFalse(step_live.bill_estimate.is_tariff_verified)
        self.assertIsNone(step_live.estimated_bill_amount)
        self.assertEqual(res_live.summary.months_with_bill_estimates, 0)
        self.assertIsNone(res_live.summary.total_estimated_bill_inr)
        self.assertTrue(any("no verified tariff schedule is active" in d.lower() for d in step_live.bill_estimate.disclosures))


    def test_16_forecast_response_semantics_distinguish_kwh_and_inr(self):
        """16. Each forecast step clearly distinguishes forecast_kwh from estimated_bill_amount."""
        req = MonthlyForecastRequest(
            records=[
                MonthlyConsumptionRecord(
                    billing_period_start=date(2025, 4, 1),
                    billing_period_end=date(2025, 4, 30),
                    consumption_kwh=167.0,
                )
            ],
            horizon_months=2,
            provider_id=None,  # No provider
        )
        res = monthly_forecast_service.generate_forecast(req)

        for step in res.forecast_steps:
            self.assertEqual(step.forecast_kwh, 167.0)
            self.assertIsNone(step.estimated_bill_amount)
            self.assertIsNone(step.bill_estimate)

        self.assertEqual(res.summary.total_forecast_kwh, 334.0)
        self.assertEqual(res.summary.average_monthly_kwh, 167.0)
        self.assertEqual(res.summary.months_with_bill_estimates, 0)
        self.assertIsNone(res.summary.total_estimated_bill_inr)
        self.assertIn("no utility provider was specified", res.summary.bill_estimate_notes.lower())


class TestMonthlyForecastApi(unittest.TestCase):
    """API integration tests for POST /api/v1/forecasts/monthly/run."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX
        cls.run_url = f"{cls.prefix}/forecasts/monthly/run"

        # Register and authenticate a test user
        cls.username = f"monthly_fc_user_{uuid.uuid4().hex[:8]}"
        cls.password = "ValidPassword123!"
        cls.client.post(
            f"{cls.prefix}/auth/register",
            json={"username": cls.username, "password": cls.password},
        )
        login_res = cls.client.post(
            f"{cls.prefix}/auth/login",
            json={"username": cls.username, "password": cls.password},
        )
        cls.token = login_res.json()["access_token"]
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

    def test_12a_unauthenticated_request_rejected_401(self):
        """12a. Unauthenticated requests to /monthly/run return HTTP 401."""
        payload = {
            "records": [
                {
                    "billing_period_start": "2025-04-01",
                    "billing_period_end": "2025-04-30",
                    "consumption_kwh": 167.0,
                }
            ],
            "horizon_months": 1,
        }
        res = self.client.post(self.run_url, json=payload)
        self.assertEqual(res.status_code, 401)

    def test_12b_authenticated_single_month_success_200(self):
        """12b. Authenticated request with single month returns HTTP 200 with baseline-only results.
        For live forecasts in late 2026 (FY 2026-27), kWh forecast is preserved while unverified
        future monetary estimates are withheld.
        """
        payload = {
            "records": [
                {
                    "billing_period_start": "2025-04-01",
                    "billing_period_end": "2025-04-30",
                    "consumption_kwh": 167.0,
                    "bill_amount": 1760.0,
                    "period_label": "Apr 2025",
                }
            ],
            "horizon_months": 3,
            "provider_id": "msedcl",
        }
        res = self.client.post(self.run_url, json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["history_length_months"], 1)
        self.assertEqual(data["selected_method"], "baseline_only")
        self.assertEqual(len(data["forecast_steps"]), 3)
        self.assertEqual(data["forecast_steps"][0]["forecast_kwh"], 167.0)
        self.assertIsNotNone(data["forecast_steps"][0]["bill_estimate"])
        # In live mode (Oct 2026 reference), Nov 2026 falls into FY 2026-27 (unverified tariff)
        self.assertFalse(data["forecast_steps"][0]["bill_estimate"]["is_tariff_verified"])
        self.assertIsNone(data["forecast_steps"][0]["estimated_bill_amount"])
        self.assertEqual(data["summary"]["months_with_bill_estimates"], 0)
        self.assertIsNone(data["summary"]["total_estimated_bill_inr"])
        self.assertEqual(data["last_observed_end_date"], "2025-04-30")

    def test_12g_authenticated_within_verified_tariff_window(self):
        """12g. Authenticated request with dates within verified tariff window (FY 2024-25) returns full monetary estimates."""
        payload = {
            "records": [
                {
                    "billing_period_start": "2024-04-01",
                    "billing_period_end": "2024-04-30",
                    "consumption_kwh": 167.0,
                    "bill_amount": 1760.0,
                    "period_label": "Apr 2024",
                }
            ],
            "horizon_months": 3,
            "is_live": False,  # Projects May, Jun, Jul 2024
            "provider_id": "msedcl",
        }
        res = self.client.post(self.run_url, json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(len(data["forecast_steps"]), 3)
        self.assertEqual(data["forecast_steps"][0]["forecast_kwh"], 167.0)
        self.assertEqual(data["forecast_steps"][0]["month_label"], "May 2024")
        self.assertTrue(data["forecast_steps"][0]["bill_estimate"]["is_tariff_verified"])
        self.assertEqual(data["forecast_steps"][0]["estimated_bill_amount"], 1635.91)
        self.assertEqual(data["summary"]["months_with_bill_estimates"], 3)
        self.assertEqual(data["summary"]["total_estimated_bill_inr"], 4907.73)

    def test_12e_live_api_distinguishes_offline_flag(self):
        """12e. Live API endpoint respects is_live flag for offline fixtures."""
        payload_offline = {
            "records": [
                {
                    "billing_period_start": "2025-04-01",
                    "billing_period_end": "2025-04-30",
                    "consumption_kwh": 167.0,
                    "bill_amount": 1760.0,
                    "period_label": "Apr 2025",
                }
            ],
            "horizon_months": 2,
            "is_live": False,
        }
        res = self.client.post(self.run_url, json=payload_offline, headers=self.headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["forecast_steps"][0]["month_label"], "May 2025")
        self.assertEqual(data["forecast_steps"][1]["month_label"], "Jun 2025")
        self.assertEqual(data["last_observed_end_date"], "2025-04-30")

    def test_12c_invalid_duplicate_records_rejected_422(self):
        """12c. Duplicate periods return HTTP 422 with a safe error message."""
        payload = {
            "records": [
                {
                    "billing_period_start": "2025-04-01",
                    "billing_period_end": "2025-04-30",
                    "consumption_kwh": 167.0,
                },
                {
                    "billing_period_start": "2025-04-01",
                    "billing_period_end": "2025-04-30",
                    "consumption_kwh": 170.0,
                },
            ],
            "horizon_months": 1,
        }
        res = self.client.post(self.run_url, json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 422)
        self.assertIn("duplicate", res.json()["detail"].lower())

    def test_12d_invalid_empty_records_rejected_422(self):
        """12d. Empty records array rejected with HTTP 422."""
        payload = {
            "records": [],
            "horizon_months": 1,
        }
        res = self.client.post(self.run_url, json=payload, headers=self.headers)
        self.assertEqual(res.status_code, 422)

    def test_12f_single_month_all_horizons_api_consistency(self):
        """12f. Verifies that single-month history returns exactly 1, 3, and 6 steps with matching summaries.
        In live mode (FY 2026-27), unverified monetary estimates are withheld while physical forecasts match.
        In verified mode (FY 2024-25), verified monetary estimates are computed and properly summed.
        """
        single_record = [
            {
                "billing_period_start": "2025-04-01",
                "billing_period_end": "2025-04-30",
                "consumption_kwh": 167.0,
                "bill_amount": 1760.0,
                "period_label": "Apr 2025",
            }
        ]
        for h in [1, 3, 6]:
            payload = {
                "records": single_record,
                "horizon_months": h,
                "provider_id": "msedcl",
                "is_live": True,
            }
            res = self.client.post(self.run_url, json=payload, headers=self.headers)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["history_length_months"], 1)
            self.assertEqual(data["selected_method"], "baseline_only")
            self.assertEqual(len(data["forecast_steps"]), h)
            self.assertEqual(data["forecast_steps"][-1]["month_index"], h)
            self.assertEqual(data["summary"]["total_forecast_kwh"], round(167.0 * h, 2))
            self.assertEqual(data["summary"]["average_monthly_kwh"], 167.0)
            # Live forecast in 2026: unverified future schedule -> 0 months with bill estimates
            self.assertEqual(data["summary"]["months_with_bill_estimates"], 0)
            self.assertIsNone(data["summary"]["total_estimated_bill_inr"])

            # Verify consecutive future months starting from Nov 2026 (anchored for Oct 2026 current time)
            labels = [s["month_label"] for s in data["forecast_steps"]]
            expected_labels = ["Nov 2026", "Dec 2026", "Jan 2027", "Feb 2027", "Mar 2027", "Apr 2027"][:h]
            self.assertEqual(labels, expected_labels)

            # Step bill estimates should all be None
            for s in data["forecast_steps"]:
                self.assertIsNone(s["estimated_bill_amount"])

        # Verified window check
        verified_record = [
            {
                "billing_period_start": "2024-04-01",
                "billing_period_end": "2024-04-30",
                "consumption_kwh": 167.0,
                "bill_amount": 1760.0,
                "period_label": "Apr 2024",
            }
        ]
        for h in [1, 3, 6]:
            payload = {
                "records": verified_record,
                "horizon_months": h,
                "provider_id": "msedcl",
                "is_live": False,
            }
            res = self.client.post(self.run_url, json=payload, headers=self.headers)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(len(data["forecast_steps"]), h)
            self.assertEqual(data["summary"]["months_with_bill_estimates"], h)
            step_bills_sum = round(sum(s["estimated_bill_amount"] for s in data["forecast_steps"]), 2)
            self.assertAlmostEqual(data["summary"]["total_estimated_bill_inr"], step_bills_sum, places=2)
            self.assertEqual(data["summary"]["total_estimated_bill_inr"], round(1635.91 * h, 2))


if __name__ == "__main__":
    unittest.main()

