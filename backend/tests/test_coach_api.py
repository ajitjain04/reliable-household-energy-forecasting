"""
Integration tests for Energy Coach API endpoints.

Covers:
- GET /api/v1/coach/overview: authenticated vs unauthenticated (401).
- POST /api/v1/coach/simulate-appliance: valid vs invalid payload (422).
- GET /api/v1/coach/presets: returns standard presets.
- Record isolation: User A sees User A's data, not User B's.
"""

import unittest
import uuid
from datetime import date
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.monthly_forecast import (
    ConfidenceRating,
    MonthlyForecastMethod,
    MonthlyForecastRequest,
    MonthlyForecastResponse,
    MonthlyForecastStep,
    MonthlyForecastSummary,
)
from backend.app.services.monthly_record_service import monthly_record_service


class TestCoachAPI(unittest.TestCase):
    """Test suite for /api/v1/coach router."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

        # User A
        cls.user_a = f"coach_user_a_{uuid.uuid4().hex[:8]}"
        cls.pwd_a = "ValidPass123!"
        cls.client.post(f"{cls.prefix}/auth/register", json={"username": cls.user_a, "password": cls.pwd_a})
        res_a = cls.client.post(f"{cls.prefix}/auth/login", json={"username": cls.user_a, "password": cls.pwd_a})
        cls.token_a = res_a.json()["access_token"]
        cls.headers_a = {"Authorization": f"Bearer {cls.token_a}"}

        # User B
        cls.user_b = f"coach_user_b_{uuid.uuid4().hex[:8]}"
        cls.pwd_b = "ValidPass123!"
        cls.client.post(f"{cls.prefix}/auth/register", json={"username": cls.user_b, "password": cls.pwd_b})
        res_b = cls.client.post(f"{cls.prefix}/auth/login", json={"username": cls.user_b, "password": cls.pwd_b})
        cls.token_b = res_b.json()["access_token"]
        cls.headers_b = {"Authorization": f"Bearer {cls.token_b}"}

    def test_unauthenticated_overview_returns_401(self):
        """Unauthenticated GET /coach/overview returns HTTP 401."""
        res = self.client.get(f"{self.prefix}/coach/overview")
        self.assertEqual(res.status_code, 401)

    def test_authenticated_overview_empty_user(self):
        """User with no records receives 200 with has_data=False and onboarding guidance."""
        res = self.client.get(f"{self.prefix}/coach/overview", headers=self.headers_b)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["has_data"])
        self.assertEqual(data["data_freshness_status"], "no_records")
        self.assertGreaterEqual(len(data["insights"]), 1)
        self.assertGreaterEqual(len(data["recommendations"]), 1)

    def test_authenticated_overview_populated_user(self):
        """User with confirmed monthly forecast receives 200 with full coaching metrics and insights."""
        # Seed a forecast run for User A
        me_res = self.client.get(f"{self.prefix}/auth/me", headers=self.headers_a)
        user_a_id = me_res.json()["id"]

        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 1, 1),
                billing_period_end=date(2024, 1, 31),
                consumption_kwh=190.0,
                bill_amount=1850.0,
                period_label="Jan 2024",
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 2, 1),
                billing_period_end=date(2024, 2, 29),
                consumption_kwh=210.0,
                bill_amount=2100.0,
                period_label="Feb 2024",
            ),
        ]
        steps = [
            MonthlyForecastStep(
                month_index=1,
                month_label="Mar 2024",
                period_start=date(2024, 3, 1),
                period_end=date(2024, 3, 31),
                forecast_kwh=230.0,
                uncertainty_lower_kwh=200.0,
                uncertainty_upper_kwh=260.0,
                estimated_bill_amount=2400.0,
            )
        ]
        run_id = f"coach_api_run_{uuid.uuid4().hex[:8]}"
        monthly_record_service.save_monthly_forecast(
            owner_user_id=user_a_id,
            request=MonthlyForecastRequest(records=records, horizon_months=1, provider_id="msedcl"),
            response=MonthlyForecastResponse(
                forecast_id=run_id,
                created_at="2024-03-01T10:00:00Z",
                selected_method=MonthlyForecastMethod.WEIGHTED_MOVING_AVERAGE,
                method_description="Weighted Moving Average Baseline",
                confidence_rating=ConfidenceRating.MODERATE,
                history_length_months=2,
                last_observed_period="Feb 2024",
                last_observed_end_date=date(2024, 2, 29),
                is_stale_history=False,
                forecast_steps=steps,
                summary=MonthlyForecastSummary(
                    total_forecast_kwh=230.0,
                    average_monthly_kwh=230.0,
                    total_estimated_bill_inr=2400.0,
                    months_with_bill_estimates=1,
                ),
                warnings=[],
                limitations=[],
            ),
            forecast_id=run_id,
        )

        res = self.client.get(f"{self.prefix}/coach/overview", headers=self.headers_a)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["has_data"])
        self.assertEqual(data["latest_consumption_kwh"], 210.0)
        self.assertEqual(data["previous_consumption_kwh"], 190.0)
        self.assertEqual(data["consumption_change_kwh"], 20.0)
        self.assertEqual(data["provider_id"], "msedcl")
        self.assertTrue(data["is_tariff_verified"])

    def test_simulate_appliance_endpoint_success(self):
        """POST /coach/simulate-appliance returns 200 with energy delta and disclosures."""
        payload = {
            "baseline": {
                "category": "air_conditioner",
                "name": "AC Living Room",
                "power_watts": 1400.0,
                "hours_per_day": 8.0,
                "days_per_month": 30,
                "quantity": 1,
            },
            "modified": {
                "category": "air_conditioner",
                "name": "AC Living Room",
                "power_watts": 1400.0,
                "hours_per_day": 7.0,
                "days_per_month": 30,
                "quantity": 1,
            },
            "provider_id": "msedcl",
            "phase_type": "single_phase",
            "current_monthly_total_kwh": 250.0,
        }
        res = self.client.post(
            f"{self.prefix}/coach/simulate-appliance",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["baseline_kwh"], 336.0)
        self.assertEqual(data["modified_kwh"], 294.0)
        self.assertEqual(data["delta_kwh"], 42.0)
        self.assertTrue(data["is_tariff_verified"])
        self.assertGreater(data["estimated_savings_inr"], 0.0)

    def test_simulate_appliance_endpoint_validation_error(self):
        """POST /coach/simulate-appliance with invalid values returns 422."""
        invalid_payload = {
            "baseline": {
                "category": "air_conditioner",
                "name": "Impossible AC",
                "power_watts": 1400.0,
                "hours_per_day": 26.0,  # Invalid: > 24 hours
                "days_per_month": 30,
                "quantity": 1,
            },
            "modified": {
                "category": "air_conditioner",
                "name": "Impossible AC",
                "power_watts": 1400.0,
                "hours_per_day": 8.0,
                "days_per_month": 30,
                "quantity": 1,
            },
        }
        res = self.client.post(
            f"{self.prefix}/coach/simulate-appliance",
            json=invalid_payload,
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 422)

    def test_get_appliance_presets(self):
        """GET /coach/presets returns appliance categories with default wattages."""
        res = self.client.get(f"{self.prefix}/coach/presets")
        self.assertEqual(res.status_code, 200)
        presets = res.json()
        self.assertIn("air_conditioner", presets)
        self.assertIn("refrigerator", presets)
        self.assertIn("water_heater", presets)
        self.assertIn("ceiling_fan", presets)

    def test_compare_household_scenario_api_success(self):
        """POST /coach/compare-household-scenario returns 200 with multi-step comparison and verified MSEDCL savings."""
        payload = {
            "baseline_forecast_kwh": [450.0, 480.0],
            "month_labels": ["May 2024", "Jun 2024"],
            "appliance_scenario": {
                "baseline": {
                    "category": "air_conditioner",
                    "name": "AC",
                    "power_watts": 1500.0,
                    "hours_per_day": 8.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
                "modified": {
                    "category": "air_conditioner",
                    "name": "AC",
                    "power_watts": 1500.0,
                    "hours_per_day": 6.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
            },
            "provider_id": "msedcl",
            "phase_type": "single_phase",
        }
        res = self.client.post(
            f"{self.prefix}/coach/compare-household-scenario",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_baseline_kwh"], 930.0)
        self.assertEqual(data["appliance_result"]["baseline_kwh"], 360.0)
        self.assertEqual(data["appliance_result"]["modified_kwh"], 270.0)
        self.assertEqual(data["appliance_result"]["delta_kwh"], 90.0)
        self.assertEqual(data["total_delta_kwh"], 180.0)
        self.assertEqual(data["total_adjusted_kwh"], 750.0)
        self.assertTrue(data["is_household_adjustment_supported"])
        self.assertTrue(data["is_verified_tariff"])
        self.assertIsNotNone(data["total_savings_inr"])
        self.assertGreater(data["total_savings_inr"], 0.0)
        self.assertEqual(len(data["steps"]), 2)
        self.assertGreaterEqual(len(data["recommendations"]), 1)

    def test_compare_household_scenario_api_exact_bug_regression(self):
        """POST /coach/compare-household-scenario regression for 1450W AC (8h->7h) vs 2.0 kWh baseline.
        Must return 200, preserve appliance delta 43.5 kWh, mark adjustment unsupported, and withhold
        household adjusted kWh and bill savings (NOT clamp to 0.0 kWh).
        """
        payload = {
            "baseline_forecast_kwh": [2.0, 2.0, 2.0],
            "month_labels": ["Jan 2025", "Feb 2025", "Mar 2025"],
            "appliance_scenario": {
                "baseline": {
                    "category": "air_conditioner",
                    "name": "1.5 Ton Split AC",
                    "power_watts": 1450.0,
                    "hours_per_day": 8.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
                "modified": {
                    "category": "air_conditioner",
                    "name": "1.5 Ton Split AC",
                    "power_watts": 1450.0,
                    "hours_per_day": 7.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
            },
            "provider_id": "msedcl",
            "phase_type": "single_phase",
        }
        res = self.client.post(
            f"{self.prefix}/coach/compare-household-scenario",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Appliance-level calculation is correct and labeled hypothetical
        self.assertEqual(data["appliance_result"]["baseline_kwh"], 348.0)
        self.assertEqual(data["appliance_result"]["modified_kwh"], 304.5)
        self.assertEqual(data["appliance_result"]["delta_kwh"], 43.5)
        self.assertTrue(data["appliance_result"]["is_hypothetical_estimate"])

        # Household adjustment is NOT supported
        self.assertFalse(data["is_household_adjustment_supported"])
        self.assertIsNotNone(data["household_adjustment_reason"])
        self.assertIsNone(data["total_adjusted_kwh"])
        self.assertIsNone(data["total_delta_kwh"])
        self.assertIsNone(data["total_savings_inr"])

        # Steps must have adjusted_kwh and savings withheld (NOT 0.0 or ₹12.95)
        self.assertEqual(len(data["steps"]), 3)
        for step in data["steps"]:
            self.assertFalse(step["is_adjustment_supported"])
            self.assertEqual(step["baseline_kwh"], 2.0)
            self.assertIsNone(step["adjusted_kwh"])
            self.assertIsNone(step["delta_kwh"])
            self.assertIsNone(step["savings_amount"])
            self.assertIsNotNone(step["unsupported_reason"])

    def test_compare_household_scenario_api_unauthenticated(self):
        """Unauthenticated POST /coach/compare-household-scenario returns 401."""
        payload = {
            "baseline_forecast_kwh": [100.0],
            "month_labels": ["May 2024"],
            "appliance_scenario": {
                "baseline": {
                    "category": "lighting",
                    "name": "LEDs",
                    "power_watts": 50.0,
                    "hours_per_day": 6.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
                "modified": {
                    "category": "lighting",
                    "name": "LEDs",
                    "power_watts": 50.0,
                    "hours_per_day": 4.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
            },
        }
        res = self.client.post(
            f"{self.prefix}/coach/compare-household-scenario",
            json=payload,
        )
        self.assertEqual(res.status_code, 401)

    def test_compare_household_scenario_api_validation_error(self):
        """POST /coach/compare-household-scenario with invalid power returns 422."""
        invalid_payload = {
            "baseline_forecast_kwh": [100.0],
            "month_labels": ["May 2024"],
            "appliance_scenario": {
                "baseline": {
                    "category": "lighting",
                    "name": "LEDs",
                    "power_watts": -10.0,  # Invalid negative watts
                    "hours_per_day": 6.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
                "modified": {
                    "category": "lighting",
                    "name": "LEDs",
                    "power_watts": 50.0,
                    "hours_per_day": 4.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
            },
        }
        res = self.client.post(
            f"{self.prefix}/coach/compare-household-scenario",
            json=invalid_payload,
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 422)

    def test_compare_household_scenario_api_unverified_tariff_withholding(self):
        """POST /coach/compare-household-scenario with unverified provider suppresses financial savings."""
        payload = {
            "baseline_forecast_kwh": [150.0],
            "month_labels": ["May 2024"],
            "appliance_scenario": {
                "baseline": {
                    "category": "television",
                    "name": "TV",
                    "power_watts": 120.0,
                    "hours_per_day": 5.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
                "modified": {
                    "category": "television",
                    "name": "TV",
                    "power_watts": 120.0,
                    "hours_per_day": 3.0,
                    "days_per_month": 30,
                    "quantity": 1,
                },
            },
            "provider_id": "bescom",
            "phase_type": "single_phase",
        }
        res = self.client.post(
            f"{self.prefix}/coach/compare-household-scenario",
            json=payload,
            headers=self.headers_a,
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertFalse(data["is_verified_tariff"])
        self.assertIsNone(data["total_savings_inr"])
        self.assertIsNone(data["steps"][0]["baseline_bill_amount"])
        self.assertIsNone(data["steps"][0]["savings_amount"])
        self.assertTrue(any("withheld" in d.lower() for d in data["disclosures"]))


if __name__ == "__main__":
    unittest.main()
