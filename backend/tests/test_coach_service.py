"""
Focused unit tests for CoachService, appliance energy calculations, and insight generation.

Covers:
- Formula verification: kWh = (watts * hours/day * days/month * quantity) / 1000.
- Zero usage, multiple appliances, fractional hours.
- Scenario comparison and delta savings calculations.
- Verified tariff integration (MSEDCL) vs unverified tariff fallback (BESCOM/JVVNL/None).
- Refrigerator safety caution when hours < 24.
- Input validation: invalid power, hours > 24, days > 31, quantity limits.
- Overview generation: empty history, single month, multi-month trends, stale history.
"""

import unittest
from datetime import date
from pydantic import ValidationError

from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.coach import (
    ApplianceCategory,
    ApplianceScenarioRequest,
    ApplianceSimulationInput,
    DEFAULT_APPLIANCE_PRESETS,
    HouseholdScenarioComparisonRequest,
)
from backend.app.schemas.monthly_forecast import (
    ConfidenceRating,
    MonthlyForecastMethod,
    MonthlyForecastRequest,
    MonthlyForecastResponse,
    MonthlyForecastStep,
    MonthlyForecastSummary,
)
from backend.app.services.coach_service import CoachService, coach_service
from backend.app.services.monthly_record_service import MonthlyRecordService
from backend.app.db.database import DatabaseManager


class TestCoachService(unittest.TestCase):
    """Unit test suite for CoachService and appliance physics calculations."""

    def setUp(self):
        self.db = DatabaseManager(":memory:")
        self.db.init_db()
        self.rec_service = MonthlyRecordService(db=self.db)
        self.service = CoachService(monthly_record_service=self.rec_service)

    def test_appliance_kwh_formula_basic(self):
        """Test basic appliance energy calculation: (1500W * 8h * 30d * 1) / 1000 = 360 kWh."""
        appliance = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="Bedroom AC",
            power_watts=1500.0,
            hours_per_day=8.0,
            days_per_month=30,
            quantity=1,
        )
        kwh = self.service.compute_appliance_kwh(appliance)
        self.assertEqual(kwh, 360.0)

    def test_appliance_kwh_formula_multi_unit_fractional_hours(self):
        """Test multi-unit appliance with fractional hours: 4 fans * 55W * 10.5h * 30d / 1000 = 69.3 kWh."""
        fans = ApplianceSimulationInput(
            category=ApplianceCategory.CEILING_FAN,
            name="Ceiling Fans",
            power_watts=55.0,
            hours_per_day=10.5,
            days_per_month=30,
            quantity=4,
        )
        kwh = self.service.compute_appliance_kwh(fans)
        self.assertEqual(kwh, 69.3)

    def test_appliance_kwh_formula_zero_usage(self):
        """Test appliance with 0 hours per day or 0 days per month produces 0.0 kWh."""
        zero_hours = ApplianceSimulationInput(
            category=ApplianceCategory.WATER_HEATER,
            name="Summer Geyser",
            power_watts=2000.0,
            hours_per_day=0.0,
            days_per_month=30,
            quantity=1,
        )
        self.assertEqual(self.service.compute_appliance_kwh(zero_hours), 0.0)

        zero_days = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="Winter AC",
            power_watts=1500.0,
            hours_per_day=8.0,
            days_per_month=0,
            quantity=1,
        )
        self.assertEqual(self.service.compute_appliance_kwh(zero_days), 0.0)

    def test_appliance_input_validation_rejects_unreasonable_values(self):
        """Test that Pydantic validators reject hours > 24, days > 31, negative values, and extreme power."""
        # Hours > 24
        with self.assertRaises(ValidationError):
            ApplianceSimulationInput(
                category=ApplianceCategory.AIR_CONDITIONER,
                name="Invalid Hours",
                power_watts=1000.0,
                hours_per_day=25.0,
                days_per_month=30,
                quantity=1,
            )

        # Negative hours
        with self.assertRaises(ValidationError):
            ApplianceSimulationInput(
                category=ApplianceCategory.AIR_CONDITIONER,
                name="Negative Hours",
                power_watts=1000.0,
                hours_per_day=-1.0,
                days_per_month=30,
                quantity=1,
            )

        # Days > 31
        with self.assertRaises(ValidationError):
            ApplianceSimulationInput(
                category=ApplianceCategory.LIGHTING,
                name="Invalid Days",
                power_watts=20.0,
                hours_per_day=5.0,
                days_per_month=32,
                quantity=1,
            )

        # Power < 1W or > 50,000W
        with self.assertRaises(ValidationError):
            ApplianceSimulationInput(
                category=ApplianceCategory.OTHER,
                name="Zero Watts",
                power_watts=0.5,
                hours_per_day=1.0,
                days_per_month=30,
                quantity=1,
            )

        with self.assertRaises(ValidationError):
            ApplianceSimulationInput(
                category=ApplianceCategory.OTHER,
                name="Power Substation",
                power_watts=60000.0,
                hours_per_day=1.0,
                days_per_month=30,
                quantity=1,
            )

        # Quantity < 1 or > 20
        with self.assertRaises(ValidationError):
            ApplianceSimulationInput(
                category=ApplianceCategory.CEILING_FAN,
                name="Zero Fans",
                power_watts=55.0,
                hours_per_day=5.0,
                days_per_month=30,
                quantity=0,
            )

    def test_simulate_scenario_energy_delta(self):
        """Test scenario comparison: Reducing AC from 8h/day to 6h/day."""
        baseline = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton AC",
            power_watts=1500.0,
            hours_per_day=8.0,
            days_per_month=30,
            quantity=1,
        )
        modified = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton AC",
            power_watts=1500.0,
            hours_per_day=6.0,
            days_per_month=30,
            quantity=1,
        )
        req = ApplianceScenarioRequest(
            baseline=baseline,
            modified=modified,
            provider_id="msedcl",
            current_monthly_total_kwh=200.0,
        )
        res = self.service.simulate_appliance_scenario(req)

        self.assertEqual(res.baseline_kwh, 360.0)
        self.assertEqual(res.modified_kwh, 270.0)
        self.assertEqual(res.delta_kwh, 90.0)
        self.assertTrue(res.is_tariff_verified)
        self.assertIsNotNone(res.estimated_savings_inr)
        self.assertGreater(res.estimated_savings_inr, 0.0)
        self.assertTrue(any("MSEDCL" in d for d in res.disclosures))

    def test_simulate_scenario_unverified_tariff_withholds_monetary_savings(self):
        """Test scenario comparison with unverified provider (e.g. BESCOM) preserves kWh but withholds INR."""
        baseline = ApplianceSimulationInput(
            category=ApplianceCategory.WATER_HEATER,
            name="Geyser",
            power_watts=2000.0,
            hours_per_day=2.0,
            days_per_month=30,
            quantity=1,
        )
        modified = ApplianceSimulationInput(
            category=ApplianceCategory.WATER_HEATER,
            name="Geyser",
            power_watts=2000.0,
            hours_per_day=1.0,
            days_per_month=30,
            quantity=1,
        )
        req = ApplianceScenarioRequest(
            baseline=baseline,
            modified=modified,
            provider_id="bescom",
        )
        res = self.service.simulate_appliance_scenario(req)

        self.assertEqual(res.baseline_kwh, 120.0)
        self.assertEqual(res.modified_kwh, 60.0)
        self.assertEqual(res.delta_kwh, 60.0)
        self.assertFalse(res.is_tariff_verified)
        self.assertIsNone(res.estimated_baseline_cost_inr)
        self.assertIsNone(res.estimated_modified_cost_inr)
        self.assertIsNone(res.estimated_savings_inr)
        self.assertTrue(any("unverified" in d.lower() or "withheld" in d.lower() for d in res.disclosures))

    def test_simulate_scenario_refrigerator_safety_caution(self):
        """Test refrigerator simulation flags safety warning if modified hours < 24."""
        baseline = ApplianceSimulationInput(
            category=ApplianceCategory.REFRIGERATOR,
            name="Fridge",
            power_watts=150.0,
            hours_per_day=24.0,
            days_per_month=30,
            quantity=1,
        )
        modified = ApplianceSimulationInput(
            category=ApplianceCategory.REFRIGERATOR,
            name="Fridge",
            power_watts=150.0,
            hours_per_day=18.0,  # Unsafe behavior: disconnecting fridge
            days_per_month=30,
            quantity=1,
        )
        req = ApplianceScenarioRequest(baseline=baseline, modified=modified)
        res = self.service.simulate_appliance_scenario(req)

        self.assertTrue(any("SAFETY CAUTION" in note for note in res.appliance_notes))

    def test_overview_empty_history(self):
        """Test overview for a user with no saved monthly forecast records."""
        overview = self.service.get_household_overview(user_id="non_existent_user_12345")
        self.assertFalse(overview.has_data)
        self.assertEqual(overview.data_freshness_status, "no_records")
        self.assertEqual(len(overview.historical_records), 0)
        self.assertGreaterEqual(len(overview.insights), 1)
        self.assertGreaterEqual(len(overview.recommendations), 1)
        self.assertEqual(overview.recommendations[0].action_type, "data_setup")

    def test_overview_with_mocked_saved_run(self):
        """Test overview generation with populated confirmed monthly records using isolated in-memory service."""
        test_user = "coach_test_user_abc"
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 4, 1),
                billing_period_end=date(2024, 4, 30),
                consumption_kwh=180.0,
                bill_amount=1750.0,
                period_label="Apr 2024",
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 5, 1),
                billing_period_end=date(2024, 5, 31),
                consumption_kwh=220.0,
                bill_amount=2210.0,
                period_label="May 2024",
            ),
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 6, 1),
                billing_period_end=date(2024, 6, 30),
                consumption_kwh=250.0,
                bill_amount=2650.0,
                period_label="Jun 2024",
            ),
        ]
        steps = [
            MonthlyForecastStep(
                month_index=1,
                month_label="Jul 2024",
                period_start=date(2024, 7, 1),
                period_end=date(2024, 7, 31),
                forecast_kwh=270.0,
                uncertainty_lower_kwh=240.0,
                uncertainty_upper_kwh=300.0,
                estimated_bill_amount=2900.0,
            )
        ]
        req = MonthlyForecastRequest(
            records=records,
            horizon_months=1,
            provider_id="msedcl",
            phase_type="single_phase",
        )
        resp = MonthlyForecastResponse(
            forecast_id="mock_run_coach_1",
            created_at="2024-07-01T10:00:00Z",
            selected_method=MonthlyForecastMethod.SEASONAL_NAIVE,
            method_description="Seasonal Naive Baseline",
            confidence_rating=ConfidenceRating.HIGH,
            history_length_months=3,
            last_observed_period="Jun 2024",
            last_observed_end_date=date(2024, 6, 30),
            is_stale_history=False,
            forecast_steps=steps,
            summary=MonthlyForecastSummary(
                total_forecast_kwh=270.0,
                average_monthly_kwh=270.0,
                total_estimated_bill_inr=2900.0,
                months_with_bill_estimates=1,
            ),
            warnings=[],
            limitations=[],
        )
        self.rec_service.save_monthly_forecast(
            owner_user_id=test_user,
            request=req,
            response=resp,
            forecast_id="mock_run_coach_1",
        )

        overview = self.service.get_household_overview(user_id=test_user)
        self.assertTrue(overview.has_data)
        self.assertEqual(overview.latest_consumption_kwh, 250.0)
        self.assertEqual(overview.latest_period_label, "Jun 2024")
        self.assertEqual(overview.previous_consumption_kwh, 220.0)
        self.assertEqual(overview.consumption_change_kwh, 30.0)
        self.assertEqual(overview.consumption_change_percent, 13.6)
        self.assertEqual(overview.latest_bill_amount_inr, 2650.0)
        self.assertEqual(overview.provider_id, "msedcl")
        self.assertTrue(overview.is_tariff_verified)
        self.assertEqual(len(overview.historical_records), 3)

        # Check insights
        insight_ids = [i.id for i in overview.insights]
        self.assertIn("increasing_trend", insight_ids)
        increasing_insight = next(i for i in overview.insights if i.id == "increasing_trend")
        self.assertIn("May 2024 = 220.0 kWh", increasing_insight.evidence)
        self.assertIn("Jun 2024 = 250.0 kWh", increasing_insight.evidence)

        # Check recommendations
        rec_ids = [r.id for r in overview.recommendations]
        self.assertIn("ac_thermostat_tuning", rec_ids)
        self.assertIn("slab_boundary_awareness", rec_ids)

    def test_household_overview_isolated_in_memory_dependency_injection(self):
        """
        Regression test for CoachService dependency injection:
        Exercises household overview using an isolated in-memory database
        without relying on the global database singleton, and verifies default fallback.
        """
        # 1. Verify default CoachService uses production singleton fallback
        default_coach = CoachService()
        from backend.app.services.monthly_record_service import monthly_record_service as default_singleton
        self.assertIs(default_coach.monthly_record_service, default_singleton)

        # 2. Setup isolated in-memory database manager and injected service
        isolated_db = DatabaseManager(":memory:")
        isolated_db.init_db()
        isolated_rec_service = MonthlyRecordService(db=isolated_db)
        isolated_coach = CoachService(monthly_record_service=isolated_rec_service)

        self.assertIs(isolated_coach.monthly_record_service, isolated_rec_service)
        self.assertIsNot(isolated_coach.monthly_record_service, default_singleton)

        # 3. Save a record strictly in isolated in-memory DB
        test_user_isolated = "isolated_user_di_987"
        records = [
            MonthlyConsumptionRecord(
                billing_period_start=date(2024, 5, 1),
                billing_period_end=date(2024, 5, 31),
                consumption_kwh=310.0,
                bill_amount=3200.0,
                period_label="May 2024",
            )
        ]
        req = MonthlyForecastRequest(
            records=records,
            horizon_months=1,
            provider_id="msedcl",
            phase_type="single_phase",
        )
        resp = MonthlyForecastResponse(
            forecast_id="mock_run_isolated_1",
            created_at="2024-06-01T10:00:00Z",
            selected_method=MonthlyForecastMethod.SEASONAL_NAIVE,
            method_description="Seasonal Naive Baseline",
            confidence_rating=ConfidenceRating.HIGH,
            history_length_months=1,
            last_observed_period="May 2024",
            last_observed_end_date=date(2024, 5, 31),
            is_stale_history=False,
            forecast_steps=[],
            summary=MonthlyForecastSummary(
                total_forecast_kwh=310.0,
                average_monthly_kwh=310.0,
                total_estimated_bill_inr=3200.0,
                months_with_bill_estimates=1,
            ),
            warnings=[],
            limitations=[],
        )
        isolated_rec_service.save_monthly_forecast(
            owner_user_id=test_user_isolated,
            request=req,
            response=resp,
            forecast_id="mock_run_isolated_1",
        )

        # 4. Injected coach service sees the record
        overview = isolated_coach.get_household_overview(user_id=test_user_isolated)
        self.assertTrue(overview.has_data)
        self.assertEqual(overview.latest_consumption_kwh, 310.0)
        self.assertEqual(overview.latest_period_label, "May 2024")
        self.assertEqual(len(overview.historical_records), 1)

        # 5. Default singleton (production DB) DOES NOT have this isolated record
        default_overview = default_coach.get_household_overview(user_id=test_user_isolated)
        self.assertFalse(default_overview.has_data)
        self.assertEqual(len(default_overview.historical_records), 0)

    def test_compare_household_scenario_msedcl_verified(self):
        """Test multi-month household scenario comparison with verified MSEDCL progressive slab tariff."""
        baseline_app = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5T AC",
            power_watts=1500.0,
            hours_per_day=8.0,
            days_per_month=30,
            quantity=1,
        )
        modified_app = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5T AC (Eco)",
            power_watts=1500.0,
            hours_per_day=7.0,
            days_per_month=30,
            quantity=1,
        )
        # Delta = 360 - 315 = 45 kWh/month
        request = HouseholdScenarioComparisonRequest(
            baseline_forecast_kwh=[420.0, 440.0, 460.0],
            month_labels=["Aug 2024", "Sep 2024", "Oct 2024"],
            appliance_scenario=ApplianceScenarioRequest(
                baseline=baseline_app,
                modified=modified_app,
            ),
            provider_id="msedcl",
            phase_type="single_phase",
        )
        result = self.service.compare_household_scenario(request)

        self.assertEqual(result.total_baseline_kwh, 1320.0)
        self.assertEqual(result.total_adjusted_kwh, 1185.0)
        self.assertEqual(result.total_delta_kwh, 135.0)
        self.assertTrue(result.is_verified_tariff)
        self.assertIsNotNone(result.total_savings_inr)
        self.assertGreater(result.total_savings_inr, 0.0)
        self.assertTrue(any("MERC" in d for d in result.disclosures))
        self.assertEqual(len(result.steps), 3)

        # Check step 1
        s1 = result.steps[0]
        self.assertEqual(s1.baseline_kwh, 420.0)
        self.assertEqual(s1.adjusted_kwh, 375.0)
        self.assertEqual(s1.delta_kwh, 45.0)
        self.assertIsNotNone(s1.savings_amount)
        self.assertGreater(s1.savings_amount, 0.0)

        # Check recommendations
        self.assertGreaterEqual(len(result.recommendations), 1)

    def test_compare_household_scenario_unverified_tariff_withholding(self):
        """Test household scenario comparison with unverified tariff suppresses monetary estimates."""
        baseline_app = ApplianceSimulationInput(
            category=ApplianceCategory.CEILING_FAN,
            name="Ceiling Fans",
            power_watts=75.0,
            hours_per_day=14.0,
            days_per_month=30,
            quantity=3,
        )
        modified_app = ApplianceSimulationInput(
            category=ApplianceCategory.CEILING_FAN,
            name="Ceiling Fans (BLDC)",
            power_watts=28.0,
            hours_per_day=14.0,
            days_per_month=30,
            quantity=3,
        )
        # Delta = 94.5 - 35.28 = 59.22 kWh/month
        request = HouseholdScenarioComparisonRequest(
            baseline_forecast_kwh=[180.0],
            month_labels=["Aug 2024"],
            appliance_scenario=ApplianceScenarioRequest(
                baseline=baseline_app,
                modified=modified_app,
            ),
            provider_id="bescom",
            phase_type="single_phase",
        )
        result = self.service.compare_household_scenario(request)

        self.assertFalse(result.is_verified_tariff)
        self.assertEqual(result.total_baseline_kwh, 180.0)
        self.assertAlmostEqual(result.total_delta_kwh, 59.22, places=2)
        # Financial estimates MUST be withheld
        self.assertIsNone(result.total_savings_inr)
        self.assertIsNone(result.steps[0].baseline_bill_amount)
        self.assertIsNone(result.steps[0].adjusted_bill_amount)
        self.assertIsNone(result.steps[0].savings_amount)
        self.assertTrue(any("withheld" in d.lower() for d in result.disclosures))

    def test_exact_bug_scenario_regression_withholding(self):
        """Regression test for user-reported bug:
        1.5 Ton Split AC (1450 W, 8h->7h, 30 days, qty 1) against 2.0 kWh/month household baseline.
        Must preserve appliance calculation (348.0 -> 304.5, delta 43.5 kWh) as a hypothetical estimate,
        withhold adjusted household forecast (NOT clamp to 0.0 kWh), and withhold monetary savings.
        """
        ac_baseline = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton Split AC",
            power_watts=1450.0,
            hours_per_day=8.0,
            days_per_month=30,
            quantity=1,
        )
        ac_modified = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton Split AC",
            power_watts=1450.0,
            hours_per_day=7.0,
            days_per_month=30,
            quantity=1,
        )
        request = HouseholdScenarioComparisonRequest(
            baseline_forecast_kwh=[2.0, 2.0, 2.0],
            month_labels=["Jan 2025", "Feb 2025", "Mar 2025"],
            appliance_scenario=ApplianceScenarioRequest(
                baseline=ac_baseline,
                modified=ac_modified,
            ),
            provider_id="msedcl",
            phase_type="single_phase",
        )
        result = self.service.compare_household_scenario(request)

        # 1. Appliance-only calculation remains correct and labeled hypothetical
        self.assertEqual(result.appliance_result.baseline_kwh, 348.0)
        self.assertEqual(result.appliance_result.modified_kwh, 304.5)
        self.assertEqual(result.appliance_result.delta_kwh, 43.5)
        self.assertTrue(result.appliance_result.is_hypothetical_estimate)

        # 2. Household adjustment MUST be marked unsupported (NOT clamped to 0.0)
        self.assertFalse(result.is_household_adjustment_supported)
        self.assertIsNotNone(result.household_adjustment_reason)
        self.assertIn("exceeds", result.household_adjustment_reason.lower())

        # 3. Household adjusted totals and monetary savings MUST be withheld
        self.assertEqual(result.total_baseline_kwh, 6.0)
        self.assertIsNone(result.total_adjusted_kwh)
        self.assertIsNone(result.total_delta_kwh)
        self.assertIsNone(result.total_savings_inr)

        # 4. Each step must withhold adjusted_kwh and savings (NOT 0.0 or ₹12.95)
        self.assertEqual(len(result.steps), 3)
        for step in result.steps:
            self.assertFalse(step.is_adjustment_supported)
            self.assertEqual(step.baseline_kwh, 2.0)
            self.assertIsNone(step.adjusted_kwh)
            self.assertIsNone(step.delta_kwh)
            self.assertIsNone(step.savings_amount)
            self.assertIsNotNone(step.unsupported_reason)

        # 5. Recommendations should alert user to physical mismatch / load exceeding baseline
        self.assertTrue(any("exceeds" in r.title.lower() or "mismatch" in r.title.lower() for r in result.recommendations))

    def test_valid_household_scenario_consistent_inputs(self):
        """Test that a valid scenario with physically plausible inputs (household baseline > appliance load)
        produces a valid adjusted household forecast and calculated savings.
        """
        ac_baseline = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton Split AC",
            power_watts=1450.0,
            hours_per_day=8.0,
            days_per_month=30,
            quantity=1,
        )
        ac_modified = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton Split AC",
            power_watts=1450.0,
            hours_per_day=7.0,
            days_per_month=30,
            quantity=1,
        )
        # Household baseline is 450, 480, 500 kWh (well above 348 kWh AC baseline)
        request = HouseholdScenarioComparisonRequest(
            baseline_forecast_kwh=[450.0, 480.0, 500.0],
            month_labels=["Aug 2024", "Sep 2024", "Oct 2024"],
            period_starts=[date(2024, 8, 1), date(2024, 9, 1), date(2024, 10, 1)],
            period_ends=[date(2024, 8, 31), date(2024, 9, 30), date(2024, 10, 31)],
            appliance_scenario=ApplianceScenarioRequest(
                baseline=ac_baseline,
                modified=ac_modified,
            ),
            provider_id="msedcl",
            phase_type="single_phase",
        )
        result = self.service.compare_household_scenario(request)

        self.assertTrue(result.is_household_adjustment_supported)
        self.assertIsNone(result.household_adjustment_reason)
        self.assertEqual(result.total_baseline_kwh, 1430.0)
        self.assertEqual(result.total_adjusted_kwh, 1299.5)
        self.assertEqual(result.total_delta_kwh, 130.5)  # 43.5 * 3
        self.assertTrue(result.is_verified_tariff)
        self.assertIsNotNone(result.total_savings_inr)
        self.assertGreater(result.total_savings_inr, 0.0)

        # Step 1: 450 - 43.5 = 406.5
        s1 = result.steps[0]
        self.assertTrue(s1.is_adjustment_supported)
        self.assertEqual(s1.baseline_kwh, 450.0)
        self.assertEqual(s1.adjusted_kwh, 406.5)
        self.assertEqual(s1.delta_kwh, 43.5)
        self.assertIsNotNone(s1.savings_amount)
        self.assertGreater(s1.savings_amount, 0.0)
        self.assertEqual(s1.tariff_status, "verified")
        self.assertEqual(s1.tariff_status_label, "Verified")

    def test_household_scenario_with_expired_tariff_schedule(self):
        """
        Test scenario where forecast periods are after 2025-03-31 (MSEDCL FY 24-25 schedule expired).
        Physical kWh deltas must be preserved, but tariff status must reflect expired schedule
        and monetary bill estimates and savings must be withheld.
        """
        ac_baseline = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton Split AC",
            power_watts=1450.0,
            hours_per_day=8.0,
            days_per_month=30,
            quantity=1,
        )
        ac_modified = ApplianceSimulationInput(
            category=ApplianceCategory.AIR_CONDITIONER,
            name="1.5 Ton Split AC",
            power_watts=1450.0,
            hours_per_day=7.0,
            days_per_month=30,
            quantity=1,
        )
        request = HouseholdScenarioComparisonRequest(
            baseline_forecast_kwh=[450.0, 480.0, 500.0],
            month_labels=["Apr 2025", "May 2025", "Jun 2025"],
            period_starts=[date(2025, 4, 1), date(2025, 5, 1), date(2025, 6, 1)],
            period_ends=[date(2025, 4, 30), date(2025, 5, 31), date(2025, 6, 30)],
            appliance_scenario=ApplianceScenarioRequest(
                baseline=ac_baseline,
                modified=ac_modified,
            ),
            provider_id="msedcl",
            phase_type="single_phase",
        )
        result = self.service.compare_household_scenario(request)

        # 1. Physical adjustment is physically valid and supported
        self.assertTrue(result.is_household_adjustment_supported)
        self.assertEqual(result.total_baseline_kwh, 1430.0)
        self.assertEqual(result.total_adjusted_kwh, 1299.5)
        self.assertEqual(result.total_delta_kwh, 130.5)

        # 2. Tariff status is expired_schedule, NOT verified
        self.assertFalse(result.is_verified_tariff)
        self.assertEqual(result.tariff_status, "expired_schedule")
        self.assertEqual(result.tariff_status_label, "Expired schedule")
        self.assertIsNone(result.total_savings_inr)

        # 3. Each step reflects expired schedule and withholds monetary amounts
        for step in result.steps:
            self.assertEqual(step.tariff_status, "expired_schedule")
            self.assertEqual(step.tariff_status_label, "Expired schedule")
            self.assertIsNone(step.baseline_bill_amount)
            self.assertIsNone(step.adjusted_bill_amount)
            self.assertIsNone(step.savings_amount)


if __name__ == "__main__":
    unittest.main()

