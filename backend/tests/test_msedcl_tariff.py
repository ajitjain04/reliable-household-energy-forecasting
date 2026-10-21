"""
Unit tests for verified MSEDCL LT-I(B) Residential tariff calculator.

Covers:
- Zero consumption (vacant house fixed charges + duty).
- Exact Slab 1 consumption and boundary checks (0 - 100 kWh).
- Multi-slab consumption crossing Slabs 1 and 2 (250 kWh).
- High consumption crossing all 4 slabs (650 kWh).
- Three-phase vs Single-phase fixed charge divergence.
- Fuel Adjustment Charge (FAC) parameter integration.
- Billing cycle pro-rata scaling (non-30-day billing windows).
- Sanctioned load charge for high domestic loads (> 10 kW).
- Registry integration and peer Discom (BESCOM, JVVNL, TANGEDCO) isolation.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.msedcl_tariff_service import msedcl_calculator
from backend.app.services.tariff_service import tariff_registry


class TestMSEDCLTariffCalculator(unittest.TestCase):
    """Direct mathematical unit tests for verified MSEDCL calculation rules."""

    def test_zero_consumption(self):
        """
        0 kWh consumed (30 days, single phase):
        - Energy = 0.00
        - Wheeling = 0.00
        - Fixed = 128.00
        - Taxable = 128.00
        - Duty (16%) = 20.48
        - Total = 148.48
        """
        res = msedcl_calculator.calculate(consumption_kwh=0.0, phase_type="single_phase", cycle_days=30)
        self.assertEqual(res.forecast_kwh, 0.0)
        self.assertTrue(res.is_tariff_supported)
        self.assertTrue(res.is_tariff_verified)
        self.assertIsNotNone(res.estimated_bill_amount)
        self.assertEqual(res.estimated_bill_amount, 148.48)
        self.assertEqual(res.charge_breakdown["energy_charges_inr"], 0.0)
        self.assertEqual(res.charge_breakdown["wheeling_charges_inr"], 0.0)
        self.assertEqual(res.charge_breakdown["fixed_charges_inr"], 128.0)
        self.assertEqual(res.charge_breakdown["electricity_duty_16pct_inr"], 20.48)

    def test_slab_1_consumption(self):
        """
        50 kWh consumed (30 days, single phase):
        - Energy (50 * 4.41) = 220.50
        - Wheeling (50 * 1.17) = 58.50
        - Fixed = 128.00
        - Taxable = 407.00
        - Duty (16%) = 65.12
        - Total = 472.12
        """
        res = msedcl_calculator.calculate(consumption_kwh=50.0, phase_type="single_phase", cycle_days=30)
        self.assertEqual(res.estimated_bill_amount, 472.12)
        self.assertEqual(res.charge_breakdown["slab_1_cost_inr"], 220.50)
        self.assertEqual(res.charge_breakdown["energy_charges_inr"], 220.50)
        self.assertEqual(res.charge_breakdown["wheeling_charges_inr"], 58.50)
        self.assertEqual(res.charge_breakdown["fixed_charges_inr"], 128.00)
        self.assertEqual(res.charge_breakdown["taxable_subtotal_inr"], 407.00)
        self.assertEqual(res.charge_breakdown["electricity_duty_16pct_inr"], 65.12)

    def test_slab_1_exact_boundary(self):
        """
        100 kWh consumed (30 days, single phase):
        - Energy (100 * 4.41) = 441.00
        - Wheeling (100 * 1.17) = 117.00
        - Fixed = 128.00
        - Taxable = 686.00
        - Duty (16%) = 109.76
        - Total = 795.76
        """
        res = msedcl_calculator.calculate(consumption_kwh=100.0, phase_type="single_phase", cycle_days=30)
        self.assertEqual(res.estimated_bill_amount, 795.76)
        self.assertEqual(res.charge_breakdown["slab_1_units"], 100.0)
        self.assertEqual(res.charge_breakdown["slab_2_units"], 0.0)
        self.assertEqual(res.charge_breakdown["energy_charges_inr"], 441.00)

    def test_multi_slab_consumption_250_kwh(self):
        """
        250 kWh consumed (30 days, single phase):
        - Slab 1 (100 * 4.41) = 441.00
        - Slab 2 (150 * 9.64) = 1446.00
        - Energy Total = 1887.00
        - Wheeling (250 * 1.17) = 292.50
        - Fixed = 128.00
        - Taxable = 2307.50
        - Duty (16%) = 369.20
        - Total = 2676.70
        """
        res = msedcl_calculator.calculate(consumption_kwh=250.0, phase_type="single_phase", cycle_days=30)
        self.assertEqual(res.estimated_bill_amount, 2676.70)
        self.assertEqual(res.charge_breakdown["slab_1_units"], 100.0)
        self.assertEqual(res.charge_breakdown["slab_2_units"], 150.0)
        self.assertEqual(res.charge_breakdown["energy_charges_inr"], 1887.00)
        self.assertEqual(res.charge_breakdown["wheeling_charges_inr"], 292.50)
        self.assertEqual(res.charge_breakdown["taxable_subtotal_inr"], 2307.50)
        self.assertEqual(res.charge_breakdown["electricity_duty_16pct_inr"], 369.20)

    def test_high_consumption_all_four_slabs(self):
        """
        650 kWh consumed (30 days, single phase):
        - Slab 1 (100 * 4.41) = 441.00
        - Slab 2 (200 * 9.64) = 1928.00
        - Slab 3 (200 * 13.61) = 2722.00
        - Slab 4 (150 * 15.57) = 2335.50
        - Energy Total = 7426.50
        - Wheeling (650 * 1.17) = 760.50
        - Fixed = 128.00
        - Taxable = 8315.00
        - Duty (16%) = 1330.40
        - Total = 9645.40
        """
        res = msedcl_calculator.calculate(consumption_kwh=650.0, phase_type="single_phase", cycle_days=30)
        self.assertEqual(res.estimated_bill_amount, 9645.40)
        self.assertEqual(res.charge_breakdown["slab_1_units"], 100.0)
        self.assertEqual(res.charge_breakdown["slab_2_units"], 200.0)
        self.assertEqual(res.charge_breakdown["slab_3_units"], 200.0)
        self.assertEqual(res.charge_breakdown["slab_4_units"], 150.0)
        self.assertEqual(res.charge_breakdown["energy_charges_inr"], 7426.50)
        self.assertEqual(res.charge_breakdown["wheeling_charges_inr"], 760.50)
        self.assertEqual(res.charge_breakdown["fixed_charges_inr"], 128.00)
        self.assertEqual(res.charge_breakdown["electricity_duty_16pct_inr"], 1330.40)

    def test_three_phase_fixed_charge_divergence(self):
        """
        250 kWh consumed with Three-Phase connection:
        Fixed charge is 385.00 instead of 128.00 (+257.00 taxable, +298.12 total).
        - Energy = 1887.00
        - Wheeling = 292.50
        - Fixed = 385.00
        - Taxable = 2564.50
        - Duty (16%) = 410.32
        - Total = 2974.82
        """
        res_3p = msedcl_calculator.calculate(consumption_kwh=250.0, phase_type="three_phase", cycle_days=30)
        self.assertEqual(res_3p.estimated_bill_amount, 2974.82)
        self.assertEqual(res_3p.charge_breakdown["fixed_charges_inr"], 385.00)

        res_1p = msedcl_calculator.calculate(consumption_kwh=250.0, phase_type="single_phase", cycle_days=30)
        delta = round(res_3p.estimated_bill_amount - res_1p.estimated_bill_amount, 2)
        self.assertEqual(delta, 298.12)

    def test_fuel_adjustment_charge_inclusion(self):
        """
        250 kWh with explicit FAC rate of ₹0.25/kWh:
        - FAC charge = 250 * 0.25 = 62.50
        - Taxable increases by 62.50 (2307.50 + 62.50 = 2370.00)
        - Duty (16%) = 2370.00 * 0.16 = 379.20 (+10.00)
        - Total = 2370.00 + 379.20 = 2749.20 (+72.50)
        """
        res_fac = msedcl_calculator.calculate(
            consumption_kwh=250.0,
            phase_type="single_phase",
            cycle_days=30,
            fac_rate_per_kwh=0.25,
        )
        self.assertEqual(res_fac.charge_breakdown["fuel_adjustment_charges_inr"], 62.50)
        self.assertEqual(res_fac.estimated_bill_amount, 2749.20)

    def test_cycle_prorata_scaling_45_days(self):
        """
        45-day billing cycle (prorata_factor = 1.5):
        - Slabs expand by 1.5: Slab 1 is 0 to 150 kWh.
        - Fixed charge scales by 1.5: 128 * 1.5 = 192.00.
        - 150 kWh consumed in 45 days: all 150 kWh fall into Slab 1 (150 * 4.41 = 661.50).
        - Wheeling = 150 * 1.17 = 175.50.
        - Taxable = 661.50 + 175.50 + 192.00 = 1029.00.
        - Duty (16%) = 164.64.
        - Total = 1193.64.
        """
        res_45d = msedcl_calculator.calculate(
            consumption_kwh=150.0,
            phase_type="single_phase",
            cycle_days=45,
        )
        self.assertEqual(res_45d.charge_breakdown["slab_1_units"], 150.0)
        self.assertEqual(res_45d.charge_breakdown["slab_2_units"], 0.0)
        self.assertEqual(res_45d.charge_breakdown["fixed_charges_inr"], 192.00)
        self.assertEqual(res_45d.estimated_bill_amount, 1193.64)

    def test_sanctioned_load_excess_over_10kw(self):
        """
        15 kW load (5 kW excess -> 1 block of 10 kW -> +200 INR fixed charge).
        Fixed charge for single phase becomes 128 + 200 = 328.00.
        """
        res_high_load = msedcl_calculator.calculate(
            consumption_kwh=100.0,
            phase_type="single_phase",
            cycle_days=30,
            sanctioned_load_kw=15.0,
        )
        self.assertEqual(res_high_load.charge_breakdown["fixed_charges_inr"], 328.00)

    def test_audit_167_kwh_standard_30_day_cycle(self):
        """
        Audit Task 2.3: Trace arithmetic for 167 kWh (standard 30-day billing cycle):
        - Slab 1 (100 * 4.41) = 441.00
        - Slab 2 (67 * 9.64) = 645.88
        - Energy Charges = 1086.88
        - Wheeling Charges (167 * 1.17) = 195.39
        - Fixed Charges (Single Phase) = 128.00
        - FAC = 0.00
        - Taxable Subtotal = 1086.88 + 195.39 + 128.00 = 1410.27
        - Electricity Duty (16%) = 1410.27 * 0.16 = 225.64
        - Estimated Total = 1410.27 + 225.64 = 1635.91
        """
        res = msedcl_calculator.calculate(
            consumption_kwh=167.0,
            phase_type="single_phase",
            cycle_days=30,
        )
        bd = res.charge_breakdown
        self.assertEqual(bd["slab_1_units"], 100.0)
        self.assertEqual(bd["slab_1_cost_inr"], 441.00)
        self.assertEqual(bd["slab_2_units"], 67.0)
        self.assertEqual(bd["slab_2_cost_inr"], 645.88)
        self.assertEqual(bd["energy_charges_inr"], 1086.88)
        self.assertEqual(bd["wheeling_charges_inr"], 195.39)
        self.assertEqual(bd["fixed_charges_inr"], 128.00)
        self.assertEqual(bd["fuel_adjustment_charges_inr"], 0.0)
        self.assertEqual(bd["taxable_subtotal_inr"], 1410.27)
        self.assertEqual(bd["electricity_duty_16pct_inr"], 225.64)
        self.assertEqual(bd["estimated_total_bill_inr"], 1635.91)
        self.assertEqual(res.estimated_bill_amount, 1635.91)

    def test_calendar_month_duration_nov_dec_jan(self):
        """
        Task 2.3A Issue 1: Complete calendar months (Nov 1-30, Dec 1-31, Jan 1-31)
        must all use the standard monthly 1.0 slab and fixed charge treatment.
        167 kWh on single phase produces 1635.91 INR for all complete calendar months.
        """
        # November 1-30 (30 days): Complete calendar month
        res_nov = msedcl_calculator.calculate(
            consumption_kwh=167.0,
            phase_type="single_phase",
            billing_period_start=date(2024, 11, 1),
            billing_period_end=date(2024, 11, 30),
        )
        self.assertEqual(res_nov.estimated_bill_amount, 1635.91)
        self.assertEqual(res_nov.charge_breakdown["slab_1_units"], 100.0)
        self.assertEqual(res_nov.charge_breakdown["slab_2_units"], 67.0)
        self.assertEqual(res_nov.charge_breakdown["energy_charges_inr"], 1086.88)
        self.assertEqual(res_nov.charge_breakdown["fixed_charges_inr"], 128.0)
        self.assertEqual(res_nov.charge_breakdown["electricity_duty_16pct_inr"], 225.64)
        self.assertFalse(any("prorated" in d.lower() for d in res_nov.disclosures))

        # December 1-31 (31 days): Complete calendar month
        res_dec = msedcl_calculator.calculate(
            consumption_kwh=167.0,
            phase_type="single_phase",
            billing_period_start=date(2024, 12, 1),
            billing_period_end=date(2024, 12, 31),
        )
        self.assertEqual(res_dec.estimated_bill_amount, 1635.91)
        self.assertEqual(res_dec.charge_breakdown["slab_1_units"], 100.0)
        self.assertEqual(res_dec.charge_breakdown["fixed_charges_inr"], 128.0)
        self.assertFalse(any("prorated" in d.lower() for d in res_dec.disclosures))

        # January 1-31 (31 days): Complete calendar month
        res_jan = msedcl_calculator.calculate(
            consumption_kwh=167.0,
            phase_type="single_phase",
            billing_period_start=date(2025, 1, 1),
            billing_period_end=date(2025, 1, 31),
        )
        self.assertEqual(res_jan.estimated_bill_amount, 1635.91)
        self.assertEqual(res_jan.charge_breakdown["slab_1_units"], 100.0)
        self.assertEqual(res_jan.charge_breakdown["fixed_charges_inr"], 128.0)
        self.assertFalse(any("prorated" in d.lower() for d in res_jan.disclosures))

    def test_shorter_billing_cycle_pro_rata_scaling(self):
        """
        Task 2.3A Issue 1: Genuinely shorter non-calendar billing cycle (e.g. 15 days)
        must retain justified pro-rata scaling:
        - 15 days (2024-04-01 to 2024-04-15 inclusive): factor = 15 / 30 = 0.5.
        - Fixed charge: 128.0 * 0.5 = 64.00.
        - Slabs: Slab 1 is 0 to 50.0 kWh @ 4.41.
        - 50 kWh consumed:
          - Energy = 50 * 4.41 = 220.50
          - Wheeling = 50 * 1.17 = 58.50
          - Fixed = 64.00
          - Taxable = 220.50 + 58.50 + 64.00 = 343.00
          - Duty (16%) = 343.00 * 0.16 = 54.88
          - Total = 397.88
        """
        res_15d = msedcl_calculator.calculate(
            consumption_kwh=50.0,
            phase_type="single_phase",
            billing_period_start=date(2024, 4, 1),
            billing_period_end=date(2024, 4, 15),
        )
        self.assertEqual(res_15d.charge_breakdown["slab_1_units"], 50.0)
        self.assertEqual(res_15d.charge_breakdown["energy_charges_inr"], 220.50)
        self.assertEqual(res_15d.charge_breakdown["wheeling_charges_inr"], 58.50)
        self.assertEqual(res_15d.charge_breakdown["fixed_charges_inr"], 64.00)
        self.assertEqual(res_15d.charge_breakdown["taxable_subtotal_inr"], 343.00)
        self.assertEqual(res_15d.charge_breakdown["electricity_duty_16pct_inr"], 54.88)
        self.assertEqual(res_15d.estimated_bill_amount, 397.88)
        self.assertTrue(any("prorated by 0.500" in d for d in res_15d.disclosures))

    def test_future_unverified_tariff_withholds_monetary_estimate(self):
        """
        Task 2.3A Issue 2: A future date (e.g. Nov 2026 in FY 2026-27) beyond the
        verified FY 2024-25 baseline (which ended 2025-03-31) must retain physical
        forecast but return NO monetary bill estimate.
        """
        res_future = tariff_registry.estimate_bill(
            forecast_kwh=167.0,
            provider_id="msedcl",
            as_of_date=date(2026, 11, 1),
            billing_period_start=date(2026, 11, 1),
            billing_period_end=date(2026, 11, 30),
        )
        self.assertEqual(res_future.forecast_kwh, 167.0)
        self.assertFalse(res_future.is_tariff_verified)
        self.assertFalse(res_future.is_tariff_supported)
        self.assertIsNone(res_future.estimated_bill_amount)
        self.assertIsNone(res_future.charge_breakdown)
        self.assertTrue(any("no verified tariff schedule is active" in d.lower() for d in res_future.disclosures))


class TestRegistryMSEDCLIntegration(unittest.TestCase):

    """Integration tests verifying registry routing to MSEDCL and fallback for peers."""

    def test_registry_routes_to_verified_msedcl(self):
        """Verify that tariff_registry delegates MSEDCL queries within the verified window."""
        res = tariff_registry.estimate_bill(
            forecast_kwh=200.0,
            provider_id="msedcl",
            as_of_date=date(2024, 6, 1),
            connection_params={"phase_type": "single_phase", "cycle_days": 30},
        )
        self.assertTrue(res.is_tariff_supported)
        self.assertTrue(res.is_tariff_verified)
        self.assertEqual(res.provider_id, "msedcl")
        self.assertIsNotNone(res.estimated_bill_amount)
        self.assertGreater(res.estimated_bill_amount, 0.0)

    def test_registry_peer_discoms_remain_unverified(self):
        """Verify that BESCOM, JVVNL, and TANGEDCO remain strictly unverified and fall back."""
        for peer in ["bescom", "jvvnl", "tangedco"]:
            res = tariff_registry.estimate_bill(
                forecast_kwh=200.0,
                provider_id=peer,
            )
            self.assertFalse(res.is_tariff_supported, f"{peer} must not be supported yet")
            self.assertFalse(res.is_tariff_verified, f"{peer} must not be verified yet")
            self.assertIsNone(res.estimated_bill_amount, f"{peer} must not have monetary estimate")
            self.assertEqual(res.forecast_kwh, 200.0)
            self.assertIn("currently unverified", res.disclosures[0])


if __name__ == "__main__":
    unittest.main()
