"""
Unit tests for Pan-India Tariff Architecture Foundation.

Covers:
- Tariff schema validation (EnergySlab bounds, contiguity, effective dates).
- Provider registry initialization and case-insensitive lookup.
- Effective date resolution (active, historical, expired schedules).
- Verification enforcement (unverified providers rejected for monetary billing).
- Decoupled fallback bill estimation (forecast_kwh preserved, monetary bill withheld with disclosure).
- Extensibility of billing frequencies, fixed-charge models, and duty types.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import ValidationError
from backend.app.schemas.tariffs import (
    BillEstimateResponse,
    BillingFrequency,
    DutyType,
    EnergySlab,
    FixedChargeType,
    TariffSchedule,
    TariffScheduleMetadata,
)
from backend.app.services.tariff_service import TariffRegistry, tariff_registry


class TestTariffSchemas(unittest.TestCase):
    """Tests for tariff data models and constraints."""

    def test_valid_energy_slab_bounded(self):
        """Test valid bounded energy tier."""
        slab = EnergySlab(slab_index=1, min_units=0.0, max_units=100.0, rate_per_kwh=4.05)
        self.assertEqual(slab.slab_index, 1)
        self.assertEqual(slab.min_units, 0.0)
        self.assertEqual(slab.max_units, 100.0)
        self.assertEqual(slab.rate_per_kwh, 4.05)

    def test_valid_energy_slab_unbounded(self):
        """Test top-tier unbounded slab (max_units is None)."""
        slab = EnergySlab(slab_index=4, min_units=500.0, max_units=None, rate_per_kwh=16.50)
        self.assertIsNone(slab.max_units)
        self.assertEqual(slab.min_units, 500.0)

    def test_invalid_energy_slab_bounds(self):
        """Test rejection when min_units >= max_units."""
        with self.assertRaises(ValidationError) as ctx:
            EnergySlab(slab_index=1, min_units=100.0, max_units=50.0, rate_per_kwh=5.0)
        self.assertIn("must be strictly less than", str(ctx.exception))

    def test_invalid_negative_rates(self):
        """Test rejection of negative rates or negative units."""
        with self.assertRaises(ValidationError):
            EnergySlab(slab_index=1, min_units=-1.0, max_units=50.0, rate_per_kwh=5.0)
        with self.assertRaises(ValidationError):
            EnergySlab(slab_index=1, min_units=0.0, max_units=50.0, rate_per_kwh=-2.5)

    def test_invalid_schedule_dates_from_after_to(self):
        """Test rejection when effective_from is after effective_to."""
        with self.assertRaises(ValidationError) as ctx:
            TariffScheduleMetadata(
                provider_id="test_discom",
                provider_name="Test Electricity Board",
                state="Test State",
                regulatory_commission="TERC",
                category_code="LT-1",
                effective_from=date(2025, 4, 1),
                effective_to=date(2024, 4, 1),  # earlier than from
                fixed_charge_type=FixedChargeType.FLAT_MONTHLY,
                duty_type=DutyType.EXEMPT,
            )
        self.assertIn("cannot be after effective_to", str(ctx.exception))

    def test_contiguous_slabs_validation(self):
        """Test that non-contiguous slabs raise ValueError."""
        slab1 = EnergySlab(slab_index=1, min_units=0.0, max_units=100.0, rate_per_kwh=3.0)
        # Gap between 100 and 150
        slab2 = EnergySlab(slab_index=2, min_units=150.0, max_units=300.0, rate_per_kwh=6.0)

        with self.assertRaises(ValidationError) as ctx:
            TariffSchedule(
                provider_id="test_discom",
                provider_name="Test Electricity Board",
                state="Test State",
                regulatory_commission="TERC",
                category_code="LT-1",
                effective_from=date(2024, 1, 1),
                fixed_charge_type=FixedChargeType.ZERO,
                duty_type=DutyType.EXEMPT,
                slabs=[slab1, slab2],
            )
        self.assertIn("Non-contiguous slabs", str(ctx.exception))


class TestTariffRegistry(unittest.TestCase):
    """Tests for Provider Registry lookups, normalization, and effective dating."""

    def setUp(self):
        # Create an isolated test registry to avoid mutating global singleton
        self.registry = TariffRegistry()

    def test_known_providers_status(self):
        """
        Verify provider registry states:
        - MSEDCL is officially verified under MERC Case No. 226 of 2022 (Task 1.1D).
        - BESCOM, JVVNL, and TANGEDCO are strictly unverified pending future tasks.
        """
        providers = {p["provider_id"]: p for p in self.registry.list_providers()}
        self.assertIn("msedcl", providers)
        self.assertIn("bescom", providers)
        self.assertIn("jvvnl", providers)
        self.assertIn("tangedco", providers)

        # MSEDCL verified in Task 1.1D
        self.assertTrue(providers["msedcl"]["is_verified"])
        self.assertTrue(self.registry.is_provider_supported("msedcl"))

        # Other providers strictly unverified
        self.assertFalse(providers["bescom"]["is_verified"])
        self.assertFalse(providers["jvvnl"]["is_verified"])
        self.assertFalse(providers["tangedco"]["is_verified"])
        self.assertFalse(self.registry.is_provider_supported("bescom"))
        self.assertFalse(self.registry.is_provider_supported("jvvnl"))
        self.assertFalse(self.registry.is_provider_supported("tangedco"))

    def test_case_insensitive_lookup(self):
        """Test that provider lookups are normalized and case-insensitive."""
        meta1 = self.registry.get_provider_metadata("MSEDCL")
        meta2 = self.registry.get_provider_metadata("  msedcl  ")
        meta3 = self.registry.get_provider_metadata("Msedcl")

        self.assertIsNotNone(meta1)
        self.assertEqual(meta1.provider_id, "msedcl")
        self.assertEqual(meta1, meta2)
        self.assertEqual(meta1, meta3)

    def test_unknown_provider_lookup(self):
        """Test that querying an unregistered provider returns None."""
        self.assertIsNone(self.registry.get_provider_metadata("non_existent_discom"))
        self.assertIsNone(self.registry.get_schedule("non_existent_discom"))
        self.assertFalse(self.registry.is_provider_supported("non_existent_discom"))

    def test_effective_date_schedule_resolution(self):
        """Test effective-date schedule resolution across historical versions."""
        schedule_2023 = TariffSchedule(
            provider_id="mock_discom",
            provider_name="Mock Discom",
            state="Mock State",
            regulatory_commission="MERC_MOCK",
            category_code="LT-1",
            effective_from=date(2023, 4, 1),
            effective_to=date(2024, 3, 31),
            fixed_charge_type=FixedChargeType.FLAT_MONTHLY,
            duty_type=DutyType.PERCENTAGE,
            is_verified=True,
            verification_notes="FY 2023-24 Mock Schedule",
        )
        schedule_2024 = TariffSchedule(
            provider_id="mock_discom",
            provider_name="Mock Discom",
            state="Mock State",
            regulatory_commission="MERC_MOCK",
            category_code="LT-1",
            effective_from=date(2024, 4, 1),
            effective_to=None,  # active onwards
            fixed_charge_type=FixedChargeType.FLAT_MONTHLY,
            duty_type=DutyType.PERCENTAGE,
            is_verified=True,
            verification_notes="FY 2024-25 Active Mock Schedule",
        )

        self.registry.register_schedule(schedule_2023)
        self.registry.register_schedule(schedule_2024)

        # 1. Before earliest schedule -> None
        self.assertIsNone(
            self.registry.get_schedule("mock_discom", as_of_date=date(2023, 1, 1))
        )

        # 2. Inside 2023 schedule
        s_mid_2023 = self.registry.get_schedule("mock_discom", as_of_date=date(2023, 10, 15))
        self.assertIsNotNone(s_mid_2023)
        self.assertEqual(s_mid_2023.verification_notes, "FY 2023-24 Mock Schedule")

        # 3. Inside 2024 active schedule
        s_mid_2024 = self.registry.get_schedule("mock_discom", as_of_date=date(2024, 8, 1))
        self.assertIsNotNone(s_mid_2024)
        self.assertEqual(s_mid_2024.verification_notes, "FY 2024-25 Active Mock Schedule")

        # 4. Verified support flag
        self.assertTrue(self.registry.is_provider_supported("mock_discom", as_of_date=date(2024, 8, 1)))

    def test_unverified_provider_not_supported(self):
        """Test that a schedule with is_verified=False is NOT considered supported for billing."""
        unverified_schedule = TariffSchedule(
            provider_id="unverified_discom",
            provider_name="Unverified Electricity Co",
            state="State",
            regulatory_commission="SERC",
            category_code="LT-1",
            effective_from=date(2024, 1, 1),
            fixed_charge_type=FixedChargeType.ZERO,
            duty_type=DutyType.EXEMPT,
            is_verified=False,
        )
        self.registry.register_schedule(unverified_schedule)
        self.assertFalse(self.registry.is_provider_supported("unverified_discom"))


class TestDecoupledBillEstimateFallback(unittest.TestCase):
    """Tests guaranteeing that physical energy forecasts are preserved without monetary fabrication."""

    def setUp(self):
        self.registry = TariffRegistry()

    def test_fallback_with_unverified_registered_discom(self):
        """
        Test bill estimation for BESCOM (registered as unverified):
        Must preserve 100% of forecast_kwh, withhold estimated_bill_amount, and emit disclosure.
        """
        res = self.registry.estimate_bill(
            forecast_kwh=312.45,
            provider_id="bescom",
            billing_period_start=date(2024, 4, 1),
            billing_period_end=date(2024, 5, 1),
        )

        self.assertIsInstance(res, BillEstimateResponse)
        self.assertEqual(res.forecast_kwh, 312.45)
        self.assertFalse(res.is_tariff_supported)
        self.assertFalse(res.is_tariff_verified)
        self.assertIsNone(res.estimated_bill_amount)
        self.assertIsNone(res.charge_breakdown)
        self.assertEqual(res.provider_id, "bescom")
        self.assertEqual(res.state, "Karnataka")

        # Must contain explicit disclosure
        self.assertGreater(len(res.disclosures), 0)
        self.assertIn("currently unverified", res.disclosures[0])
        self.assertIn("No monetary bill estimate was fabricated", res.disclosures[0])

    def test_fallback_with_unknown_provider(self):
        """Test bill estimation for completely unknown provider."""
        res = self.registry.estimate_bill(
            forecast_kwh=150.0,
            provider_id="unknown_power_corp",
        )

        self.assertEqual(res.forecast_kwh, 150.0)
        self.assertFalse(res.is_tariff_supported)
        self.assertIsNone(res.estimated_bill_amount)
        self.assertIn("not currently configured", res.disclosures[0])

    def test_fallback_without_provider(self):
        """Test bill estimation when user provides no provider at all."""
        res = self.registry.estimate_bill(
            forecast_kwh=420.7,
            provider_id=None,
        )

        self.assertEqual(res.forecast_kwh, 420.7)
        self.assertFalse(res.is_tariff_supported)
        self.assertIsNone(res.estimated_bill_amount)
        self.assertIn("No electricity distribution company was specified", res.disclosures[0])

    def test_multiple_billing_frequencies_support(self):
        """Verify that multiple billing frequencies and fixed charge types are supported."""
        # Monthly MSEDCL
        msedcl_meta = self.registry.get_provider_metadata("msedcl")
        self.assertEqual(msedcl_meta.billing_frequency, BillingFrequency.MONTHLY)
        self.assertEqual(msedcl_meta.fixed_charge_type, FixedChargeType.PHASE_BASED)

        # Bi-monthly TANGEDCO
        tangedco_meta = self.registry.get_provider_metadata("tangedco")
        self.assertEqual(tangedco_meta.billing_frequency, BillingFrequency.BI_MONTHLY)
        self.assertEqual(tangedco_meta.fixed_charge_type, FixedChargeType.ZERO)

        # Per-kW BESCOM
        bescom_meta = self.registry.get_provider_metadata("bescom")
        self.assertEqual(bescom_meta.fixed_charge_type, FixedChargeType.PER_KW_SANCTIONED)

    def test_get_tariff_status_distinguishes_four_concepts(self):
        """
        Verify that get_tariff_status clearly distinguishes:
        1. Provider identification confirmed.
        2. Historical tariff schedule known.
        3. Tariff schedule currently applicable to the forecast period.
        4. Monetary estimate available or withheld.
        """
        # Concept 1, 2, 3: Active verified schedule in FY 24-25
        info_valid = self.registry.get_tariff_status("msedcl", as_of_date=date(2024, 8, 1))
        self.assertTrue(info_valid["is_provider_known"])
        self.assertTrue(info_valid["has_historical_verified_schedule"])
        self.assertTrue(info_valid["is_applicable_schedule_verified"])
        self.assertEqual(info_valid["status"], "verified")
        self.assertEqual(info_valid["status_label"], "Verified")

        # Concept 1, 2 vs 3: Known provider and historical schedule, but expired for forecast period
        info_exp = self.registry.get_tariff_status("msedcl", as_of_date=date(2025, 4, 1))
        self.assertTrue(info_exp["is_provider_known"])
        self.assertTrue(info_exp["has_historical_verified_schedule"])
        self.assertFalse(info_exp["is_applicable_schedule_verified"])
        self.assertEqual(info_exp["status"], "expired_schedule")
        self.assertEqual(info_exp["status_label"], "Expired schedule")

        # Future date in 2026 also resolves to expired_schedule
        info_future = self.registry.get_tariff_status("msedcl", as_of_date=date(2026, 10, 21))
        self.assertTrue(info_future["has_historical_verified_schedule"])
        self.assertFalse(info_future["is_applicable_schedule_verified"])
        self.assertEqual(info_future["status"], "expired_schedule")
        self.assertEqual(info_future["status_label"], "Expired schedule")

        # Known provider but unverified schedule
        info_unver = self.registry.get_tariff_status("bescom", as_of_date=date(2024, 8, 1))
        self.assertTrue(info_unver["is_provider_known"])
        self.assertFalse(info_unver["has_historical_verified_schedule"])
        self.assertFalse(info_unver["is_applicable_schedule_verified"])
        self.assertEqual(info_unver["status"], "unverified_schedule")
        self.assertEqual(info_unver["status_label"], "Current tariff unverified")

        # No provider specified
        info_none = self.registry.get_tariff_status(None, as_of_date=date(2024, 8, 1))
        self.assertFalse(info_none["is_provider_known"])
        self.assertFalse(info_none["has_historical_verified_schedule"])
        self.assertFalse(info_none["is_applicable_schedule_verified"])
        self.assertEqual(info_none["status"], "no_provider")
        self.assertEqual(info_none["status_label"], "No provider selected")

    def test_estimate_bill_expired_schedule_withholds_monetary_amount(self):
        """
        Ensure estimate_bill for MSEDCL after 2025-03-31 sets tariff_status='expired_schedule'
        and withholds estimated_bill_amount.
        """
        res = self.registry.estimate_bill(
            forecast_kwh=250.0,
            provider_id="msedcl",
            billing_period_start=date(2025, 4, 1),
            billing_period_end=date(2025, 4, 30),
        )
        self.assertEqual(res.forecast_kwh, 250.0)
        self.assertFalse(res.is_tariff_verified)
        self.assertIsNone(res.estimated_bill_amount)
        self.assertEqual(res.tariff_status, "expired_schedule")
        self.assertEqual(res.tariff_status_label, "Expired schedule")
        self.assertTrue(any("expired on 2025-03-31" in d for d in res.disclosures))

    def test_estimate_bill_valid_schedule_produces_verified_amount(self):
        """
        Ensure estimate_bill for MSEDCL on or before 2025-03-31 sets tariff_status='verified'
        and produces non-null estimated_bill_amount.
        """
        res = self.registry.estimate_bill(
            forecast_kwh=250.0,
            provider_id="msedcl",
            billing_period_start=date(2024, 8, 1),
            billing_period_end=date(2024, 8, 31),
        )
        self.assertEqual(res.forecast_kwh, 250.0)
        self.assertTrue(res.is_tariff_verified)
        self.assertIsNotNone(res.estimated_bill_amount)
        self.assertGreater(res.estimated_bill_amount, 0.0)
        self.assertEqual(res.tariff_status, "verified")
        self.assertEqual(res.tariff_status_label, "Verified")


if __name__ == "__main__":
    unittest.main()
