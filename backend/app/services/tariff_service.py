"""
Pan-India Tariff Service and Provider Registry.

Implements a provider-independent architecture for:
- Registering distribution companies (Discoms) across Indian states.
- Managing effective-dated tariff schedules.
- Performing safe, normalized provider lookups.
- Enforcing strict verification boundaries: unverified or unknown providers NEVER
  receive fabricated monetary bill estimates.
- Preserving physical energy forecasts (kWh) independently of tariff availability.
"""

from datetime import date, timedelta
from typing import Any, Dict, List, Optional
from backend.app.schemas.tariffs import (
    BillEstimateResponse,
    BillingFrequency,
    DutyType,
    EnergySlab,
    FixedChargeType,
    TariffSchedule,
    TariffScheduleMetadata,
)


class TariffRegistry:
    """
    In-memory registry managing provider metadata and versioned, effective-dated tariff schedules.
    """

    def __init__(self):
        # provider_id (normalized slug) -> list of TariffSchedule objects (sorted by effective_from)
        self._schedules: Dict[str, List[TariffSchedule]] = {}
        # provider_id -> TariffScheduleMetadata profile
        self._providers: Dict[str, TariffScheduleMetadata] = {}
        self._initialize_known_discoms()

    def _normalize_id(self, provider_id: Optional[str]) -> str:
        """Sanitize and normalize provider identifier slug."""
        if not provider_id:
            return ""
        return provider_id.strip().lower().replace(" ", "_").replace("-", "_")

    def _initialize_known_discoms(self) -> None:
        """
        Seed initial representative Discom metadata across India.

        STRICT ARCHITECTURAL REQUIREMENT:
        In accordance with Task 1.1C specifications, MSEDCL, JVVNL, BESCOM, and TANGEDCO
        are registered as metadata profiles with 'is_verified=False' and NO active verified
        tariff schedules. Official rates and formulas must be verified against primary regulatory
        documents in subsequent tasks before being marked verified.
        """
        # 1. MSEDCL - Officially Verified under MERC Case No. 226 of 2022 (Task 1.1D)
        msedcl_slabs = [
            EnergySlab(slab_index=1, min_units=0.0, max_units=100.0, rate_per_kwh=4.41),
            EnergySlab(slab_index=2, min_units=100.0, max_units=300.0, rate_per_kwh=9.64),
            EnergySlab(slab_index=3, min_units=300.0, max_units=500.0, rate_per_kwh=13.61),
            EnergySlab(slab_index=4, min_units=500.0, max_units=None, rate_per_kwh=15.57),
        ]
        msedcl_schedule = TariffSchedule(
            provider_id="msedcl",
            provider_name="Maharashtra State Electricity Distribution Company Limited (Mahavitaran)",
            state="Maharashtra",
            regulatory_commission="MERC (Maharashtra Electricity Regulatory Commission)",
            category_code="LT-I(B)",
            effective_from=date(2024, 4, 1),
            effective_to=date(2025, 3, 31),
            billing_frequency=BillingFrequency.MONTHLY,
            fixed_charge_type=FixedChargeType.PHASE_BASED,
            duty_type=DutyType.PERCENTAGE,
            is_verified=True,
            official_citation="MERC MTR Order Case No. 226 of 2022 (31 March 2023) & Circular No. 341",
            verification_notes="Verified FY 2024-25 baseline rates for LT-I(B) Residential (effective 1 April 2024 to 31 March 2025).",
            slabs=msedcl_slabs,
            fixed_charge_rates={"single_phase": 128.0, "three_phase": 385.0},
            wheeling_charge_per_kwh=1.17,
            duty_rate=0.16,
            additional_charges={"load_excess_per_10kw": 200.0},
        )
        self._providers["msedcl"] = TariffScheduleMetadata(
            provider_id="msedcl",
            provider_name="Maharashtra State Electricity Distribution Company Limited (Mahavitaran)",
            state="Maharashtra",
            regulatory_commission="MERC (Maharashtra Electricity Regulatory Commission)",
            category_code="LT-I(B)",
            effective_from=date(2024, 4, 1),
            effective_to=date(2025, 3, 31),
            billing_frequency=BillingFrequency.MONTHLY,
            fixed_charge_type=FixedChargeType.PHASE_BASED,
            duty_type=DutyType.PERCENTAGE,
            is_verified=True,
            official_citation="MERC MTR Order Case No. 226 of 2022 (31 March 2023) & Circular No. 341",
            verification_notes="Verified FY 2024-25 baseline rates for LT-I(B) Residential (effective 1 April 2024 to 31 March 2025).",
        )
        self._schedules["msedcl"] = [msedcl_schedule]

        # 2. Other representative Discoms - STRICTLY UNVERIFIED pending future tasks
        unverified_providers = [
            TariffScheduleMetadata(
                provider_id="bescom",
                provider_name="Bangalore Electricity Supply Company Limited",
                state="Karnataka",
                regulatory_commission="KERC (Karnataka Electricity Regulatory Commission)",
                category_code="LT-1",
                effective_from=date(2024, 4, 1),
                billing_frequency=BillingFrequency.MONTHLY,
                fixed_charge_type=FixedChargeType.PER_KW_SANCTIONED,
                duty_type=DutyType.PERCENTAGE,
                is_verified=False,
                official_citation="KERC Tariff Orders & Gruha Jyothi Guidelines",
                verification_notes="Single-slab domestic model verified; rate schedules awaiting verification.",
            ),
            TariffScheduleMetadata(
                provider_id="jvvnl",
                provider_name="Jaipur Vidyut Vitran Nigam Limited",
                state="Rajasthan",
                regulatory_commission="RERC (Rajasthan Electricity Regulatory Commission)",
                category_code="LT-1",
                effective_from=date(2024, 4, 1),
                billing_frequency=BillingFrequency.MONTHLY,
                fixed_charge_type=FixedChargeType.SLAB_TIERED,
                duty_type=DutyType.FLAT_PER_UNIT,
                is_verified=False,
                official_citation="RERC Multi-Year Tariff Orders",
                verification_notes="Load-tiered and urban cess framework verified; rates awaiting verification.",
            ),
            TariffScheduleMetadata(
                provider_id="tangedco",
                provider_name="Tamil Nadu Generation and Distribution Corporation Limited (TNPDCL)",
                state="Tamil Nadu",
                regulatory_commission="TNERC (Tamil Nadu Electricity Regulatory Commission)",
                category_code="Tariff-IA",
                effective_from=date(2024, 7, 1),
                billing_frequency=BillingFrequency.BI_MONTHLY,
                fixed_charge_type=FixedChargeType.ZERO,
                duty_type=DutyType.EXEMPT,
                is_verified=False,
                official_citation="TNERC Comprehensive Tariff Order",
                verification_notes="Bi-monthly cycle & ₹0 fixed charge verified; rates awaiting verification.",
            ),
        ]

        for p in unverified_providers:
            self._providers[p.provider_id] = p
            self._schedules[p.provider_id] = []

    def register_provider(self, metadata: TariffScheduleMetadata) -> None:
        """Register or update a provider metadata profile."""
        norm_id = self._normalize_id(metadata.provider_id)
        self._providers[norm_id] = metadata
        if norm_id not in self._schedules:
            self._schedules[norm_id] = []

    def register_schedule(self, schedule: TariffSchedule) -> None:
        """Register an effective-dated tariff schedule for a provider."""
        norm_id = self._normalize_id(schedule.provider_id)
        if norm_id not in self._schedules:
            self._schedules[norm_id] = []

        # Maintain sorted order by effective_from
        self._schedules[norm_id].append(schedule)
        self._schedules[norm_id].sort(key=lambda s: s.effective_from)

        # Update metadata profile if present
        if norm_id not in self._providers:
            meta = TariffScheduleMetadata(
                provider_id=schedule.provider_id,
                provider_name=schedule.provider_name,
                state=schedule.state,
                regulatory_commission=schedule.regulatory_commission,
                category_code=schedule.category_code,
                effective_from=schedule.effective_from,
                effective_to=schedule.effective_to,
                billing_frequency=schedule.billing_frequency,
                fixed_charge_type=schedule.fixed_charge_type,
                duty_type=schedule.duty_type,
                is_verified=schedule.is_verified,
                official_citation=schedule.official_citation,
                verification_notes=schedule.verification_notes,
            )
            self._providers[norm_id] = meta

    def get_provider_metadata(self, provider_id: Optional[str]) -> Optional[TariffScheduleMetadata]:
        """Retrieve metadata profile for a provider slug, if registered."""
        norm_id = self._normalize_id(provider_id)
        return self._providers.get(norm_id)

    def get_schedule(
        self,
        provider_id: Optional[str],
        as_of_date: Optional[date] = None,
        category_code: Optional[str] = None,
    ) -> Optional[TariffSchedule]:
        """
        Retrieve the legally active tariff schedule for a provider on a given date.

        Parameters:
            provider_id: Provider slug (case-insensitive).
            as_of_date: Reference calendar date (defaults to today if omitted).
            category_code: Optional category filter.

        Returns:
            The matching TariffSchedule or None if no active schedule covers the date.
        """
        norm_id = self._normalize_id(provider_id)
        schedules = self._schedules.get(norm_id, [])
        if not schedules:
            return None

        ref_date = as_of_date or date.today()

        # Find all candidate schedules covering ref_date
        matching = []
        for s in schedules:
            if category_code and s.category_code.lower() != category_code.strip().lower():
                continue
            if s.effective_from <= ref_date:
                if s.effective_to is not None:
                    if ref_date <= s.effective_to:
                        matching.append(s)
                else:
                    # Enforce Issue 2: A schedule cannot be treated as verified indefinitely
                    # merely because effective_to is missing.
                    # Without an explicit effective_to, tariff validity is capped at 365 days from effective_from.
                    if ref_date <= s.effective_from + timedelta(days=365):
                        matching.append(s)

        if not matching:
            return None

        # Return the most recent matching schedule
        return matching[-1]

    def is_provider_supported(
        self,
        provider_id: Optional[str],
        as_of_date: Optional[date] = None,
    ) -> bool:
        """
        Check whether an active and verified tariff schedule is available.

        - If as_of_date is provided: Returns True ONLY if an active schedule covers that
          specific date AND schedule.is_verified is True.
        - If as_of_date is omitted: Returns True if the provider is configured with
          any officially verified tariff schedules in the registry.
        """
        norm_id = self._normalize_id(provider_id)
        if as_of_date is not None:
            schedule = self.get_schedule(provider_id=norm_id, as_of_date=as_of_date)
            if schedule is None:
                return False
            return bool(schedule.is_verified)

        meta = self._providers.get(norm_id)
        schedules = self._schedules.get(norm_id, [])
        return bool(meta and meta.is_verified and any(s.is_verified for s in schedules))

    def get_tariff_status(
        self,
        provider_id: Optional[str],
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate precise tariff status distinguishing:
        1. Provider identification confirmed
        2. Historical tariff schedule known
        3. Tariff schedule currently applicable to the forecast period
        4. Monetary estimate available or withheld
        """
        norm_id = self._normalize_id(provider_id)
        if not norm_id:
            return {
                "provider_id": None,
                "provider_name": "Unassigned",
                "is_provider_known": False,
                "has_historical_verified_schedule": False,
                "is_applicable_schedule_verified": False,
                "status": "no_provider",
                "status_label": "No provider selected",
                "schedule": None,
                "reason": "No electricity distribution company was specified.",
            }

        meta = self._providers.get(norm_id)
        schedules = self._schedules.get(norm_id, [])
        has_historical_verified = bool(meta and meta.is_verified and any(s.is_verified for s in schedules))
        applicable_schedule = self.get_schedule(provider_id=norm_id, as_of_date=as_of_date)
        is_applicable_verified = bool(applicable_schedule and applicable_schedule.is_verified)

        prov_name = meta.provider_name if meta else (provider_id or "Unassigned Provider")

        if is_applicable_verified:
            return {
                "provider_id": norm_id,
                "provider_name": prov_name,
                "is_provider_known": True,
                "has_historical_verified_schedule": True,
                "is_applicable_schedule_verified": True,
                "status": "verified",
                "status_label": "Verified",
                "schedule": applicable_schedule,
                "reason": f"Official verified tariff schedule is active for {prov_name}.",
            }

        if has_historical_verified:
            return {
                "provider_id": norm_id,
                "provider_name": prov_name,
                "is_provider_known": True,
                "has_historical_verified_schedule": True,
                "is_applicable_schedule_verified": False,
                "status": "expired_schedule",
                "status_label": "Expired schedule",
                "schedule": None,
                "reason": f"Official tariff schedule for {prov_name} expired on 2025-03-31 and is not verified for the forecast period.",
            }

        if meta is not None:
            return {
                "provider_id": norm_id,
                "provider_name": prov_name,
                "is_provider_known": True,
                "has_historical_verified_schedule": False,
                "is_applicable_schedule_verified": False,
                "status": "unverified_schedule",
                "status_label": "Current tariff unverified",
                "schedule": None,
                "reason": f"Official tariff rate schedules for '{prov_name}' are currently unverified.",
            }

        return {
            "provider_id": norm_id,
            "provider_name": prov_name,
            "is_provider_known": False,
            "has_historical_verified_schedule": False,
            "is_applicable_schedule_verified": False,
            "status": "unverified_schedule",
            "status_label": "Current tariff unverified",
            "schedule": None,
            "reason": f"Provider '{provider_id}' is not configured in the tariff registry.",
        }

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers with current support status."""
        result = []
        for norm_id, meta in sorted(self._providers.items()):
            schedules = self._schedules.get(norm_id, [])
            has_verified = bool(meta.is_verified and any(s.is_verified for s in schedules))
            result.append({
                "provider_id": meta.provider_id,
                "provider_name": meta.provider_name,
                "state": meta.state,
                "regulatory_commission": meta.regulatory_commission,
                "category_code": meta.category_code,
                "billing_frequency": meta.billing_frequency.value,
                "is_verified": has_verified,
                "official_citation": meta.official_citation,
            })
        return result

    def build_fallback_bill_estimate(
        self,
        forecast_kwh: float,
        provider_id: Optional[str] = None,
        billing_period_start: Optional[date] = None,
        billing_period_end: Optional[date] = None,
        custom_disclosure: Optional[str] = None,
        tariff_status: Optional[str] = None,
        tariff_status_label: Optional[str] = None,
    ) -> BillEstimateResponse:
        """
        Construct a transparent fallback bill response for unsupported or unverified providers.

        Guarantees:
        - The forecasted electricity volume (forecast_kwh) is 100% preserved.
        - estimated_bill_amount is explicitly set to None.
        - charge_breakdown is explicitly set to None.
        - Adds explicit disclosure explaining why a monetary calculation was withheld.
        """
        clean_kwh = max(0.0, round(float(forecast_kwh), 2))
        norm_id = self._normalize_id(provider_id)
        meta = self._providers.get(norm_id)

        prov_name = meta.provider_name if meta else (provider_id or "Unassigned Provider")
        state_name = meta.state if meta else None

        if tariff_status is None or tariff_status_label is None:
            derived_status = self.get_tariff_status(provider_id=provider_id, as_of_date=billing_period_start)
            tariff_status = derived_status["status"]
            tariff_status_label = derived_status["status_label"]

        disclosures = []
        if custom_disclosure:
            disclosures.append(custom_disclosure)
        elif not provider_id:
            disclosures.append(
                "No electricity distribution company was specified. Physical consumption forecast "
                f"({clean_kwh} kWh) is preserved. Monetary bill estimate cannot be computed without a provider."
            )
        elif tariff_status == "expired_schedule":
            disclosures.append(
                f"No verified tariff schedule is active for '{prov_name}' as of {billing_period_start or 'the requested date'}. "
                "The official tariff schedule expired on 2025-03-31 and is not verified for the forecast period. "
                f"Physical consumption forecast ({clean_kwh} kWh) is preserved. Monetary bill estimate is withheld."
            )
        elif meta and not meta.is_verified:
            disclosures.append(
                f"Official tariff rate schedules for '{prov_name}' ({meta.state}) are currently unverified. "
                f"Physical consumption forecast ({clean_kwh} kWh) is preserved. No monetary bill estimate "
                "was fabricated."
            )
        else:
            disclosures.append(
                f"Provider '{provider_id}' is not currently configured in the tariff registry. "
                f"Physical consumption forecast ({clean_kwh} kWh) is preserved. Estimated bill amount is withheld."
            )

        return BillEstimateResponse(
            provider_id=provider_id,
            provider_name=prov_name,
            state=state_name,
            is_tariff_supported=False,
            is_tariff_verified=False,
            tariff_status=tariff_status,
            tariff_status_label=tariff_status_label,
            forecast_kwh=clean_kwh,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            estimated_bill_amount=None,
            charge_breakdown=None,
            currency="INR",
            disclosures=disclosures,
        )

    def estimate_bill(
        self,
        forecast_kwh: float,
        provider_id: Optional[str] = None,
        as_of_date: Optional[date] = None,
        billing_period_start: Optional[date] = None,
        billing_period_end: Optional[date] = None,
        connection_params: Optional[Dict[str, Any]] = None,
    ) -> BillEstimateResponse:
        """
        Entrypoint for bill estimation.

        Evaluates tariff applicability against the actual forecast date, distinguishing:
        - Verified applicable schedule (computes official bill)
        - Expired schedule (withholds bill, emits 'Expired schedule' status)
        - Unverified provider schedule (withholds bill, emits 'Current tariff unverified' status)
        - No provider (withholds bill, emits 'No provider selected' status)
        """
        clean_kwh = max(0.0, round(float(forecast_kwh), 2))
        query_date = as_of_date or billing_period_start
        status_info = self.get_tariff_status(provider_id=provider_id, as_of_date=query_date)
        schedule = status_info["schedule"]

        # Safe fallback if schedule does not exist or is not verified
        if schedule is None or not schedule.is_verified:
            custom_disc = None
            if status_info["status"] == "expired_schedule":
                eff_date_str = str(billing_period_start or as_of_date or "specified date")
                custom_disc = (
                    f"No verified tariff schedule is active for '{status_info['provider_name']}' as of {eff_date_str}. "
                    "The official tariff schedule expired on 2025-03-31 and is not verified for the forecast period. "
                    f"Physical consumption forecast ({clean_kwh} kWh) is preserved. Monetary bill estimate is withheld."
                )
            return self.build_fallback_bill_estimate(
                forecast_kwh=clean_kwh,
                provider_id=provider_id,
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
                custom_disclosure=custom_disc,
                tariff_status=status_info["status"],
                tariff_status_label=status_info["status_label"],
            )

        norm_id = self._normalize_id(provider_id)
        if norm_id == "msedcl" and schedule.is_verified:
            from backend.app.services.msedcl_tariff_service import msedcl_calculator

            params = connection_params or {}
            res = msedcl_calculator.calculate(
                consumption_kwh=clean_kwh,
                phase_type=params.get("phase_type"),
                cycle_days=params.get("cycle_days"),
                fac_rate_per_kwh=params.get("fac_rate_per_kwh"),
                sanctioned_load_kw=params.get("sanctioned_load_kw"),
                billing_period_start=billing_period_start,
                billing_period_end=billing_period_end,
            )
            res.tariff_status = "verified"
            res.tariff_status_label = "Verified"
            return res

        # Other verified providers will be wired as they are verified in subsequent tasks
        return self.build_fallback_bill_estimate(
            forecast_kwh=clean_kwh,
            provider_id=provider_id,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            tariff_status=status_info["status"],
            tariff_status_label=status_info["status_label"],
        )


# Global singleton tariff registry
tariff_registry = TariffRegistry()
