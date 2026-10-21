"""
Pydantic schemas for pan-India electricity tariff models, charge specifications, and bill estimates.

Design principles:
- Provider-independent: supports varied state structures, billing frequencies, fixed-charge types,
  and statutory duty models across Indian distribution companies.
- Strictly decoupled: physical energy forecasting (kWh) is preserved even when tariff calculation
  is unavailable or unverified.
- No fabricated bills: if tariff rules are missing or unverified, estimated monetary amounts are
  withheld (None) with explicit transparent disclosures.
- Effective-dated: supports regulatory tariff revisions over time.
"""

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class BillingFrequency(str, Enum):
    """Billing interval cadence for electricity accounts."""

    MONTHLY = "monthly"        # 30-day standard cycle (e.g. MSEDCL, BESCOM, JVVNL Urban)
    BI_MONTHLY = "bi_monthly"  # 60-day cycle (e.g. TANGEDCO / TNPDCL, JVVNL Rural)
    CUSTOM = "custom"


class FixedChargeType(str, Enum):
    """Method by which standing or demand fixed charges are assessed."""

    PHASE_BASED = "phase_based"               # Flat rate per phase type (e.g. 1-phase vs 3-phase)
    PER_KW_SANCTIONED = "per_kw_sanctioned"   # Linear charge per kW of sanctioned/contract load
    SLAB_TIERED = "slab_tiered"               # Fixed fee determined by total consumption bracket
    FLAT_MONTHLY = "flat_monthly"             # Fixed fee uniform for all domestic accounts
    ZERO = "zero"                             # No fixed charge levied for category (e.g. TANGEDCO LT-IA)
    CUSTOM = "custom"


class DutyType(str, Enum):
    """Statutory electricity duty or municipal cess assessment structure."""

    PERCENTAGE = "percentage"        # Assessed as a percentage of taxable charges (e.g. 16% in Maharashtra)
    FLAT_PER_UNIT = "flat_per_unit"  # Assessed as a fixed currency rate per kWh consumed (e.g. ₹0.40/kWh)
    EXEMPT = "exempt"                # Category is exempt from electricity duty
    CUSTOM = "custom"


class EnergySlab(BaseModel):
    """
    A single consumption tier within an energy charge schedule.

    For bounded tiers, consumption within [min_units, max_units] is charged at rate_per_kwh.
    For unbounded top tiers, max_units is None.
    """

    model_config = {"extra": "forbid"}

    slab_index: int = Field(..., ge=1, description="Sequential 1-based index of the slab tier.")
    min_units: float = Field(..., ge=0.0, description="Lower unit threshold of this tier (inclusive).")
    max_units: Optional[float] = Field(
        default=None,
        description="Upper unit threshold of this tier (exclusive or inclusive). None signifies no upper ceiling.",
    )
    rate_per_kwh: float = Field(..., ge=0.0, description="Base energy charge rate per kWh in Indian Rupees.")

    @model_validator(mode="after")
    def validate_bounds(self) -> "EnergySlab":
        if self.max_units is not None and self.min_units >= self.max_units:
            raise ValueError(
                f"Slab {self.slab_index}: min_units ({self.min_units}) must be strictly less than "
                f"max_units ({self.max_units})."
            )
        return self


class TariffScheduleMetadata(BaseModel):
    """
    Descriptive and regulatory provenance metadata for an effective-dated tariff schedule.
    """

    model_config = {"extra": "forbid"}

    provider_id: str = Field(
        ...,
        description="Unique normalized provider slug (e.g. 'msedcl', 'bescom', 'jvvnl', 'tangedco').",
    )
    provider_name: str = Field(..., description="Official legal name of the distribution company.")
    state: str = Field(..., description="Indian state or union territory.")
    regulatory_commission: str = Field(
        ...,
        description="State Electricity Regulatory Commission overseeing this tariff (e.g. 'MERC', 'KERC').",
    )
    category_code: str = Field(
        ...,
        description="Consumer tariff category code (e.g. 'LT-1(B)', 'LT-1', 'Tariff-IA').",
    )
    effective_from: date = Field(..., description="Date from which this schedule is legally effective.")
    effective_to: Optional[date] = Field(
        default=None,
        description="Date until which this schedule was legally effective. None indicates currently active.",
    )
    billing_frequency: BillingFrequency = Field(
        default=BillingFrequency.MONTHLY,
        description="Billing cycle frequency.",
    )
    fixed_charge_type: FixedChargeType = Field(
        ...,
        description="Rule used to compute standing fixed charges.",
    )
    duty_type: DutyType = Field(
        ...,
        description="Statutory rule used to calculate electricity duty.",
    )
    is_verified: bool = Field(
        default=False,
        description=(
            "Strict integrity flag: True only if rates and formulas have been verified against "
            "official published regulatory tariff orders."
        ),
    )
    official_citation: Optional[str] = Field(
        default=None,
        description="Official tariff order number, case number, or regulatory gazette citation.",
    )
    verification_notes: Optional[str] = Field(
        default=None,
        description="Verification audit commentary or limitations.",
    )

    @model_validator(mode="after")
    def validate_effective_dates(self) -> "TariffScheduleMetadata":
        if self.effective_to is not None and self.effective_from > self.effective_to:
            raise ValueError(
                f"effective_from ({self.effective_from}) cannot be after effective_to ({self.effective_to})."
            )
        return self


class TariffSchedule(TariffScheduleMetadata):
    """
    Complete tariff specification including energy slabs, fixed rates, network charges, and taxes.
    """

    slabs: List[EnergySlab] = Field(
        default_factory=list,
        description="Ordered list of telescopic consumption slabs.",
    )
    fixed_charge_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Fixed charge determinant map (e.g. {'single_phase': 130.0, 'three_phase': 435.0} or {'per_kw': 150.0}).",
    )
    wheeling_charge_per_kwh: float = Field(
        default=0.0,
        ge=0.0,
        description="Uniform distribution wheeling charge per unit, if applicable.",
    )
    duty_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Electricity duty rate (e.g. 0.16 for 16% or 0.40 for ₹0.40/unit).",
    )
    additional_charges: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional statutory or fuel adjustments (e.g. {'urban_cess': 0.15, 'meter_rent': 20.0}).",
    )

    @model_validator(mode="after")
    def validate_slabs_ordering(self) -> "TariffSchedule":
        """If slabs are specified, ensure they are strictly ordered and contiguous."""
        if not self.slabs:
            return self

        sorted_slabs = sorted(self.slabs, key=lambda s: s.min_units)
        for i in range(len(sorted_slabs) - 1):
            curr_slab = sorted_slabs[i]
            next_slab = sorted_slabs[i + 1]

            if curr_slab.max_units is None:
                raise ValueError(
                    f"Slab {curr_slab.slab_index} is unbounded (max_units=None) but is not the final slab."
                )

            if curr_slab.max_units != next_slab.min_units:
                raise ValueError(
                    f"Non-contiguous slabs: slab {curr_slab.slab_index} ends at {curr_slab.max_units} "
                    f"but slab {next_slab.slab_index} starts at {next_slab.min_units}."
                )

        return self


class BillEstimateResponse(BaseModel):
    """
    Standard response payload for household bill estimation.

    Guarantees:
    - Physical electricity forecast (forecast_kwh) is ALWAYS preserved.
    - If provider is unsupported or unverified, estimated_bill_amount is None with clear disclosures.
    - No synthetic or arbitrary monetary figures are ever generated.
    """

    model_config = {"extra": "forbid"}

    provider_id: Optional[str] = Field(
        default=None,
        description="Provider identifier queried (e.g. 'msedcl', 'bescom') or None if unassigned.",
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Full provider name if known.",
    )
    state: Optional[str] = Field(
        default=None,
        description="State or union territory.",
    )
    is_tariff_supported: bool = Field(
        ...,
        description="True if verified tariff rules are available to compute an official bill estimate.",
    )
    is_tariff_verified: bool = Field(
        default=False,
        description="True if the active tariff schedule is verified against official regulatory orders.",
    )
    tariff_status: Optional[str] = Field(
        default=None,
        description="Tariff applicability status: 'verified', 'expired_schedule', 'unverified_schedule', or 'no_provider'.",
    )
    tariff_status_label: Optional[str] = Field(
        default=None,
        description="Human-readable tariff status label (e.g. 'Verified', 'Expired schedule', 'Current tariff unverified').",
    )
    forecast_kwh: float = Field(
        ...,
        ge=0.0,
        description="Forecasted electricity consumption in kilowatt-hours (kWh). Always preserved.",
    )
    billing_period_start: Optional[date] = Field(
        default=None,
        description="Start date of the estimated billing period if provided.",
    )
    billing_period_end: Optional[date] = Field(
        default=None,
        description="End date of the estimated billing period if provided.",
    )
    estimated_bill_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated bill amount in Indian Rupees. None if tariff is unsupported or unverified.",
    )
    charge_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Itemized charge breakdown (energy, wheeling, fixed, taxes) if calculated.",
    )
    currency: str = Field(
        default="INR",
        description="Currency code for billing amounts.",
    )
    disclosures: List[str] = Field(
        default_factory=list,
        description="Mandatory regulatory disclosures, assumption notices, or missing rate explanations.",
    )
