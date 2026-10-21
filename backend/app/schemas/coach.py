"""
Pydantic schemas for Household Energy Coach and Appliance What-If Simulator.
"""

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.monthly_forecast import MonthlyForecastStep


class ApplianceCategory(str, Enum):
    AIR_CONDITIONER = "air_conditioner"
    REFRIGERATOR = "refrigerator"
    WATER_HEATER = "water_heater"
    WASHING_MACHINE = "washing_machine"
    CEILING_FAN = "ceiling_fan"
    LIGHTING = "lighting"
    TELEVISION = "television"
    OTHER = "other"


# Sensible default presets for appliances based on Indian Bureau of Energy Efficiency (BEE) standards
DEFAULT_APPLIANCE_PRESETS: Dict[str, Dict[str, Any]] = {
    ApplianceCategory.AIR_CONDITIONER.value: {
        "name": "1.5 Ton Split AC (Inverter)",
        "default_power_watts": 1450.0,
        "default_hours_per_day": 8.0,
        "default_days_per_month": 30,
        "default_quantity": 1,
        "duty_cycle_factor": 0.65,  # Thermostat cycles on/off in typical 24-26°C setting
        "notes": "Inverter ACs cycle power dynamically. Average operational power is typically 50-70% of rated capacity once room temperature stabilizes.",
    },
    ApplianceCategory.REFRIGERATOR.value: {
        "name": "Frost-Free Refrigerator (~260L)",
        "default_power_watts": 150.0,
        "default_hours_per_day": 24.0,
        "default_days_per_month": 30,
        "default_quantity": 1,
        "duty_cycle_factor": 0.35,  # Compressor runs ~8-9 hours per day total
        "notes": "Refrigerators stay plugged in 24 hours, but the compressor only cycles on for approximately 30-40% of the day. NEVER disconnect refrigeration for energy savings.",
    },
    ApplianceCategory.WATER_HEATER.value: {
        "name": "Storage Geyser (15-25L)",
        "default_power_watts": 2000.0,
        "default_hours_per_day": 1.5,
        "default_days_per_month": 30,
        "default_quantity": 1,
        "duty_cycle_factor": 1.0,
        "notes": "High continuous resistive draw when heating. Reducing thermostat setting from 60°C to 45-50°C and turning off after bathing yields reliable reductions.",
    },
    ApplianceCategory.WASHING_MACHINE.value: {
        "name": "Fully-Automatic Washing Machine",
        "default_power_watts": 500.0,
        "default_hours_per_day": 1.0,
        "default_days_per_month": 20,
        "default_quantity": 1,
        "duty_cycle_factor": 0.8,
        "notes": "Motor and heating cycles vary. Cold-water cycles significantly reduce energy compared to high-temperature hot cycles.",
    },
    ApplianceCategory.CEILING_FAN.value: {
        "name": "Standard / BLDC Ceiling Fan",
        "default_power_watts": 55.0,
        "default_hours_per_day": 12.0,
        "default_days_per_month": 30,
        "default_quantity": 3,
        "duty_cycle_factor": 1.0,
        "notes": "Standard induction fans consume 70-80W; BLDC fans consume 28-35W. Multi-fan households see compounding savings with BLDC upgrades.",
    },
    ApplianceCategory.LIGHTING.value: {
        "name": "LED Lighting Fixtures",
        "default_power_watts": 12.0,
        "default_hours_per_day": 6.0,
        "default_days_per_month": 30,
        "default_quantity": 6,
        "duty_cycle_factor": 1.0,
        "notes": "LED lights operate near their full rated wattage continuously when switched on.",
    },
    ApplianceCategory.TELEVISION.value: {
        "name": "Smart LED TV (43-55 inch)",
        "default_power_watts": 90.0,
        "default_hours_per_day": 4.0,
        "default_days_per_month": 30,
        "default_quantity": 1,
        "duty_cycle_factor": 1.0,
        "notes": "Operates continuously at rated wattage while on. Standby consumption is negligible (<1W).",
    },
    ApplianceCategory.OTHER.value: {
        "name": "Custom Appliance",
        "default_power_watts": 100.0,
        "default_hours_per_day": 2.0,
        "default_days_per_month": 30,
        "default_quantity": 1,
        "duty_cycle_factor": 1.0,
        "notes": "Custom simulation based on user-entered parameters.",
    },
}


class ApplianceSimulationInput(BaseModel):
    """Input parameters for a single appliance usage scenario."""

    category: ApplianceCategory = Field(
        default=ApplianceCategory.AIR_CONDITIONER,
        description="Category of the household appliance",
    )
    name: str = Field(
        default="Air Conditioner",
        min_length=1,
        max_length=100,
        description="User-friendly name or label for the appliance",
    )
    power_watts: float = Field(
        default=1500.0,
        ge=1.0,
        le=50000.0,
        description="Rated power in Watts (1W to 50,000W)",
    )
    hours_per_day: float = Field(
        default=8.0,
        ge=0.0,
        le=24.0,
        description="Operating hours per day (0 to 24)",
    )
    days_per_month: int = Field(
        default=30,
        ge=0,
        le=31,
        description="Days used per month (0 to 31)",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of identical appliances (1 to 20)",
    )

    @field_validator("power_watts")
    @classmethod
    def validate_power(cls, v: float) -> float:
        if v < 1.0 or v > 50000.0:
            raise ValueError("Rated power must be between 1.0 Watt and 50,000.0 Watts.")
        return round(float(v), 1)

    @field_validator("hours_per_day")
    @classmethod
    def validate_hours(cls, v: float) -> float:
        if v < 0.0 or v > 24.0:
            raise ValueError("Operating hours per day cannot exceed 24.0 hours.")
        return round(float(v), 2)

    @field_validator("days_per_month")
    @classmethod
    def validate_days(cls, v: int) -> int:
        if v < 0 or v > 31:
            raise ValueError("Days used per month must be between 0 and 31.")
        return int(v)

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("Appliance quantity must be between 1 and 20.")
        return int(v)


class ApplianceScenarioRequest(BaseModel):
    """Request payload comparing baseline appliance usage against a modified scenario."""

    baseline: ApplianceSimulationInput = Field(
        ...,
        description="Baseline current appliance configuration",
    )
    modified: ApplianceSimulationInput = Field(
        ...,
        description="Hypothetical modified appliance configuration (e.g. reduced hours or upgraded model)",
    )
    provider_id: Optional[str] = Field(
        default=None,
        description="Distribution utility identifier (e.g. 'msedcl') to estimate monetary bill impact",
    )
    phase_type: Optional[str] = Field(
        default="single_phase",
        description="Electrical phase ('single_phase' or 'three_phase')",
    )
    current_monthly_total_kwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=50000.0,
        description="Current baseline monthly total household kWh to compute accurate telescopic slab savings",
    )
    as_of_date: Optional[date] = Field(
        default=None,
        description="Optional date to verify tariff schedule applicability",
    )


class ApplianceScenarioResponse(BaseModel):
    """Simulation results comparing energy and cost between baseline and modified scenarios."""

    baseline_kwh: float = Field(..., description="Estimated monthly energy for baseline scenario (kWh)")
    modified_kwh: float = Field(..., description="Estimated monthly energy for modified scenario (kWh)")
    delta_kwh: float = Field(
        ...,
        description="Energy difference (baseline_kwh - modified_kwh). Positive indicates energy saved.",
    )
    is_hypothetical_estimate: bool = Field(
        default=True,
        description="True: calculated from rated power and user-entered hours, not sub-metered from household lines",
    )
    is_tariff_verified: bool = Field(
        ...,
        description="Whether a verified official tariff calculation was applied for cost estimates",
    )
    estimated_baseline_cost_inr: Optional[float] = Field(
        default=None,
        description="Estimated incremental monthly cost in INR for baseline scenario (if verified tariff available)",
    )
    estimated_modified_cost_inr: Optional[float] = Field(
        default=None,
        description="Estimated incremental monthly cost in INR for modified scenario (if verified tariff available)",
    )
    estimated_savings_inr: Optional[float] = Field(
        default=None,
        description="Estimated monthly cost reduction in INR. Positive indicates cost saved.",
    )
    disclosures: List[str] = Field(
        default_factory=list,
        description="Methodological notices, tariff verification statements, and estimation caveats",
    )
    appliance_notes: List[str] = Field(
        default_factory=list,
        description="Appliance-specific duty cycle and operational behavior notes",
    )


class CoachInsight(BaseModel):
    """A data-grounded, transparent insight explaining household energy patterns."""

    id: str = Field(..., description="Unique slug for the insight")
    category: str = Field(..., description="Insight category: 'trend', 'baseline', 'forecast', 'data_quality'")
    title: str = Field(..., description="Concise summary header")
    description: str = Field(..., description="Detailed explanation of the observation")
    evidence: str = Field(..., description="Exact historical records, dates, and numbers supporting this insight")
    severity: str = Field(
        default="info",
        description="Visual indicator: 'positive', 'warning', 'info', 'neutral'",
    )


class CoachRecommendation(BaseModel):
    """An actionable, transparent energy-saving recommendation with explicit rationale."""

    id: str = Field(..., description="Unique slug for the recommendation")
    title: str = Field(..., description="Concise action item")
    description: str = Field(..., description="Step-by-step guidance")
    action_type: str = Field(..., description="Action type: 'behavioral', 'appliance', 'data_setup'")
    rationale: str = Field(..., description="Transparent reason why this recommendation is suggested")
    is_hypothetical: bool = Field(
        default=True,
        description="Whether savings are hypothetical rather than measured",
    )
    estimated_monthly_kwh_reduction: Optional[float] = Field(
        default=None,
        description="Estimated potential energy reduction in kWh/month",
    )
    estimated_monthly_savings_inr: Optional[float] = Field(
        default=None,
        description="Estimated monetary savings in INR/month (only if verified tariff applies)",
    )
    savings_disclaimer: Optional[str] = Field(
        default=None,
        description="Explicit limitation explaining assumptions and non-guaranteed nature of savings",
    )


class EnergyCoachOverviewResponse(BaseModel):
    """Comprehensive household energy coaching profile with overview metrics, trends, and advice."""

    has_data: bool = Field(..., description="Whether confirmed historical records exist for this household")
    latest_consumption_kwh: Optional[float] = Field(
        default=None,
        description="Latest recorded monthly consumption in kWh",
    )
    latest_period_label: Optional[str] = Field(
        default=None,
        description="Label for latest billing period (e.g. 'Jun 2024')",
    )
    latest_billing_end_date: Optional[date] = Field(
        default=None,
        description="End date of the latest recorded billing period",
    )
    latest_bill_amount_inr: Optional[float] = Field(
        default=None,
        description="Confirmed bill amount for the latest period (only if extracted/confirmed)",
    )
    previous_consumption_kwh: Optional[float] = Field(
        default=None,
        description="Consumption in preceding monthly period in kWh",
    )
    previous_period_label: Optional[str] = Field(
        default=None,
        description="Label for preceding billing period",
    )
    consumption_change_kwh: Optional[float] = Field(
        default=None,
        description="Absolute change from previous to latest month (latest - previous)",
    )
    consumption_change_percent: Optional[float] = Field(
        default=None,
        description="Percentage change from previous to latest month",
    )
    historical_records: List[MonthlyConsumptionRecord] = Field(
        default_factory=list,
        description="Chronological historical monthly consumption records",
    )
    latest_forecast_run_id: Optional[str] = Field(
        default=None,
        description="Identifier of the latest saved monthly forecast run",
    )
    latest_forecast_created_at: Optional[str] = Field(
        default=None,
        description="Timestamp when the latest forecast run was created",
    )
    latest_forecast_horizon_months: Optional[int] = Field(
        default=None,
        description="Forecast horizon length in months",
    )
    latest_forecast_period_range: Optional[str] = Field(
        default=None,
        description="Text range of forecast months (e.g. 'Jul 2024 – Dec 2024')",
    )
    latest_forecast_total_kwh: Optional[float] = Field(
        default=None,
        description="Total forecasted energy across horizon (kWh)",
    )
    latest_forecast_average_kwh: Optional[float] = Field(
        default=None,
        description="Average forecasted monthly energy (kWh)",
    )
    latest_forecast_confidence: Optional[str] = Field(
        default=None,
        description="Confidence rating of latest forecast ('high', 'medium', 'low', 'very_low')",
    )
    forecast_steps: List[MonthlyForecastStep] = Field(
        default_factory=list,
        description="Monthly forecast steps with predictions and uncertainty bounds",
    )
    provider_id: Optional[str] = Field(
        default=None,
        description="Distribution company identifier (e.g. 'msedcl')",
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Human-readable provider name",
    )
    is_tariff_verified: bool = Field(
        default=False,
        description="Whether the utility tariff is officially verified",
    )
    data_freshness_status: str = Field(
        default="no_records",
        description="Freshness status: 'current', 'stale', 'insufficient_history', 'no_records'",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Data quality warnings and missing-data notes",
    )
    insights: List[CoachInsight] = Field(
        default_factory=list,
        description="Personalized, data-grounded energy insights",
    )
    recommendations: List[CoachRecommendation] = Field(
        default_factory=list,
        description="Actionable, transparent energy-saving recommendations",
    )


class ScenarioComparisonStep(BaseModel):
    """Step-level comparison between baseline monthly forecast and adjusted scenario."""

    month_index: int = Field(..., ge=1, description="1-based forecast month index")
    month_label: str = Field(..., description="Forecast month label e.g. 'Nov 2026'")
    baseline_kwh: float = Field(..., ge=0.0, description="Projected monthly baseline consumption in kWh")
    adjusted_kwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Projected consumption in kWh after scenario adjustments (None if withheld)",
    )
    delta_kwh: Optional[float] = Field(
        default=None,
        description="Estimated energy reduction in kWh (baseline - adjusted, None if withheld)",
    )
    baseline_bill_amount: Optional[float] = Field(
        default=None,
        description="Estimated monthly bill in INR before intervention (verified and active tariffs only)",
    )
    adjusted_bill_amount: Optional[float] = Field(
        default=None,
        description="Estimated monthly bill in INR after intervention (verified and active tariffs only)",
    )
    savings_amount: Optional[float] = Field(
        default=None,
        description="Estimated monetary savings in INR (baseline_bill - adjusted_bill)",
    )
    is_adjustment_supported: bool = Field(
        default=True,
        description="Whether this step's household adjustment is physically plausible and supported",
    )
    unsupported_reason: Optional[str] = Field(
        default=None,
        description="Explanation if household adjustment is withheld for this step",
    )
    tariff_status: Optional[str] = Field(
        default=None,
        description="Applicable tariff status for this step: 'verified', 'expired_schedule', 'unverified_schedule', or 'no_provider'",
    )
    tariff_status_label: Optional[str] = Field(
        default=None,
        description="Human-readable tariff status badge label (e.g. 'Verified', 'Expired schedule', 'Current tariff unverified')",
    )


class HouseholdScenarioComparisonRequest(BaseModel):
    """Request to evaluate a household-level forecast against an appliance what-if scenario."""

    baseline_forecast_kwh: List[float] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Baseline projected monthly consumption series in kWh",
    )
    month_labels: List[str] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="Labels for each forecast step e.g. ['Nov 2026', 'Dec 2026']",
    )
    appliance_scenario: ApplianceScenarioRequest = Field(
        ...,
        description="Appliance baseline vs proposed modification scenario",
    )
    provider_id: Optional[str] = Field(
        default="msedcl",
        description="Electricity provider identifier e.g. 'msedcl'",
    )
    phase_type: Optional[str] = Field(
        default="single_phase",
        description="'single_phase' or 'three_phase'",
    )
    sanctioned_load_kw: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=150.0,
        description="Optional sanctioned load in kW",
    )
    period_starts: Optional[List[date]] = Field(
        default=None,
        description="Optional projected start dates for each billing cycle to verify tariff schedule dates",
    )
    period_ends: Optional[List[date]] = Field(
        default=None,
        description="Optional projected end dates for each billing cycle",
    )

    @field_validator("baseline_forecast_kwh")
    @classmethod
    def validate_baseline(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("At least one monthly forecast step is required.")
        for kwh in v:
            if kwh < 0.0:
                raise ValueError("Projected monthly consumption cannot be negative.")
        return [round(float(k), 2) for k in v]


class HouseholdScenarioComparisonResponse(BaseModel):
    """Comparative response detailing household baseline vs adjusted scenario projections."""

    steps: List[ScenarioComparisonStep] = Field(
        ...,
        description="Step-by-step monthly comparison between baseline forecast and adjusted scenario",
    )
    total_baseline_kwh: float = Field(
        ...,
        description="Total projected baseline consumption across horizon (kWh)",
    )
    total_adjusted_kwh: Optional[float] = Field(
        default=None,
        description="Total projected adjusted consumption across horizon (kWh). Set to None if withheld.",
    )
    total_delta_kwh: Optional[float] = Field(
        default=None,
        description="Total projected energy reduction across horizon (kWh). Set to None if withheld.",
    )
    total_savings_inr: Optional[float] = Field(
        default=None,
        description="Total estimated bill savings in INR across horizon (verified tariffs only)",
    )
    is_verified_tariff: bool = Field(
        default=False,
        description="Whether official verified tariff calculations could be defensibly applied",
    )
    tariff_status: Optional[str] = Field(
        default=None,
        description="Overall tariff applicability status across horizon: 'verified', 'expired_schedule', 'unverified_schedule', or 'no_provider'",
    )
    tariff_status_label: Optional[str] = Field(
        default=None,
        description="Human-readable overall tariff status label across horizon",
    )
    is_household_adjustment_supported: bool = Field(
        default=True,
        description="Whether the appliance scenario can be defensibly reconciled with the household baseline",
    )
    household_adjustment_reason: Optional[str] = Field(
        default=None,
        description="Reason if household scenario adjustment is withheld",
    )
    appliance_result: ApplianceScenarioResponse = Field(
        ...,
        description="Detailed appliance-level simulation results",
    )
    recommendations: List[CoachRecommendation] = Field(
        default_factory=list,
        description="Contextual, evidence-based recommendations for this scenario",
    )
    disclosures: List[str] = Field(
        default_factory=list,
        description="Transparency notices, methodology disclaimers, and tariff notes",
    )

