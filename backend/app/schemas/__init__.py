"""Pydantic data schemas for API requests and responses."""

from backend.app.schemas.bills import (
    BillExtractionSource,
    ExtractedBillData,
    MonthlyConsumptionRecord,
    UserCorrectedBillData,
    validate_and_sort_billing_records,
)
from backend.app.schemas.tariffs import (
    BillEstimateResponse,
    BillingFrequency,
    DutyType,
    EnergySlab,
    FixedChargeType,
    TariffSchedule,
    TariffScheduleMetadata,
)
from backend.app.schemas.monthly_forecast import (
    BacktestMetrics,
    ConfidenceRating,
    MonthlyForecastMethod,
    MonthlyForecastRequest,
    MonthlyForecastResponse,
    MonthlyForecastStep,
    MonthlyForecastSummary,
)

from backend.app.schemas.coach import (
    ApplianceCategory,
    ApplianceSimulationInput,
    ApplianceScenarioRequest,
    ApplianceScenarioResponse,
    CoachInsight,
    CoachRecommendation,
    EnergyCoachOverviewResponse,
    DEFAULT_APPLIANCE_PRESETS,
    ScenarioComparisonStep,
    HouseholdScenarioComparisonRequest,
    HouseholdScenarioComparisonResponse,
)

__all__ = [
    "BillExtractionSource",
    "ExtractedBillData",
    "MonthlyConsumptionRecord",
    "UserCorrectedBillData",
    "validate_and_sort_billing_records",
    "BillEstimateResponse",
    "BillingFrequency",
    "DutyType",
    "EnergySlab",
    "FixedChargeType",
    "TariffSchedule",
    "TariffScheduleMetadata",
    "BacktestMetrics",
    "ConfidenceRating",
    "MonthlyForecastMethod",
    "MonthlyForecastRequest",
    "MonthlyForecastResponse",
    "MonthlyForecastStep",
    "MonthlyForecastSummary",
    "ApplianceCategory",
    "ApplianceSimulationInput",
    "ApplianceScenarioRequest",
    "ApplianceScenarioResponse",
    "CoachInsight",
    "CoachRecommendation",
    "EnergyCoachOverviewResponse",
    "DEFAULT_APPLIANCE_PRESETS",
    "ScenarioComparisonStep",
    "HouseholdScenarioComparisonRequest",
    "HouseholdScenarioComparisonResponse",
]

