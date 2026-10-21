"""
Pydantic schemas for monthly household electricity consumption forecasting.

Provides structured request and response models for:
- Inputting confirmed household monthly consumption records.
- Configurable forecast horizons (1, 3, 6, up to 12 months).
- Conservative history-dependent method selection.
- Defensible uncertainty bounds and honest reporting.
- Chronological backtesting metrics (MAE, RMSE, sMAPE).
- Decoupled, optional tariff bill estimates with regulatory disclosures.
"""

import hashlib
import json
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.tariffs import BillEstimateResponse


class MonthlyForecastMethod(str, Enum):
    """Forecasting method selected based on data availability and backtesting."""

    BASELINE_ONLY = "baseline_only"
    WEIGHTED_MOVING_AVERAGE = "weighted_moving_average"
    DAMPED_TREND_MOVING_AVERAGE = "damped_trend_moving_average"
    SEASONAL_NAIVE = "seasonal_naive"
    HOLT_WINTERS = "holt_winters"


class ConfidenceRating(str, Enum):
    """Categorical confidence level reflecting available historical depth."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class MonthlyForecastRequest(BaseModel):
    """
    Request payload to execute a monthly consumption forecast.
    """

    model_config = {"extra": "forbid"}

    records: List[MonthlyConsumptionRecord] = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Chronological or un-ordered confirmed monthly consumption records. Must contain at least one valid record (up to 120 cycles).",
    )
    horizon_months: int = Field(
        default=3,
        ge=1,
        le=12,
        description="Number of future billing cycles to project (1, 3, 6, up to 12). Default is 3.",
    )
    provider_id: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Optional utility provider ID (e.g. 'msedcl', 'bescom') for optional decoupled tariff bill estimation.",
    )
    sanctioned_load_kw: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1000.0,
        description="Optional contract / sanctioned load in kW for tariff fixed-charge calculations.",
    )
    phase_type: Optional[str] = Field(
        default="single_phase",
        max_length=32,
        description="Optional electrical phase connection ('single_phase' or 'three_phase').",
    )
    is_live: bool = Field(
        default=True,
        description=(
            "True if this is a live household forecast for upcoming calendar cycles; "
            "False for offline/historical backtests projecting immediately after the last record."
        ),
    )
    reference_date: Optional[date] = Field(
        default=None,
        description=(
            "Optional reference calendar anchor date. In live mode, future forecast periods "
            "are anchored relative to this date (defaults to today's date if omitted)."
        ),
    )


class MonthlyForecastStep(BaseModel):
    """
    Individual projected monthly billing cycle.
    """

    model_config = {"extra": "forbid"}

    month_index: int = Field(
        ...,
        ge=1,
        le=24,
        description="Sequential 1-based index of the forecast step (1 = next billing cycle).",
    )
    month_label: str = Field(
        ...,
        max_length=64,
        description="Human-readable label for the forecast period (e.g., 'May 2025').",
    )
    period_start: date = Field(
        ...,
        description="Projected start date of the billing period.",
    )
    period_end: date = Field(
        ...,
        description="Projected end date of the billing period.",
    )
    forecast_kwh: float = Field(
        ...,
        ge=0.0,
        le=10_000_000.0,
        description="Projected electricity consumption in kilowatt-hours (kWh).",
    )
    uncertainty_lower_kwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10_000_000.0,
        description="Lower uncertainty bound in kWh (None if history is insufficient to compute defensible interval).",
    )
    uncertainty_upper_kwh: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10_000_000.0,
        description="Upper uncertainty bound in kWh (None if history is insufficient to compute defensible interval).",
    )
    bill_estimate: Optional[BillEstimateResponse] = Field(
        default=None,
        description="Optional decoupled tariff bill estimate. Set to None if tariff is unsupported or unverified.",
    )
    estimated_bill_amount: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100_000_000.0,
        description="Estimated electricity bill amount in INR for this cycle. Explicitly None if tariff is unverified, unsupported, or withheld.",
    )
    tariff_status: Optional[str] = Field(
        default=None,
        description="Applicable tariff status for this cycle: 'verified', 'expired_schedule', 'unverified_schedule', or 'no_provider'.",
    )
    tariff_status_label: Optional[str] = Field(
        default=None,
        description="Human-readable tariff status badge label (e.g. 'Verified', 'Expired schedule', 'Current tariff unverified').",
    )


class BacktestMetrics(BaseModel):
    """
    Out-of-sample chronological backtesting evaluation metrics.
    """

    model_config = {"extra": "forbid"}

    backtest_periods: int = Field(
        ...,
        ge=1,
        description="Number of chronological historical periods evaluated.",
    )
    mae_kwh: float = Field(
        ...,
        ge=0.0,
        description="Mean Absolute Error in kWh on backtest partition.",
    )
    rmse_kwh: float = Field(
        ...,
        ge=0.0,
        description="Root Mean Squared Error in kWh on backtest partition.",
    )
    baseline_mae_kwh: float = Field(
        ...,
        ge=0.0,
        description="Benchmark Naive Persistence Mean Absolute Error in kWh.",
    )
    smape_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=200.0,
        description="Symmetric Mean Absolute Percentage Error (0% to 200%). Bounded to avoid division-by-zero.",
    )


class MonthlyForecastSummary(BaseModel):
    """
    Aggregate metrics summarizing the forecast horizon.
    """

    model_config = {"extra": "forbid"}

    total_forecast_kwh: float = Field(
        ...,
        ge=0.0,
        le=120_000_000.0,
        description="Cumulative projected consumption over the entire horizon (kWh).",
    )
    average_monthly_kwh: float = Field(
        ...,
        ge=0.0,
        le=10_000_000.0,
        description="Average monthly consumption projected across the horizon (kWh).",
    )
    total_estimated_bill_inr: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1_200_000_000.0,
        description="Total estimated bill in INR. Computed only when all months in the horizon have verified tariff estimates; None otherwise.",
    )
    months_with_bill_estimates: int = Field(
        default=0,
        ge=0,
        le=24,
        description="Number of forecast months with valid, verified monetary bill estimates.",
    )
    bill_estimate_notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Explanatory disclosure regarding whether cumulative monetary totals are complete, partial, or withheld.",
    )
    currency: str = Field(
        default="INR",
        max_length=10,
        description="Currency code for monetary amounts.",
    )


class MonthlyForecastResponse(BaseModel):
    """
    Complete response payload for monthly electricity consumption forecasting.
    """

    model_config = {"extra": "forbid"}

    forecast_id: str = Field(
        ...,
        max_length=64,
        description="Unique identifier for this forecast run.",
    )
    created_at: str = Field(
        ...,
        max_length=64,
        description="ISO 8601 UTC timestamp of forecast execution.",
    )
    history_length_months: int = Field(
        ...,
        ge=1,
        le=120,
        description="Count of valid confirmed monthly billing records utilized.",
    )
    last_observed_period: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Label or dates of the most recent confirmed billing cycle.",
    )
    last_observed_end_date: Optional[date] = Field(
        default=None,
        description="End date of the most recent confirmed billing cycle.",
    )
    selected_method: MonthlyForecastMethod = Field(
        ...,
        description="Algorithm selected based on history depth and backtesting.",
    )
    method_description: str = Field(
        ...,
        max_length=500,
        description="Clear explanation of why this forecasting method was chosen.",
    )
    confidence_rating: ConfidenceRating = Field(
        ...,
        description="Categorical confidence assessment reflecting available historical depth.",
    )
    is_stale_history: bool = Field(
        default=False,
        description="True if the most recent billing observation is older than 90 days from the current date.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        max_length=50,
        description="Informational notices regarding gaps, history length, or staleness.",
    )
    limitations: List[str] = Field(
        default_factory=list,
        max_length=50,
        description="Explicit caveats regarding missing intervals, seasonal unobservability, or tariff status.",
    )
    forecast_steps: List[MonthlyForecastStep] = Field(
        ...,
        max_length=24,
        description="Chronological step-by-step monthly forecasts.",
    )
    backtest_metrics: Optional[BacktestMetrics] = Field(
        default=None,
        description="Historical backtest accuracy metrics if sufficient observations were available.",
    )
    summary: MonthlyForecastSummary = Field(
        ...,
        description="Cumulative and average summary metrics across the horizon.",
    )


# -----------------------------------------------------------------------------
# Persistent Monthly Forecast Record Schemas (Task 2.4)
# -----------------------------------------------------------------------------
class MonthlyForecastRecordSummary(BaseModel):
    """Summary of a saved monthly forecast run for listing and history tables."""

    run_id: str = Field(..., max_length=64, description="Unique monthly forecast run identifier (e.g. 'mfc_...')")
    created_at: str = Field(..., max_length=64, description="UTC ISO timestamp of run creation")
    owner_user_id: str = Field(..., max_length=64, description="Authenticated owner user ID")
    horizon_months: int = Field(..., ge=1, le=12, description="Forecast horizon length in months (1, 3, 6, etc.)")
    provider_id: Optional[str] = Field(default=None, max_length=32, description="Utility provider ID if specified")
    phase_type: Optional[str] = Field(default="single_phase", max_length=32, description="Connection phase type")
    sanctioned_load_kw: Optional[float] = Field(default=None, ge=0.0, le=1000.0, description="Sanctioned load in kW")
    is_live: bool = Field(default=True, description="True for live forecasts, False for historical")
    selected_method: MonthlyForecastMethod = Field(..., description="Forecasting method used")
    confidence_rating: ConfidenceRating = Field(..., description="Confidence assessment rating")
    history_length_months: int = Field(..., ge=1, le=120, description="Number of historical billing cycles supplied")
    last_observed_period: Optional[str] = Field(default=None, max_length=128, description="Last confirmed period label")
    total_forecast_kwh: float = Field(..., ge=0.0, le=120_000_000.0, description="Total cumulative kWh across horizon")
    average_monthly_kwh: float = Field(..., ge=0.0, le=10_000_000.0, description="Average monthly kWh")
    total_estimated_bill_inr: Optional[float] = Field(default=None, ge=0.0, le=1_200_000_000.0, description="Total estimated bill in INR if available")
    months_with_bill_estimates: int = Field(default=0, ge=0, le=24, description="Count of months with bill estimates")
    bill_estimate_notes: Optional[str] = Field(default=None, max_length=500, description="Tariff estimate status notes")
    forecast_period_range: Optional[str] = Field(default=None, max_length=128, description="Display period range, e.g. 'Nov 2026 – Jan 2027'")
    warnings_count: int = Field(default=0, ge=0, le=100, description="Number of warnings/disclosures attached")


class MonthlyForecastRecordDetail(MonthlyForecastRecordSummary):
    """Full detail of a saved monthly forecast run, including predictions, warnings, and input snapshot."""

    method_description: str = Field(..., max_length=500, description="Description of the selected method")
    last_observed_end_date: Optional[date] = Field(default=None, description="End date of the most recent confirmed billing cycle")
    is_stale_history: bool = Field(default=False, description="Whether historical data is stale")
    warnings: List[str] = Field(default_factory=list, max_length=50, description="All warnings and notices")
    limitations: List[str] = Field(default_factory=list, max_length=50, description="Method and data limitations")
    forecast_steps: List[MonthlyForecastStep] = Field(..., max_length=24, description="Chronological monthly projection steps")
    summary: MonthlyForecastSummary = Field(..., description="Horizon summary metrics")
    backtest_metrics: Optional[BacktestMetrics] = Field(default=None, description="Backtesting metrics if computed")
    input_records: List[MonthlyConsumptionRecord] = Field(..., max_length=120, description="Immutable snapshot of input billing records")


class MonthlyForecastRecordListResponse(BaseModel):
    """Paginated list of saved monthly forecast records."""

    total_count: int = Field(..., ge=0, description="Total count of matching records")
    limit: int = Field(..., ge=1, le=100, description="Page size limit")
    offset: int = Field(..., ge=0, description="Pagination offset")
    records: List[MonthlyForecastRecordSummary] = Field(..., description="List of saved monthly forecast summaries")


class SaveMonthlyForecastRequest(BaseModel):
    """Request payload to save a completed monthly forecast run with server-authoritative results."""

    model_config = {"extra": "forbid"}

    request: MonthlyForecastRequest = Field(..., description="Input configuration and billing records")
    forecast_id: Optional[str] = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]{4,64}$",
        description="Optional forecast run identifier to correlate ID and ensure idempotency",
    )


def compute_monthly_input_fingerprint(request: MonthlyForecastRequest) -> str:
    """
    Compute a deterministic SHA-256 fingerprint of the complete validated input.

    Canonicalizes billing records (sorted by start and end dates) and all forecast
    generation parameters (horizon, provider, load, phase, live mode, reference date)
    so equivalent validated inputs produce the exact same stable fingerprint.
    """
    sorted_records = sorted(
        request.records,
        key=lambda r: (r.billing_period_start.isoformat(), r.billing_period_end.isoformat()),
    )
    canonical_records = [
        {
            "billing_period_start": r.billing_period_start.isoformat(),
            "billing_period_end": r.billing_period_end.isoformat(),
            "consumption_kwh": round(float(r.consumption_kwh), 4),
            "bill_amount": round(float(r.bill_amount), 4) if r.bill_amount is not None else None,
            "period_label": r.period_label.strip() if (r.period_label and r.period_label.strip()) else None,
        }
        for r in sorted_records
    ]
    canonical_data = {
        "records": canonical_records,
        "horizon_months": int(request.horizon_months),
        "provider_id": (request.provider_id.strip().lower() if request.provider_id and request.provider_id.strip() else None),
        "sanctioned_load_kw": (
            round(float(request.sanctioned_load_kw), 4)
            if request.sanctioned_load_kw is not None
            else None
        ),
        "phase_type": (request.phase_type.strip().lower() if request.phase_type and request.phase_type.strip() else None),
        "is_live": bool(request.is_live),
        "reference_date": request.reference_date.isoformat() if request.reference_date else None,
    }
    canonical_bytes = json.dumps(canonical_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()

