"""Pydantic schemas for persistent forecast run records and metadata."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.forecasts import ConformalIntervalStep


class ForecastRecordSummary(BaseModel):
    """Summary of a stored forecast run record for listings and console overviews."""

    run_id: str = Field(..., description="Unique forecast run identifier")
    created_at: str = Field(..., description="UTC timestamp of run creation")
    status: str = Field(..., description="Execution status: pending, succeeded, or failed")
    input_method: str = Field(..., description="Ingestion method: 'api' or 'csv'")
    user_id: Optional[str] = Field(default=None, description="Pseudonymous user ID if supplied")
    owner_user_id: Optional[str] = Field(default=None, description="Authenticated owner user ID enforcing record authorization")
    selected_model: Optional[str] = Field(default=None, description="Selected model display name")
    model_key: Optional[str] = Field(default=None, description="Internal model key")
    selection_mode: Optional[str] = Field(default=None, description="Selection mode: 'auto' or 'override'")
    selection_reason: Optional[str] = Field(default=None, description="Detailed rationale for model selection")
    forecast_origin_timestamp: Optional[str] = Field(default=None, description="Last historical observation timestamp")
    mean_interval_width_watts: Optional[float] = Field(default=None, description="Average conformal interval width in Watts")
    error_code: Optional[str] = Field(default=None, description="Sanitized error code if run failed")
    error_message: Optional[str] = Field(default=None, description="Sanitized error description if run failed")


class ForecastRecordDetail(ForecastRecordSummary):
    """Comprehensive details of a stored forecast run, including predictions and intervals."""

    predictions_watts: Optional[List[float]] = Field(default=None, description="24 predicted active power values in Watts")
    intervals: Optional[List[ConformalIntervalStep]] = Field(default=None, description="24 conformal prediction intervals")
    input_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Sanitized summary of input dimensions and features (no raw data)")
    model_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Model parameter count, architecture, and scaler info")


class ForecastRecordListResponse(BaseModel):
    """Paginated list of persistent forecast run records."""

    total_count: int = Field(..., description="Total matching records count")
    limit: int = Field(..., description="Page size limit")
    offset: int = Field(..., description="Pagination offset")
    records: List[ForecastRecordSummary] = Field(..., description="Page of forecast summaries")


class ForecastStatusResponse(BaseModel):
    """Lightweight run status response."""

    run_id: str = Field(..., description="Unique forecast run identifier")
    created_at: str = Field(..., description="UTC timestamp of run creation")
    status: str = Field(..., description="Execution status: pending, succeeded, or failed")
    input_method: str = Field(..., description="Ingestion method: 'api' or 'csv'")
    selected_model: Optional[str] = Field(default=None, description="Selected model if succeeded")
    error_code: Optional[str] = Field(default=None, description="Error code if failed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
