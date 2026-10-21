"""Pydantic schemas for multi-step test forecasts, window metadata, and live inference."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Precomputed Research Forecast Schemas (Retained for Step 10 & Explorer)
# -----------------------------------------------------------------------------
class TestWindowsMetadataResponse(BaseModel):
    """Metadata describing the withheld rolling test evaluation partition."""

    total_windows: int = Field(default=587, description="Total rolling evaluation windows")
    history_length_hours: int = Field(default=168, description="Lookback historical context length (7 days)")
    forecast_horizon_hours: int = Field(default=24, description="Forecast horizon length (24 hours)")
    first_forecast_origin: str = Field(..., description="Timestamp of first test forecast origin")
    last_forecast_origin: str = Field(..., description="Timestamp of last test forecast origin")
    first_target_start: str = Field(..., description="Timestamp of first target hour")
    last_target_end: str = Field(..., description="Timestamp of last target hour")
    available_models: List[str] = Field(..., description="Available evaluated model names")
    window_index_range: List[int] = Field(default=[0, 586], description="Valid range of test window indices")


class WindowModelMetrics(BaseModel):
    """Window-level error metrics for a single model."""

    mae_watts: float = Field(..., description="Window MAE in Watts")
    rmse_watts: float = Field(..., description="Window RMSE in Watts")
    max_error_watts: float = Field(..., description="Maximum absolute error across the 24 hours")


class WindowForecastResponse(BaseModel):
    """Forecast rollout comparison across all 4 models for a specific test window."""

    window_idx: int = Field(..., description="Test window index (0 to 586)")
    forecast_origin_timestamp: str = Field(..., description="Forecast origin timestamp")
    horizon_hours: List[int] = Field(..., description="Horizon offsets 1 to 24")
    target_timestamps: List[str] = Field(..., description="Target hour timestamps for t+1 to t+24")
    actual_watts: List[float] = Field(..., description="Ground truth observed active power (Watts)")
    predictions_watts: Dict[str, List[float]] = Field(..., description="Predicted power by model name (Watts)")
    metrics: Dict[str, WindowModelMetrics] = Field(..., description="Window-level evaluation metrics by model")


# -----------------------------------------------------------------------------
# 2. Live Inference Request & Response Schemas
# -----------------------------------------------------------------------------
class WeatherSeriesInput(BaseModel):
    """Column-oriented weather series covering exactly 168 hours."""

    temp: List[float] = Field(..., description="Hourly temperature (°C)")
    feels_like: List[float] = Field(..., description="Hourly perceived temperature (°C)")
    humidity: List[float] = Field(..., description="Hourly relative humidity (%)")
    pressure: List[float] = Field(..., description="Hourly atmospheric pressure (hPa)")
    speed: List[float] = Field(..., description="Hourly wind speed (m/s)")
    wind_sin: Optional[List[float]] = Field(default=None, description="Hourly sin component of wind direction")
    wind_cos: Optional[List[float]] = Field(default=None, description="Hourly cos component of wind direction")
    deg: Optional[List[float]] = Field(default=None, description="Hourly meteorological wind direction in degrees (0-360)")


class LocationInput(BaseModel):
    """Geographic location of household for automatic weather retrieval."""

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees (-90.0 to 90.0)")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees (-180.0 to 180.0)")
    label: Optional[str] = Field(default=None, max_length=128, description="Optional human-readable location or city label")


class ForecastRunRequest(BaseModel):
    """Request payload to generate a 24-hour forecast from user-provided history."""

    electricity_active_power: List[float] = Field(
        ...,
        description="Exactly 168 consecutive hourly active power observations in Watts.",
    )
    timestamps: Optional[List[str]] = Field(
        default=None,
        description="Optional list of 168 ISO timestamps corresponding to the electricity observations.",
    )
    weather: Optional[Union[WeatherSeriesInput, Dict[str, List[float]], List[Dict[str, float]]]] = Field(
        default=None,
        description=(
            "Optional 168-hour weather observations. Can be provided as a structured series, "
            "column dictionary, or row record list. Required for multimodal models."
        ),
    )
    location: Optional[LocationInput] = Field(
        default=None,
        description=(
            "Optional household coordinates to automatically retrieve historical weather observations "
            "when user-supplied weather is omitted."
        ),
    )
    model: Optional[str] = Field(
        default="auto",
        description=(
            "Model selection mode. Default is 'auto' (selects Early Fusion if valid weather is present, "
            "or Energy-Only if weather is absent). Explicit authorized overrides: "
            "'energy_only', 'early_fusion', 'late_fusion', 'hybrid_fusion'."
        ),
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional pseudonymous user identifier for forecast tracking.",
    )


class ConformalIntervalStep(BaseModel):
    """Conformal prediction interval for a single forecast horizon step."""

    horizon_hour: int = Field(..., description="Horizon offset step (1 to 24)")
    target_timestamp: Optional[str] = Field(default=None, description="Target timestamp if input timestamps were supplied")
    prediction_watts: float = Field(..., description="Point forecast in Watts")
    lower_bound_watts: float = Field(..., description="Unclipped lower prediction interval bound in Watts")
    upper_bound_watts: float = Field(..., description="Upper prediction interval bound in Watts")
    lower_bound_clipped_watts: float = Field(..., description="Non-negative clipped lower bound in Watts (max(0, lower))")
    interval_width_watts: float = Field(..., description="Interval width (upper - lower) in Watts")


class ModelMetadataInfo(BaseModel):
    """Metadata regarding the selected model checkpoint and scalers."""

    model_name: str = Field(..., description="Standardized display name")
    model_class: str = Field(..., description="PyTorch model class name")
    trainable_parameters: int = Field(..., description="Exact trainable parameter count")
    history_length_hours: int = Field(default=168, description="History context length in hours")
    forecast_horizon_hours: int = Field(default=24, description="Forecast horizon in hours")
    energy_scaler_type: str = Field(default="StandardScaler (mean=307.17 W, std=147.99 W)")
    weather_scaler_type: Optional[str] = Field(default=None, description="Weather scaler status if used")


class ForecastRunResponse(BaseModel):
    """Response containing live 24-hour predictions and conformal uncertainty intervals."""

    run_id: str = Field(..., description="Unique UUID for this forecast execution")
    selected_model: str = Field(..., description="Selected model display name")
    model_key: str = Field(..., description="Internal model key")
    selection_mode: str = Field(..., description="'auto', 'override', or 'fallback'")
    selection_reason: str = Field(..., description="Detailed explanation for model selection")
    forecast_origin_timestamp: Optional[str] = Field(default=None, description="Timestamp of the final historical input observation")
    horizon_hours: List[int] = Field(..., description="Horizon offsets from 1 to 24")
    forecast_timestamps: Optional[List[str]] = Field(default=None, description="Forecast timestamps for t+1 to t+24")
    predictions_watts: List[float] = Field(..., description="24 predicted active power values in Watts")
    intervals: List[ConformalIntervalStep] = Field(..., description="24 conformal prediction intervals")
    mean_interval_width_watts: float = Field(..., description="Mean prediction interval width across the 24 hours")
    nominal_coverage_pct: float = Field(default=90.0, description="Nominal split conformal coverage level")
    confidence_explanation: str = Field(
        ...,
        description="Explanation of marginal conformal coverage and its limitations for individual households.",
    )
    units: str = Field(default="Watts", description="Physical unit of prediction")
    model_metadata: ModelMetadataInfo = Field(..., description="Technical metadata for the model used")
