"""Pydantic schemas for Explainable AI (XAI) and attribution analysis."""

from typing import List, Optional
from pydantic import BaseModel, Field


class XAISummaryRecord(BaseModel):
    """Clean and degraded modality attribution summary for a model."""

    model: str = Field(..., description="Model architecture name")
    clean_electricity_share_pct: float = Field(..., description="Clean electricity attribution share (%)")
    clean_weather_share_pct: float = Field(..., description="Clean weather attribution share (%)")
    clean_electricity_density_share_pct: float = Field(..., description="Single-channel electricity density share (%)")
    clean_weather_density_share_pct: float = Field(..., description="Single-channel weather density share (%)")
    miss_NA_elec_shift_pct_pts: float = Field(..., description="Shift in electricity attribution when weather is masked (p.p.)")
    miss_AN_elec_shift_pct_pts: float = Field(..., description="Shift in electricity attribution when electricity is masked (p.p.)")
    miss_AA_elec_shift_pct_pts: float = Field(..., description="Shift in electricity attribution under dual missingness (p.p.)")
    noise_AA_elec_shift_pct_pts: float = Field(..., description="Shift in electricity attribution under dual noise (p.p.)")
    electricity_temporal_recent_24h_pct: float = Field(..., description="Electricity attribution in recent 24 hours (%)")
    electricity_temporal_intermediate_48h_pct: float = Field(..., description="Electricity attribution in intermediate 48 hours (%)")
    electricity_temporal_distant_96h_pct: float = Field(..., description="Electricity attribution in distant 96 hours (%)")
    top_weather_features_clean: str = Field(..., description="Top ranking clean weather features")


class XAISummaryResponse(BaseModel):
    """Response schema for XAI modality attribution summary."""

    source_artifact: str = Field(..., description="Underlying verified artifact filename")
    method: str = Field(default="Integrated Gradients (m=20 steps, zero-mean baseline)", description="XAI method")
    records: List[XAISummaryRecord] = Field(..., description="Per-model attribution summary records")


class TemporalLagRecord(BaseModel):
    """Temporal attribution lag breakdown."""

    model: str = Field(..., description="Model name")
    experiment: str = Field(..., description="Experiment: missingness or noise")
    condition: str = Field(..., description="Condition: NN, NA, AN, AA")
    bin_name: str = Field(..., description="Lag bin name: Recent (0-23h), Intermediate (24-71h), Distant (72-167h)")
    description: str = Field(..., description="Bin description")
    n_hours: int = Field(..., description="Number of hours in this temporal bin")
    attribution_magnitude_mean: float = Field(..., description="Mean attribution magnitude")
    temporal_share_pct: float = Field(..., description="Percentage share of attribution within electricity modality")


class WeatherFeatureRecord(BaseModel):
    """Weather channel attribution importance and rank."""

    model: str = Field(..., description="Model name")
    experiment: str = Field(..., description="Experiment: missingness or noise")
    condition: str = Field(..., description="Condition: NN, NA, AN, AA")
    feature_name: str = Field(..., description="Weather variable name")
    attribution_magnitude_mean: float = Field(..., description="Mean attribution magnitude")
    attribution_share_within_weather_pct: float = Field(..., description="Percentage share within weather modality")
    feature_rank: int = Field(..., description="Feature importance rank (1 = most influential)")
