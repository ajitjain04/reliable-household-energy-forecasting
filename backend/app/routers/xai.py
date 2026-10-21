"""XAI router for Integrated Gradients modality attribution analysis."""

from typing import List, Optional
from fastapi import APIRouter, Query
from backend.app.schemas.xai import (
    TemporalLagRecord,
    WeatherFeatureRecord,
    XAISummaryResponse,
)
from backend.app.services.artifact_service import artifact_service

router = APIRouter(prefix="/xai", tags=["Explainable AI"])


@router.get(
    "/summary",
    response_model=XAISummaryResponse,
    summary="Get XAI Modality Attribution Summary",
    description=(
        "Retrieve Integrated Gradients attribution shares, single-channel density concentration, "
        "dynamic gradient reallocation shifts under missingness, and temporal memory lag distributions "
        "for all four evaluated models."
    ),
)
def get_xai_summary() -> XAISummaryResponse:
    """Return high-level modality attribution summary."""
    return artifact_service.get_xai_summary()


@router.get(
    "/temporal",
    response_model=List[TemporalLagRecord],
    summary="Get Electricity Temporal Lag Attributions",
    description=(
        "Retrieve electricity attribution partitioned across temporal lag intervals: "
        "Recent (0-23h), Intermediate (24-71h), and Distant (72-167h) lookbacks."
    ),
)
def get_xai_temporal(
    model: Optional[str] = Query(None, description="Filter by model name"),
    condition: Optional[str] = Query(None, description="Filter by condition code (NN, NA, AN, AA)"),
    experiment: Optional[str] = Query(None, description="Filter by experiment (missingness, noise)"),
) -> List[TemporalLagRecord]:
    """Return temporal lag breakdown records."""
    return artifact_service.get_xai_temporal(
        model=model, condition=condition, experiment=experiment
    )


@router.get(
    "/weather-features",
    response_model=List[WeatherFeatureRecord],
    summary="Get Weather Feature Attribution Rankings",
    description=(
        "Retrieve relative importance shares and feature hierarchy rankings for all 7 "
        "exogenous weather channels (temperature, feels_like, humidity, pressure, speed, wind_sin, wind_cos)."
    ),
)
def get_xai_weather_features(
    model: Optional[str] = Query(None, description="Filter by model name"),
    condition: Optional[str] = Query(None, description="Filter by condition code"),
    experiment: Optional[str] = Query(None, description="Filter by experiment (missingness, noise)"),
) -> List[WeatherFeatureRecord]:
    """Return weather feature rankings and importance shares."""
    return artifact_service.get_xai_weather_features(
        model=model, condition=condition, experiment=experiment
    )
