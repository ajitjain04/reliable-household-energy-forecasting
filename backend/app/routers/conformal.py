"""Conformal router for uncertainty quantification and distribution-free prediction intervals."""

from fastapi import APIRouter, Path, Query
from backend.app.schemas.conformal import (
    ConformalSummaryResponse,
    ConformalWindowIntervalResponse,
)
from backend.app.services.artifact_service import artifact_service

router = APIRouter(prefix="/conformal", tags=["Conformal Prediction"])


@router.get(
    "/summary",
    response_model=ConformalSummaryResponse,
    summary="Get Conformal Calibration & Uncertainty Summary",
    description=(
        "Retrieve horizon-wise absolute residual split conformal prediction metrics (alpha=0.10, "
        "90% nominal coverage) across all four models, including empirical PICP, interval width MPIW, "
        "clean Winkler scores, degraded coverage erosion, and oracle required expansion widths."
    ),
)
def get_conformal_summary() -> ConformalSummaryResponse:
    """Return conformal uncertainty calibration and performance summary."""
    return artifact_service.get_conformal_summary()


@router.get(
    "/intervals/{model}/{condition}",
    response_model=ConformalWindowIntervalResponse,
    summary="Get 90% Conformal Interval Forecast for Test Window",
    description=(
        "Retrieve precomputed 90% prediction intervals [L, U], point forecasts, and ground truth "
        "active load for a specific test window, model, and degradation condition."
    ),
)
def get_conformal_intervals(
    model: str = Path(..., description="Model key: energy_only, early_fusion, late_fusion, hybrid_fusion"),
    condition: str = Path(..., description="Condition key: clean_NN, missingness_NA, missingness_AN, missingness_AA, noise_NA, noise_AN, noise_AA"),
    window_idx: int = Query(0, ge=0, le=586, description="Test window index (0 to 586)"),
) -> ConformalWindowIntervalResponse:
    """Return 90% conformal interval arrays for selected window."""
    return artifact_service.get_conformal_interval(
        model=model, condition=condition, window_idx=window_idx
    )
