"""Robustness router for sensor degradation and operational reliability experiments."""

from typing import Optional
from fastapi import APIRouter, Query
from backend.app.schemas.robustness import RobustnessResponse
from backend.app.services.artifact_service import artifact_service

router = APIRouter(prefix="/robustness", tags=["Robustness"])


@router.get(
    "/matrix",
    response_model=RobustnessResponse,
    summary="Get 2x2 Factorial Robustness Performance Matrix",
    description=(
        "Retrieve model performance across all 28 experimental configurations evaluating "
        "30% MCAR sensor missingness and 20% Additive Gaussian Noise across the 2x2 factorial "
        "modality grid (Clean NN, Weather-Degraded NA, Electricity-Degraded AN, Joint-Degraded AA)."
    ),
)
def get_robustness_matrix(
    experiment: Optional[str] = Query(
        None,
        description="Filter by degradation mechanism: 'missingness_30' or 'noise_20'",
    ),
    model: Optional[str] = Query(
        None,
        description="Filter by model name: 'Energy-Only', 'Early Fusion', 'Late Fusion', 'Hybrid Fusion'",
    ),
    condition: Optional[str] = Query(
        None,
        description="Filter by 2x2 factorial condition code: 'NN', 'NA', 'AN', 'AA'",
    ),
) -> RobustnessResponse:
    """Return filtered master robustness performance metrics."""
    return artifact_service.get_robustness_matrix(
        experiment=experiment, model=model, condition=condition
    )
