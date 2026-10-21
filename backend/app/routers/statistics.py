"""Statistics router for hypothesis testing battery and inferential evaluations."""

from fastapi import APIRouter, Path
from backend.app.schemas.statistics import FamilyDetailResponse, StatisticalSummaryResponse
from backend.app.services.artifact_service import artifact_service

router = APIRouter(prefix="/statistics", tags=["Statistical Inference"])


@router.get(
    "/summary",
    response_model=StatisticalSummaryResponse,
    summary="Get 78-Test Statistical Hypothesis Overview",
    description=(
        "Retrieve inferential summary across all 6 planned hypothesis families (78 total tests) "
        "evaluating clean model superiority, missingness impact, noise impact, within-model degradation, "
        "and cross-degradation sensitivity under Holm-Bonferroni and Benjamini-Hochberg FDR control."
    ),
)
def get_statistical_summary() -> StatisticalSummaryResponse:
    """Return overview of statistical testing battery across all 6 families."""
    return artifact_service.get_statistical_summary()


@router.get(
    "/family/{family_id}",
    response_model=FamilyDetailResponse,
    summary="Get Detailed Family Hypothesis Results",
    description=(
        "Retrieve exact inferential statistics, mean differences, 95% confidence intervals, "
        "Wilcoxon signed-rank statistics, Holm-adjusted p-values, and Diebold-Mariano statistics "
        "for a specific hypothesis family (IDs 1 through 6)."
    ),
)
def get_family_detail(
    family_id: int = Path(..., description="Hypothesis Family ID (1 to 6)"),
) -> FamilyDetailResponse:
    """Return narrative synthesis and granular test records for a family."""
    return artifact_service.get_statistical_family_detail(family_id)
