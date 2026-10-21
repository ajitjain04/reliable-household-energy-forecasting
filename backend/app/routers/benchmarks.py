"""Benchmarks router for clean research model comparisons."""

from fastapi import APIRouter
from backend.app.schemas.benchmarks import CleanBenchmarkResponse
from backend.app.services.artifact_service import artifact_service

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])


@router.get(
    "/clean",
    response_model=CleanBenchmarkResponse,
    summary="Get Clean Benchmark Comparisons",
    description=(
        "Retrieve out-of-sample test and validation performance metrics across all four "
        "evaluated architectures (Energy-Only Baseline, Early Fusion, Late Fusion, Hybrid Fusion) "
        "under nominal clean operating conditions."
    ),
)
def get_clean_benchmarks() -> CleanBenchmarkResponse:
    """Return verified clean benchmark performance metrics."""
    return artifact_service.get_clean_benchmarks()
