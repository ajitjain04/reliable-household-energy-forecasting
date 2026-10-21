"""Health check router."""

from datetime import datetime, timezone
from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Return service health status and verify research artifact path presence.
    """
    artifact_dirs = {
        "outputs": settings.OUTPUTS_DIR.exists(),
        "tables": settings.TABLES_DIR.exists(),
        "models": settings.MODELS_DIR.exists(),
        "scalers": settings.SCALERS_DIR.exists(),
        "predictions": settings.PREDICTIONS_DIR.exists(),
        "uncertainty": settings.UNCERTAINTY_DIR.exists(),
        "xai": settings.XAI_DIR.exists(),
        "reports": settings.REPORTS_DIR.exists(),
    }

    # All primary tables, models, and scalers must be present
    artifacts_present = all([
        artifact_dirs["outputs"],
        artifact_dirs["tables"],
        artifact_dirs["models"],
        artifact_dirs["scalers"],
    ])

    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc).isoformat(),
        project_root=str(settings.PROJECT_ROOT),
        artifacts_present=artifacts_present,
        artifact_dirs=artifact_dirs,
    )
