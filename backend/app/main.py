"""
Main entry point for Reliable Multimodal Household Energy Forecasting API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.db.database import db_manager
from backend.app.routers import (
    auth,
    benchmarks,
    coach,
    conformal,
    forecasts,
    health,
    robustness,
    statistics,
    xai,
)


from backend.app.services.user_service import user_service


def create_app() -> FastAPI:
    """Application factory for FastAPI instance."""
    # Ensure database schema is initialized and bootstrap dev if configured
    db_manager.init_db()
    user_service.bootstrap_initial_developer()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "REST API powering the research dashboard for reliable multimodal "
            "household energy forecasting under degraded weather information."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        swagger_ui_parameters={"persistAuthorization": True},
    )

    # Configure CORS for local development (React/Vite)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Routers under /api/v1
    app.include_router(health.router, prefix=settings.API_PREFIX)
    app.include_router(auth.router, prefix=settings.API_PREFIX)
    app.include_router(benchmarks.router, prefix=settings.API_PREFIX)
    app.include_router(robustness.router, prefix=settings.API_PREFIX)
    app.include_router(statistics.router, prefix=settings.API_PREFIX)
    app.include_router(xai.router, prefix=settings.API_PREFIX)
    app.include_router(conformal.router, prefix=settings.API_PREFIX)
    app.include_router(forecasts.router, prefix=settings.API_PREFIX)
    app.include_router(coach.router, prefix=settings.API_PREFIX)


    @app.get("/", tags=["Root"])
    def root():
        """Root welcome endpoint with API discovery links."""
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "docs_url": "/docs",
            "health_url": f"{settings.API_PREFIX}/health",
            "endpoints": {
                "benchmarks_clean": f"{settings.API_PREFIX}/benchmarks/clean",
                "robustness_matrix": f"{settings.API_PREFIX}/robustness/matrix",
                "statistics_summary": f"{settings.API_PREFIX}/statistics/summary",
                "xai_summary": f"{settings.API_PREFIX}/xai/summary",
                "xai_temporal": f"{settings.API_PREFIX}/xai/temporal",
                "xai_weather": f"{settings.API_PREFIX}/xai/weather-features",
                "conformal_summary": f"{settings.API_PREFIX}/conformal/summary",
                "forecasts_windows": f"{settings.API_PREFIX}/forecasts/test-windows",
            },
        }

    return app


app = create_app()
