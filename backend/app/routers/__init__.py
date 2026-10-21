"""API router handlers."""

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

__all__ = [
    "auth",
    "benchmarks",
    "coach",
    "conformal",
    "forecasts",
    "health",
    "robustness",
    "statistics",
    "xai",
]
