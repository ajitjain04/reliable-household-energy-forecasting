"""Pydantic schemas for service health and diagnostic status."""

from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check diagnostic response schema."""

    status: str = Field(default="ok", description="Service health status indicator")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="API version string")
    environment: str = Field(..., description="Deployment environment")
    timestamp: str = Field(..., description="ISO 8601 UTC server timestamp")
    project_root: str = Field(..., description="Resolved absolute path to project root")
    artifacts_present: bool = Field(..., description="Whether required research artifacts are present")
    artifact_dirs: Dict[str, bool] = Field(..., description="Existence status of core artifact directories")
