"""Pydantic schemas for operational robustness and sensor degradation experiments."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RobustnessRecord(BaseModel):
    """Single experimental evaluation record across the 2x2 factorial matrix."""

    experiment: str = Field(..., description="Experiment key: missingness_30 or noise_20")
    experiment_label: str = Field(..., description="Human-readable experiment label")
    model: str = Field(..., description="Evaluated model name")
    condition: str = Field(..., description="Condition code: NN, NA, AN, or AA")
    condition_description: str = Field(..., description="Descriptive status of electricity and weather channels")
    test_mae_watts: float = Field(..., description="Test MAE in Watts under this condition")
    delta_mae_vs_clean_watts: float = Field(..., description="Error increase vs nominal clean NN (Watts)")
    relative_mae_change_pct: float = Field(..., description="Relative error change vs clean NN (%)")
    test_rmse_watts: float = Field(..., description="Test RMSE in Watts")
    delta_rmse_vs_clean_watts: float = Field(..., description="RMSE delta vs clean NN (Watts)")
    relative_rmse_change_pct: float = Field(..., description="Relative RMSE change vs clean NN (%)")
    test_r2: float = Field(..., description="Coefficient of determination R^2")
    conformal_picp_pct: float = Field(..., description="Empirical conformal prediction interval coverage (%)")
    conformal_ace_pct: float = Field(..., description="Average Coverage Error: |PICP - 90%| (%)")
    conformal_mpiw_watts: float = Field(..., description="Mean Prediction Interval Width (Watts)")
    winkler_interval_score: float = Field(..., description="Winkler interval penalty score")
    oracle_mpiw_for_90_coverage_watts: float = Field(..., description="Oracle test interval width needed for 90% coverage")


class RobustnessResponse(BaseModel):
    """Response schema for robustness evaluations across the 28 conditions."""

    source_artifact: str = Field(..., description="Underlying verified artifact filename")
    total_evaluations: int = Field(..., description="Total records returned matching filters")
    available_experiments: List[str] = Field(..., description="Available degradation experiments")
    available_models: List[str] = Field(..., description="Evaluated model names")
    available_conditions: List[str] = Field(..., description="2x2 factorial condition codes")
    records: List[RobustnessRecord] = Field(..., description="List of robustness performance records")
