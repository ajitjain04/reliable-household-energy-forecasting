"""Pydantic schemas for clean research benchmarks."""

from typing import List, Optional
from pydantic import BaseModel, Field


class CleanBenchmarkItem(BaseModel):
    """Clean benchmark model performance record."""

    model: str = Field(..., description="Model architecture name")
    architecture_type: str = Field(..., description="High-level architecture classification")
    trainable_parameters: int = Field(..., description="Exact trainable parameter count")
    history_hours: int = Field(..., description="Lookback window length (168 hours)")
    horizon_hours: int = Field(..., description="Forecast horizon length (24 hours)")
    input_features: str = Field(..., description="Description of input modalities/channels")
    val_mae_watts: float = Field(..., description="Validation MAE in Watts")
    val_rmse_watts: float = Field(..., description="Validation RMSE in Watts")
    val_r2: float = Field(..., description="Validation coefficient of determination R^2")
    val_mape_pct: float = Field(..., description="Validation MAPE in percent")
    val_smape_pct: float = Field(..., description="Validation sMAPE in percent")
    test_mae_watts: float = Field(..., description="Out-of-sample Test MAE in Watts")
    test_rmse_watts: float = Field(..., description="Out-of-sample Test RMSE in Watts")
    test_r2: float = Field(..., description="Out-of-sample Test R^2")
    test_mape_pct: float = Field(..., description="Out-of-sample Test MAPE in percent")
    test_smape_pct: float = Field(..., description="Out-of-sample Test sMAPE in percent")
    delta_mae_vs_energy_only_watts: float = Field(..., description="Delta MAE relative to Energy-Only baseline (W)")
    relative_delta_mae_pct: float = Field(..., description="Relative MAE change vs baseline (%)")
    clean_test_rank: int = Field(..., description="Performance rank on clean test set")


class CleanBenchmarkResponse(BaseModel):
    """Response schema for clean benchmark comparisons."""

    source_artifact: str = Field(..., description="Underlying verified artifact filename")
    total_models: int = Field(..., description="Number of evaluated models")
    benchmark_leader: str = Field(..., description="Model achieving lowest test MAE")
    records: List[CleanBenchmarkItem] = Field(..., description="Benchmark records for each model")
