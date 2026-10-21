"""Pydantic schemas for Conformal Prediction and Uncertainty Quantification."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ConformalSummaryRecord(BaseModel):
    """Conformal uncertainty calibration and evaluation summary for a model."""

    model: str = Field(..., description="Model architecture name")
    calibration_mae_watts: float = Field(..., description="Calibration partition MAE in Watts")
    conformal_quantile_qh_mean_watts: float = Field(..., description="Mean horizon conformal quantile q_h threshold in Watts")
    mpiw_watts: float = Field(..., description="Mean Prediction Interval Width (2 * q_h) in Watts")
    clean_picp_pct: float = Field(..., description="Clean test Prediction Interval Coverage Probability (%)")
    clean_ace_pct: float = Field(..., description="Clean Average Coverage Error: |PICP - 90%| (%)")
    clean_winkler_interval_score: float = Field(..., description="Clean test Winkler interval score")
    miss_NA_picp_pct: float = Field(..., description="Coverage under weather missingness (%)")
    miss_AN_picp_pct: float = Field(..., description="Coverage under electricity missingness (%)")
    miss_AA_picp_pct: float = Field(..., description="Coverage under dual missingness (%)")
    miss_AA_winkler_score: float = Field(..., description="Winkler score under dual missingness")
    noise_AA_picp_pct: float = Field(..., description="Coverage under dual noise (%)")
    noise_AA_winkler_score: float = Field(..., description="Winkler score under dual noise")
    oracle_mpiw_miss_AA_watts: float = Field(..., description="Oracle test interval width needed to restore 90% coverage (Watts)")
    oracle_width_expansion_req_pct: float = Field(..., description="Oracle percentage expansion required (%)")


class ConformalSummaryResponse(BaseModel):
    """Response schema for Conformal Prediction summary."""

    source_artifact: str = Field(..., description="Underlying verified artifact filename")
    confidence_level_pct: float = Field(default=90.0, description="Nominal confidence level (1 - alpha)")
    alpha: float = Field(default=0.10, description="Significance level alpha")
    calibration_samples: int = Field(default=587, description="Number of independent calibration windows")
    records: List[ConformalSummaryRecord] = Field(..., description="Per-model conformal uncertainty records")


class ConformalWindowIntervalResponse(BaseModel):
    """Window-specific conformal interval predictions and ground truth."""

    model: str = Field(..., description="Model key")
    condition: str = Field(..., description="Condition key")
    window_idx: int = Field(..., description="Test window index (0 to 586)")
    forecast_origin_timestamp: str = Field(..., description="Forecast origin timestamp")
    horizon_hours: List[int] = Field(..., description="Horizon step offsets (1 to 24)")
    target_timestamps: List[str] = Field(..., description="Timestamps corresponding to forecast steps")
    L_raw: List[float] = Field(..., description="Lower prediction interval bounds in Watts")
    U_raw: List[float] = Field(..., description="Upper prediction interval bounds in Watts")
    y_hat: List[float] = Field(..., description="Model point predictions in Watts")
    y_true: List[float] = Field(..., description="Ground truth observed values in Watts")
    is_covered: List[bool] = Field(..., description="Whether ground truth lies inside [L, U] at each step")
    window_picp_pct: float = Field(..., description="Window empirical coverage percentage")
    window_interval_score: float = Field(..., description="Window Winkler interval penalty score")
    dataset_mean_picp_pct: float = Field(..., description="Overall dataset PICP for this condition (%)")
    dataset_mean_mpiw_watts: float = Field(..., description="Overall dataset MPIW for this condition (Watts)")
