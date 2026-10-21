"""Pydantic schemas for formal statistical inference and hypothesis testing."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HypothesisFamilySummary(BaseModel):
    """Overview record for a single hypothesis testing family."""

    family_id: int = Field(..., description="Hypothesis Family ID (1 to 6)")
    family_name: str = Field(..., description="Full family name")
    total_tests: int = Field(..., description="Total planned tests within this family")
    significant_tests_holm: int = Field(..., description="Tests rejecting null under Holm-Bonferroni correction")
    significant_tests_fdr: int = Field(..., description="Tests rejecting null under Benjamini-Hochberg FDR")
    primary_inference: str = Field(..., description="Audited statistical synthesis narrative with exact statistics")


class StatisticalSummaryResponse(BaseModel):
    """Response schema for the 78-test statistical hypothesis battery overview."""

    source_artifact: str = Field(..., description="Underlying verified artifact filename")
    total_families: int = Field(default=6, description="Number of planned hypothesis families")
    total_planned_tests: int = Field(default=78, description="Total hypothesis tests planned")
    total_significant_holm: int = Field(default=60, description="Total tests achieving Holm significance")
    overall_rejection_rate_pct: float = Field(default=76.92, description="Overall null rejection percentage")
    families: List[HypothesisFamilySummary] = Field(..., description="Per-family summaries")


class GranularTestRecord(BaseModel):
    """Granular pairwise hypothesis test record."""

    test_id: int = Field(..., description="Unique test ID (1 to 78)")
    family_id: int = Field(..., description="Family ID (1 to 6)")
    comparison_name: str = Field(..., description="Human-readable comparison title")
    model_1: str = Field(..., description="First compared model")
    cond_1: str = Field(..., description="First condition code")
    model_2: str = Field(..., description="Second compared model")
    cond_2: str = Field(..., description="Second condition code")
    n_windows: int = Field(..., description="Sample size of test windows (587)")
    mean_diff_watts: float = Field(..., description="Mean paired difference in MAE (Watts)")
    mean_ci_95_low: float = Field(..., description="95% CI lower bound of mean difference")
    mean_ci_95_high: float = Field(..., description="95% CI upper bound of mean difference")
    rel_diff_pct: float = Field(..., description="Relative percentage difference (%)")
    rank_biserial_r_rb: float = Field(..., description="Rank-biserial correlation effect size r_rb")
    wilcoxon_stat: float = Field(..., description="Wilcoxon signed-rank test statistic W")
    wilcoxon_p_raw: float = Field(..., description="Unadjusted two-sided Wilcoxon p-value")
    wilcoxon_p_holm: float = Field(..., description="Holm-Bonferroni family-wise adjusted p-value")
    sig_holm: bool = Field(..., description="Whether test rejects null under Holm correction (alpha=0.05)")
    wilcoxon_p_fdr: float = Field(..., description="Benjamini-Hochberg FDR adjusted p-value")
    sig_fdr: bool = Field(..., description="Whether test rejects null under FDR control")
    hln_stat: Optional[float] = Field(None, description="Harvey-Leybourne-Newbery modified DM statistic")
    hln_p_raw: Optional[float] = Field(None, description="Diebold-Mariano p-value with Newey-West HAC adjustment")
    decision: str = Field(..., description="Audited statistical decision")


class FamilyDetailResponse(BaseModel):
    """Detailed response for a specific hypothesis testing family."""

    family_id: int = Field(..., description="Family ID")
    family_name: str = Field(..., description="Family Name")
    total_tests: int = Field(..., description="Total tests in family")
    significant_tests_holm: int = Field(..., description="Significant tests under Holm correction")
    primary_inference: str = Field(..., description="Primary narrative synthesis")
    granular_tests: List[GranularTestRecord] = Field(..., description="Granular test records")
