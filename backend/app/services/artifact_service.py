"""
Artifact Service Layer.
Safely loads, caches, and parses precomputed research artifacts from outputs/ and data/.
Ensures paths are anchored to settings.PROJECT_ROOT and keeps parsing logic out of route handlers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.app.core.config import settings
from backend.app.schemas.benchmarks import CleanBenchmarkItem, CleanBenchmarkResponse
from backend.app.schemas.conformal import (
    ConformalSummaryRecord,
    ConformalSummaryResponse,
    ConformalWindowIntervalResponse,
)
from backend.app.schemas.forecasts import (
    TestWindowsMetadataResponse,
    WindowForecastResponse,
    WindowModelMetrics,
)
from backend.app.schemas.robustness import RobustnessRecord, RobustnessResponse
from backend.app.schemas.statistics import (
    FamilyDetailResponse,
    GranularTestRecord,
    HypothesisFamilySummary,
    StatisticalSummaryResponse,
)
from backend.app.schemas.xai import (
    TemporalLagRecord,
    WeatherFeatureRecord,
    XAISummaryRecord,
    XAISummaryResponse,
)


class ArtifactService:
    """Singleton service for loading and serving verified research artifacts."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._test_origins: Optional[List[str]] = None
        self._test_targets: Optional[List[List[str]]] = None

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------
    def _read_csv(self, file_path: Path, keep_default_na: bool = True) -> pd.DataFrame:
        """Safely read and cache a CSV artifact."""
        cache_key = str(file_path)
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        if not file_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Required research artifact '{file_path.name}' is missing.",
            )

        df = pd.read_csv(file_path, keep_default_na=keep_default_na)
        self._cache[cache_key] = df
        return df.copy()

    def _get_test_time_mapping(self) -> Tuple[List[str], List[List[str]]]:
        """Precompute and return the 587 rolling forecast origin and target timestamps."""
        if self._test_origins is not None and self._test_targets is not None:
            return self._test_origins, self._test_targets

        meta_file = settings.PROCESSED_DATA_DIR / "sequences" / "test" / "metadata.json"
        first_origin = "2023-12-08 14:00:00"
        sample_count = 587
        horizon = 24

        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    first_origin = meta.get("first_forecast_origin", first_origin)
                    sample_count = meta.get("sample_count", sample_count)
                    horizon = meta.get("forecast_horizon", horizon)
            except Exception:
                pass

        origins = [
            ts.strftime("%Y-%m-%d %H:%M:%S")
            for ts in pd.date_range(first_origin, periods=sample_count, freq="h")
        ]

        targets = []
        for orig_str in origins:
            orig_dt = pd.to_datetime(orig_str)
            tgt = [
                (orig_dt + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
                for h in range(1, horizon + 1)
            ]
            targets.append(tgt)

        self._test_origins = origins
        self._test_targets = targets
        return self._test_origins, self._test_targets

    def _normalize_model_key(self, model: str) -> str:
        """Normalize model string to standard archive key format."""
        m = model.strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "energy_only": "energy_only",
            "early_fusion": "early_fusion",
            "late_fusion": "late_fusion",
            "hybrid_fusion": "hybrid_fusion",
        }
        if m in mapping:
            return mapping[m]
        raise HTTPException(
            status_code=404,
            detail=f"Model identifier '{model}' is invalid. Supported: Energy-Only, Early Fusion, Late Fusion, Hybrid Fusion.",
        )

    def _normalize_condition_key(self, condition: str) -> str:
        """Normalize condition string to archive file condition format."""
        c = condition.strip().lower().replace("-", "_")
        valid_conditions = {
            "clean_nn": "clean_NN",
            "nn": "clean_NN",
            "clean": "clean_NN",
            "missingness_na": "missingness_NA",
            "na": "missingness_NA",
            "missingness_an": "missingness_AN",
            "an": "missingness_AN",
            "missingness_aa": "missingness_AA",
            "aa": "missingness_AA",
            "noise_na": "noise_NA",
            "noise_an": "noise_AN",
            "noise_aa": "noise_AA",
        }
        if c in valid_conditions:
            return valid_conditions[c]
        raise HTTPException(
            status_code=404,
            detail=f"Condition identifier '{condition}' is invalid. Supported: clean_NN, missingness_NA, missingness_AN, missingness_AA, noise_NA, noise_AN, noise_AA.",
        )

    # -------------------------------------------------------------------------
    # 1. Clean Benchmark Endpoints
    # -------------------------------------------------------------------------
    def get_clean_benchmarks(self) -> CleanBenchmarkResponse:
        """Load and return the revised clean benchmark comparisons."""
        file_path = settings.TABLES_DIR / "step10_clean_benchmark_comparison_revised.csv"
        df = self._read_csv(file_path)

        records = [
            CleanBenchmarkItem(
                model=str(row["model"]),
                architecture_type=str(row["architecture_type"]),
                trainable_parameters=int(row["trainable_parameters"]),
                history_hours=int(row["history_hours"]),
                horizon_hours=int(row["horizon_hours"]),
                input_features=str(row["input_features"]),
                val_mae_watts=float(row["val_mae_watts"]),
                val_rmse_watts=float(row["val_rmse_watts"]),
                val_r2=float(row["val_r2"]),
                val_mape_pct=float(row["val_mape_pct"]),
                val_smape_pct=float(row["val_smape_pct"]),
                test_mae_watts=float(row["test_mae_watts"]),
                test_rmse_watts=float(row["test_rmse_watts"]),
                test_r2=float(row["test_r2"]),
                test_mape_pct=float(row["test_mape_pct"]),
                test_smape_pct=float(row["test_smape_pct"]),
                delta_mae_vs_energy_only_watts=float(row["delta_mae_vs_energy_only_watts"]),
                relative_delta_mae_pct=float(row["relative_delta_mae_pct"]),
                clean_test_rank=int(row["clean_test_rank"]),
            )
            for _, row in df.iterrows()
        ]

        leader = min(records, key=lambda r: r.test_mae_watts).model

        return CleanBenchmarkResponse(
            source_artifact="step10_clean_benchmark_comparison_revised.csv",
            total_models=len(records),
            benchmark_leader=leader,
            records=records,
        )

    # -------------------------------------------------------------------------
    # 2. Operational Robustness Matrix Endpoints
    # -------------------------------------------------------------------------
    def get_robustness_matrix(
        self,
        experiment: Optional[str] = None,
        model: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> RobustnessResponse:
        """Load and return the revised master performance table with filtering."""
        file_path = settings.TABLES_DIR / "step10_master_performance_table_revised.csv"
        df = self._read_csv(file_path, keep_default_na=False)

        # Correct for empty string condition where NA was string-parsed
        for idx, row in df.iterrows():
            if not row["condition"] or pd.isna(row["condition"]):
                if "Weather-Degraded" in str(row["condition_description"]):
                    df.at[idx, "condition"] = "NA"

        # Apply filters
        if experiment:
            exp_clean = experiment.strip().lower()
            df = df[df["experiment"].str.lower() == exp_clean]
        if model:
            mod_clean = model.strip().lower()
            df = df[df["model"].str.lower() == mod_clean]
        if condition:
            cond_clean = condition.strip().upper()
            df = df[df["condition"].str.upper() == cond_clean]

        records = [
            RobustnessRecord(
                experiment=str(row["experiment"]),
                experiment_label=str(row["experiment_label"]),
                model=str(row["model"]),
                condition=str(row["condition"]),
                condition_description=str(row["condition_description"]),
                test_mae_watts=float(row["test_mae_watts"]),
                delta_mae_vs_clean_watts=float(row["delta_mae_vs_clean_watts"]),
                relative_mae_change_pct=float(row["relative_mae_change_pct"]),
                test_rmse_watts=float(row["test_rmse_watts"]),
                delta_rmse_vs_clean_watts=float(row["delta_rmse_vs_clean_watts"]),
                relative_rmse_change_pct=float(row["relative_rmse_change_pct"]),
                test_r2=float(row["test_r2"]),
                conformal_picp_pct=float(row["conformal_picp_pct"]),
                conformal_ace_pct=float(row["conformal_ace_pct"]),
                conformal_mpiw_watts=float(row["conformal_mpiw_watts"]),
                winkler_interval_score=float(row["winkler_interval_score"]),
                oracle_mpiw_for_90_coverage_watts=float(row["oracle_mpiw_for_90_coverage_watts"]),
            )
            for _, row in df.iterrows()
        ]

        return RobustnessResponse(
            source_artifact="step10_master_performance_table_revised.csv",
            total_evaluations=len(records),
            available_experiments=["missingness_30", "noise_20"],
            available_models=["Energy-Only", "Early Fusion", "Late Fusion", "Hybrid Fusion"],
            available_conditions=["NN", "NA", "AN", "AA"],
            records=records,
        )

    # -------------------------------------------------------------------------
    # 3. Statistical Inference Endpoints
    # -------------------------------------------------------------------------
    def get_statistical_summary(self) -> StatisticalSummaryResponse:
        """Load and return the 6-family statistical significance overview."""
        file_path = settings.TABLES_DIR / "step10_statistical_significance_summary_revised.csv"
        df = self._read_csv(file_path)

        families = [
            HypothesisFamilySummary(
                family_id=int(row["family_id"]),
                family_name=str(row["family_name"]),
                total_tests=int(row["total_tests"]),
                significant_tests_holm=int(row["significant_tests_holm"]),
                significant_tests_fdr=int(row["significant_tests_fdr"]),
                primary_inference=str(row["primary_inference"]),
            )
            for _, row in df.iterrows()
        ]

        total_planned = sum(f.total_tests for f in families)
        total_sig = sum(f.significant_tests_holm for f in families)
        rejection_rate = (total_sig / total_planned * 100.0) if total_planned else 0.0

        return StatisticalSummaryResponse(
            source_artifact="step10_statistical_significance_summary_revised.csv",
            total_families=len(families),
            total_planned_tests=total_planned,
            total_significant_holm=total_sig,
            overall_rejection_rate_pct=round(rejection_rate, 2),
            families=families,
        )

    def get_statistical_family_detail(self, family_id: int) -> FamilyDetailResponse:
        """Load detailed narrative and granular tests for a specific hypothesis family."""
        if family_id < 1 or family_id > 6:
            raise HTTPException(
                status_code=404,
                detail=f"Hypothesis family ID {family_id} not found. Must be an integer between 1 and 6.",
            )

        summary_df = self._read_csv(
            settings.TABLES_DIR / "step10_statistical_significance_summary_revised.csv"
        )
        fam_row = summary_df[summary_df["family_id"] == family_id]
        if fam_row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Hypothesis family {family_id} not present in summary table.",
            )

        fam_info = fam_row.iloc[0]

        # Load granular pairwise tests from Step 7 directory
        all_tests_path = settings.OUTPUTS_DIR / "robustness" / "step7_analysis" / "tables" / "all_78_pairwise_tests_full.csv"
        granular_records = []
        if all_tests_path.exists():
            tests_df = self._read_csv(all_tests_path)
            fam_tests = tests_df[tests_df["family_id"] == family_id]
            for _, r in fam_tests.iterrows():
                granular_records.append(
                    GranularTestRecord(
                        test_id=int(r["test_id"]),
                        family_id=int(r["family_id"]),
                        comparison_name=str(r["comparison_name"]),
                        model_1=str(r["model_1"]),
                        cond_1=str(r["cond_1"]),
                        model_2=str(r["model_2"]),
                        cond_2=str(r["cond_2"]),
                        n_windows=int(r["n_windows"]),
                        mean_diff_watts=float(r["mean_diff_watts"]),
                        mean_ci_95_low=float(r["mean_ci_95_low"]),
                        mean_ci_95_high=float(r["mean_ci_95_high"]),
                        rel_diff_pct=float(r["rel_diff_pct"]),
                        rank_biserial_r_rb=float(r["rank_biserial_r_rb"]),
                        wilcoxon_stat=float(r["wilcoxon_stat"]),
                        wilcoxon_p_raw=float(r["wilcoxon_p_raw"]),
                        wilcoxon_p_holm=float(r["wilcoxon_p_holm"]),
                        sig_holm=bool(r["sig_holm"]),
                        wilcoxon_p_fdr=float(r["wilcoxon_p_fdr"]),
                        sig_fdr=bool(r["sig_fdr"]),
                        hln_stat=float(r["hln_stat"]) if pd.notna(r.get("hln_stat")) else None,
                        hln_p_raw=float(r["hln_p_raw"]) if pd.notna(r.get("hln_p_raw")) else None,
                        decision=str(r["decision"]),
                    )
                )

        return FamilyDetailResponse(
            family_id=family_id,
            family_name=str(fam_info["family_name"]),
            total_tests=int(fam_info["total_tests"]),
            significant_tests_holm=int(fam_info["significant_tests_holm"]),
            primary_inference=str(fam_info["primary_inference"]),
            granular_tests=granular_records,
        )

    # -------------------------------------------------------------------------
    # 4. Explainable AI (XAI) Endpoints
    # -------------------------------------------------------------------------
    def get_xai_summary(self) -> XAISummaryResponse:
        """Load and return the revised XAI modality attribution summary."""
        file_path = settings.TABLES_DIR / "step10_xai_modality_attribution_summary_revised.csv"
        df = self._read_csv(file_path)

        records = [
            XAISummaryRecord(
                model=str(row["model"]),
                clean_electricity_share_pct=float(row["clean_electricity_share_pct"]),
                clean_weather_share_pct=float(row["clean_weather_share_pct"]),
                clean_electricity_density_share_pct=float(row["clean_electricity_density_share_pct"]),
                clean_weather_density_share_pct=float(row["clean_weather_density_share_pct"]),
                miss_NA_elec_shift_pct_pts=float(row["miss_NA_elec_shift_pct_pts"]),
                miss_AN_elec_shift_pct_pts=float(row["miss_AN_elec_shift_pct_pts"]),
                miss_AA_elec_shift_pct_pts=float(row["miss_AA_elec_shift_pct_pts"]),
                noise_AA_elec_shift_pct_pts=float(row["noise_AA_elec_shift_pct_pts"]),
                electricity_temporal_recent_24h_pct=float(row["electricity_temporal_recent_24h_pct"]),
                electricity_temporal_intermediate_48h_pct=float(row["electricity_temporal_intermediate_48h_pct"]),
                electricity_temporal_distant_96h_pct=float(row["electricity_temporal_distant_96h_pct"]),
                top_weather_features_clean=str(row["top_weather_features_clean"]),
            )
            for _, row in df.iterrows()
        ]

        return XAISummaryResponse(
            source_artifact="step10_xai_modality_attribution_summary_revised.csv",
            records=records,
        )

    def get_xai_temporal(
        self,
        model: Optional[str] = None,
        condition: Optional[str] = None,
        experiment: Optional[str] = None,
    ) -> List[TemporalLagRecord]:
        """Load granular electricity temporal lag attributions."""
        file_path = settings.XAI_TABLES_DIR / "electricity_temporal_attribution.csv"
        df = self._read_csv(file_path)

        if model:
            df = df[df["model"].str.lower() == model.strip().lower()]
        if condition:
            df = df[df["condition"].str.upper() == condition.strip().upper()]
        if experiment:
            df = df[df["experiment"].str.lower() == experiment.strip().lower()]

        return [
            TemporalLagRecord(
                model=str(row["model"]),
                experiment=str(row["experiment"]),
                condition=str(row["condition"]),
                bin_name=str(row["bin_name"]),
                description=str(row["description"]),
                n_hours=int(row["n_hours"]),
                attribution_magnitude_mean=float(row["attribution_magnitude_mean"]),
                temporal_share_pct=float(row["temporal_share_pct"]),
            )
            for _, row in df.iterrows()
        ]

    def get_xai_weather_features(
        self,
        model: Optional[str] = None,
        condition: Optional[str] = None,
        experiment: Optional[str] = None,
    ) -> List[WeatherFeatureRecord]:
        """Load weather feature rankings and percentage shares."""
        file_path = settings.XAI_TABLES_DIR / "weather_feature_attribution.csv"
        df = self._read_csv(file_path)

        if model:
            df = df[df["model"].str.lower() == model.strip().lower()]
        if condition:
            df = df[df["condition"].str.upper() == condition.strip().upper()]
        if experiment:
            df = df[df["experiment"].str.lower() == experiment.strip().lower()]

        return [
            WeatherFeatureRecord(
                model=str(row["model"]),
                experiment=str(row["experiment"]),
                condition=str(row["condition"]),
                feature_name=str(row["feature_name"]),
                attribution_magnitude_mean=float(row["attribution_magnitude_mean"]),
                attribution_share_within_weather_pct=float(row["attribution_share_within_weather_pct"]),
                feature_rank=int(row["feature_rank"]),
            )
            for _, row in df.iterrows()
        ]

    # -------------------------------------------------------------------------
    # 5. Conformal Prediction & Uncertainty Endpoints
    # -------------------------------------------------------------------------
    def get_conformal_summary(self) -> ConformalSummaryResponse:
        """Load and return the revised conformal uncertainty summary."""
        file_path = settings.TABLES_DIR / "step10_conformal_uncertainty_summary_revised.csv"
        df = self._read_csv(file_path)

        records = [
            ConformalSummaryRecord(
                model=str(row["model"]),
                calibration_mae_watts=float(row["calibration_mae_watts"]),
                conformal_quantile_qh_mean_watts=float(row["conformal_quantile_qh_mean_watts"]),
                mpiw_watts=float(row["mpiw_watts"]),
                clean_picp_pct=float(row["clean_picp_pct"]),
                clean_ace_pct=float(row["clean_ace_pct"]),
                clean_winkler_interval_score=float(row["clean_winkler_interval_score"]),
                miss_NA_picp_pct=float(row["miss_NA_picp_pct"]),
                miss_AN_picp_pct=float(row["miss_AN_picp_pct"]),
                miss_AA_picp_pct=float(row["miss_AA_picp_pct"]),
                miss_AA_winkler_score=float(row["miss_AA_winkler_score"]),
                noise_AA_picp_pct=float(row["noise_AA_picp_pct"]),
                noise_AA_winkler_score=float(row["noise_AA_winkler_score"]),
                oracle_mpiw_miss_AA_watts=float(row["oracle_mpiw_miss_AA_watts"]),
                oracle_width_expansion_req_pct=float(row["oracle_width_expansion_req_pct"]),
            )
            for _, row in df.iterrows()
        ]

        return ConformalSummaryResponse(
            source_artifact="step10_conformal_uncertainty_summary_revised.csv",
            records=records,
        )

    def get_conformal_interval(
        self, model: str, condition: str, window_idx: int = 0
    ) -> ConformalWindowIntervalResponse:
        """Load precomputed conformal prediction intervals for a window and model condition."""
        if window_idx < 0 or window_idx >= 587:
            raise HTTPException(
                status_code=404,
                detail=f"Test window index {window_idx} out of range (must be between 0 and 586).",
            )

        norm_model = self._normalize_model_key(model)
        norm_cond = self._normalize_condition_key(condition)

        archive_name = f"intervals_{norm_model}_{norm_cond}.npz"
        file_path = settings.UNCERTAINTY_ARTIFACTS_DIR / archive_name

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Conformal intervals archive for model '{model}' under condition '{condition}' is not available.",
            )

        cache_key = f"npz_{archive_name}"
        if cache_key not in self._cache:
            try:
                npz = np.load(file_path)
                self._cache[cache_key] = {k: npz[k] for k in npz.files}
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load interval archive for '{model}'.",
                )

        data = self._cache[cache_key]
        L_w = data["L_raw"][window_idx].tolist()
        U_w = data["U_raw"][window_idx].tolist()
        y_hat_w = data["y_hat"][window_idx].tolist()
        y_true_w = data["y_true"][window_idx].tolist()

        is_covered = [
            bool(L <= y <= U) for L, U, y in zip(L_w, U_w, y_true_w)
        ]
        win_picp = (sum(is_covered) / len(is_covered)) * 100.0
        win_is = float(data["per_window_is"][window_idx]) if "per_window_is" in data else 0.0

        origins, targets = self._get_test_time_mapping()
        origin_ts = origins[window_idx]
        target_ts = targets[window_idx]

        # Overall dataset PICP and MPIW for this condition
        mean_picp = float(np.mean(data["picp_by_h"])) * 100.0 if "picp_by_h" in data else win_picp
        mean_mpiw = float(np.mean(data["U_raw"] - data["L_raw"]))

        return ConformalWindowIntervalResponse(
            model=norm_model,
            condition=norm_cond,
            window_idx=window_idx,
            forecast_origin_timestamp=origin_ts,
            horizon_hours=list(range(1, 25)),
            target_timestamps=target_ts,
            L_raw=[round(v, 2) for v in L_w],
            U_raw=[round(v, 2) for v in U_w],
            y_hat=[round(v, 2) for v in y_hat_w],
            y_true=[round(v, 2) for v in y_true_w],
            is_covered=is_covered,
            window_picp_pct=round(win_picp, 2),
            window_interval_score=round(win_is, 2),
            dataset_mean_picp_pct=round(mean_picp, 2),
            dataset_mean_mpiw_watts=round(mean_mpiw, 2),
        )

    # -------------------------------------------------------------------------
    # 6. Test Forecast Endpoints
    # -------------------------------------------------------------------------
    def get_test_windows_metadata(self) -> TestWindowsMetadataResponse:
        """Return metadata for the 587 rolling test windows."""
        origins, targets = self._get_test_time_mapping()
        return TestWindowsMetadataResponse(
            total_windows=len(origins),
            history_length_hours=168,
            forecast_horizon_hours=24,
            first_forecast_origin=origins[0],
            last_forecast_origin=origins[-1],
            first_target_start=targets[0][0],
            last_target_end=targets[-1][-1],
            available_models=["Energy-Only", "Early Fusion", "Late Fusion", "Hybrid Fusion"],
            window_index_range=[0, len(origins) - 1],
        )

    def get_test_window_forecast(self, window_idx: int) -> WindowForecastResponse:
        """Return ground truth and multi-model forecasts for a test window."""
        if window_idx < 0 or window_idx >= 587:
            raise HTTPException(
                status_code=404,
                detail=f"Test window index {window_idx} out of range (must be between 0 and 586).",
            )

        models_map = {
            "Energy-Only": "energy_only",
            "Early Fusion": "early_fusion",
            "Late Fusion": "late_fusion",
            "Hybrid Fusion": "hybrid_fusion",
        }

        # Cache predictions
        if "test_predictions_dict" not in self._cache:
            preds_dict = {}
            actuals_arr = None
            for display_name, folder in models_map.items():
                p = settings.PREDICTIONS_DIR / folder / "test_predictions.npz"
                if p.exists():
                    npz = np.load(p)
                    for k in ["predictions_watts", "predicted_watts", "predictions"]:
                        if k in npz:
                            preds_dict[display_name] = npz[k]
                            break
                    if actuals_arr is None and "actual_watts" in npz:
                        actuals_arr = npz["actual_watts"]
            if actuals_arr is None:
                raise HTTPException(
                    status_code=500,
                    detail="Ground truth test predictions archive is missing.",
                )
            self._cache["test_predictions_dict"] = preds_dict
            self._cache["test_actuals"] = actuals_arr

        actuals = self._cache["test_actuals"][window_idx].tolist()
        predictions = {}
        metrics = {}

        for display_name, pred_mat in self._cache["test_predictions_dict"].items():
            p_seq = pred_mat[window_idx].tolist()
            predictions[display_name] = [round(v, 2) for v in p_seq]

            err = np.array(p_seq) - np.array(actuals)
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            max_err = float(np.max(np.abs(err)))
            metrics[display_name] = WindowModelMetrics(
                mae_watts=round(mae, 2),
                rmse_watts=round(rmse, 2),
                max_error_watts=round(max_err, 2),
            )

        origins, targets = self._get_test_time_mapping()

        return WindowForecastResponse(
            window_idx=window_idx,
            forecast_origin_timestamp=origins[window_idx],
            horizon_hours=list(range(1, 25)),
            target_timestamps=targets[window_idx],
            actual_watts=[round(v, 2) for v in actuals],
            predictions_watts=predictions,
            metrics=metrics,
        )


# Global service instance
artifact_service = ArtifactService()
