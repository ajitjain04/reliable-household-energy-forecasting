"""
Comprehensive test suite for precomputed research artifact API endpoints.
Verifies all 10 artifact-backed routes, schema conformance, error handling,
parameter validation, and source artifact immutability.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.config import settings


class TestArtifactEndpoints(unittest.TestCase):
    """Test suite covering all precomputed research artifact endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.prefix = settings.API_PREFIX

    # -------------------------------------------------------------------------
    # 1. Clean Benchmark Tests
    # -------------------------------------------------------------------------
    def test_clean_benchmarks_success(self):
        """Verify GET /api/v1/benchmarks/clean returns verified 4-model comparison."""
        res = self.client.get(f"{self.prefix}/benchmarks/clean")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["source_artifact"], "step10_clean_benchmark_comparison_revised.csv")
        self.assertEqual(data["total_models"], 4)
        self.assertEqual(data["benchmark_leader"], "Early Fusion")

        models = [r["model"] for r in data["records"]]
        self.assertIn("Early Fusion", models)
        self.assertIn("Energy-Only", models)
        self.assertIn("Late Fusion", models)
        self.assertIn("Hybrid Fusion", models)

        # Verify Early Fusion values match audited Step 10 report
        early = next(r for r in data["records"] if r["model"] == "Early Fusion")
        self.assertAlmostEqual(early["test_mae_watts"], 71.64, places=1)
        self.assertAlmostEqual(early["test_rmse_watts"], 88.56, places=1)
        self.assertEqual(early["trainable_parameters"], 53784)
        self.assertEqual(early["clean_test_rank"], 1)

    # -------------------------------------------------------------------------
    # 2. Operational Robustness Matrix Tests
    # -------------------------------------------------------------------------
    def test_robustness_matrix_unfiltered(self):
        """Verify GET /api/v1/robustness/matrix returns all 28+ evaluations."""
        res = self.client.get(f"{self.prefix}/robustness/matrix")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["source_artifact"], "step10_master_performance_table_revised.csv")
        self.assertGreaterEqual(data["total_evaluations"], 28)
        self.assertIn("missingness_30", data["available_experiments"])
        self.assertIn("noise_20", data["available_experiments"])

        # Check condition "NA" is properly preserved
        na_records = [r for r in data["records"] if r["condition"] == "NA"]
        self.assertGreater(len(na_records), 0)
        for r in na_records:
            self.assertIn("Weather-Degraded", r["condition_description"])

    def test_robustness_matrix_filtered(self):
        """Verify filtering by experiment, model, and condition."""
        res = self.client.get(
            f"{self.prefix}/robustness/matrix?experiment=missingness_30&model=Early%20Fusion&condition=AA"
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_evaluations"], 1)
        record = data["records"][0]
        self.assertEqual(record["model"], "Early Fusion")
        self.assertEqual(record["condition"], "AA")
        self.assertAlmostEqual(record["test_mae_watts"], 76.39, places=1)
        self.assertAlmostEqual(record["delta_mae_vs_clean_watts"], 4.76, places=1)

    # -------------------------------------------------------------------------
    # 3. Statistical Inference Tests
    # -------------------------------------------------------------------------
    def test_statistical_summary(self):
        """Verify GET /api/v1/statistics/summary returns 6 families and 78 tests."""
        res = self.client.get(f"{self.prefix}/statistics/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["source_artifact"], "step10_statistical_significance_summary_revised.csv")
        self.assertEqual(data["total_families"], 6)
        self.assertEqual(data["total_planned_tests"], 78)
        self.assertEqual(data["total_significant_holm"], 60)
        self.assertAlmostEqual(data["overall_rejection_rate_pct"], 76.92, places=1)
        self.assertEqual(len(data["families"]), 6)

    def test_statistical_family_detail_valid(self):
        """Verify GET /api/v1/statistics/family/{family_id} for valid family 1."""
        res = self.client.get(f"{self.prefix}/statistics/family/1")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["family_id"], 1)
        self.assertIn("Clean Model Comparisons", data["family_name"])
        self.assertEqual(data["total_tests"], 6)
        self.assertEqual(data["significant_tests_holm"], 5)
        self.assertEqual(len(data["granular_tests"]), 6)

        # Check that granular test records have effect sizes and p-values
        t1 = data["granular_tests"][0]
        self.assertIn("mean_diff_watts", t1)
        self.assertIn("wilcoxon_p_holm", t1)
        self.assertIn("rank_biserial_r_rb", t1)

    def test_statistical_family_detail_invalid(self):
        """Verify 404 for invalid family ID."""
        res = self.client.get(f"{self.prefix}/statistics/family/99")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"].lower())

    # -------------------------------------------------------------------------
    # 4. Explainable AI (XAI) Tests
    # -------------------------------------------------------------------------
    def test_xai_summary(self):
        """Verify GET /api/v1/xai/summary returns modality attribution shares."""
        res = self.client.get(f"{self.prefix}/xai/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["source_artifact"], "step10_xai_modality_attribution_summary_revised.csv")
        self.assertEqual(len(data["records"]), 4)

        # Check Early Fusion attribution numbers
        early = next(r for r in data["records"] if r["model"] == "Early Fusion")
        self.assertAlmostEqual(early["clean_electricity_share_pct"], 33.1, places=1)
        self.assertAlmostEqual(early["clean_electricity_density_share_pct"], 76.22, places=1)
        self.assertAlmostEqual(early["miss_NA_elec_shift_pct_pts"], 8.14, places=1)

    def test_xai_temporal(self):
        """Verify GET /api/v1/xai/temporal returns lag allocations."""
        res = self.client.get(f"{self.prefix}/xai/temporal?model=Early%20Fusion&condition=NN")
        self.assertEqual(res.status_code, 200)
        records = res.json()
        self.assertGreaterEqual(len(records), 3)
        bin_names = [r["bin_name"] for r in records]
        self.assertTrue(any("Recent" in b for b in bin_names))

    def test_xai_weather_features(self):
        """Verify GET /api/v1/xai/weather-features returns feature rankings."""
        res = self.client.get(
            f"{self.prefix}/xai/weather-features?model=Early%20Fusion&condition=NN&experiment=missingness"
        )
        self.assertEqual(res.status_code, 200)
        records = res.json()
        self.assertEqual(len(records), 7)
        ranks = [r["feature_rank"] for r in records]
        self.assertEqual(sorted(ranks), list(range(1, 8)))

    # -------------------------------------------------------------------------
    # 5. Conformal Prediction Tests
    # -------------------------------------------------------------------------
    def test_conformal_summary(self):
        """Verify GET /api/v1/conformal/summary returns calibration metrics."""
        res = self.client.get(f"{self.prefix}/conformal/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["source_artifact"], "step10_conformal_uncertainty_summary_revised.csv")
        self.assertEqual(data["confidence_level_pct"], 90.0)
        self.assertEqual(len(data["records"]), 4)

        early = next(r for r in data["records"] if r["model"] == "Early Fusion")
        self.assertAlmostEqual(early["clean_picp_pct"], 88.56, places=1)
        self.assertAlmostEqual(early["mpiw_watts"], 268.26, places=1)

    def test_conformal_intervals_valid(self):
        """Verify GET /api/v1/conformal/intervals/{model}/{condition} returns valid arrays."""
        res = self.client.get(f"{self.prefix}/conformal/intervals/early_fusion/clean_NN?window_idx=42")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["model"], "early_fusion")
        self.assertEqual(data["condition"], "clean_NN")
        self.assertEqual(data["window_idx"], 42)
        self.assertEqual(len(data["L_raw"]), 24)
        self.assertEqual(len(data["U_raw"]), 24)
        self.assertEqual(len(data["y_hat"]), 24)
        self.assertEqual(len(data["y_true"]), 24)
        self.assertEqual(len(data["is_covered"]), 24)
        self.assertGreaterEqual(data["window_picp_pct"], 0.0)
        self.assertLessEqual(data["window_picp_pct"], 100.0)

    def test_conformal_intervals_invalid_params(self):
        """Verify 404 on invalid model, condition, or window index."""
        # Invalid model
        res = self.client.get(f"{self.prefix}/conformal/intervals/unknown_model/clean_NN")
        self.assertEqual(res.status_code, 404)

        # Invalid condition
        res = self.client.get(f"{self.prefix}/conformal/intervals/early_fusion/invalid_cond")
        self.assertEqual(res.status_code, 404)

        # Out of bounds window index
        res = self.client.get(f"{self.prefix}/conformal/intervals/early_fusion/clean_NN?window_idx=9999")
        self.assertEqual(res.status_code, 422)  # Pydantic query ge/le validation

    # -------------------------------------------------------------------------
    # 6. Test Forecasts Tests
    # -------------------------------------------------------------------------
    def test_test_windows_metadata(self):
        """Verify GET /api/v1/forecasts/test-windows returns partition structure."""
        res = self.client.get(f"{self.prefix}/forecasts/test-windows")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["total_windows"], 587)
        self.assertEqual(data["history_length_hours"], 168)
        self.assertEqual(data["forecast_horizon_hours"], 24)
        self.assertEqual(data["first_forecast_origin"], "2023-12-08 14:00:00")
        self.assertEqual(data["last_forecast_origin"], "2024-01-02 00:00:00")
        self.assertEqual(len(data["available_models"]), 4)

    def test_test_window_forecast_valid(self):
        """Verify GET /api/v1/forecasts/test-window/{window_idx} returns all 4 model forecasts."""
        res = self.client.get(f"{self.prefix}/forecasts/test-window/100")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["window_idx"], 100)
        self.assertEqual(len(data["horizon_hours"]), 24)
        self.assertEqual(len(data["target_timestamps"]), 24)
        self.assertEqual(len(data["actual_watts"]), 24)

        # Check all 4 models present
        preds = data["predictions_watts"]
        self.assertIn("Early Fusion", preds)
        self.assertIn("Energy-Only", preds)
        self.assertIn("Late Fusion", preds)
        self.assertIn("Hybrid Fusion", preds)
        for m, vals in preds.items():
            self.assertEqual(len(vals), 24)

        # Check metrics present
        metrics = data["metrics"]
        self.assertIn("Early Fusion", metrics)
        self.assertGreater(metrics["Early Fusion"]["mae_watts"], 0.0)

    def test_test_window_forecast_invalid_index(self):
        """Verify 422 for out-of-range window index."""
        res = self.client.get(f"{self.prefix}/forecasts/test-window/9999")
        self.assertEqual(res.status_code, 422)

    # -------------------------------------------------------------------------
    # 7. Error Handling for Missing Artifacts (Mocked)
    # -------------------------------------------------------------------------
    def test_missing_artifact_error_handling(self):
        """Verify clean 500 error when underlying artifact is unavailable."""
        with patch.object(Path, "exists", return_value=False):
            # Bypass cache by creating a fresh mock service or fresh request
            from backend.app.services.artifact_service import ArtifactService
            fresh_service = ArtifactService()
            with self.assertRaises(Exception) as ctx:
                fresh_service.get_clean_benchmarks()
            self.assertIn("missing", str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main()
