"""
Non-destructive test suite auditing trained-model loading, forward pass,
scaler compatibility, and conformal quantile integrity.

Verifies:
- Checkpoint files exist and load into PyTorch model architectures.
- Correct input dimensions (168 timesteps, 1 or 8 features).
- Forward passes output (batch, 24) without NaNs/Infs.
- Scaler transformation and inverse-scaling to Watts.
- Conformal calibration quantiles exist with length 24.
"""

import math
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.early_fusion import EarlyFusionLSTM
from src.models.energy_only import EnergyOnlyLSTM
from src.models.hybrid_fusion import HybridFusionLSTM
from src.models.late_fusion import LateFusionLSTM


class TestModelInferenceAudit(unittest.TestCase):
    """Audit test suite for the four trained PyTorch models and scalers."""

    @classmethod
    def setUpClass(cls):
        cls.models_dir = PROJECT_ROOT / "outputs" / "models"
        cls.scalers_dir = cls.models_dir / "scalers"
        cls.uncertainty_dir = PROJECT_ROOT / "outputs" / "uncertainty" / "artifacts"

        # Check directory existence
        assert cls.models_dir.exists(), "outputs/models directory must exist"
        assert cls.scalers_dir.exists(), "outputs/models/scalers directory must exist"

    def test_scalers_and_metadata_exist(self):
        """Verify scaler artifacts and parameters."""
        energy_scaler_path = self.scalers_dir / "energy_scaler.pkl"
        weather_scaler_path = self.scalers_dir / "weather_scaler.pkl"
        meta_path = self.scalers_dir / "scaler_metadata.json"

        self.assertTrue(energy_scaler_path.exists(), "energy_scaler.pkl must exist")
        self.assertTrue(weather_scaler_path.exists(), "weather_scaler.pkl must exist")
        self.assertTrue(meta_path.exists(), "scaler_metadata.json must exist")

        e_scaler = joblib.load(energy_scaler_path)
        w_scaler = joblib.load(weather_scaler_path)

        # Energy scaler checks: 1 feature (hourly_active_power)
        self.assertEqual(len(e_scaler.mean_), 1)
        self.assertEqual(len(e_scaler.scale_), 1)
        self.assertAlmostEqual(float(e_scaler.mean_[0]), 307.17, places=1)
        self.assertAlmostEqual(float(e_scaler.scale_[0]), 147.99, places=1)

        # Weather scaler checks: 7 features
        self.assertEqual(len(w_scaler.mean_), 7)
        self.assertEqual(len(w_scaler.scale_), 7)

    def test_energy_only_model_inference(self):
        """Verify Energy-Only checkpoint loading and deterministic forward pass."""
        ckpt_path = self.models_dir / "energy_only" / "best_model.pt"
        self.assertTrue(ckpt_path.exists(), "energy_only/best_model.pt must exist")

        model = EnergyOnlyLSTM()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        self.assertEqual(model.count_parameters(), 51992)

        # Input: (batch=2, lookback=168, features=1)
        dummy_x = torch.zeros(2, 168, 1, dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_x)

        self.assertEqual(out.shape, (2, 24))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

        # Test inverse-scaling to Watts
        e_scaler = joblib.load(self.scalers_dir / "energy_scaler.pkl")
        out_watts = out.numpy() * e_scaler.scale_[0] + e_scaler.mean_[0]
        self.assertEqual(out_watts.shape, (2, 24))
        self.assertFalse(np.isnan(out_watts).any())

    def test_early_fusion_model_inference(self):
        """Verify Early Fusion checkpoint loading and deterministic forward pass."""
        ckpt_path = self.models_dir / "early_fusion" / "best_model.pt"
        self.assertTrue(ckpt_path.exists(), "early_fusion/best_model.pt must exist")

        model = EarlyFusionLSTM()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        self.assertEqual(model.count_parameters(), 53784)

        # Input: (batch=2, lookback=168, features=8)
        dummy_x = torch.zeros(2, 168, 8, dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_x)

        self.assertEqual(out.shape, (2, 24))
        self.assertFalse(torch.isnan(out).any())

    def test_late_fusion_model_inference(self):
        """Verify Late Fusion checkpoint loading and deterministic forward pass."""
        ckpt_path = self.models_dir / "late_fusion" / "best_model.pt"
        self.assertTrue(ckpt_path.exists(), "late_fusion/best_model.pt must exist")

        model = LateFusionLSTM()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        self.assertEqual(model.count_parameters(), 112216)

        # Dual inputs: energy (batch=2, 168, 1), weather (batch=2, 168, 7)
        dummy_e = torch.zeros(2, 168, 1, dtype=torch.float32)
        dummy_w = torch.zeros(2, 168, 7, dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_e, dummy_w)

        self.assertEqual(out.shape, (2, 24))
        self.assertFalse(torch.isnan(out).any())

    def test_hybrid_fusion_model_inference(self):
        """Verify Hybrid Fusion checkpoint loading and deterministic forward pass."""
        ckpt_path = self.models_dir / "hybrid_fusion" / "best_model.pt"
        self.assertTrue(ckpt_path.exists(), "hybrid_fusion/best_model.pt must exist")

        model = HybridFusionLSTM()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        self.assertEqual(model.count_parameters(), 145496)

        # Dual inputs: energy (batch=2, 168, 1), weather (batch=2, 168, 7)
        dummy_e = torch.zeros(2, 168, 1, dtype=torch.float32)
        dummy_w = torch.zeros(2, 168, 7, dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_e, dummy_w)

        self.assertEqual(out.shape, (2, 24))
        self.assertFalse(torch.isnan(out).any())

    def test_conformal_quantiles_availability(self):
        """Verify frozen conformal calibration quantiles for all four models."""
        models = ["energy_only", "early_fusion", "late_fusion", "hybrid_fusion"]
        for m in models:
            calib_art = self.uncertainty_dir / f"calibration_predictions_{m}.npz"
            self.assertTrue(calib_art.exists(), f"Calibration artifact for {m} must exist")
            with np.load(calib_art) as npz:
                self.assertIn("quantiles_q_h", npz)
                q_h = npz["quantiles_q_h"]
                self.assertEqual(q_h.shape, (24,))
                self.assertTrue(np.all(q_h > 0), f"Quantiles for {m} must be positive")
                self.assertFalse(np.isnan(q_h).any())

    def test_energy_only_control_invariance_and_batch_consistency(self):
        """
        Regression test for Step 7 statistical audit:
        1. Energy-Only model does not take weather input.
        2. Clean evaluation evaluates in batches of 64. Robustness inference must use
           the exact same batch size (64), avoiding floating-point GEMM differences.
        3. Predictions under NN, NA (missingness), and NA (noise) must match exactly
           with max absolute difference of 0.0 (bit-for-bit exact identity).
        4. Scaled predictions inverse-scaled to Watts match clean Watts within floating-point
           tolerance (< 1e-4 W).
        5. Invariance holds for any weather degradation when electricity input is unchanged.
        """
        test_seq_dir = PROJECT_ROOT / "data" / "processed" / "sequences" / "test"
        clean_npz_path = PROJECT_ROOT / "outputs" / "predictions" / "energy_only" / "test_predictions.npz"
        self.assertTrue(clean_npz_path.exists(), "Clean test_predictions.npz must exist")

        with np.load(clean_npz_path) as clean_data:
            clean_preds_scaled = clean_data["predictions_scaled"]
            clean_preds_watts = clean_data["predictions_watts"]

        with np.load(test_seq_dir / "X_energy.npz") as f_e:
            X_energy = f_e["data"]
        with np.load(test_seq_dir / "X_weather.npz") as f_w:
            X_weather = f_w["data"]

        # Load Energy-Only model
        ckpt_path = self.models_dir / "energy_only" / "best_model.pt"
        model = EnergyOnlyLSTM()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        from src.robustness.missingness_experiment import run_inference as run_inference_miss
        from src.robustness.noise_experiment import run_inference as run_inference_noise

        # 1. Clean test predictions match exactly with batch_size=64
        preds_clean_b64 = run_inference_miss(model, "energy_only", X_energy, X_weather, batch_size=64)
        max_diff_clean = float(np.max(np.abs(preds_clean_b64 - clean_preds_scaled)))
        self.assertEqual(
            max_diff_clean, 0.0,
            f"Batch-64 inference must match clean test_predictions.npz exactly, got max diff: {max_diff_clean}"
        )

        # 2. Missingness NA condition invariance
        miss_dir = PROJECT_ROOT / "data" / "processed" / "degraded" / "missingness_30"
        mask_w_path = miss_dir / "mask_weather_test.npz"
        self.assertTrue(mask_w_path.exists(), "mask_weather_test.npz must exist")
        with np.load(mask_w_path) as f_mask:
            mask_w = f_mask["data"]
        X_weather_miss = X_weather.copy()
        for i in range(len(X_weather)):
            X_weather_miss[i, mask_w[i], :] = 0.0

        preds_na_miss = run_inference_miss(model, "energy_only", X_energy, X_weather_miss, batch_size=64)
        diff_miss = float(np.max(np.abs(preds_clean_b64 - preds_na_miss)))
        self.assertEqual(
            diff_miss, 0.0,
            f"Energy-Only predictions must be identical between NN and NA missingness, got max diff: {diff_miss}"
        )

        # 3. Noise NA condition invariance
        noise_dir = PROJECT_ROOT / "data" / "processed" / "degraded" / "noise_20"
        noise_w_path = noise_dir / "noise_weather_test.npz"
        self.assertTrue(noise_w_path.exists(), "noise_weather_test.npz must exist")
        with np.load(noise_w_path) as f_noise:
            noise_w = f_noise["data"]
        X_weather_noisy = X_weather + noise_w

        preds_na_noise = run_inference_noise(model, "energy_only", X_energy, X_weather_noisy, batch_size=64)
        diff_noise = float(np.max(np.abs(preds_clean_b64 - preds_na_noise)))
        self.assertEqual(
            diff_noise, 0.0,
            f"Energy-Only predictions must be identical between NN and NA noise, got max diff: {diff_noise}"
        )

        # 4. Scaled to Watts inverse-scaling invariance
        e_scaler = joblib.load(self.scalers_dir / "energy_scaler.pkl")
        watts_clean = preds_clean_b64 * e_scaler.scale_[0] + e_scaler.mean_[0]
        diff_watts = float(np.max(np.abs(watts_clean - clean_preds_watts)))
        self.assertLess(
            diff_watts, 1e-4,
            f"Inverse-scaled Watts must match within floating-point tolerance (1e-4 W), got max diff: {diff_watts}"
        )


if __name__ == "__main__":
    unittest.main()
