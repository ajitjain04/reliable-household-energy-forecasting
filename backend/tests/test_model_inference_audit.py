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
            npz = np.load(calib_art)
            self.assertIn("quantiles_q_h", npz)
            q_h = npz["quantiles_q_h"]
            self.assertEqual(q_h.shape, (24,))
            self.assertTrue(np.all(q_h > 0), f"Quantiles for {m} must be positive")
            self.assertFalse(np.isnan(q_h).any())


if __name__ == "__main__":
    unittest.main()
