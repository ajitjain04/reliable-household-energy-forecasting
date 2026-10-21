"""
Energy-Only LSTM Baseline (Step 4.1).

A lightweight LSTM model that forecasts 24 hours of household energy
consumption using only 168 hours of historical energy data.

Architecture:
    Input: (batch, 168, 1) -> LSTM(64, 2 layers) -> Linear(64, 24) -> (batch, 24)

Key design decisions:
    - No weather features (energy-only baseline)
    - Training uses MSE on scaled values
    - Early stopping on validation MSE loss (patience 7)
    - Best validation checkpoint is saved and restored
    - All evaluation metrics reported in original Watts via inverse scaling
    - Energy scaler from Step 3.8 is loaded read-only (never refit)
    - Calibration set is completely untouched
    - Test set is withheld from training/model selection; evaluated once after finalization
    - CPU-only, seed 42
"""

import json
import os
import pickle
import random
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# ============================================================================
# Dataset
# ============================================================================

class EnergyOnlyDataset(Dataset):
    """
    PyTorch Dataset for energy-only forecasting.

    Loads only X_energy and y from a partition directory.
    Does NOT load X_weather (energy-only model).

    Parameters
    ----------
    partition_dir : str
        Path to a partition directory containing X_energy.npz and y.npz.
    """

    def __init__(self, partition_dir: str):
        x_energy = np.load(os.path.join(partition_dir, "X_energy.npz"))["data"]
        y = np.load(os.path.join(partition_dir, "y.npz"))["data"]

        # Convert to float32 tensors
        self.X_energy = torch.from_numpy(x_energy).float()  # (N, 168, 1)
        self.y = torch.from_numpy(y).float()                # (N, 24)

        assert self.X_energy.shape[0] == self.y.shape[0]
        assert self.X_energy.shape[1] == 168
        assert self.X_energy.shape[2] == 1
        assert self.y.shape[1] == 24

    def __len__(self) -> int:
        return self.X_energy.shape[0]

    def __getitem__(self, idx: int):
        return self.X_energy[idx], self.y[idx]


# ============================================================================
# Model
# ============================================================================

class EnergyOnlyLSTM(nn.Module):
    """
    Energy-Only LSTM for 24-hour energy forecasting.

    Architecture:
        LSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
        -> take last timestep hidden state -> Linear(64, 24)

    Parameters
    ----------
    input_size : int
        Number of input features (1 for energy-only).
    hidden_size : int
        LSTM hidden state dimension.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (applied only if num_layers > 1).
    forecast_horizon : int
        Number of output timesteps (24).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 24,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 168, 1) — scaled energy history.

        Returns
        -------
        torch.Tensor
            Shape (batch, 24) — 24-hour scaled energy forecast.
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Project to forecast horizon
        output = self.fc(last_hidden)  # (batch, 24)
        return output

    def count_parameters(self) -> int:
        """Return the exact number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    mape_clamp_threshold: float = 10.0,
) -> dict:
    """
    Compute primary and secondary forecasting metrics in original Watts.

    Primary: MAE, RMSE, R²
    Secondary: MAPE (with low-value clamping), sMAPE

    Parameters
    ----------
    actual : np.ndarray
        Ground truth values in Watts. Shape (N, 24) or flattened.
    predicted : np.ndarray
        Predicted values in Watts. Same shape as actual.
    mape_clamp_threshold : float
        Minimum actual value (Watts) used in MAPE denominator to avoid
        division-by-near-zero. Default 10.0 W.

    Returns
    -------
    dict with keys: mae, rmse, r2, mape, smape, mape_clamp_threshold_watts
    """
    actual_flat = actual.flatten()
    predicted_flat = predicted.flatten()

    # MAE
    mae = float(np.mean(np.abs(actual_flat - predicted_flat)))

    # RMSE
    rmse = float(np.sqrt(np.mean((actual_flat - predicted_flat) ** 2)))

    # R²
    ss_res = np.sum((actual_flat - predicted_flat) ** 2)
    ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE with low-value clamping
    # Clamp actual to at least mape_clamp_threshold to avoid division by near-zero
    clamped_actual = np.maximum(np.abs(actual_flat), mape_clamp_threshold)
    mape = float(np.mean(np.abs(actual_flat - predicted_flat) / clamped_actual) * 100.0)

    # sMAPE: 200 * |actual - pred| / (|actual| + |pred|)
    # Guard against both being zero
    denominator = np.abs(actual_flat) + np.abs(predicted_flat)
    mask = denominator > 0
    if mask.any():
        smape_values = 200.0 * np.abs(actual_flat[mask] - predicted_flat[mask]) / denominator[mask]
        smape = float(np.mean(smape_values))
    else:
        smape = 0.0

    return {
        "mae_watts": round(mae, 4),
        "rmse_watts": round(rmse, 4),
        "r2": round(r2, 6),
        "mape_pct": round(mape, 4),
        "smape_pct": round(smape, 4),
        "mape_clamp_threshold_watts": mape_clamp_threshold,
    }


# ============================================================================
# Inverse scaling
# ============================================================================

def load_energy_scaler(scaler_path: str):
    """
    Load the energy scaler from Step 3.8 (read-only).

    Parameters
    ----------
    scaler_path : str
        Path to energy_scaler.pkl.

    Returns
    -------
    sklearn.preprocessing.StandardScaler
    """
    try:
        import joblib
        return joblib.load(scaler_path)
    except Exception:
        with open(scaler_path, "rb") as f:
            return pickle.load(f)


def inverse_transform_energy(
    scaled_values: np.ndarray,
    scaler_mean: float,
    scaler_std: float,
) -> np.ndarray:
    """
    Inverse-transform scaled energy values back to original Watts.

    Uses the mean and std from the training-fitted scaler directly,
    avoiding any refit or modification.

    Parameters
    ----------
    scaled_values : np.ndarray
        Scaled values (any shape).
    scaler_mean : float
        Mean from the energy scaler.
    scaler_std : float
        Standard deviation from the energy scaler.

    Returns
    -------
    np.ndarray in original Watts.
    """
    return scaled_values * scaler_std + scaler_mean


# ============================================================================
# Training
# ============================================================================

def train_energy_only(
    train_dir: str,
    val_dir: str,
    output_dir: str,
    scaler_metadata_path: str,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    max_epochs: int = 50,
    patience: int = 7,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Train the Energy-Only LSTM baseline.

    Parameters
    ----------
    train_dir : str
        Path to train sequences directory.
    val_dir : str
        Path to validation sequences directory.
    output_dir : str
        Path to save model checkpoint, history, and metadata.
    scaler_metadata_path : str
        Path to scaler_metadata.json (read-only).
    hidden_size, num_layers, dropout : model architecture params.
    batch_size, learning_rate, max_epochs, patience : training params.
    seed : int
        Random seed.
    device : str
        Device string ('cpu').

    Returns
    -------
    dict with training results (best_epoch, best_val_loss, param_count, history).
    """
    print("=" * 60)
    print("STEP 4.1: Energy-Only LSTM Baseline — Training")
    print("=" * 60)

    # Set reproducibility
    set_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load datasets
    print("\n--- Loading data ---")
    train_dataset = EnergyOnlyDataset(train_dir)
    val_dataset = EnergyOnlyDataset(val_dir)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # DataLoaders (no shuffling for reproducibility with deterministic algorithms)
    # Using a generator for reproducible worker behavior
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Build model
    print("\n--- Building model ---")
    model = EnergyOnlyLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        forecast_horizon=24,
    )
    model = model.to(device)
    param_count = model.count_parameters()
    print(f"  Architecture: LSTM({hidden_size}, {num_layers} layers, dropout={dropout}) -> Linear({hidden_size}, 24)")
    print(f"  Exact trainable parameters: {param_count:,}")

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    print(f"\n--- Training (max {max_epochs} epochs, patience {patience}) ---")
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_state_dict = None

    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses))

        history["train_loss"].append(round(avg_train_loss, 6))
        history["val_loss"].append(round(avg_val_loss, 6))

        # ---- Early stopping check ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            epochs_without_improvement += 1
            marker = ""

        print(
            f"  Epoch {epoch:3d}/{max_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}{marker}"
        )

        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s")
    print(f"  Best epoch: {best_epoch} (val loss: {best_val_loss:.6f})")

    # Restore best checkpoint
    print("\n--- Restoring best validation checkpoint ---")
    model.load_state_dict(best_state_dict)

    # Save best checkpoint
    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "input_size": 1,
            "forecast_horizon": 24,
        },
        checkpoint_path,
    )
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved: {history_path}")

    # Save model metadata
    metadata = {
        "model_name": "EnergyOnlyLSTM",
        "step": "4.1",
        "architecture": {
            "input_size": 1,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "forecast_horizon": 24,
        },
        "exact_trainable_parameters": param_count,
        "training": {
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss_function": "MSELoss",
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "seed": seed,
            "device": device,
            "total_epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "training_time_seconds": round(elapsed, 1),
        },
        "data": {
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "history_length": 168,
            "forecast_horizon": 24,
            "features_used": ["hourly_active_power"],
            "weather_used": False,
        },
        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Model metadata saved: {metadata_path}")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "param_count": param_count,
        "history": history,
        "elapsed": elapsed,
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_partition(
    model: EnergyOnlyLSTM,
    partition_dir: str,
    partition_name: str,
    predictions_dir: str,
    scaler_mean: float,
    scaler_std: float,
    mape_clamp_threshold: float = 10.0,
    batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """
    Evaluate the model on a partition and save predictions.

    Parameters
    ----------
    model : EnergyOnlyLSTM
        Trained model with best checkpoint loaded.
    partition_dir : str
        Path to partition sequences directory.
    partition_name : str
        Name of the partition ('validation' or 'test').
    predictions_dir : str
        Path to save prediction files.
    scaler_mean : float
        Energy scaler mean (from Step 3.8, read-only).
    scaler_std : float
        Energy scaler std (from Step 3.8, read-only).
    mape_clamp_threshold : float
        MAPE clamping threshold in Watts.
    batch_size : int
        Batch size for inference.
    device : str
        Device ('cpu').

    Returns
    -------
    dict with metrics and prediction shapes.
    """
    print(f"\n--- Evaluating on {partition_name} set ---")

    dataset = EnergyOnlyDataset(partition_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"  Samples: {len(dataset)}")

    # Inference
    model.eval()
    all_preds_scaled = []
    all_actual_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds_scaled.append(preds.cpu().numpy())
            all_actual_scaled.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)   # (N, 24)
    actual_scaled = np.concatenate(all_actual_scaled, axis=0)  # (N, 24)

    print(f"  Predictions shape: {preds_scaled.shape}")
    assert preds_scaled.shape == actual_scaled.shape
    assert not np.isnan(preds_scaled).any(), "NaN in predictions!"
    assert not np.isinf(preds_scaled).any(), "Inf in predictions!"

    # Inverse transform to Watts
    preds_watts = inverse_transform_energy(preds_scaled, scaler_mean, scaler_std)
    actual_watts = inverse_transform_energy(actual_scaled, scaler_mean, scaler_std)

    print(f"  Actual range (Watts): [{actual_watts.min():.1f}, {actual_watts.max():.1f}]")
    print(f"  Predicted range (Watts): [{preds_watts.min():.1f}, {preds_watts.max():.1f}]")

    # Compute metrics
    metrics = compute_metrics(actual_watts, preds_watts, mape_clamp_threshold)
    print(f"  MAE:   {metrics['mae_watts']:.2f} W")
    print(f"  RMSE:  {metrics['rmse_watts']:.2f} W")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  MAPE:  {metrics['mape_pct']:.2f}%")
    print(f"  sMAPE: {metrics['smape_pct']:.2f}%")

    # Save predictions
    os.makedirs(predictions_dir, exist_ok=True)
    pred_filename = f"{partition_name.replace('validation', 'val')}_predictions.npz"
    pred_path = os.path.join(predictions_dir, pred_filename)
    np.savez_compressed(
        pred_path,
        predictions_watts=preds_watts,
        actual_watts=actual_watts,
        predictions_scaled=preds_scaled,
        actual_scaled=actual_scaled,
    )
    print(f"  Predictions saved: {pred_path}")

    return {
        "partition": partition_name,
        "samples": len(dataset),
        "prediction_shape": list(preds_scaled.shape),
        "metrics": metrics,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_training_curves(
    history: dict,
    best_epoch: int,
    output_dir: str,
) -> None:
    """
    Plot training vs validation loss curves.

    Parameters
    ----------
    history : dict
        Contains 'train_loss' and 'val_loss' lists.
    best_epoch : int
        Best epoch (1-indexed).
    output_dir : str
        Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", linewidth=1.5, label="Train Loss (MSE)")
    ax.plot(epochs, history["val_loss"], "r-", linewidth=1.5, label="Validation Loss (MSE)")

    # Mark best epoch
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax.scatter(
        [best_epoch],
        [history["val_loss"][best_epoch - 1]],
        color="green",
        s=80,
        zorder=5,
        marker="*",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss (scaled)", fontsize=12)
    ax.set_title("Energy-Only LSTM — Training Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved: {save_path}")


def plot_test_diagnostics(
    predictions_path: str,
    output_dir: str,
) -> None:
    """
    Generate diagnostic plots from the test set predictions.

    Produces:
        - residual_histogram.png: distribution of residuals (Watts)
        - actual_vs_predicted.png: scatter plot (Watts)
        - sample_forecasts.png: 3 sample 24-hour forecast windows

    Parameters
    ----------
    predictions_path : str
        Path to test_predictions.npz.
    output_dir : str
        Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(predictions_path)
    actual = data["actual_watts"]      # (N, 24)
    predicted = data["predictions_watts"]  # (N, 24)

    n_samples = actual.shape[0]

    # ---- 1. Residual histogram ----
    residuals = (actual - predicted).flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(
        x=np.mean(residuals), color="orange", linestyle="--", linewidth=1.5,
        alpha=0.7, label=f"Mean: {np.mean(residuals):.1f} W",
    )
    ax.set_xlabel("Residual (Actual − Predicted) [Watts]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Energy-Only LSTM — Test Set Residual Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "residual_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. Actual vs Predicted scatter ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual.flatten(), predicted.flatten(), alpha=0.05, s=5, color="steelblue")
    lims = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect forecast")
    ax.set_xlabel("Actual Power [Watts]", fontsize=12)
    ax.set_ylabel("Predicted Power [Watts]", fontsize=12)
    ax.set_title("Energy-Only LSTM — Test Set: Actual vs Predicted", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 3. Sample forecasts (first, middle, last) ----
    sample_indices = [0, n_samples // 2, n_samples - 1]
    sample_labels = ["First", "Middle", "Last"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    hours = np.arange(1, 25)

    for ax, idx, label in zip(axes, sample_indices, sample_labels):
        ax.plot(hours, actual[idx], "b-o", markersize=4, linewidth=1.5, label="Actual")
        ax.plot(hours, predicted[idx], "r--s", markersize=4, linewidth=1.5, label="Predicted")
        ax.set_ylabel("Power [Watts]", fontsize=11)
        ax.set_title(f"{label} Test Window (sample {idx})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Forecast Hour (t+1 to t+24)", fontsize=12)
    fig.suptitle("Energy-Only LSTM — Sample 24h Test Forecasts", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "sample_forecasts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Diagnostic plots saved to: {output_dir}")


# ============================================================================
# Report
# ============================================================================

def generate_baseline_report(
    model_metadata: dict,
    val_results: dict,
    test_results: dict,
    history: dict,
    output_path: str,
) -> None:
    """
    Generate the comprehensive Energy-Only baseline report.

    Parameters
    ----------
    model_metadata : dict
        Model metadata from training.
    val_results : dict
        Validation evaluation results.
    test_results : dict
        Test evaluation results.
    history : dict
        Training history (train_loss, val_loss).
    output_path : str
        Path to save the report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    arch = model_metadata["architecture"]
    train_cfg = model_metadata["training"]
    data_cfg = model_metadata["data"]
    val_m = val_results["metrics"]
    test_m = test_results["metrics"]

    report_lines = [
        "=" * 70,
        "ENERGY-ONLY LSTM BASELINE REPORT",
        "Step 4.1 — Reliable Multimodal Household Energy Forecasting",
        "=" * 70,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 70,
        "1. ARCHITECTURE",
        "-" * 70,
        "",
        f"  Model:              EnergyOnlyLSTM",
        f"  Input:              (batch, 168, {arch['input_size']})",
        f"  LSTM hidden size:   {arch['hidden_size']}",
        f"  LSTM layers:        {arch['num_layers']}",
        f"  LSTM dropout:       {arch['dropout']}",
        f"  Output head:        Linear({arch['hidden_size']}, {arch['forecast_horizon']})",
        f"  Output:             (batch, {arch['forecast_horizon']})",
        f"  Trainable params:   {model_metadata['exact_trainable_parameters']:,}",
        f"  Features used:      {data_cfg['features_used']}",
        f"  Weather used:       {data_cfg['weather_used']}",
        "",
        "-" * 70,
        "2. TRAINING CONFIGURATION",
        "-" * 70,
        "",
        f"  Optimizer:          {train_cfg['optimizer']}",
        f"  Learning rate:      {train_cfg['learning_rate']}",
        f"  Loss function:      {train_cfg['loss_function']} (on scaled values)",
        f"  Batch size:         {train_cfg['batch_size']}",
        f"  Max epochs:         {train_cfg['max_epochs']}",
        f"  Early stopping:     patience {train_cfg['early_stopping_patience']}",
        f"  Seed:               {train_cfg['seed']}",
        f"  Device:             {train_cfg['device']}",
        "",
        f"  Train samples:      {data_cfg['train_samples']:,}",
        f"  Validation samples: {data_cfg['validation_samples']:,}",
        f"  History length:     {data_cfg['history_length']} hours",
        f"  Forecast horizon:   {data_cfg['forecast_horizon']} hours",
        "",
        "-" * 70,
        "3. TRAINING RESULTS",
        "-" * 70,
        "",
        f"  Total epochs run:   {train_cfg['total_epochs_run']}",
        f"  Best epoch:         {train_cfg['best_epoch']}",
        f"  Best val loss:      {train_cfg['best_val_loss']:.6f} (MSE, scaled)",
        f"  Training time:      {train_cfg['training_time_seconds']:.1f}s",
        "",
        f"  Final train loss:   {history['train_loss'][-1]:.6f}",
        f"  Final val loss:     {history['val_loss'][-1]:.6f}",
        "",
        "-" * 70,
        "4. VALIDATION METRICS (in original Watts)",
        "-" * 70,
        "",
        "  Primary metrics:",
        f"    MAE:              {val_m['mae_watts']:.2f} W",
        f"    RMSE:             {val_m['rmse_watts']:.2f} W",
        f"    R²:               {val_m['r2']:.4f}",
        "",
        "  Secondary metrics:",
        f"    MAPE:             {val_m['mape_pct']:.2f}%",
        f"    sMAPE:            {val_m['smape_pct']:.2f}%",
        "",
        "-" * 70,
        "5. TEST METRICS — FINAL BASELINE RESULTS (in original Watts)",
        "-" * 70,
        "",
        "  Primary metrics:",
        f"    MAE:              {test_m['mae_watts']:.2f} W",
        f"    RMSE:             {test_m['rmse_watts']:.2f} W",
        f"    R²:               {test_m['r2']:.4f}",
        "",
        "  Secondary metrics:",
        f"    MAPE:             {test_m['mape_pct']:.2f}%",
        f"    sMAPE:            {test_m['smape_pct']:.2f}%",
        "",
        "-" * 70,
        "6. MAPE LOW-VALUE HANDLING",
        "-" * 70,
        "",
        f"  MAPE clamping threshold: {val_m['mape_clamp_threshold_watts']} W",
        "",
        "  Rule: When computing MAPE, actual values below the clamping",
        "  threshold are clamped to the threshold before division.",
        "  This prevents inflation of percentage errors during hours",
        "  of very low consumption (e.g., standby power).",
        "",
        "  Formula: MAPE = mean(|actual - pred| / max(|actual|, threshold)) × 100",
        "",
        "  This rule is applied consistently across all model evaluations",
        "  in this project.",
        "",
        "-" * 70,
        "7. DATA INTEGRITY STATEMENT",
        "-" * 70,
        "",
        "  TRAINING SET (3,816 windows):",
        "    Used for model parameter optimization.",
        "",
        "  VALIDATION SET (881 windows):",
        "    Used ONLY for early stopping and best checkpoint selection.",
        "    NOT used for gradient updates.",
        "",
        "  CALIBRATION SET (587 windows):",
        "    COMPLETELY UNTOUCHED in Step 4.1.",
        "    Not loaded, not evaluated, not used in any way.",
        "    Reserved for conformal prediction in a later step.",
        "",
        "  TEST SET (587 windows):",
        "    WITHHELD from training and model selection.",
        "    Evaluated ONCE after the model was fully finalized",
        "    (best checkpoint restored from validation-based early stopping).",
        "    Results reported in Section 5 above.",
        "",
        "  SCALER:",
        "    Energy StandardScaler from Step 3.8 loaded read-only.",
        "    Mean: 307.1706 W, Std: 147.9934 W.",
        "    Fitted on training partition only. Never refit or modified.",
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"  Baseline report saved: {output_path}")


# ============================================================================
# Main pipeline
# ============================================================================

def run_step41(project_root: str) -> None:
    """
    Execute the full Step 4.1 pipeline.

    1. Train the model (train + validation; early stopping on val loss)
    2. Restore best validation checkpoint
    3. Evaluate on validation set (881 windows)
    4. Evaluate on test set (587 windows) — one-time final evaluation
    5. Generate diagnostic plots (from test set)
    6. Generate baseline report (validation + test metrics)
    7. Print summary

    Parameters
    ----------
    project_root : str
        Path to the project root directory.
    """
    # Paths
    seq_dir = os.path.join(project_root, "data", "processed", "sequences")
    train_dir = os.path.join(seq_dir, "train")
    val_dir = os.path.join(seq_dir, "validation")
    test_dir = os.path.join(seq_dir, "test")
    scaler_dir = os.path.join(project_root, "outputs", "models", "scalers")
    model_output_dir = os.path.join(project_root, "outputs", "models", "energy_only")
    predictions_dir = os.path.join(project_root, "outputs", "predictions", "energy_only")
    figures_dir = os.path.join(project_root, "outputs", "figures", "energy_only")
    reports_dir = os.path.join(project_root, "outputs", "reports")

    # Load scaler metadata (read-only)
    scaler_meta_path = os.path.join(scaler_dir, "scaler_metadata.json")
    with open(scaler_meta_path, "r", encoding="utf-8") as f:
        scaler_meta = json.load(f)

    scaler_mean = scaler_meta["energy_scaler"]["means"]["hourly_active_power"]
    scaler_std = scaler_meta["energy_scaler"]["standard_deviations"]["hourly_active_power"]
    print(f"Energy scaler (read-only): mean={scaler_mean:.4f}, std={scaler_std:.4f}")

    # 1. Train
    train_results = train_energy_only(
        train_dir=train_dir,
        val_dir=val_dir,
        output_dir=model_output_dir,
        scaler_metadata_path=scaler_meta_path,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=50,
        patience=7,
        seed=42,
        device="cpu",
    )

    model = train_results["model"]

    # 2. Evaluate on validation set
    val_results = evaluate_partition(
        model=model,
        partition_dir=val_dir,
        partition_name="validation",
        predictions_dir=predictions_dir,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        mape_clamp_threshold=10.0,
        batch_size=64,
        device="cpu",
    )

    # 3. Evaluate on test set (one-time final evaluation)
    test_results = evaluate_partition(
        model=model,
        partition_dir=test_dir,
        partition_name="test",
        predictions_dir=predictions_dir,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        mape_clamp_threshold=10.0,
        batch_size=64,
        device="cpu",
    )

    # 4. Save combined evaluation metrics
    eval_metrics = {
        "model": "EnergyOnlyLSTM",
        "step": "4.1",
        "validation": val_results,
        "test": test_results,
        "mape_low_value_rule": {
            "description": "Actual values below threshold are clamped before MAPE division",
            "threshold_watts": 10.0,
        },
    }

    eval_metrics_path = os.path.join(model_output_dir, "evaluation_metrics.json")
    with open(eval_metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"\n  Combined evaluation metrics saved: {eval_metrics_path}")

    # 5. Generate plots
    print("\n--- Generating plots ---")
    plot_training_curves(
        history=train_results["history"],
        best_epoch=train_results["best_epoch"],
        output_dir=figures_dir,
    )
    plot_test_diagnostics(
        predictions_path=os.path.join(predictions_dir, "test_predictions.npz"),
        output_dir=figures_dir,
    )

    # 6. Generate report
    print("\n--- Generating baseline report ---")
    model_meta_path = os.path.join(model_output_dir, "model_metadata.json")
    with open(model_meta_path, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    generate_baseline_report(
        model_metadata=model_metadata,
        val_results=val_results,
        test_results=test_results,
        history=train_results["history"],
        output_path=os.path.join(reports_dir, "energy_only_baseline_report.txt"),
    )

    # 7. Print summary
    print("\n" + "=" * 60)
    print("STEP 4.1 COMPLETE — Energy-Only LSTM Baseline")
    print("=" * 60)
    print(f"  Trainable parameters: {train_results['param_count']:,}")
    print(f"  Best epoch:           {train_results['best_epoch']}")
    print(f"  Best val loss (MSE):  {train_results['best_val_loss']:.6f}")
    print(f"")
    print(f"  VALIDATION  |  MAE: {val_results['metrics']['mae_watts']:.2f} W  |  RMSE: {val_results['metrics']['rmse_watts']:.2f} W  |  R²: {val_results['metrics']['r2']:.4f}")
    print(f"  TEST        |  MAE: {test_results['metrics']['mae_watts']:.2f} W  |  RMSE: {test_results['metrics']['rmse_watts']:.2f} W  |  R²: {test_results['metrics']['r2']:.4f}")
    print(f"")
    print(f"  Calibration set: UNTOUCHED")
    print("=" * 60)


if __name__ == "__main__":
    run_step41(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
