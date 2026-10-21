"""
Live Forecast Inference Service.

Implements the complete end-to-end inference path:
  Raw user history (168h) -> Input Validation -> Normalization -> Scaling
  -> PyTorch Model Forward Pass -> Inverse-Scaling to Watts -> Conformal Prediction Intervals
  -> Structured API Response.

Adheres strictly to the audited Step 1 findings:
- Reuses frozen PyTorch checkpoints on CPU without retraining.
- Reuses frozen StandardScaler objects without refitting.
- Enforces exact feature ordering and 168-hour history length.
- Applies frozen split-conformal calibration quantiles (90% nominal coverage).
- Isolates automatic model selection with transparent fallback rationale.
"""

import math
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException

from backend.app.core.config import settings
from backend.app.schemas.forecasts import (
    ConformalIntervalStep,
    ForecastRunRequest,
    ForecastRunResponse,
    LocationInput,
    ModelMetadataInfo,
    WeatherSeriesInput,
)
from backend.app.services.weather_service import weather_service
from src.models.early_fusion import EarlyFusionLSTM
from src.models.energy_only import EnergyOnlyLSTM
from src.models.hybrid_fusion import HybridFusionLSTM
from src.models.late_fusion import LateFusionLSTM


# Approved model display names and architectures
APPROVED_MODELS = {
    "energy_only": {
        "display_name": "Energy-Only",
        "class": EnergyOnlyLSTM,
        "requires_weather": False,
        "input_features": 1,
        "trainable_parameters": 51992,
        "checkpoint_folder": "energy_only",
    },
    "early_fusion": {
        "display_name": "Early Fusion",
        "class": EarlyFusionLSTM,
        "requires_weather": True,
        "input_features": 8,
        "trainable_parameters": 53784,
        "checkpoint_folder": "early_fusion",
    },
    "late_fusion": {
        "display_name": "Late Fusion",
        "class": LateFusionLSTM,
        "requires_weather": True,
        "input_features": 8,
        "trainable_parameters": 112216,
        "checkpoint_folder": "late_fusion",
    },
    "hybrid_fusion": {
        "display_name": "Hybrid Fusion",
        "class": HybridFusionLSTM,
        "requires_weather": True,
        "input_features": 8,
        "trainable_parameters": 145496,
        "checkpoint_folder": "hybrid_fusion",
    },
}

AUDITED_WEATHER_FEATURES = [
    "temp",
    "feels_like",
    "humidity",
    "pressure",
    "speed",
    "wind_sin",
    "wind_cos",
]


class ModelRegistry:
    """Singleton registry managing in-memory PyTorch models, scalers, and conformal quantiles."""

    def __init__(self):
        self._models: Dict[str, torch.nn.Module] = {}
        self._energy_scaler: Optional[Any] = None
        self._weather_scaler: Optional[Any] = None
        self._conformal_quantiles: Dict[str, np.ndarray] = {}

    def get_model(self, model_key: str) -> torch.nn.Module:
        """Load and cache an approved PyTorch model on CPU in eval mode."""
        if model_key in self._models:
            return self._models[model_key]

        if model_key not in APPROVED_MODELS:
            raise HTTPException(
                status_code=422,
                detail=f"Model '{model_key}' is not approved. Approved models: Energy-Only, Early Fusion, Late Fusion, Hybrid Fusion.",
            )

        spec = APPROVED_MODELS[model_key]
        ckpt_path = settings.MODELS_DIR / spec["checkpoint_folder"] / "best_model.pt"

        if not ckpt_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Checkpoint artifact for model '{spec['display_name']}' is missing.",
            )

        try:
            model = spec["class"]()
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()
            self._models[model_key] = model
            return model
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize neural network for '{spec['display_name']}': {str(e)}",
            )

    def get_scalers(self) -> Tuple[Any, Any]:
        """Load and cache frozen StandardScaler objects."""
        if self._energy_scaler is not None and self._weather_scaler is not None:
            return self._energy_scaler, self._weather_scaler

        energy_path = settings.SCALERS_DIR / "energy_scaler.pkl"
        weather_path = settings.SCALERS_DIR / "weather_scaler.pkl"

        if not energy_path.exists() or not weather_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Fitted numerical scaler artifacts are missing from project storage.",
            )

        try:
            self._energy_scaler = joblib.load(energy_path)
            self._weather_scaler = joblib.load(weather_path)
            return self._energy_scaler, self._weather_scaler
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load numerical feature scalers: {str(e)}",
            )

    def get_conformal_quantiles(self, model_key: str) -> np.ndarray:
        """Load and cache the 24 frozen calibration quantiles for split conformal intervals."""
        if model_key in self._conformal_quantiles:
            return self._conformal_quantiles[model_key]

        art_path = settings.UNCERTAINTY_ARTIFACTS_DIR / f"calibration_predictions_{model_key}.npz"
        if not art_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Conformal calibration quantile artifact for '{model_key}' is missing.",
            )

        try:
            npz = np.load(art_path)
            if "quantiles_q_h" not in npz:
                raise KeyError("Missing 'quantiles_q_h' in calibration archive.")
            q_h = npz["quantiles_q_h"]
            if q_h.shape != (24,):
                raise ValueError(f"Expected 24 conformal quantiles, got shape {q_h.shape}")
            self._conformal_quantiles[model_key] = q_h
            return q_h
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load conformal calibration quantiles for '{model_key}': {str(e)}",
            )


# Global registry instance
model_registry = ModelRegistry()


class LiveInferenceService:
    """Service handling validation, preprocessing, prediction, and uncertainty quantification."""

    def __init__(self, registry: ModelRegistry = model_registry):
        self.registry = registry

    # -------------------------------------------------------------------------
    # 1. Input Validation and Normalization
    # -------------------------------------------------------------------------
    def validate_electricity(self, values: List[float]) -> np.ndarray:
        """Validate 168 consecutive hourly active power observations."""
        if not isinstance(values, list):
            raise HTTPException(
                status_code=422,
                detail="Electricity history must be provided as a list of numbers.",
            )

        if len(values) != 168:
            raise HTTPException(
                status_code=422,
                detail=f"Electricity history must contain exactly 168 hourly active power values (received {len(values)}).",
            )

        clean_floats = []
        for i, val in enumerate(values):
            if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise HTTPException(
                    status_code=422,
                    detail=f"Electricity history contains non-finite or invalid numeric value at index {i}.",
                )
            clean_floats.append(float(val))

        return np.array(clean_floats, dtype=np.float32).reshape(168, 1)

    def validate_timestamps(
        self, timestamps: Optional[List[str]]
    ) -> Tuple[Optional[str], Optional[List[str]]]:
        """Validate optional 168 ISO timestamps and generate 24 future horizon timestamps."""
        if timestamps is None:
            return None, None

        if not isinstance(timestamps, list) or len(timestamps) != 168:
            raise HTTPException(
                status_code=422,
                detail=f"Timestamps list must contain exactly 168 values corresponding to the lookback history (received {len(timestamps) if isinstance(timestamps, list) else 'invalid'}).",
            )

        try:
            parsed = pd.to_datetime(timestamps, errors="coerce")
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="Timestamps list contains invalid datetime formats.",
            )

        if parsed.isna().any():
            first_bad = int(np.where(parsed.isna())[0][0])
            raise HTTPException(
                status_code=422,
                detail=f"Unparseable datetime string at index {first_bad}: '{timestamps[first_bad]}'.",
            )

        # Validate strictly consecutive 1-hour intervals
        diffs = pd.Series(parsed).diff()[1:]
        expected_hour = pd.Timedelta(hours=1)
        irregular_indices = diffs[diffs != expected_hour].index.tolist()

        if irregular_indices:
            bad_idx = irregular_indices[0]
            actual_delta = diffs[bad_idx]
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Timestamps must be consecutive hourly observations spaced by exactly 1 hour. "
                    f"Irregular interval detected at index {bad_idx} (timestamp '{timestamps[bad_idx]}'): delta was {actual_delta}."
                ),
            )

        origin_ts = timestamps[-1]
        last_dt = parsed[-1]
        forecast_ts = [
            (last_dt + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
            for h in range(1, 25)
        ]

        return origin_ts, forecast_ts

    def validate_weather(
        self,
        weather_input: Optional[Union[WeatherSeriesInput, Dict[str, Any], List[Dict[str, Any]]]],
    ) -> Optional[np.ndarray]:
        """Validate weather observations and construct the 7-feature matrix in audited order."""
        if weather_input is None:
            return None

        col_dict: Dict[str, List[float]] = {}

        if isinstance(weather_input, WeatherSeriesInput):
            col_dict["temp"] = weather_input.temp
            col_dict["feels_like"] = weather_input.feels_like
            col_dict["humidity"] = weather_input.humidity
            col_dict["pressure"] = weather_input.pressure
            col_dict["speed"] = weather_input.speed
            if weather_input.wind_sin is not None and weather_input.wind_cos is not None:
                col_dict["wind_sin"] = weather_input.wind_sin
                col_dict["wind_cos"] = weather_input.wind_cos
            elif weather_input.deg is not None:
                col_dict["deg"] = weather_input.deg

        elif isinstance(weather_input, dict):
            col_dict = weather_input

        elif isinstance(weather_input, list):
            # List of 168 row dictionaries
            if len(weather_input) != 168:
                raise HTTPException(
                    status_code=422,
                    detail=f"Weather row record list must contain exactly 168 hourly entries (received {len(weather_input)}).",
                )
            for key in ["temp", "feels_like", "humidity", "pressure", "speed", "wind_sin", "wind_cos", "deg"]:
                if any(key in row for row in weather_input):
                    col_dict[key] = [row.get(key) for row in weather_input]

        # Verify required primary weather features
        required_numeric = ["temp", "feels_like", "humidity", "pressure", "speed"]
        missing_keys = [k for k in required_numeric if k not in col_dict or col_dict[k] is None]
        if missing_keys:
            raise HTTPException(
                status_code=422,
                detail=f"Weather history is missing required features: {missing_keys}.",
            )

        # Check wind components
        has_sin_cos = (
            "wind_sin" in col_dict
            and "wind_cos" in col_dict
            and col_dict["wind_sin"] is not None
            and col_dict["wind_cos"] is not None
        )
        has_deg = "deg" in col_dict and col_dict["deg"] is not None

        if not has_sin_cos and not has_deg:
            raise HTTPException(
                status_code=422,
                detail="Weather history must include either ('wind_sin' and 'wind_cos') or wind direction 'deg'.",
            )

        # Ensure all columns have length 168
        for k in required_numeric:
            series = col_dict[k]
            if not isinstance(series, list) or len(series) != 168:
                raise HTTPException(
                    status_code=422,
                    detail=f"Weather feature '{k}' must contain exactly 168 hourly values (received {len(series) if isinstance(series, list) else 'invalid'}).",
                )

        # Build wind_sin and wind_cos
        if has_sin_cos:
            wind_sin = col_dict["wind_sin"]
            wind_cos = col_dict["wind_cos"]
            if len(wind_sin) != 168 or len(wind_cos) != 168:
                raise HTTPException(
                    status_code=422,
                    detail="Wind components ('wind_sin', 'wind_cos') must each contain exactly 168 values.",
                )
        else:
            deg = col_dict["deg"]
            if len(deg) != 168:
                raise HTTPException(
                    status_code=422,
                    detail="Wind direction 'deg' must contain exactly 168 hourly values.",
                )
            rad = np.deg2rad(np.array(deg, dtype=np.float32))
            wind_sin = np.sin(rad).tolist()
            wind_cos = np.cos(rad).tolist()

        # Assemble the 7 columns in exact audited order
        matrix_cols = [
            col_dict["temp"],
            col_dict["feels_like"],
            col_dict["humidity"],
            col_dict["pressure"],
            col_dict["speed"],
            wind_sin,
            wind_cos,
        ]

        matrix = np.zeros((168, 7), dtype=np.float32)
        for col_idx, col_vals in enumerate(matrix_cols):
            for row_idx, val in enumerate(col_vals):
                if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Weather feature '{AUDITED_WEATHER_FEATURES[col_idx]}' contains invalid or "
                            f"non-finite numeric value at hour {row_idx}."
                        ),
                    )
                matrix[row_idx, col_idx] = float(val)

        return matrix

    # -------------------------------------------------------------------------
    # 2. Model Selection Policy (Isolated & Conservative)
    # -------------------------------------------------------------------------
    def resolve_model(
        self,
        requested_model: Optional[str],
        has_weather: bool,
        auto_weather_location: Optional[LocationInput] = None,
        auto_weather_error: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Determine model according to conservative baseline policy.

        Returns:
            (model_key, selection_mode, selection_reason)
        """
        raw_key = (requested_model or "auto").strip().lower().replace("-", "_").replace(" ", "_")

        if raw_key in ["auto", ""]:
            if has_weather:
                if auto_weather_location is not None:
                    loc_desc = (
                        f"'{auto_weather_location.label}' ({auto_weather_location.latitude:.4f}, {auto_weather_location.longitude:.4f})"
                        if auto_weather_location.label
                        else f"coordinates ({auto_weather_location.latitude:.4f}, {auto_weather_location.longitude:.4f})"
                    )
                    return (
                        "early_fusion",
                        "auto",
                        f"Complete weather history automatically retrieved from Open-Meteo for {loc_desc}. "
                        "Early Fusion selected as initial default based on best clean test MAE (71.64 W) in research benchmarks.",
                    )
                return (
                    "early_fusion",
                    "auto",
                    "Complete valid weather history provided. Early Fusion selected as initial default "
                    "based on best clean test MAE (71.64 W) in research benchmarks. Note: clean-test "
                    "performance does not guarantee optimal performance for every new household.",
                )
            else:
                if auto_weather_error is not None:
                    return (
                        "energy_only",
                        "fallback",
                        f"Automated weather retrieval for household location failed: {auto_weather_error}. "
                        "Gracefully fell back to Energy-Only model because it operates exclusively on historical active power.",
                    )
                return (
                    "energy_only",
                    "auto",
                    "No weather history provided. Energy-Only selected because it operates exclusively "
                    "on historical active power without requiring weather observations.",
                )

        # Explicit override
        if raw_key not in APPROVED_MODELS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Requested model '{requested_model}' is not approved. "
                    "Approved models: Energy-Only, Early Fusion, Late Fusion, Hybrid Fusion."
                ),
            )

        spec = APPROVED_MODELS[raw_key]
        if spec["requires_weather"] and not has_weather:
            if auto_weather_error is not None:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"Model '{spec['display_name']}' requires 168 hours of weather history. "
                        f"Automatic weather retrieval for household location failed: {auto_weather_error}. "
                        "Provide manual weather observations or use Energy-Only."
                    ),
                )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model '{spec['display_name']}' requires 168 hours of weather history, "
                    "but no weather observations were provided. Provide complete weather data or use Energy-Only."
                ),
            )

        if auto_weather_location is not None:
            return (
                raw_key,
                "override",
                f"Authorized explicit model override: {spec['display_name']} with automatically retrieved weather.",
            )

        return (
            raw_key,
            "override",
            f"Authorized explicit model override: {spec['display_name']}.",
        )

    # -------------------------------------------------------------------------
    # 3. Prediction Execution & Conformal Quantification
    # -------------------------------------------------------------------------
    def run_forecast(
        self, request: ForecastRunRequest, run_id: Optional[str] = None
    ) -> ForecastRunResponse:
        """Execute end-to-end live forecast pipeline."""
        # 1. Validation
        electricity_raw = self.validate_electricity(request.electricity_active_power)
        origin_ts, forecast_ts = self.validate_timestamps(request.timestamps)
        weather_raw = self.validate_weather(request.weather)

        auto_weather_location: Optional[LocationInput] = None
        auto_weather_error: Optional[str] = None

        # Automatic weather retrieval if weather not supplied by user but location is provided
        if weather_raw is None and request.location is not None:
            if request.timestamps and len(request.timestamps) == 168:
                fetched_weather, fetch_err = weather_service.fetch_weather_for_timestamps(
                    latitude=request.location.latitude,
                    longitude=request.location.longitude,
                    timestamps=request.timestamps,
                )
                if fetched_weather is not None:
                    weather_raw = self.validate_weather(fetched_weather)
                    auto_weather_location = request.location
                else:
                    auto_weather_error = fetch_err
            else:
                auto_weather_error = (
                    "Timestamps missing or incomplete; exactly 168 hourly timestamps are required "
                    "to determine the historical weather window."
                )

        has_weather = weather_raw is not None

        # 2. Model Selection
        model_key, selection_mode, selection_reason = self.resolve_model(
            request.model,
            has_weather,
            auto_weather_location=auto_weather_location,
            auto_weather_error=auto_weather_error,
        )
        spec = APPROVED_MODELS[model_key]

        # 3. Model & Scaler Retrieval
        model = self.registry.get_model(model_key)
        energy_scaler, weather_scaler = self.registry.get_scalers()
        quantiles_q_h = self.registry.get_conformal_quantiles(model_key)

        # 4. Feature Scaling (Leakage-safe, never refitting)
        try:
            df_e = pd.DataFrame(electricity_raw, columns=["hourly_active_power"])
            scaled_energy = energy_scaler.transform(df_e)  # (168, 1)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error scaling active power inputs: {str(e)}",
            )

        # 5. Tensor Preparation and Forward Pass
        try:
            with torch.no_grad():
                tensor_e = torch.from_numpy(scaled_energy).float().unsqueeze(0)  # (1, 168, 1)

                if model_key == "energy_only":
                    pred_scaled = model(tensor_e).squeeze(0).numpy()  # (24,)
                elif model_key == "early_fusion":
                    df_w = pd.DataFrame(weather_raw, columns=AUDITED_WEATHER_FEATURES)
                    scaled_weather = weather_scaler.transform(df_w)  # (168, 7)
                    tensor_w = torch.from_numpy(scaled_weather).float().unsqueeze(0)  # (1, 168, 7)
                    tensor_fused = torch.cat([tensor_e, tensor_w], dim=2)  # (1, 168, 8)
                    pred_scaled = model(tensor_fused).squeeze(0).numpy()  # (24,)
                elif model_key in ["late_fusion", "hybrid_fusion"]:
                    df_w = pd.DataFrame(weather_raw, columns=AUDITED_WEATHER_FEATURES)
                    scaled_weather = weather_scaler.transform(df_w)  # (168, 7)
                    tensor_w = torch.from_numpy(scaled_weather).float().unsqueeze(0)  # (1, 168, 7)
                    pred_scaled = model(tensor_e, tensor_w).squeeze(0).numpy()  # (24,)
                else:
                    raise ValueError(f"Unhandled model key: {model_key}")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Neural network forward pass failed: {str(e)}",
            )

        # 6. Inverse Scaling to Physical Watts
        e_mean = float(energy_scaler.mean_[0])
        e_std = float(energy_scaler.scale_[0])
        pred_watts = pred_scaled * e_std + e_mean  # (24,)

        # 7. Conformal Prediction Intervals
        interval_steps: List[ConformalIntervalStep] = []
        for h in range(24):
            q_h = float(quantiles_q_h[h])
            p_val = float(pred_watts[h])
            lower = p_val - q_h
            upper = p_val + q_h
            lower_clipped = max(0.0, lower)
            width = upper - lower

            ts = forecast_ts[h] if forecast_ts is not None else None
            interval_steps.append(
                ConformalIntervalStep(
                    horizon_hour=h + 1,
                    target_timestamp=ts,
                    prediction_watts=round(p_val, 2),
                    lower_bound_watts=round(lower, 2),
                    upper_bound_watts=round(upper, 2),
                    lower_bound_clipped_watts=round(lower_clipped, 2),
                    interval_width_watts=round(width, 2),
                )
            )

        mean_mpiw = float(np.mean([step.interval_width_watts for step in interval_steps]))

        resolved_run_id = run_id or f"fc_{uuid.uuid4().hex[:12]}"
        meta_info = ModelMetadataInfo(
            model_name=spec["display_name"],
            model_class=spec["class"].__name__,
            trainable_parameters=spec["trainable_parameters"],
            history_length_hours=168,
            forecast_horizon_hours=24,
            energy_scaler_type="StandardScaler (mean=307.17 W, std=147.99 W)",
            weather_scaler_type=(
                "StandardScaler (7 features)" if spec["requires_weather"] else "Not applicable (univariate)"
            ),
        )

        explanation = (
            "Split conformal prediction intervals are computed at 90% nominal confidence using "
            "frozen calibration quantiles from 587 clean benchmark windows. Important note: marginal "
            "conformal coverage provides a population-level validity guarantee, but does not guarantee "
            "90% coverage for every individual household or under arbitrary distribution shifts."
        )

        return ForecastRunResponse(
            run_id=resolved_run_id,
            selected_model=spec["display_name"],
            model_key=model_key,
            selection_mode=selection_mode,
            selection_reason=selection_reason,
            forecast_origin_timestamp=origin_ts,
            horizon_hours=list(range(1, 25)),
            forecast_timestamps=forecast_ts,
            predictions_watts=[round(float(v), 2) for v in pred_watts],
            intervals=interval_steps,
            mean_interval_width_watts=round(mean_mpiw, 2),
            nominal_coverage_pct=90.0,
            confidence_explanation=explanation,
            units="Watts",
            model_metadata=meta_info,
        )


# Global service instance
inference_service = LiveInferenceService()
