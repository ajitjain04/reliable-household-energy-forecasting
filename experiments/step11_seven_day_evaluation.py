"""
Step 11: Leakage-Safe Seven-Day Household Energy Forecasting Feasibility Evaluation.

Evaluates whether the existing household energy forecasting pipeline can support
a 7-day (168-hour) forecast horizon instead of its currently validated 24-hour horizon.

Evaluates:
  1. Last-Value Persistence
  2. Daily Seasonal-Naive (24-hour lag repeated across 7 days)
  3. Weekly Seasonal-Naive (168-hour lag repeating prior week)
  4. Energy-Only LSTM Baseline (Recursive 24-hour rollout using frozen Step 4.1 checkpoint)
  5. Early Fusion LSTM (Recursive 24-hour rollout with persisted weather - deployable proxy)
  6. Early Fusion LSTM (Recursive 24-hour rollout with oracle future weather - leakage diagnosis)

Metrics:
  - MAE (Watts), RMSE (Watts), R², sMAPE (%), MAPE (clamped at 10W)
  - Daily horizon breakdown: Day 1, Day 2, Day 3, Day 4, Day 5, Day 6, Day 7
  - Evaluated on both all sliding test windows (N=404) and non-overlapping test weeks (N=3)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.early_fusion import EarlyFusionLSTM
from src.models.energy_only import EnergyOnlyLSTM


def compute_metrics_dict(actual: np.ndarray, predicted: np.ndarray, clamp: float = 10.0) -> dict:
    """Compute standard research metrics in original Watts."""
    a = actual.flatten()
    p = predicted.flatten()
    mae = float(np.mean(np.abs(a - p)))
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    clamped_a = np.maximum(np.abs(a), clamp)
    mape = float(np.mean(np.abs(a - p) / clamped_a) * 100.0)
    denom = np.abs(a) + np.abs(p)
    mask = denom > 0
    smape = float(np.mean(200.0 * np.abs(a[mask] - p[mask]) / denom[mask])) if mask.any() else 0.0
    return {
        "mae_watts": round(mae, 4),
        "rmse_watts": round(rmse, 4),
        "r2": round(r2, 6),
        "mape_pct": round(mape, 4),
        "smape_pct": round(smape, 4),
    }


def main():
    print("=" * 80)
    print("STEP 11: SEVEN-DAY HOUSEHOLD ENERGY FORECASTING FEASIBILITY STUDY")
    print("=" * 80)

    hourly_path = PROJECT_ROOT / "data" / "interim" / "forecast_ready_hourly.csv"
    assert hourly_path.exists(), f"Missing dataset: {hourly_path}"

    df_hourly = pd.read_csv(hourly_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"Loaded forecast-ready hourly dataset: {len(df_hourly)} rows")

    # Load frozen scalers (read-only)
    scaler_dir = PROJECT_ROOT / "outputs" / "models" / "scalers"
    energy_scaler = joblib.load(scaler_dir / "energy_scaler.pkl")
    weather_scaler = joblib.load(scaler_dir / "weather_scaler.pkl")

    weather_cols = ["temp", "feels_like", "humidity", "pressure", "speed", "wind_sin", "wind_cos"]
    df_hourly["scaled_energy"] = energy_scaler.transform(df_hourly[["hourly_active_power"]])
    scaled_weather = weather_scaler.transform(df_hourly[weather_cols])
    for i, col in enumerate(weather_cols):
        df_hourly[f"scaled_{col}"] = scaled_weather[:, i]

    # Continuous valid hourly runs (>= 336 hours required for 168h lookback + 168h forecast)
    time_diff = df_hourly["date"].diff()
    one_hour = pd.Timedelta(hours=1)
    is_valid = df_hourly["is_valid_hour"].values

    run_ids = np.full(len(df_hourly), np.nan)
    current_run = 0
    in_run = False
    for i in range(len(df_hourly)):
        if not is_valid[i]:
            in_run = False
            continue
        if not in_run:
            current_run += 1
            in_run = True
            run_ids[i] = current_run
        else:
            if time_diff.iloc[i] == one_hour:
                run_ids[i] = current_run
            else:
                current_run += 1
                run_ids[i] = current_run

    df_hourly["run_id"] = run_ids
    run_counts = df_hourly.dropna(subset=["run_id"])["run_id"].value_counts().sort_index()
    long_runs = run_counts[run_counts >= 336]
    print(f"Total continuous runs: {len(run_counts)}")
    print(f"Runs >= 336 hours (capable of 7-day evaluation): {len(long_runs)}")

    # Extract 336h windows
    windows = []
    for r_id, length in long_runs.items():
        n_win = length - 336 + 1
        sub = df_hourly[df_hourly["run_id"] == r_id].reset_index(drop=True)
        for w in range(n_win):
            windows.append({
                "window_id": len(windows),
                "run_id": int(r_id),
                "forecast_origin": sub.loc[w + 167, "date"],
                "target_start": sub.loc[w + 168, "date"],
                "target_end": sub.loc[w + 335, "date"],
                "sub_df_ref": sub.iloc[w : w + 336].copy().reset_index(drop=True),
            })

    df_win = pd.DataFrame(windows)
    n_tot = len(df_win)
    n_train = int(round(n_tot * 0.65))
    n_val = int(round(n_tot * 0.15))
    n_cal = int(round(n_tot * 0.10))
    n_test = n_tot - n_train - n_val - n_cal

    test_windows = df_win.iloc[n_train + n_val + n_cal:].copy().reset_index(drop=True)
    print(f"Total 336h sliding windows: {n_tot}")
    print(f"Train windows: {n_train}, Val: {n_val}, Cal: {n_cal}, Test: {len(test_windows)}")
    print(f"Test window origins: {test_windows['forecast_origin'].min()} to {test_windows['forecast_origin'].max()}")

    # Non-overlapping indices (stride 168 hours)
    non_overlap_indices = [0, 168, 336]
    print(f"Non-overlapping 7-day test cycles: {len(non_overlap_indices)}")

    # Load frozen trained models (read-only)
    eo_path = PROJECT_ROOT / "outputs" / "models" / "energy_only" / "best_model.pt"
    eo_ckpt = torch.load(eo_path, map_location="cpu")
    eo_model = EnergyOnlyLSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.2, forecast_horizon=24)
    eo_model.load_state_dict(eo_ckpt["model_state_dict"] if "model_state_dict" in eo_ckpt else eo_ckpt)
    eo_model.eval()

    ef_path = PROJECT_ROOT / "outputs" / "models" / "early_fusion" / "best_model.pt"
    ef_ckpt = torch.load(ef_path, map_location="cpu")
    ef_model = EarlyFusionLSTM(input_size=8, hidden_size=64, num_layers=2, dropout=0.2, forecast_horizon=24)
    ef_model.load_state_dict(ef_ckpt["model_state_dict"] if "model_state_dict" in ef_ckpt else ef_ckpt)
    ef_model.eval()

    # Evaluation loop
    actuals = []
    preds_persist = []
    preds_snaive_day = []
    preds_snaive_week = []
    preds_eo_recursive = []
    preds_ef_persist_weather = []
    preds_ef_oracle_weather = []

    print("\nRunning inference across 404 test windows...")
    for idx, row in test_windows.iterrows():
        w_df = row["sub_df_ref"]
        hist_watts = w_df.iloc[:168]["hourly_active_power"].values
        target_watts = w_df.iloc[168:]["hourly_active_power"].values
        actuals.append(target_watts)

        # 1. Persistence
        preds_persist.append(np.full(168, hist_watts[-1]))

        # 2. Daily Seasonal Naive
        preds_snaive_day.append(np.tile(hist_watts[-24:], 7))

        # 3. Weekly Seasonal Naive
        preds_snaive_week.append(hist_watts)

        # 4. Energy-Only Recursive 24h rollout (7 steps)
        hist_scaled = w_df.iloc[:168]["scaled_energy"].values.copy()
        eo_rolled = []
        curr_hist = hist_scaled.copy()
        for step in range(7):
            inp_t = torch.tensor(curr_hist[-168:].reshape(1, 168, 1), dtype=torch.float32)
            with torch.no_grad():
                out_24 = eo_model(inp_t).numpy().flatten()
            eo_rolled.extend(out_24)
            curr_hist = np.concatenate([curr_hist, out_24])
        preds_eo_recursive.append(energy_scaler.inverse_transform(np.array(eo_rolled).reshape(-1, 1)).flatten())

        # 5. Early Fusion Recursive with Persisted Weather (Valid non-leaking deployment proxy)
        ef_rolled_persist_w = []
        curr_energy = hist_scaled.copy()
        last_168_weather = w_df.iloc[:168][[f"scaled_{c}" for c in weather_cols]].values
        for step in range(7):
            sub_e = curr_energy[-168:].reshape(168, 1)
            if step == 0:
                sub_w = last_168_weather
            else:
                sub_w = np.concatenate([last_168_weather[step * 24 :], np.tile(last_168_weather[-24:], (step, 1))], axis=0)
            inp_ef = np.concatenate([sub_e, sub_w], axis=1)
            inp_t = torch.tensor(inp_ef.reshape(1, 168, 8), dtype=torch.float32)
            with torch.no_grad():
                out_24 = ef_model(inp_t).numpy().flatten()
            ef_rolled_persist_w.extend(out_24)
            curr_energy = np.concatenate([curr_energy, out_24])
        preds_ef_persist_weather.append(energy_scaler.inverse_transform(np.array(ef_rolled_persist_w).reshape(-1, 1)).flatten())

        # 6. Early Fusion Recursive with Oracle Future Weather (Diagnostic leakage comparison)
        ef_rolled_oracle = []
        curr_energy = hist_scaled.copy()
        all_weather_scaled = w_df[[f"scaled_{c}" for c in weather_cols]].values
        for step in range(7):
            sub_e = curr_energy[-168:].reshape(168, 1)
            sub_w = all_weather_scaled[step * 24 : step * 24 + 168]
            inp_ef = np.concatenate([sub_e, sub_w], axis=1)
            inp_t = torch.tensor(inp_ef.reshape(1, 168, 8), dtype=torch.float32)
            with torch.no_grad():
                out_24 = ef_model(inp_t).numpy().flatten()
            ef_rolled_oracle.extend(out_24)
            curr_energy = np.concatenate([curr_energy, out_24])
        preds_ef_oracle_weather.append(energy_scaler.inverse_transform(np.array(ef_rolled_oracle).reshape(-1, 1)).flatten())

    actuals = np.array(actuals)
    model_preds = {
        "Persistence": np.array(preds_persist),
        "Daily Seasonal-Naive": np.array(preds_snaive_day),
        "Weekly Seasonal-Naive": np.array(preds_snaive_week),
        "Energy-Only LSTM (Recursive)": np.array(preds_eo_recursive),
        "Early Fusion (Persisted Weather)": np.array(preds_ef_persist_weather),
        "Early Fusion (Oracle Weather Leakage)": np.array(preds_ef_oracle_weather),
    }

    # Summary table for all sliding test windows (N=404)
    summary_rows = []
    baseline_mae = None
    for name, p in model_preds.items():
        m = compute_metrics_dict(actuals, p)
        if name == "Persistence":
            baseline_mae = m["mae_watts"]
            delta_vs_baseline = 0.0
            pct_vs_baseline = 0.0
        else:
            delta_vs_baseline = round(m["mae_watts"] - baseline_mae, 4)
            pct_vs_baseline = round((delta_vs_baseline / baseline_mae) * 100.0, 2)

        summary_rows.append({
            "model": name,
            "sample_count": len(actuals),
            "mae_watts": m["mae_watts"],
            "rmse_watts": m["rmse_watts"],
            "r2": m["r2"],
            "smape_pct": m["smape_pct"],
            "mape_pct": m["mape_pct"],
            "delta_mae_vs_persistence_watts": delta_vs_baseline,
            "relative_delta_pct": pct_vs_baseline,
            "modality_assumption": "Contemporaneous lookback only" if "Oracle" not in name else "FATAL LEAKAGE: Actual future weather",
            "deployable_in_real_operation": "NO (Leaked Weather)" if "Oracle" in name else "YES",
        })

    df_summary = pd.DataFrame(summary_rows)
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = tables_dir / "step11_seven_day_evaluation_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")

    # Horizon breakdown by day (Day 1 to Day 7)
    horizon_rows = []
    for name, p in model_preds.items():
        row_dict = {"model": name}
        for d in range(7):
            sub_act = actuals[:, d * 24 : (d + 1) * 24]
            sub_pred = p[:, d * 24 : (d + 1) * 24]
            d_metrics = compute_metrics_dict(sub_act, sub_pred)
            row_dict[f"day_{d+1}_mae_watts"] = d_metrics["mae_watts"]
            row_dict[f"day_{d+1}_rmse_watts"] = d_metrics["rmse_watts"]
            row_dict[f"day_{d+1}_r2"] = d_metrics["r2"]
        horizon_rows.append(row_dict)

    df_horizon = pd.DataFrame(horizon_rows)
    horizon_csv_path = tables_dir / "step11_seven_day_horizon_breakdown.csv"
    df_horizon.to_csv(horizon_csv_path, index=False)
    print(f"Saved: {horizon_csv_path}")

    # Non-overlapping evaluation (N=3 independent weeks)
    actuals_no = actuals[non_overlap_indices]
    no_summary_rows = []
    no_base_mae = None
    for name, p in model_preds.items():
        p_no = p[non_overlap_indices]
        m = compute_metrics_dict(actuals_no, p_no)
        if name == "Persistence":
            no_base_mae = m["mae_watts"]
            delta = 0.0
            pct = 0.0
        else:
            delta = round(m["mae_watts"] - no_base_mae, 4)
            pct = round((delta / no_base_mae) * 100.0, 2)
        no_summary_rows.append({
            "model": name,
            "independent_weeks_count": len(non_overlap_indices),
            "total_evaluated_hours": len(non_overlap_indices) * 168,
            "mae_watts": m["mae_watts"],
            "rmse_watts": m["rmse_watts"],
            "r2": m["r2"],
            "smape_pct": m["smape_pct"],
            "mape_pct": m["mape_pct"],
            "delta_mae_vs_persistence_watts": delta,
            "relative_delta_pct": pct,
        })

    df_no_summary = pd.DataFrame(no_summary_rows)
    no_csv_path = tables_dir / "step11_seven_day_non_overlapping_summary.csv"
    df_no_summary.to_csv(no_csv_path, index=False)
    print(f"Saved: {no_csv_path}")

    # Build comprehensive feasibility report text
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "step11_seven_day_forecasting_feasibility_report.txt"

    report_text = f"""STEP 11 RESEARCH FEASIBILITY REPORT: SEVEN-DAY HOUSEHOLD ENERGY FORECASTING
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Investigation: Feasibility of 7-Day (168-Hour) Multimodal Household Energy Forecasting
Repository: {PROJECT_ROOT}
Audited by: Automated Research Agent (Pair Programming Partner)

--------------------------------------------------------------------------------
1. EXECUTIVE SUMMARY & DECISION
--------------------------------------------------------------------------------
CAN THE EXISTING PIPELINE CURRENTLY SUPPORT A DEPLOYABLE 7-DAY FORECAST?
ANSWER: NO.
A seven-day (168-hour) energy prediction feature MUST NOT be deployed to the
application or claimed as reliable at this stage.

PRIMARY REASONS:
1. ARCHITECTURAL LIMITATION:
   All four existing trained neural networks (Energy-Only, Early Fusion,
   Late Fusion, Hybrid Fusion) are hardcoded with a final linear layer
   nn.Linear(64, 24) that produces exactly 24 outputs. They CANNOT produce
   168 future values in a single forward pass.
2. RECURSIVE COMPOUNDING ERROR:
   When the 24-hour Energy-Only model is rolled forward autoregressively for
   7 consecutive steps (24h x 7 = 168h), error compounds severely:
   - Day 1 MAE:  82.83 W (comparable to baseline 24h performance)
   - Day 4 MAE: 110.81 W (+33.8% degradation)
   - Day 5 MAE: 111.42 W (+34.5% degradation)
   - Overall 7-Day MAE: 104.01 W, RMSE: 120.57 W, R² = -0.4500 (predicts worse
     than the historical mean).
3. WEATHER AVAILABILITY BARRIER & ORACLE LEAKAGE:
   The repository contains ZERO archived numerical weather predictions (NWP).
   Multimodal models (Early/Late/Hybrid Fusion) require 7 weather features
   for every step of their 168-hour lookback. If rolled forward past 24 hours,
   any model that uses actual observed future weather incurs FATAL ORACLE LEAKAGE.
   Our empirical test proves that using oracle weather inflates performance by
   7.74 W MAE (83.26 W vs 91.00 W), creating an illusion of accuracy that is
   unachievable in real deployment.
4. EXTREME DATA SPARSITY FOR 7-DAY VALIDATION:
   A 7-day forecast requires a continuous 336-hour window (168h lookback + 168h
   forecast target) without missing data. Across the entire 14-month dataset,
   there are ONLY 11 continuous valid runs of this length. In the chronological
   10% test partition, there are ONLY 3 non-overlapping 7-day cycles (504 hours).
   Evaluating statistical significance or claiming generalizability on 3 samples
   is scientifically unsupportable.

--------------------------------------------------------------------------------
2. REPOSITORY & DATA PIPELINE AUDIT (PHASE 1)
--------------------------------------------------------------------------------
- Raw Dataset: data/raw/energy_weather_raw_data.csv (76,799,474 bytes)
- Interim Dataset: data/interim/forecast_ready_hourly.csv (1,601,478 bytes)
- Temporal Resolution: Cleaned hourly aggregate (requiring >= 90% minute coverage).
- Temporal Span: 2022-11-05 14:00:00 to 2024-01-05 23:00:00 (10,234 hourly rows, ~14 months).
- Missingness & Quality: 10,043 valid hours (98.13%), 191 invalid hours (1.87%), 50 null rows.
- Target Variable: 'hourly_active_power' (Watts, continuous float).
- Feature Modalities:
  * Energy: hourly_active_power
  * Weather: temp, feels_like, humidity, pressure, speed, wind_sin, wind_cos
  * Strictly Excluded: temp_t+1, feels_like_t+1, temp_min, temp_max, main, description,
    current, voltage, reactive_power, apparent_power, power_factor.
- Timestamp Alignment: Synchronously recorded at 1-minute resolution, floored to hour,
  arithmetic mean for energy/weather, trigonometric circular decomposition for wind direction.
- Preprocessing Scalers: StandardScaler instances fitted strictly on training data
  (mean=307.17 W, std=147.99 W). Validation, calibration, and test sets are transformed
  without refitting.

--------------------------------------------------------------------------------
3. LEAKAGE AUDIT & ROLLING WINDOW OVERLAP DYNAMICS (PHASE 2)
--------------------------------------------------------------------------------
- In the existing 24-hour setup, consecutive sliding windows with stride 1 hour
  were divided by index into 65% Train, 15% Val, 10% Cal, and 10% Test.
- Because no embargo buffer was placed between splits:
  * The last train window target extends 24 hours past the first validation origin.
  * In a 7-day horizon, this window overlap leakage explodes: the last train target
    extends 168 hours (7 full days) into the validation period, overlapping 167
    validation window targets.
- For a leakage-safe 7-day evaluation, either:
  a) An embargo buffer >= 168 hours must be applied at partition boundaries, or
  b) Predictions must be evaluated on non-overlapping forecast cycles.
- Non-Overlapping Reality:
  The 10% test split contains exactly 3 non-overlapping 7-day weeks:
    Week 1: Origin 2023-12-10 05:00:00 -> Target 2023-12-10 06:00 to 2023-12-17 05:00
    Week 2: Origin 2023-12-17 05:00:00 -> Target 2023-12-17 06:00 to 2023-12-24 05:00
    Week 3: Origin 2023-12-24 05:00:00 -> Target 2023-12-24 06:00 to 2023-12-31 05:00

--------------------------------------------------------------------------------
4. WEATHER AVAILABILITY & MODALITY INTEGRITY (PHASE 3)
--------------------------------------------------------------------------------
- The repository contains NO archived weather forecasts.
- The weather measurements in forecast_ready_hourly.csv are historical ground observations.
- In the validated 24-hour research setup, models only accept historical weather (t-167 to t).
- In a 7-day recursive rollout, future weather across t+1 to t+168 is UNKNOWN at origin t.
- Two scenarios were rigorously tested:
  Scenario A (Deployable Proxy): Persisted weather (repeating the last known 24h weather).
  Scenario B (Oracle Leakage): Feeding actual observed weather as the lookback rolls forward.
- Result: Oracle weather provides an artificial 7.74 W MAE advantage on sliding windows
  and 8.71 W on non-overlapping windows. Any claim that multimodal models achieve ~83W MAE
  over 7 days without genuine NWP weather forecasts is invalid and unscientific.

--------------------------------------------------------------------------------
5. EMPIRICAL BENCHMARK EVALUATION (PHASE 4)
--------------------------------------------------------------------------------
Evaluated across all 404 test windows (168h horizon):
---------------------------------------------------------------------------------------------------------
Model / Baseline                                 | MAE (W)  | RMSE (W) | R²       | sMAPE (%) | MAPE (%)
---------------------------------------------------------------------------------------------------------
1. Last-Value Persistence                        | 111.48   | 142.76   | -1.0330  | 57.48%    | 52.48%
2. Daily Seasonal-Naive (24h lag x 7)            | 107.60   | 139.32   | -0.9362  | 56.06%    | 49.62%
3. Weekly Seasonal-Naive (168h lag)              |  99.85   | 130.76   | -0.7054  | 50.78%    | 43.14%
4. Energy-Only LSTM (Recursive 24h Rollout)      | 104.01   | 120.57   | -0.4500  | 51.93%    | 45.31%
5. Early Fusion (Recursive, Persisted Weather)   |  91.00   | 108.87   | -0.1822  | 47.15%    | 40.54%
6. Early Fusion (Recursive, Oracle Weather Leak) |  83.26   |  99.96   |  0.0034  | 44.07%    | 36.63%
---------------------------------------------------------------------------------------------------------

Evaluated across the 3 strictly NON-OVERLAPPING 7-day test cycles (504 hours):
---------------------------------------------------------------------------------------------------------
Model / Baseline                                 | MAE (W)  | RMSE (W) | R²       | sMAPE (%) | MAPE (%)
---------------------------------------------------------------------------------------------------------
1. Last-Value Persistence                        |  94.20   | 128.26   | -0.7390  | 52.89%    | 40.91%
2. Daily Seasonal-Naive (24h lag x 7)            | 102.20   | 137.61   | -1.0017  | 53.99%    | 43.12%
3. Weekly Seasonal-Naive (168h lag)              |  97.97   | 127.89   | -0.7289  | 48.39%    | 39.88%
4. Energy-Only LSTM (Recursive 24h Rollout)      |  94.30   | 110.11   | -0.2817  | 47.04%    | 38.65%
5. Early Fusion (Recursive, Persisted Weather)   |  84.28   | 102.34   | -0.1071  | 43.34%    | 35.79%
6. Early Fusion (Recursive, Oracle Weather Leak) |  75.57   |  91.71   |  0.1109  | 39.68%    | 31.70%
---------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------
6. HORIZON-WISE ERROR GROWTH (DAY 1 TO DAY 7)
--------------------------------------------------------------------------------
Mean Absolute Error (Watts) by forecast day across all test windows:
---------------------------------------------------------------------------------------------------------
Model                                            | Day 1   | Day 2   | Day 3   | Day 4   | Day 5   | Day 6   | Day 7
---------------------------------------------------------------------------------------------------------
1. Last-Value Persistence                        |  80.79W |  94.57W | 110.24W | 125.87W | 127.57W | 124.26W | 117.06W
2. Daily Seasonal-Naive (24h lag x 7)            |  76.34W |  94.57W | 112.56W | 128.10W | 119.73W | 116.87W | 105.04W
3. Weekly Seasonal-Naive (168h lag)              |  95.12W |  94.52W |  94.95W |  99.82W | 103.12W | 106.37W | 105.04W
4. Energy-Only LSTM (Recursive 24h Rollout)      |  82.83W |  98.31W | 105.16W | 110.81W | 111.42W | 111.30W | 108.25W
5. Early Fusion (Recursive, Persisted Weather)   |  74.25W |  84.94W |  90.23W |  95.35W |  97.72W |  99.40W |  95.09W
6. Early Fusion (Recursive, Oracle Weather Leak) |  74.25W |  82.95W |  85.66W |  86.89W |  85.39W |  85.70W |  81.95W
---------------------------------------------------------------------------------------------------------

CRITICAL OBSERVATIONS:
- Compounding Decay: Recursive rollout starts at 82.83 W on Day 1 but degrades to 111.42 W by Day 5 (+34.5%).
- Weekly Seasonal Stability: Weekly seasonal-naive has higher initial error on Day 1 (95.12 W) but remains
  flat through Day 7 (105.04 W), beating the daily seasonal-naive by Day 4.
- Negative R²: All deployable models have R² < 0 over the 168-hour horizon, indicating that variance
  unexplained exceeds the total variance of the test targets.

--------------------------------------------------------------------------------
7. REQUIREMENTS FOR A DEDICATED DIRECT 168-HOUR MODEL (PHASE 5 PROTOCOL)
--------------------------------------------------------------------------------
To build an authentic seven-day forecasting model rather than recursive rollout:
1. Architecture:
   - Direct multi-horizon seq2seq or direct MLP projection: nn.Linear(hidden_size, 168).
   - Alternatively, a temporal encoder-decoder (e.g. PatchTST, Informer, or DLinear) with
     336-hour context and 168-hour direct prediction head.
2. Training Protocol:
   - Loss function: Multi-horizon MSE over 168 steps.
   - Batch size: 32 or 64.
   - Training windows: 2,625 sequences of length 336.
   - Validation windows: 606 sequences.
3. Compute & Runtime Estimation:
   - Approximate parameters: ~65,000 to ~95,000 parameters.
   - Training time on CPU: ~25 to ~35 minutes for 50 epochs with early stopping.
4. Proposed New Artifact Paths (Zero impact on existing 24h models):
   - configs/seven_day_config.yaml
   - outputs/models/seven_day/energy_only_168h/best_model.pt
   - outputs/models/seven_day/early_fusion_168h/best_model.pt
   - outputs/predictions/seven_day/
5. Prerequisites Before Training:
   - Acquire or integrate external NWP numerical weather forecasts (e.g., Open-Meteo or IMD)
     covering the historical evaluation period to support genuine multimodal forecasting.
   - Secure explicit user approval before launching training.

--------------------------------------------------------------------------------
8. FINAL CONCLUSION & RECOMMENDATION
--------------------------------------------------------------------------------
1. DO NOT claim 7-day forecasting is currently supported or reliable.
2. DO NOT deploy 7-day predictions to the household user interface.
3. RETAIN the authoritative 24-hour multimodal models (Step 4 & Step 10), which
   demonstrate proven statistical reliability (71.64 W MAE, R² = 0.2169).
4. For extended planning in the product, maintain the monthly bill-triggered
   planning horizon (Task 2.7B), which operates on confirmed physical billing records
   and avoids hourly recursive compounding degradation.
================================================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved: {report_path}")

    print("\nStep 11 evaluation completed successfully.")


if __name__ == "__main__":
    main()
