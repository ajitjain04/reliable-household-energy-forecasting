# Reliable Multimodal Household Energy Forecasting Under Degraded Weather Information

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Complete%20%26%20Audited-success.svg)](outputs/reports/step10_final_research_synthesis_report_revised.txt)
[![Streamlit App](https://img.shields.io/badge/App-Streamlit%20Dashboard-FF4B4B.svg)](app/streamlit_app.py)

An end-to-end empirical study investigating whether multimodal fusion between past household electricity consumption and auxiliary weather telemetry improves multi-step ahead residential load forecasting, and whether multimodal neural architectures maintain reliable, calibrated predictions under operational sensor degradation (packet missingness and measurement noise).

---

## 🔬 Key Research Findings

1. **Multimodal Early Fusion Superiority**:
   Early Fusion achieved the lowest test MAE across all models (**71.64 W** vs **77.88 W** for Energy-Only baseline, an **8.02% relative error reduction**), an RMSE of **88.56 W** (-5.14%), an $R^2$ of **0.2169** (vs 0.1296), and a MAPE of **48.58%** (vs 56.01%).
2. **Decoupled Architectural Capacity vs Generalization**:
   Decoupled multimodal architectures (Late Fusion: 112K parameters; Hybrid Fusion: 145K parameters) increased model capacity and training time, but did not outperform the lightweight Early Fusion model (53K parameters) or the unimodal baseline on the held-out test partition.
3. **Modality Sensitivity Asymmetry**:
   Electricity history is the dominant operational signal. Losing electricity data caused substantial error increases (+1.89 W to +9.13 W), whereas losing weather data caused minute increments (+0.17 W to +0.65 W). Weather provides secondary contextual gains, but cannot substitute for missing electricity history.
4. **Noise Invariance vs Missingness Vulnerability**:
   All neural models demonstrated strong invariance to 20% Additive Gaussian Noise ($\le 0.10$ W MAE increase, non-significant). In contrast, 30% MCAR missingness (with zero/mean imputation) caused statistically significant performance drops across all models ($p < 10^{-15}$).
5. **Statistical Test Battery**:
   In formal inferential evaluation across 78 planned hypothesis tests, **60 of 78 tests rejected the null hypothesis (76.9% rejection rate)** under Holm-Bonferroni ($p < 0.05$) and Benjamini-Hochberg FDR control.
6. **Conformal Uncertainty Quantification**:
   Split conformal prediction calibrated at 90% nominal coverage ($\alpha = 0.10$) maintained constant interval width ($268.26\text{ W}$ for Early Fusion), resulting in empirical coverage erosion under sensor missingness.

---

## 📊 Clean Benchmark Performance Comparison

| Model Architecture | Trainable Parameters | Validation MAE (W) | Test MAE (W) | Test RMSE (W) | Test $R^2$ | Test MAPE (%) | $\Delta$ MAE vs Baseline | Relative MAE Change |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Energy-Only Baseline** | 51,992 | 77.25 W | 77.88 W | 93.36 W | 0.1296 | 56.01% | 0.00 W | 0.00% |
| **Early Fusion LSTM** | **53,784** | **77.47 W** | **71.64 W** | **88.56 W** | **0.2169** | **48.58%** | **-6.25 W** | **-8.02%** |
| **Late Fusion LSTM** | 112,216 | 79.61 W | 78.90 W | 95.07 W | 0.0975 | 52.46% | +1.02 W | +1.31% |
| **Hybrid Fusion LSTM** | 145,496 | 80.24 W | 82.19 W | 98.08 W | 0.0395 | 56.35% | +4.30 W | +5.53% |

---

## 🏗️ Neural Architectures & Task Setup

- **Lookback Window ($W$)**: 168 hours (7 days of hourly historical dynamics)
- **Forecast Horizon ($H$)**: 24 hours ahead (multi-step autoregressive rollout)
- **Sampling Resolution**: 1-hour aggregation (`hourly_active_power` in Watts)
- **Feature Modalities**:
  - Electrical: `hourly_active_power` (1 channel)
  - Weather: `temp`, `feels_like`, `humidity`, `pressure`, `speed`, `wind_sin`, `wind_cos` (7 channels)
- **Chronological Split**:
  - Train: 4,400 hours (Nov 2022 – May 2023, 4,209 windows)
  - Validation: 881 hours (May 2023 – Jul 2023, 881 windows)
  - Calibration: 587 hours (Jul 2023 – Sep 2023, 587 windows)
  - Test: 587 hours (Sep 2023 – Nov 2023, 587 windows, 14,088 test points)

---

## 🚀 Interactive Streamlit Research Dashboard

An interactive research dashboard is included in `app/streamlit_app.py` providing:
- Executive Summary & Findings KPI overview
- Interactive 24-hour Multi-Step Forecast Explorer across all 587 test windows
- Operational Robustness & 2x2 Factorial Degradation Lab
- Inferential Statistics & 78 Hypothesis Significance Explorer
- Explainable AI & Modality Attribution Inspector (Integrated Gradients)
- Conformal Prediction & 90% Uncertainty Bounds Visualizer
- Architectural Specifications & Mathematical Formulations

To launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Repository Structure

```text
├── app/
│   └── streamlit_app.py                   # Interactive research web application
├── data/
│   ├── raw/                               # Raw minute-level smart meter & weather data
│   ├── interim/                           # Duplicate audit logs & intermediate cleaning
│   └── processed/                         # Processed hourly datasets and sequence tensors
├── docs/
│   ├── dataset_report.md                  # Comprehensive EDA and aggregation specifications
│   ├── methodology.md                     # Mathematical and architectural research methodology
│   ├── decisions.md                       # Architectural Decision Records (ADR-001 through 008)
│   └── experiment_log.md                  # Chronological experiment log across Steps 1-10
├── notebooks/                             # Jupyter research and validation notebooks
├── outputs/
│   ├── figures/                           # High-resolution research plots (EDA, models, residual)
│   ├── models/                            # Trained PyTorch model checkpoints & scalers
│   ├── predictions/                       # Precomputed multi-step test and validation predictions
│   ├── reports/                           # Audited step-by-step synthesis and audit reports
│   ├── robustness/                        # 30% missingness and 20% noise experiment artifacts
│   ├── tables/                            # Consolidated CSV benchmark and statistical tables
│   ├── uncertainty/                       # Conformal prediction calibration artifacts and bounds
│   └── xai/                               # Integrated Gradients attribution tables and figures
├── src/
│   ├── analysis/                          # XAI and conformal prediction runner scripts
│   ├── data/                              # Ingestion, cleaning, aggregation, scaling routines
│   ├── evaluation/                        # Forecasting metrics (MAE, RMSE, R^2, MAPE, Winkler)
│   ├── models/                            # PyTorch model definitions (Energy-only, Early, Late, Hybrid)
│   ├── robustness/                        # MCAR missingness and AWGN noise experiment pipelines
│   └── utils/                             # Random seeding, data loaders, and helper utilities
└── requirements.txt                       # Core Python scientific dependencies
```

---

## 📜 Complete Documentation Links

- [Methodology Specification](docs/methodology.md)
- [Architectural Decision Records](docs/decisions.md)
- [Research Experiment Log](docs/experiment_log.md)
- [Dataset Preprocessing Report](docs/dataset_report.md)
- [Final Research Synthesis Report (Revised)](outputs/reports/step10_final_research_synthesis_report_revised.txt)
- [Step 10 Discrepancy & Verification Report](outputs/reports/step10_discrepancy_and_verification_report.txt)
- [System Setup & Deployment Guide](README_SETUP.md)

---

## 🌐 Full-Stack Application & Operational Workflows

Beyond the empirical research study, this repository provides a production-grade web application integrating the evaluated forecasting models, bill ingestion, and energy advisory tools:

- **System Architecture**:
  - **Backend**: FastAPI (`http://127.0.0.1:8000`) with Argon2id authentication, role-based access control (`household` vs `developer`), anti-enumeration ownership protection, and vector PDF report generation.
  - **Frontend**: React 18, Vite, TypeScript, Tailwind CSS, and Lucide icons (`http://localhost:5173`) featuring dark/light themes and dynamic Recharts visualizers.
  - **Persistence**: Embedded SQLite (`data/forecast_records.db`) storing durable hourly and monthly forecast snapshots and user accounts.

- **Key Operational Features**:
  1. **24-Hour Hourly Forecasting**: Interactive rollout across all four evaluated LSTM architectures with calibrated 90% conformal prediction uncertainty bands.
  2. **Guided Electricity Account Connection & Bill Upload**: Guided workflow for four major Discoms (MSEDCL, BESCOM, JVVNL, TANGEDCO/TNPDCL). Users click verified official portal links opening in a new tab, download their official e-bill, and drop it into the in-app upload zone on the same screen. The platform **never** collects utility passwords or OTPs and does not perform automated scraping.
  3. **Multi-Format Bill Parsing**: Automated ingestion of digital PDFs (cross-platform), scanned PDFs, and images (with Windows Native OCR). Extracted records populate an interactive review table requiring **explicit user confirmation** before forecasting.
  4. **Monthly Forecasting & Tariff Integration**: 1, 3, and 6-month horizons with trend modeling and historical backtesting. Features verified MERC LT-I(B) domestic slab calculations for MSEDCL; monetary estimates are safely withheld for unverified or expired rate schedules.
  5. **Energy Coach Appliance Simulator**: Physics-based load simulator (`watts * hours * days * quantity / 1000`) with refrigerator continuous-operation safety cautions and baseline household scenario comparison.
  6. **Vector PDF Reports**: Genuine vector PDF reports for hourly and monthly forecasts with anti-enumeration security.

For detailed setup, environment variables, portal links, and test execution procedures, refer to the [System Setup and Manual Launch Guide](README_SETUP.md).

