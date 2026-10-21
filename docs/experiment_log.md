# Research Experiment Log

Project: **Reliable Multimodal Household Energy Forecasting Under Degraded Weather Information**

---

## Experiment Summary & Milestone Tracking

| Step ID | Description | Primary Artifacts | Date | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | Exploratory Data Analysis & Quality Audit | `outputs/reports/eda_report.txt`, `outputs/figures/eda/` | 2026-10-08 | Completed |
| **Step 2** | Preprocessing, Aggregation & Feature Engineering | `outputs/reports/hourly_coverage_report.txt`, `docs/dataset_report.md` | 2026-10-08 | Completed |
| **Step 3** | Chronological Splitting, Scaling & Windowing | `outputs/reports/step37_split_report.txt`, `step39_sequence_report.txt` | 2026-10-09 | Completed |
| **Step 4** | Clean Benchmark Model Training & Evaluation | `outputs/reports/*_fusion_report.txt`, `step10_clean_benchmark_comparison_revised.csv` | 2026-10-09 | Completed |
| **Step 5A** | Controlled 30% MCAR Missingness Degradation | `outputs/robustness/missingness_30/` | 2026-10-09 | Completed |
| **Step 5B** | Controlled 20% Additive Gaussian Noise Degradation | `outputs/robustness/noise_20/` | 2026-10-09 | Completed |
| **Step 6** | Operational Reliability & Factorial Matrix Synthesis | `outputs/reports/step6_reliability_analysis_report.txt`, `step10_master_performance_table_revised.csv` | 2026-10-10 | Completed |
| **Step 7** | Inferential Battery & 78 Hypothesis Significance Tests | `outputs/reports/step7_statistical_analysis_report.txt`, `step10_statistical_significance_summary_revised.csv` | 2026-10-10 | Completed |
| **Step 8** | Explainable AI & Modality Attribution (Integrated Gradients) | `outputs/reports/step8_xai_attribution_report.txt`, `step10_xai_modality_attribution_summary_revised.csv` | 2026-10-10 | Completed |
| **Step 9** | Uncertainty Quantification & Split Conformal Prediction | `outputs/reports/step9_conformal_prediction_report.txt`, `step10_conformal_uncertainty_summary_revised.csv` | 2026-10-10 | Completed |
| **Step 10** | Final Research Results Synthesis & Audit Verification | `outputs/reports/step10_final_research_synthesis_report_revised.txt` | 2026-10-10 | Audited & Finalized |

---

## Detailed Step Logs

### Step 1: Exploratory Data Analysis (EDA)
- **Dataset**: `data/raw/energy_weather_raw_data.csv` (605,260 rows, 19 columns).
- **Time Range**: 2022-11-05 14:05:00 to 2024-01-05 23:59:00.
- **Key Findings**:
  - 2 exact duplicate rows removed.
  - Dominant sampling rate: 1 minute (99.80%).
  - Active power distribution is positively skewed with diurnal residential peaks (morning 07:00-09:00, evening 18:00-22:00).
  - Weather features display seasonal and daily temperature cycling.

### Step 2 & 3: Preprocessing, Aggregation, Chronological Splitting
- **Resolution**: 1-hour aggregation (`hourly_active_power`).
- **Feature Set**: `hourly_active_power` (target & autoregressive input) + 7 weather channels (`temp`, `feels_like`, `humidity`, `pressure`, `speed`, `wind_sin`, `wind_cos`).
- **Lookback & Horizon**: $W = 168$ hours (1 week), $H = 24$ hours (1 day).
- **Split Breakdown**:
  - Train: 4,400 hours (4,209 rolling windows)
  - Validation: 881 hours (881 rolling windows)
  - Calibration: 587 hours (587 rolling windows)
  - Test: 587 hours (587 rolling windows, 14,088 evaluation time points)
- **Standardization**: Training-only `StandardScaler` (Mean = 307.1706 W, Std = 147.9934 W).

### Step 4: Clean Model Benchmarks
Evaluated 4 neural models on 587 test windows:
- **Energy-Only Baseline (51,992 params)**: MAE = 77.88 W, RMSE = 93.36 W, $R^2 = 0.1296$, MAPE = 56.01%
- **Early Fusion LSTM (53,784 params)**: **MAE = 71.64 W**, **RMSE = 88.56 W**, **$R^2 = 0.2169$**, **MAPE = 48.58%** (Best clean model, $-8.02\%$ relative MAE reduction vs baseline)
- **Late Fusion LSTM (112,216 params)**: MAE = 78.90 W, RMSE = 95.07 W, $R^2 = 0.0975$, MAPE = 52.46%
- **Hybrid Fusion LSTM (145,496 params)**: MAE = 82.19 W, RMSE = 98.08 W, $R^2 = 0.0395$, MAPE = 56.35%

### Steps 5A & 5B: Controlled Degradation Experiments
- **Missingness (30% MCAR with mean imputation)**:
  - Electricity missingness caused severe degradation across all models (+1.89 W to +9.13 W MAE increase).
  - Weather missingness caused minimal degradation (+0.17 W to +0.65 W MAE increase).
  - Dual missingness (A-A) degraded Energy-Only to 87.01 W, Early Fusion to 76.39 W, Late Fusion to 83.14 W, Hybrid Fusion to 84.61 W.
- **Noise (20% AWGN, $\sigma = 0.20$)**:
  - All models showed high invariance ($\le 0.10$ W MAE change).
  - Early Fusion degraded by only +0.09 W under dual noise (A-A).

### Step 7: Formal Statistical Significance Battery
- Evaluated 78 planned hypothesis tests across 6 families.
- **60 of 78 tests rejected null hypothesis (76.9% rejection rate)** under Holm-Bonferroni ($p < 0.05$) and Benjamini-Hochberg FDR.
- Early Fusion's superiority over all three architectures in Clean (N-N), Missingness (A-A), and Noise (A-A) confirmed at $p_{\text{holm}} < 10^{-16}$.
- HLN-modified Diebold-Mariano tests confirmed significant forecast accuracy divergence ($p < 0.001$).

### Step 8: Explainable AI & Attribution Analysis
- Integrated Gradients across all 587 test windows ($m = 20$ interpolation steps).
- **Per-Channel Density Share**: Electricity captures 76.2% to 88.4% of attribution density.
- **Dynamic Reallocation**: Masking weather caused models to reallocate $+4.88\%$ to $+8.14\%$ attribution toward electricity.
- **Temporal Memory**: Late and Hybrid Fusion concentrated $>99\%$ of electricity attribution in the most recent 24 hours, whereas Early Fusion distributed 24.9% into intermediate lags (24-71h).

### Step 9: Conformal Prediction & Uncertainty Quantification
- Split conformal prediction calibrated at $\alpha = 0.10$ (90% nominal coverage) on 587 clean calibration windows.
- Fixed intervals: Early Fusion MPIW = 268.26 W (PICP = 88.56%, Winkler = 360.65). Late Fusion MPIW = 286.60 W (PICP = 89.84%, Winkler = 382.96).
- Under joint missingness (A-A), fixed intervals experienced coverage erosion (Energy-Only dropped to 79.92%, Early Fusion to 85.75%, Late Fusion to 88.42%).

### Step 10: Final Research Synthesis & Audit Verification
- Full reconciliation of all 28 experimental configurations across metrics, tables, reports, and scripts.
- Generated consolidated revised reports and CSV tables ensuring 100% numerical consistency.
