# Dataset Report

## Step 3.5: Preprocessing and Hourly Aggregation

### Raw Dataset

- **File**: `data/raw/energy_weather_raw_data.csv`
- **Rows**: 605,260
- **Columns**: 19
- **Time range**: 2022-11-05 14:05:00 to 2024-01-05 23:59:00
- **Dominant sampling interval**: 1 minute (99.80% of consecutive pairs)
- **Timezone**: Naive (no timezone information)
- **Timestamps sorted**: Yes (monotonically increasing)

### Duplicate Findings

- **Exact duplicate rows detected**: 2
- **Exact duplicate rows removed**: 2 (kept first occurrence)
- **Duplicate timestamps after exact-duplicate removal**: To be confirmed during execution
- Non-identical duplicate-timestamp records (if any) are saved to `data/interim/duplicate_timestamp_records.csv` for audit.

### Hourly Aggregation Methodology

The cleaned minute-level data is aggregated to hourly resolution.

**Grouping key**: Timestamps are floored to the hour.

**Primary target**:
- `hourly_active_power` = hourly mean of `active_power` (unit: W)
- This represents hourly average active power, not energy consumption.

**Secondary electrical variables** (hourly mean, preserved for future use):
- `current`, `voltage`, `reactive_power`, `apparent_power`, `power_factor`

**Primary weather features** (hourly mean):
- `temp`, `feels_like`, `humidity`, `pressure`, `speed`

### Wind Direction Encoding

Wind direction (`deg`) is a circular variable. Taking an arithmetic mean directly on degrees would be incorrect (e.g., averaging 350 and 10 degrees would give 180, not 0).

Instead:
1. Convert each raw `deg` observation to radians.
2. Compute `wind_sin = sin(deg_radians)` and `wind_cos = cos(deg_radians)`.
3. Aggregate `wind_sin` and `wind_cos` via hourly mean.

The primary hourly dataset uses `wind_sin` and `wind_cos` rather than the raw `deg`.

### Future-Weather Exclusion

The following columns are **excluded** from the primary hourly dataset because they represent future weather values and would cause data leakage:
- `temp_t+1`
- `feels_like_t+1`

These columns are NOT deleted from the raw file. They are simply not carried into the hourly aggregation.

### Excluded Categorical and Range Weather Columns

The following columns are excluded from the primary feature set:
- `main` (categorical weather condition)
- `description` (categorical weather description)
- `temp_min` (hourly API range value, not a direct measurement)
- `temp_max` (hourly API range value, not a direct measurement)

### Coverage Methodology

- **Expected observations per complete hour**: 60 (1-minute sampling)
- **n_observations**: Count of raw records in each hourly slot
- **coverage_ratio**: `n_observations / 60`
- A **complete hourly index** is constructed from the first to the last hour, so hours with zero observations appear explicitly with `n_observations=0` and `coverage_ratio=0.0`.
- **No interpolation, forward-fill, or backward-fill** is performed on missing hourly slots.

### Candidate Coverage Thresholds

The following thresholds are candidates for defining "valid" hours. The final threshold has NOT been selected yet:
- 80% (>= 48 observations)
- 90% (>= 54 observations)
- 95% (>= 57 observations)

The threshold will be chosen after examining the actual coverage distribution.

### Output Files

| File | Description |
|---|---|
| `data/interim/hourly_aggregated.csv` | Hourly interim dataset |
| `data/interim/duplicate_timestamp_records.csv` | Duplicate-timestamp records (if any) |
| `outputs/reports/hourly_coverage_report.txt` | Hourly coverage analysis |

---

## Step 3.6: Chronological Forecasting Preparation & Window Construction

### Forecasting Formulation

- **Historical Context ($H$)**: 168 hours (7 days)
- **Forecast Horizon ($F$)**: 24 hours (1 day)
- **Total Window Span ($W = H + F$)**: 192 hours
- **Forecast Origin ($t$)**:
  - Energy input: $[t-167, \dots, t]$ (168 historical hours)
  - Weather input: $[t-167, \dots, t]$ (168 historical hours)
  - Target: $[t+1, \dots, t+24]$ (24 future hours of `hourly_active_power`)
- **Strict Leakage Barrier**: The model NEVER observes weather features at $t+1 \dots t+24$. Future weather columns (`temp_t+1`, `feels_like_t+1`) are strictly excluded.

### Validity & Continuity Criteria

1. **Validity Threshold**: An hour is considered valid if `coverage_ratio >= 0.90` (at least 54 raw minute-level observations in that hour).
   - Valid hours: **10,043** (98.13%)
   - Invalid hours: **191** (1.87%)
2. **Invalid Hour Policy**: Invalid hours are **not deleted or dropped** from the hourly sequence to avoid silent timestamp warping. They are explicitly marked with `is_valid_hour = False`.
3. **Continuous Run Identification**:
   - A valid run consists strictly of contiguous valid hours where consecutive timestamps are exactly 1 hour apart (`t[i] - t[i-1] == 1h`) and all hours have `is_valid_hour == True`.
   - Any invalid hour or missing timestamp breaks the run.
   - Identified: **71 continuous runs**. Longest run: **1,501 hours** (~62.5 days).

### Forecasting Windows

- **Window Slicing Rule**: A valid 192-hour window can only be extracted from within a continuous run of length $\ge 192$.
- **Window Count Formula**: For a run of length $L \ge 192$, the number of sliding windows is $L - 192 + 1$.
- **Total Valid Windows**: **5,871 windows** across all runs.
- **Earliest Forecast Origin**: `2022-12-09 09:00:00`
- **Latest Forecast Origin**: `2024-01-02 00:00:00` (target extends to `2024-01-03 00:00:00`).

### Generated Artifacts

| File | Description | Shape / Rows |
|---|---|---|
| `data/interim/forecast_ready_hourly.csv` | Forecast-ready hourly data with validation flags | 10,234 rows × 12 columns |
| `data/interim/window_index.csv` | Metadata index for all 5,871 valid forecasting windows | 5,871 rows × 6 columns |
| `outputs/reports/step36_forecast_window_report.txt` | Complete audit log and summary statistics | Text report |

---

## Step 3.7: Chronological Dataset Splitting & Boundary Audit

### Splitting Strategy & Proportions

The 5,871 valid forecasting windows are partitioned strictly chronologically based on `forecast_origin` ascending. No random shuffling or random sampling is permitted.

- **Intended Proportions**:
  - **Train**: 65%
  - **Validation**: 15%
  - **Calibration**: 10%
  - **Test**: 10%
- **Partition Roles**:
  - **Train**: Used exclusively for model optimization and future scaler parameter fitting.
  - **Validation**: Used for hyperparameter tuning, model selection, and early stopping.
  - **Calibration**: Held out exclusively for conformal prediction interval calibration. Kept separate from test.
  - **Test**: Held out untouched for final out-of-sample evaluation and robustness stress tests.

### Resulting Partition Counts

| Partition | Window Count | Percentage | Earliest Forecast Origin | Latest Forecast Origin |
|---|---|---|---|---|
| **TRAIN** | 3,816 | 65.00% | 2022-12-09 09:00:00 | 2023-09-22 15:00:00 |
| **VALIDATION** | 881 | 15.01% | 2023-09-22 16:00:00 | 2023-11-14 02:00:00 |
| **CALIBRATION** | 587 | 10.00% | 2023-11-14 03:00:00 | 2023-12-08 13:00:00 |
| **TEST** | 587 | 10.00% | 2023-12-08 14:00:00 | 2024-01-02 00:00:00 |
| **TOTAL** | **5,871** | **100.00%** | **2022-12-09 09:00:00** | **2024-01-02 00:00:00** |

### Chronological Ordering & Disjointness

Strict ordering is verified via assertions:
$$\max(\text{train.origin}) < \min(\text{val.origin}) < \max(\text{val.origin}) < \min(\text{cal.origin}) < \max(\text{cal.origin}) < \min(\text{test.origin})$$

All window IDs are 100% mutually exclusive across partitions (zero duplicate window IDs).

### Boundary Overlap Audit

Due to sliding windows having 168 hours of history:
- Early validation windows use historical observations that occurred during the final training target window.
- Early calibration windows use historical observations from the validation period.
- Early test windows use historical observations from the calibration period.
- **Critical Leakage Barrier**: The **TEST target period** (`2023-12-08 15:00` to `2024-01-03 00:00`) has **ZERO** overlap with Train target or Train history. Train ends months before test begins (`2023-09-23 15:00`). Test targets remain completely unobserved.

### Generated Artifacts

| File | Description | Rows |
|---|---|---|
| `data/interim/splits/train_windows.csv` | Train split window metadata | 3,816 rows |
| `data/interim/splits/validation_windows.csv` | Validation split window metadata | 881 rows |
| `data/interim/splits/calibration_windows.csv` | Calibration split window metadata | 587 rows |
| `data/interim/splits/test_windows.csv` | Test split window metadata | 587 rows |
| `outputs/reports/step37_split_report.txt` | Complete audit log and boundary report | Text report |

---

## Step 3.8: Leakage-Safe Numerical Feature Scaling

### Scaler Methodology

To prevent data leakage, **all scaling parameters are fitted exclusively on training data**. Neither validation, calibration, nor test data have any influence on fitted scaling statistics.

- **Scaler Type**: `StandardScaler` ($z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$)
- **Modality Separation**: Two independent scalers are fitted to maintain modality autonomy:
  1. **Energy Scaler**: Fits and transforms `hourly_active_power` (also used for future target scaling).
  2. **Weather Scaler**: Fits and transforms 7 primary weather variables: `temp`, `feels_like`, `humidity`, `pressure`, `speed`, `wind_sin`, `wind_cos`.
- **Fitting Region**:
  - Training window temporal boundary: `2022-12-02 10:00:00` to `2023-09-23 15:00:00`.
  - Scalers are fitted on the **6,943 unique valid hourly observations** in this period (not on overlapping window concatenations).
- **Target Scaling Consistency**: The forecast target (`hourly_active_power`) is scaled using the same `EnergyScaler` fitted on training energy consumption.
- **Unscaled Metadata**: `date`, `n_observations`, `coverage_ratio`, and `is_valid_hour` are preserved unscaled.
- **Excluded Features**: Categorical weather, range weather (`temp_min`, `temp_max`), future weather (`temp_t+1`, `feels_like_t+1`), and secondary electrical features remain excluded.

### Fitted Training Scaling Parameters

| Modality | Feature | Training Mean ($\mu_{\text{train}}$) | Training Std ($\sigma_{\text{train}}$) |
|---|---|---|---|
| **Energy** | `hourly_active_power` | 307.1706 W | 147.9934 W |
| **Weather** | `temp` | 20.8340 °C | 6.6328 °C |
| | `feels_like` | 20.0103 °C | 6.7180 °C |
| | `humidity` | 44.3086 % | 18.6776 % |
| | `pressure` | 1014.2579 hPa | 5.6040 hPa |
| | `speed` | 2.8659 m/s | 2.0485 m/s |
| | `wind_sin` | 0.2090 | 0.4146 |
| | `wind_cos` | 0.2102 | 0.7899 |

### Scaled Partition Statistics (Valid Hours)

| Partition | Date Span | Rows | Mean (Active Power) | Std (Active Power) | Mean (Temp) | Mean (Humidity) |
|---|---|---|---|---|---|---|
| **Train** | `2022-12-02 10:00` to `2023-09-23 15:00` | 7,086 | -0.0000 | 1.0000 | 0.0000 | 0.0000 |
| **Validation** | `2023-09-15 17:00` to `2023-11-15 02:00` | 1,450 | -0.3209 | 0.8060 | -0.1957 | 0.4747 |
| **Calibration** | `2023-11-07 04:00` to `2023-12-09 13:00` | 778 | -0.6017 | 0.6250 | -0.7014 | 0.7944 |
| **Test** | `2023-12-01 15:00` to `2024-01-03 00:00` | 778 | -0.6423 | 0.6722 | -0.9964 | 0.7436 |

*Note: Validation, calibration, and test means and standard deviations naturally shift due to seasonal temperature drops and energy consumption patterns in late autumn and winter.*

### Numerical Verification & Leakage Checks

- **Numerical Round-Trip Check**: Inverse-transform reconstruction on training data achieves max error of $1.14 \times 10^{-13}$ ($< 10^{-10}$).
- **Leakage Barrier**: Validation, calibration, and test data were transformed strictly using `scaler.transform()` with zero calls to `fit()` or `fit_transform()`.
- **Raw File Invariance**: `energy_weather_raw_data.csv` verified untouched (76,799,474 bytes).

### Generated Artifacts

| File | Description | Details |
|---|---|---|
| `outputs/models/scalers/energy_scaler.pkl` | Serialized scikit-learn StandardScaler for energy | 1 feature (`hourly_active_power`) |
| `outputs/models/scalers/weather_scaler.pkl` | Serialized scikit-learn StandardScaler for weather | 7 weather features |
| `outputs/models/scalers/scaler_metadata.json` | Machine-readable scaler parameters & metadata | JSON metadata |
| `data/processed/scaled/train_scaled.csv` | Scaled training hourly dataset | 7,086 rows × 12 cols |
| `data/processed/scaled/validation_scaled.csv` | Scaled validation hourly dataset | 1,450 rows × 12 cols |
| `data/processed/scaled/calibration_scaled.csv` | Scaled calibration hourly dataset | 778 rows × 12 cols |
| `data/processed/scaled/test_scaled.csv` | Scaled test hourly dataset | 778 rows × 12 cols |

---

## Step 3.9: Model-Ready Sequence Construction

### Sequence Structure & Causal Design

The scaled hourly datasets from Step 3.8 and the authoritative window metadata from Step 3.7 were transformed into multidimensional NumPy sequence tensors:

- **Historical Context ($H$)**: 168 hours ($t-167 \dots t$)
- **Forecast Horizon ($F$)**: 24 hours ($t+1 \dots t+24$)
- **Causal Architecture**:
  - `X_energy`: $[N, 168, 1]$ (scaled `hourly_active_power` history)
  - `X_weather`: $[N, 168, 7]$ (scaled `temp`, `feels_like`, `humidity`, `pressure`, `speed`, `wind_sin`, `wind_cos` history)
  - `y`: $[N, 24]$ (scaled future `hourly_active_power` target)
- **Zero Future-Weather Leakage**: $X_{\text{weather}}$ contains exclusively history up to forecast origin $t$. Observed future weather ($t+1 \dots t+24$) is never supplied to any modality.

### Partition Shapes & Sample Counts

| Partition | Samples ($N$) | `X_energy` Shape | `X_weather` Shape | `y` Shape | Date Range (Origins) |
|---|---|---|---|---|---|
| **TRAIN** | **3,816** | `(3816, 168, 1)` | `(3816, 168, 7)` | `(3816, 24)` | `2022-12-09 09:00` to `2023-09-22 15:00` |
| **VALIDATION** | **881** | `(881, 168, 1)` | `(881, 168, 7)` | `(881, 24)` | `2023-09-22 16:00` to `2023-11-14 02:00` |
| **CALIBRATION** | **587** | `(587, 168, 1)` | `(587, 168, 7)` | `(587, 24)` | `2023-11-14 03:00` to `2023-12-08 13:00` |
| **TEST** | **587** | `(587, 168, 1)` | `(587, 168, 7)` | `(587, 24)` | `2023-12-08 14:00` to `2024-01-02 00:00` |
| **TOTAL** | **5,871** | — | — | — | Exactly matches Step 3.7 |

### Sequence Integrity & Quality Verification

1. **Strict Continuity**: Every sequence was verified to consist of 192 consecutive valid hours ($168\text{h history} + 24\text{h target}$). Zero sequences span invalid hours or missing timestamps.
2. **No Imputation / Interpolation**: Zero missing value filling or synthetic data generation was applied.
3. **No Rescaling**: All sequences remain in the Step 3.8 training-fitted standard scale.
4. **NaN & Inf Immunity**: 0 NaNs and 0 infinite values across all generated arrays.
5. **Inverse-Scale Sanity Check**: De-normalizing $y$ targets using the training `EnergyScaler` yields realistic household power values in watts:
   - Train target: min = 53.98 W, max = 909.60 W, mean = 297.70 W
   - Validation target: min = 57.50 W, max = 969.08 W, mean = 265.42 W
   - Calibration target: min = 58.49 W, max = 660.69 W, mean = 219.53 W
   - Test target: min = 47.87 W, max = 538.19 W, mean = 203.59 W
6. **Raw CSV Invariance**: `energy_weather_raw_data.csv` verified untouched (76,799,474 bytes).

### Generated Artifacts

| Partition | Directory | Saved Files |
|---|---|---|
| Train | `data/processed/sequences/train/` | `X_energy.npz`, `X_weather.npz`, `y.npz`, `metadata.json` |
| Validation | `data/processed/sequences/validation/` | `X_energy.npz`, `X_weather.npz`, `y.npz`, `metadata.json` |
| Calibration | `data/processed/sequences/calibration/` | `X_energy.npz`, `X_weather.npz`, `y.npz`, `metadata.json` |
| Test | `data/processed/sequences/test/` | `X_energy.npz`, `X_weather.npz`, `y.npz`, `metadata.json` |
| Audit Report | `outputs/reports/` | `step39_sequence_report.txt` |




