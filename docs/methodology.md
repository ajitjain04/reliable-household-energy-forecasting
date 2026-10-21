# Methodology: Reliable Multimodal Household Energy Forecasting Under Degraded Weather Information

## 1. Problem Formulation and Forecasting Task

### 1.1 Context and Task Definition
The objective of this research is multi-step ahead residential electricity load forecasting under nominal and degraded operating regimes. Specifically, given a fixed temporal historical lookback window of 168 hours (7 days) of hourly electrical power observations and exogenous weather observations, the models forecast the subsequent 24 hours of hourly active electricity consumption:

$$\hat{\mathbf{y}}_{t+1:t+H} = f_\theta(\mathbf{x}^{\text{energy}}_{t-W+1:t}, \mathbf{x}^{\text{weather}}_{t-W+1:t})$$

where:
- Lookback window length: $W = 168$ hours (1 week of continuous historical dynamics).
- Forecast horizon: $H = 24$ hours ahead (multi-step autoregressive rollout).
- Target variable: $\mathbf{y} \in \mathbb{R}^H$, representing `hourly_active_power` in original Watts ($W$).
- Historical electrical modality: $\mathbf{x}^{\text{energy}} \in \mathbb{R}^{W \times 1}$ (`hourly_active_power`).
- Historical exogenous weather modality: $\mathbf{x}^{\text{weather}} \in \mathbb{R}^{W \times 7}$.

### 1.2 Modal Features and Circular Transformations
The raw minute-level dataset spanning November 2022 to January 2024 is aggregated to hourly resolution with rigorous data hygiene:
1. **Target Construction**: `hourly_active_power` is computed as the hourly arithmetic mean of instantaneous `active_power` (W).
2. **Exogenous Weather Variables**: Seven engineered weather channels are incorporated:
   - `temp`: Ambient temperature ($^\circ\text{C}$)
   - `feels_like`: Apparent temperature / heat index ($^\circ\text{C}$)
   - `humidity`: Relative humidity (%)
   - `pressure`: Atmospheric sea-level pressure (hPa)
   - `speed`: Wind velocity (m/s)
   - `wind_sin`: Sine component of circular wind direction ($\sin(\theta_{\text{rad}})$)
   - `wind_cos`: Cosine component of circular wind direction ($\cos(\theta_{\text{rad}})$)
3. **Leakage Prevention**: Future forecast weather columns (`temp_t+1`, `feels_like_t+1`) are strictly excluded from the observation matrix to prevent temporal look-ahead leakage.

### 1.3 Data Standardization
Features are standardized independently using parameters fitted solely on the training partition:
$$z_t = \frac{x_t - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- Energy scaler: $\mu = 307.1706\text{ W}$, $\sigma = 147.9934\text{ W}$.
- Scalers are saved via `joblib` and accessed read-only; test partitions are never refit.

---

## 2. Chronological Data Partitioning

To preserve temporal ordering and prevent cross-partition leakage, a four-way chronological split is enforced:

| Partition | Time Interval | Total Hours | Usable Rolling Windows ($W=168, H=24$) | Role in Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | Nov 2022 – May 2023 | 4,400 h | 4,209 | Model training & gradient updates |
| **Validation** | May 2023 – Jul 2023 | 881 h | 881 | Checkpoint selection & early stopping |
| **Calibration** | Jul 2023 – Sep 2023 | 587 h | 587 | Conformal quantile calibration ($q_h$) |
| **Test** | Sep 2023 – Nov 2023 | 587 h | 587 | Final out-of-sample benchmark (14,088 pts) |

---

## 3. Neural Architecture Designs

Four distinct architectures evaluate different degrees of cross-modal inductive bias:

```
Energy-Only Baseline:
  x_energy (168, 1) ---> LSTM(64, 2 layers) ---> Dense(64, 24) ---> y_hat (24)

Early Fusion:
  [x_energy, x_weather] (168, 8) ---> LSTM(64, 2 layers) ---> Dense(64, 24) ---> y_hat (24)

Late Fusion:
  x_energy (168, 1)  ---> LSTM_e(64, 2) ---> h_e (64) ---\
                                                            [h_e, h_w] (128) ---> Dense(128, 24) ---> y_hat (24)
  x_weather (168, 7) ---> LSTM_w(64, 2) ---> h_w (64) ---/

Hybrid Fusion:
  x_energy (168, 1)  ---> LSTM_e(64, 2) ---\
                                             ---> Dense_fuse(128, 64) ---> Shared_LSTM(64, 1) ---> Dense(64, 24) ---> y_hat (24)
  x_weather (168, 7) ---> LSTM_w(64, 2) ---/
```

### 3.1 Architectural Specifications
1. **Energy-Only Baseline (51,992 parameters)**:
   - Input: $(B, 168, 1)$
   - Recurrent Backbone: 2-layer stacked LSTM with hidden dimension $d_h = 64$
   - Readout Head: Linear projection layer $\mathbf{W}_{\text{out}} \in \mathbb{R}^{64 \times 24}$
2. **Early Fusion LSTM (53,784 parameters)**:
   - Input: $(B, 168, 8)$ (joint channel-wise concatenation at each time step)
   - Recurrent Backbone: 2-layer stacked LSTM with hidden dimension $d_h = 64$
   - Readout Head: Linear projection $\mathbb{R}^{64 \times 24}$
3. **Late Fusion LSTM (112,216 parameters)**:
   - Modal Encoders: Two independent 2-layer LSTMs ($d_{h,e} = 64$, $d_{h,w} = 64$)
   - Fusion Bottleneck: Concatenation of terminal hidden states $\mathbf{h} = [\mathbf{h}_e^{(T)}; \mathbf{h}_w^{(T)}] \in \mathbb{R}^{128}$
   - Readout Head: Linear projection $\mathbb{R}^{128 \times 24}$
4. **Hybrid Fusion LSTM (145,496 parameters)**:
   - Modal Encoders: Two independent 2-layer LSTMs producing sequence representations
   - Intermediate Fusion: Linear feature projection $\mathbb{R}^{128 \to 64}$ at every lag step
   - Shared Temporal Encoder: 1-layer recurrent LSTM ($d_h = 64$) processing fused sequences
   - Readout Head: Linear projection $\mathbb{R}^{64 \times 24}$

---

## 4. Controlled Sensor Degradation Framework

To evaluate operational reliability under real-world telemetry failures, models are audited across a $2 \times 2$ factorial matrix under two distinct degradation modalities:

### 4.1 Missingness Regime (30% MCAR)
Simulates packet loss, intermittent sensor dropouts, or communication transmission failure:
- **Mechanism**: Missing Completely At Random (MCAR) with probability $p_{\text{miss}} = 0.30$.
- **Imputation Strategy**: Mean-value imputation in standardized space ($z_{\text{imputed}} = 0.0$, corresponding to training mean $307.17\text{ W}$ for power and seasonal means for weather).

### 4.2 Noise Regime (20% Additive Gaussian Noise)
Simulates calibration drift, thermal instrumentation noise, or analog-to-digital conversion noise:
- **Mechanism**: Zero-mean Additive White Gaussian Noise (AWGN):
  $$\tilde{z}_t = z_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{noise}}^2), \quad \sigma_{\text{noise}} = 0.20$$

### 4.3 Factorial Experimental Conditions
- **N-N (Nominal-Nominal)**: Clean historical energy, clean historical weather.
- **N-A (Nominal-Anomalous)**: Clean historical energy, degraded historical weather.
- **A-N (Anomalous-Nominal)**: Degraded historical energy, clean historical weather.
- **A-A (Anomalous-Anomalous)**: Degraded historical energy, degraded historical weather.

---

## 5. Statistical Inference & Hypothesis Testing Battery

The inferential protocol rigorously evaluates 78 paired hypothesis tests across 6 structured families using paired window-level MAE ($N = 587$ windows):

1. **Family 1**: Clean Model Comparisons (N-N) — 6 pairwise model tests.
2. **Family 2**: Model Comparisons under 30% Missingness — 18 tests ($3 \times 6$ pairwise tests).
3. **Family 3**: Model Comparisons under 20% Noise — 18 tests ($3 \times 6$ pairwise tests).
4. **Family 4**: Within-Model Missingness Degradation — 12 tests (vs Clean N-N).
5. **Family 5**: Within-Model Noise Degradation — 12 tests (vs Clean N-N).
6. **Family 6**: Cross-Degradation Sensitivity (Missingness vs Noise) — 12 tests.

### 5.1 Inferential Test Battery
- **Primary Nonparametric Test**: Two-sided Wilcoxon Signed-Rank Test.
- **Family-Wise Error Rate (FWER) Control**: Step-down Holm-Bonferroni correction ($p_{\text{holm}}$).
- **False Discovery Rate (FDR) Control**: Benjamini-Hochberg adjustment ($q_{\text{BH}}$).
- **Exact Resampling**: Paired Permutation Test ($B = 10,000$ iterations).
- **Time-Series Forecast Comparison**: Harvey-Leybourne-Newbery (HLN) modified Diebold-Mariano test with Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) covariance adjustments ($h = 24$ lags).

---

## 6. Explainable AI (XAI) & Attribution Protocol

To quantify modality reliance and temporal memory across architectures, **Integrated Gradients (IG)** is computed:

$$\text{Attribution}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

- **Baseline ($x'$)**: Zero-vector in standardized space ($x' = \mathbf{0}$), representing long-term training feature means.
- **Interpolation Steps**: $m = 20$ Gauss-Legendre quadrature steps.
- **Scope**: Computed across all 587 out-of-sample test windows ($N_{\text{eval}} = 587$) without subsampling.
- **Metrics**:
  - Modality Share (% Attribution allocated to Electricity vs Weather).
  - Per-Channel Attribution Density (Modality Share divided by number of channels).
  - Temporal Attribution Profiles across lookback lags $t-168$ to $t-1$.

---

## 7. Uncertainty Quantification: Split Conformal Prediction

Distribution-free predictive uncertainty intervals are constructed using **Horizon-Wise Absolute-Residual Split Conformal Prediction**:

### 7.1 Calibration Protocol
Given calibration residuals $e_{i, h} = |y_{i, h} - \hat{y}_{i, h}|$ for window $i \in \{1, \dots, N_{\text{cal}}\}$ ($N_{\text{cal}} = 587$) at horizon $h \in \{1, \dots, 24\}$, the conformal quantile threshold $q_h$ is defined at significance level $\alpha = 0.10$ (90% nominal coverage):

$$k = \left\lceil (N_{\text{cal}} + 1)(1 - \alpha) \right\rceil = \lceil 588 \times 0.90 \rceil = 530$$
$$q_h = e_{(k), h}$$

### 7.2 Interval Evaluation
Prediction intervals for test window $t$ are given by:
$$\mathcal{C}_h(\mathbf{x}_t) = [\hat{y}_{t, h} - q_h, \; \hat{y}_{t, h} + q_h]$$

Evaluated metrics:
- **Prediction Interval Coverage Probability (PICP)**:
  $$\text{PICP} = \frac{1}{N_{\text{test}} \times H} \sum_{i=1}^{N_{\text{test}}} \sum_{h=1}^H \mathbb{I}(y_{i, h} \in \mathcal{C}_h(\mathbf{x}_i))$$
- **Mean Prediction Interval Width (MPIW)**:
  $$\text{MPIW} = \frac{1}{H} \sum_{h=1}^H 2 q_h$$
- **Winkler Interval Score (IS)**: Penalizes interval width and boundary violations:
  $$\text{IS}_\alpha(l, u, y) = (u - l) + \frac{2}{\alpha}(l - y)\mathbb{I}(y < l) + \frac{2}{\alpha}(y - u)\mathbb{I}(y > u)$$
