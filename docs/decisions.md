# Architectural Decision Records (ADRs)

Project: **Reliable Multimodal Household Energy Forecasting Under Degraded Weather Information**

---

## ADR-001: 1-Hour Temporal Aggregation and Target Formulation

- **Status**: Accepted & Audited
- **Context**: The raw smart meter telemetry is sampled at 1-minute intervals (605,260 rows). Forecasting minute-by-minute active power over a multi-day horizon introduces excessive high-frequency noise and extreme computational burden without corresponding utility for grid demand response planning.
- **Decision**: 
  1. Floor all timestamps to the beginning of each hour.
  2. Compute `hourly_active_power` as the arithmetic mean of minute-level `active_power` within the hour (unit: Watts).
  3. Define `hourly_active_power` as the primary regression target.
- **Consequences**:
  - Eliminates minute-level transient spikes.
  - Represents continuous average power output rather than cumulative kWh consumption.
  - Yields 10,234 hourly observations for model training, validation, calibration, and testing.

---

## ADR-002: Circular Wind Feature Encoding & Future Weather Exclusion

- **Status**: Accepted & Audited
- **Context**: Wind direction is recorded in degrees ($0^\circ - 360^\circ$). Standard arithmetic averaging fails across the $0^\circ / 360^\circ$ discontinuity. Furthermore, raw datasets often contain forward-shifted weather columns (`temp_t+1`, `feels_like_t+1`).
- **Decision**:
  1. Transform wind direction to radians: $\theta_{\text{rad}} = \theta \times \pi / 180$.
  2. Compute orthogonal trigonometric components: $\text{wind\_sin} = \sin(\theta_{\text{rad}})$ and $\text{wind\_cos} = \cos(\theta_{\text{rad}})$.
  3. Strictly exclude `temp_t+1` and `feels_like_t+1` from the feature matrix.
- **Consequences**:
  - Ensures continuous, smooth wind representations.
  - Guarantees zero lookahead data leakage in autoregressive forecast rollouts.

---

## ADR-003: 4-Way Chronological Data Split Without Reshuffling

- **Status**: Accepted & Audited
- **Context**: Standard cross-validation with random shuffling breaks temporal autocorrelation and causes cross-contamination between training lookbacks and test targets. Furthermore, conformal prediction requires an independent, exchangeable calibration split distinct from training checkpoint validation.
- **Decision**: Implement a strictly chronological 4-way split:
  - **Train**: 4,400 hours (Nov 2022 – May 2023)
  - **Validation**: 881 hours (May 2023 – Jul 2023)
  - **Calibration**: 587 hours (Jul 2023 – Sep 2023)
  - **Test**: 587 hours (Sep 2023 – Nov 2023)
- **Consequences**:
  - Prevents data leakage.
  - Guarantees independent evaluation of conformal interval coverage.
  - Reflects real-world deployment conditions over changing seasons.

---

## ADR-004: Multimodal Fusion Topology Selection

- **Status**: Accepted & Audited
- **Context**: We seek to determine whether auxiliary weather information improves load forecasting and whether complex multimodal decoupling yields superior operational robustness.
- **Decision**: Implement four comparative architectures:
  1. **Energy-Only Baseline**: 2-layer unimodal LSTM (51,992 parameters).
  2. **Early Fusion**: Step-wise channel concatenation feeding a joint 2-layer LSTM (53,784 parameters).
  3. **Late Fusion**: Decoupled unimodal LSTM encoders concatenated at terminal hidden state (112,216 parameters).
  4. **Hybrid Fusion**: Decoupled encoders projected into a shared sequence representation before recurrent readout (145,496 parameters).
- **Consequences**:
  - Avoids overly complex attention mechanisms not suited for modest sample sizes.
  - Directly tests early vs late vs hybrid fusion hypotheses under identical capacity and training budgets.

---

## ADR-005: 2x2 Factorial Sensor Degradation Protocols

- **Status**: Accepted & Audited
- **Context**: Field sensor networks face communication dropouts (missingness) and instrumentation drift/interference (noise). Testing only clean data provides a misleading assessment of real-world reliability.
- **Decision**: Audit all models across a $2 \times 2$ factorial grid under two degradation modalities:
  1. **30% MCAR Missingness**: $p = 0.30$ random masking with zero-imputation in standardized feature space (mean imputation).
  2. **20% AWGN Noise**: Additive Gaussian noise with $\sigma = 0.20$ in standardized space.
  3. **Grid**: Clean (N-N), Weather-Degraded (N-A), Electricity-Degraded (A-N), and Joint-Degraded (A-A).
- **Consequences**:
  - Isolates individual and interactive modality sensitivities.
  - Demonstrates that electricity telemetry is significantly more critical than weather telemetry.

---

## ADR-006: Split Conformal Prediction for Uncertainty Quantification

- **Status**: Accepted & Audited
- **Context**: Parametric prediction intervals (e.g., Gaussian assumptions) frequently fail in non-Gaussian, skewed residential load distributions.
- **Decision**: Adopt Horizon-Wise Absolute-Residual Split Conformal Prediction:
  1. Calibrate conformal nonconformity scores on the held-out calibration split ($N = 587$) at $\alpha = 0.10$.
  2. Apply the finite-sample order statistic $k = \lceil(N+1)(1-\alpha)\rceil = 530$.
  3. Evaluate fixed empirical intervals across all clean and degraded test regimes.
- **Consequences**:
  - Guarantees finite-sample marginal coverage under clean exchangeability.
  - Transparently exposes coverage erosion under sensor degradation without artificially widening intervals.

---

## ADR-007: Multiple Testing Correction and HAC Covariance Adjustments

- **Status**: Accepted & Audited
- **Context**: Conducting 78 paired hypothesis tests across 6 families inflates the Family-Wise Type I Error rate ($1 - (1 - 0.05)^{78} \approx 98.2\%$). In addition, rolling test windows share 23 hours of overlap, violating classical i.i.d. assumptions.
- **Decision**:
  1. Group tests into 6 coherent hypothesis families.
  2. Apply step-down Holm-Bonferroni correction within each family as the primary inferential criterion.
  3. Supplement with Benjamini-Hochberg False Discovery Rate (FDR) adjustments.
  4. Perform Harvey-Leybourne-Newbery modified Diebold-Mariano tests with Newey-West HAC covariance corrections ($h=24$ lags) and paired permutation tests ($B=10,000$).
- **Consequences**:
  - Rigorous control of false positives.
  - Statistically sound treatment of autocorrelated rolling-window time series.

---

## ADR-008: Dual Joblib/Pickle Serialization Layer

- **Status**: Accepted & Audited
- **Context**: Scikit-learn `StandardScaler` objects saved via `joblib.dump` can fail when loaded via raw Python standard library `pickle.load` due to internal joblib array wrappers.
- **Decision**: Implement a unified loader that attempts `joblib.load` first, followed by a transparent fallback to `pickle.load`.
- **Consequences**:
  - Guarantees bulletproof serialization compatibility across CLI scripts, tests, notebooks, and web applications.
