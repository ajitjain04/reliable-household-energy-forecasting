/**
 * TypeScript definitions for research API responses.
 * Matched strictly to FastAPI Pydantic schemas.
 */

export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  timestamp: string;
  project_root: string;
  artifacts_present: boolean;
  artifact_dirs: Record<string, boolean>;
}

export interface CleanBenchmarkItem {
  model: string;
  architecture_type: string;
  trainable_parameters: number;
  history_hours: number;
  horizon_hours: number;
  input_features: string;
  val_mae_watts: number;
  val_rmse_watts: number;
  val_r2: number;
  val_mape_pct: number;
  val_smape_pct: number;
  test_mae_watts: number;
  test_rmse_watts: number;
  test_r2: number;
  test_mape_pct: number;
  test_smape_pct: number;
  delta_mae_vs_energy_only_watts: number;
  relative_delta_mae_pct: number;
  clean_test_rank: number;
}

export interface CleanBenchmarkResponse {
  source_artifact: string;
  total_models: number;
  benchmark_leader: string;
  records: CleanBenchmarkItem[];
}

export interface RobustnessRecord {
  experiment: string;
  experiment_label: string;
  model: string;
  condition: string;
  condition_description: string;
  test_mae_watts: number;
  delta_mae_vs_clean_watts: number;
  relative_mae_change_pct: number;
  test_rmse_watts: number;
  delta_rmse_vs_clean_watts: number;
  relative_rmse_change_pct: number;
  test_r2: number;
  conformal_picp_pct: number;
  conformal_ace_pct: number;
  conformal_mpiw_watts: number;
  winkler_interval_score: number;
  oracle_mpiw_for_90_coverage_watts: number;
}

export interface RobustnessResponse {
  source_artifact: string;
  total_evaluations: number;
  available_experiments: string[];
  available_models: string[];
  available_conditions: string[];
  records: RobustnessRecord[];
}

// Statistics Types
export interface HypothesisFamilySummary {
  family_id: number;
  family_name: string;
  total_tests: number;
  significant_tests_holm: number;
  significant_tests_fdr: number;
  primary_inference: string;
}

export interface StatisticalSummaryResponse {
  source_artifact: string;
  total_families: number;
  total_planned_tests: number;
  total_significant_holm: number;
  overall_rejection_rate_pct: number;
  families: HypothesisFamilySummary[];
}

export interface GranularTestRecord {
  test_id: number;
  family_id: number;
  comparison_name: string;
  model_1: string;
  cond_1: string;
  model_2: string;
  cond_2: string;
  n_windows: number;
  mean_diff_watts: number;
  mean_ci_95_low: number;
  mean_ci_95_high: number;
  rel_diff_pct: number;
  rank_biserial_r_rb: number;
  wilcoxon_stat: number;
  wilcoxon_p_raw: number;
  wilcoxon_p_holm: number;
  sig_holm: boolean;
  wilcoxon_p_fdr: number;
  sig_fdr: boolean;
  hln_stat: number | null;
  hln_p_raw: number | null;
  decision: string;
}

export interface FamilyDetailResponse {
  family_id: number;
  family_name: string;
  total_tests: number;
  significant_tests_holm: number;
  primary_inference: string;
  granular_tests: GranularTestRecord[];
}

// XAI Types
export interface XAISummaryRecord {
  model: string;
  clean_electricity_share_pct: number;
  clean_weather_share_pct: number;
  clean_electricity_density_share_pct: number;
  clean_weather_density_share_pct: number;
  miss_NA_elec_shift_pct_pts: number;
  miss_AN_elec_shift_pct_pts: number;
  miss_AA_elec_shift_pct_pts: number;
  noise_AA_elec_shift_pct_pts: number;
  electricity_temporal_recent_24h_pct: number;
  electricity_temporal_intermediate_48h_pct: number;
  electricity_temporal_distant_96h_pct: number;
  top_weather_features_clean: string;
}

export interface XAISummaryResponse {
  source_artifact: string;
  method: string;
  records: XAISummaryRecord[];
}

export interface TemporalLagRecord {
  model: string;
  experiment: string;
  condition: string;
  bin_name: string;
  description: string;
  n_hours: number;
  attribution_magnitude_mean: number;
  temporal_share_pct: number;
}

export interface WeatherFeatureRecord {
  model: string;
  experiment: string;
  condition: string;
  feature_name: string;
  attribution_magnitude_mean: number;
  attribution_share_within_weather_pct: number;
  feature_rank: number;
}

// Conformal Prediction Types
export interface ConformalSummaryRecord {
  model: string;
  calibration_mae_watts: number;
  conformal_quantile_qh_mean_watts: number;
  mpiw_watts: number;
  clean_picp_pct: number;
  clean_ace_pct: number;
  clean_winkler_interval_score: number;
  miss_NA_picp_pct: number;
  miss_AN_picp_pct: number;
  miss_AA_picp_pct: number;
  miss_AA_winkler_score: number;
  noise_AA_picp_pct: number;
  noise_AA_winkler_score: number;
  oracle_mpiw_miss_AA_watts: number;
  oracle_width_expansion_req_pct: number;
}

export interface ConformalSummaryResponse {
  source_artifact: string;
  confidence_level_pct: number;
  alpha: number;
  calibration_samples: number;
  records: ConformalSummaryRecord[];
}

export interface ConformalWindowIntervalResponse {
  model: string;
  condition: string;
  window_idx: number;
  forecast_origin_timestamp: string;
  horizon_hours: number[];
  target_timestamps: string[];
  L_raw: number[];
  U_raw: number[];
  y_hat: number[];
  y_true: number[];
  is_covered: boolean[];
  window_picp_pct: number;
  window_interval_score: number;
  dataset_mean_picp_pct: number;
  dataset_mean_mpiw_watts: number;
}

// Forecast Explorer Types
export interface TestWindowsMetadataResponse {
  total_windows: number;
  history_length_hours: number;
  forecast_horizon_hours: number;
  first_forecast_origin: string;
  last_forecast_origin: string;
  first_target_start: string;
  last_target_end: string;
  available_models: string[];
  window_index_range: number[];
}

export interface WindowModelMetrics {
  mae_watts: number;
  rmse_watts: number;
  max_error_watts: number;
}

export interface WindowForecastResponse {
  window_idx: number;
  forecast_origin_timestamp: string;
  horizon_hours: number[];
  target_timestamps: string[];
  actual_watts: number[];
  predictions_watts: Record<string, number[]>;
  metrics: Record<string, WindowModelMetrics>;
}

// ----------------------------------------------------------------------------
// Live Forecast & Persistent Record Types
// ----------------------------------------------------------------------------
export interface WeatherSeriesInput {
  temp: number[];
  feels_like: number[];
  humidity: number[];
  pressure: number[];
  speed: number[];
  wind_sin?: number[];
  wind_cos?: number[];
  deg?: number[];
}

export interface LocationInput {
  latitude: number;
  longitude: number;
  label?: string;
}

export interface ForecastRunRequest {
  electricity_active_power: number[];
  timestamps?: string[];
  weather?: WeatherSeriesInput | Record<string, number[]>;
  location?: LocationInput;
  model?: string;
  user_id?: string;
}

export interface ConformalIntervalStep {
  horizon_hour: number;
  target_timestamp?: string | null;
  prediction_watts: number;
  lower_bound_watts: number;
  upper_bound_watts: number;
  lower_bound_clipped_watts: number;
  interval_width_watts: number;
}

export interface ModelMetadataInfo {
  model_name: string;
  model_class: string;
  trainable_parameters: number;
  history_length_hours: number;
  forecast_horizon_hours: number;
  energy_scaler_type: string;
  weather_scaler_type?: string | null;
}

export interface ForecastRunResponse {
  run_id: string;
  selected_model: string;
  model_key: string;
  selection_mode: string;
  selection_reason: string;
  forecast_origin_timestamp?: string | null;
  horizon_hours: number[];
  forecast_timestamps?: string[] | null;
  predictions_watts: number[];
  intervals: ConformalIntervalStep[];
  mean_interval_width_watts: number;
  nominal_coverage_pct: number;
  confidence_explanation: string;
  units: string;
  model_metadata: ModelMetadataInfo;
}

export interface ForecastRecordSummary {
  run_id: string;
  created_at: string;
  status: 'pending' | 'succeeded' | 'failed' | string;
  input_method: 'api' | 'csv' | string;
  user_id?: string | null;
  owner_user_id?: string | null;
  selected_model?: string | null;
  model_key?: string | null;
  selection_mode?: string | null;
  selection_reason?: string | null;
  forecast_origin_timestamp?: string | null;
  mean_interval_width_watts?: number | null;
  error_code?: string | null;
  error_message?: string | null;
}

export interface ForecastRecordDetail extends ForecastRecordSummary {
  predictions_watts?: number[] | null;
  intervals?: ConformalIntervalStep[] | null;
  input_metadata?: Record<string, any> | null;
  model_metadata?: Record<string, any> | null;
}

export interface ForecastRecordListResponse {
  total_count: number;
  limit: number;
  offset: number;
  records: ForecastRecordSummary[];
}

export interface ForecastStatusResponse {
  run_id: string;
  created_at: string;
  status: string;
  input_method: string;
  selected_model?: string | null;
  error_code?: string | null;
  error_message?: string | null;
}

// -----------------------------------------------------------------------------
// Authentication and Access Control Types
// -----------------------------------------------------------------------------
export interface UserProfile {
  id: string;
  username: string;
  role: 'household' | 'developer' | string;
  created_at: string;
  is_active: boolean;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: UserProfile;
}

// -----------------------------------------------------------------------------
// Monthly Electricity Bill Ingestion & Review Types (Phase 2)
// -----------------------------------------------------------------------------
export interface MonthlyConsumptionRecord {
  billing_period_start: string;
  billing_period_end: string;
  consumption_kwh: number;
  bill_amount?: number | null;
  period_label?: string | null;
}

export interface ExtractedBillData {
  records: MonthlyConsumptionRecord[];
  extraction_source: 'pdf' | 'image' | 'manual' | 'provider_api' | 'mock' | string;
  extraction_confidence?: number | null;
  warnings: string[];
  consumer_number?: string | null;
  billing_unit?: string | null;
  rr_number?: string | null;
  utility_provider?: string | null;
  raw_metadata?: Record<string, any> | null;
}

export interface UserCorrectedBillData {
  records: MonthlyConsumptionRecord[];
  source: string;
  notes?: string | null;
  confirmed_by_user: boolean;
  warnings: string[];
}

// -----------------------------------------------------------------------------
// Monthly Electricity Demand Forecasting Types (Phase 2)
// -----------------------------------------------------------------------------
export interface BillEstimateResponse {
  provider_id?: string | null;
  provider_name?: string | null;
  state?: string | null;
  is_tariff_supported: boolean;
  is_tariff_verified: boolean;
  tariff_status?: string | null;
  tariff_status_label?: string | null;
  forecast_kwh: number;
  billing_period_start?: string | null;
  billing_period_end?: string | null;
  estimated_bill_amount?: number | null;
  charge_breakdown?: Record<string, any> | null;
  currency: string;
  disclosures: string[];
}

export interface MonthlyForecastStep {
  month_index: number;
  month_label: string;
  period_start: string;
  period_end: string;
  forecast_kwh: number;
  uncertainty_lower_kwh?: number | null;
  uncertainty_upper_kwh?: number | null;
  bill_estimate?: BillEstimateResponse | null;
  estimated_bill_amount?: number | null;
  tariff_status?: string | null;
  tariff_status_label?: string | null;
}

export interface BacktestMetrics {
  backtest_periods: number;
  mae_kwh: number;
  rmse_kwh: number;
  baseline_mae_kwh: number;
  smape_percent?: number | null;
}

export interface MonthlyForecastSummary {
  total_forecast_kwh: number;
  average_monthly_kwh: number;
  total_estimated_bill_inr?: number | null;
  months_with_bill_estimates?: number;
  bill_estimate_notes?: string | null;
  currency: string;
}

export interface MonthlyForecastRequest {
  records: MonthlyConsumptionRecord[];
  horizon_months?: number;
  provider_id?: string | null;
  sanctioned_load_kw?: number | null;
  phase_type?: 'single_phase' | 'three_phase' | string;
  is_live?: boolean;
  reference_date?: string | null;
}

export interface MonthlyForecastResponse {
  forecast_id: string;
  created_at: string;
  history_length_months: number;
  last_observed_period?: string | null;
  last_observed_end_date?: string | null;
  selected_method:
    | 'baseline_only'
    | 'weighted_moving_average'
    | 'damped_trend_moving_average'
    | 'seasonal_naive'
    | 'holt_winters'
    | string;
  method_description: string;
  confidence_rating: 'none' | 'low' | 'moderate' | 'high' | string;
  is_stale_history: boolean;
  warnings: string[];
  limitations: string[];
  forecast_steps: MonthlyForecastStep[];
  backtest_metrics?: BacktestMetrics | null;
  summary: MonthlyForecastSummary;
}

// -----------------------------------------------------------------------------
// Persistent Monthly Forecast History Types (Task 2.4)
// -----------------------------------------------------------------------------
export interface MonthlyForecastRecordSummary {
  run_id: string;
  created_at: string;
  owner_user_id: string;
  horizon_months: number;
  provider_id?: string | null;
  phase_type?: string | null;
  sanctioned_load_kw?: number | null;
  is_live: boolean;
  selected_method: string;
  confidence_rating: string;
  history_length_months: number;
  last_observed_period?: string | null;
  total_forecast_kwh: number;
  average_monthly_kwh: number;
  total_estimated_bill_inr?: number | null;
  months_with_bill_estimates: number;
  bill_estimate_notes?: string | null;
  forecast_period_range?: string | null;
  warnings_count: number;
}

export interface MonthlyForecastRecordDetail extends MonthlyForecastRecordSummary {
  method_description: string;
  last_observed_end_date?: string | null;
  is_stale_history: boolean;
  warnings: string[];
  limitations: string[];
  forecast_steps: MonthlyForecastStep[];
  summary: MonthlyForecastSummary;
  backtest_metrics?: BacktestMetrics | null;
  input_records: MonthlyConsumptionRecord[];
}

export interface MonthlyForecastRecordListResponse {
  total_count: number;
  limit: number;
  offset: number;
  records: MonthlyForecastRecordSummary[];
}

export interface SaveMonthlyForecastRequest {
  request: MonthlyForecastRequest;
  forecast_id?: string;
}

// -----------------------------------------------------------------------------
// Household Energy Coach Types (Task 2.7)
// -----------------------------------------------------------------------------
export type ApplianceCategory =
  | 'air_conditioner'
  | 'refrigerator'
  | 'water_heater'
  | 'washing_machine'
  | 'ceiling_fan'
  | 'lighting'
  | 'television'
  | 'other';

export interface ApplianceSimulationInput {
  category: ApplianceCategory;
  name: string;
  power_watts: number;
  hours_per_day: number;
  days_per_month: number;
  quantity: number;
}

export interface ApplianceScenarioRequest {
  baseline: ApplianceSimulationInput;
  modified: ApplianceSimulationInput;
  provider_id?: string | null;
  phase_type?: string | null;
  current_monthly_total_kwh?: number | null;
}

export interface ApplianceScenarioResponse {
  baseline_kwh: number;
  modified_kwh: number;
  delta_kwh: number;
  is_tariff_verified: boolean;
  estimated_baseline_cost_inr?: number | null;
  estimated_modified_cost_inr?: number | null;
  estimated_savings_inr?: number | null;
  disclosures: string[];
  appliance_notes: string[];
}

export interface CoachInsight {
  id: string;
  category: 'trend' | 'baseline' | 'forecast' | 'data_quality';
  title: string;
  description: string;
  evidence: string;
  severity: 'positive' | 'warning' | 'info' | 'neutral';
}

export interface CoachRecommendation {
  id: string;
  title: string;
  description: string;
  action_type: 'behavioral' | 'appliance' | 'data_setup';
  rationale: string;
  is_hypothetical: boolean;
  estimated_monthly_kwh_reduction?: number | null;
  estimated_monthly_savings_inr?: number | null;
  savings_disclaimer?: string | null;
}

export interface EnergyCoachOverviewResponse {
  has_data: boolean;
  latest_consumption_kwh?: number | null;
  latest_period_label?: string | null;
  latest_billing_end_date?: string | null;
  latest_bill_amount_inr?: number | null;
  previous_consumption_kwh?: number | null;
  previous_period_label?: string | null;
  consumption_change_kwh?: number | null;
  consumption_change_percent?: number | null;
  historical_records: MonthlyConsumptionRecord[];
  latest_forecast_run_id?: string | null;
  latest_forecast_created_at?: string | null;
  latest_forecast_horizon_months?: number | null;
  latest_forecast_period_range?: string | null;
  latest_forecast_total_kwh?: number | null;
  latest_forecast_average_kwh?: number | null;
  latest_forecast_confidence?: string | null;
  forecast_steps: MonthlyForecastStep[];
  provider_id?: string | null;
  provider_name?: string | null;
  is_tariff_verified: boolean;
  data_freshness_status: 'current' | 'stale' | 'insufficient_history' | 'no_records';
  warnings: string[];
  insights: CoachInsight[];
  recommendations: CoachRecommendation[];
}

export interface AppliancePreset {
  name: string;
  default_power_watts: number;
  default_hours_per_day: number;
  default_days_per_month: number;
  default_quantity: number;
  duty_cycle_factor: number;
  notes: string;
}

export interface ScenarioComparisonStep {
  month_index: number;
  month_label: string;
  baseline_kwh: number;
  adjusted_kwh?: number | null;
  delta_kwh?: number | null;
  baseline_bill_amount?: number | null;
  adjusted_bill_amount?: number | null;
  savings_amount?: number | null;
  is_adjustment_supported?: boolean;
  unsupported_reason?: string | null;
  tariff_status?: string | null;
  tariff_status_label?: string | null;
}

export interface HouseholdScenarioComparisonRequest {
  baseline_forecast_kwh: number[];
  month_labels: string[];
  appliance_scenario: ApplianceScenarioRequest;
  provider_id?: string | null;
  phase_type?: string | null;
  sanctioned_load_kw?: number | null;
  period_starts?: string[] | null;
  period_ends?: string[] | null;
}

export interface HouseholdScenarioComparisonResponse {
  steps: ScenarioComparisonStep[];
  total_baseline_kwh: number;
  total_adjusted_kwh?: number | null;
  total_delta_kwh?: number | null;
  total_savings_inr?: number | null;
  is_verified_tariff: boolean;
  tariff_status?: string | null;
  tariff_status_label?: string | null;
  is_household_adjustment_supported?: boolean;
  household_adjustment_reason?: string | null;
  appliance_result: ApplianceScenarioResponse;
  recommendations: CoachRecommendation[];
  disclosures: string[];
}



