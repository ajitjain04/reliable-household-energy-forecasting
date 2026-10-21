/**
 * API client module for the FastAPI backend.
 * Points to http://127.0.0.1:8000/api/v1 by default.
 */

import {
  HealthResponse,
  CleanBenchmarkResponse,
  RobustnessResponse,
  StatisticalSummaryResponse,
  FamilyDetailResponse,
  XAISummaryResponse,
  TemporalLagRecord,
  WeatherFeatureRecord,
  ConformalSummaryResponse,
  ConformalWindowIntervalResponse,
  TestWindowsMetadataResponse,
  WindowForecastResponse,
  LocationInput,
  ForecastRunRequest,
  ForecastRunResponse,
  ForecastRecordDetail,
  ForecastRecordListResponse,
  ForecastStatusResponse,
  UserProfile,
  AuthResponse,
  ExtractedBillData,
  MonthlyForecastRequest,
  MonthlyForecastResponse,
  MonthlyForecastRecordDetail,
  MonthlyForecastRecordListResponse,
  SaveMonthlyForecastRequest,
  ApplianceScenarioRequest,
  ApplianceScenarioResponse,
  EnergyCoachOverviewResponse,
  AppliancePreset,
  HouseholdScenarioComparisonRequest,
  HouseholdScenarioComparisonResponse,
  ScenarioComparisonStep,
} from './types';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';
const TOKEN_KEY = 'household_energy_auth_token';

class ApiClient {
  getToken(): string | null {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  }

  setToken(token: string | null): void {
    try {
      if (token) {
        localStorage.setItem(TOKEN_KEY, token);
      } else {
        localStorage.removeItem(TOKEN_KEY);
      }
    } catch {
      // ignore storage write errors
    }
  }

  private async request<T>(
    endpoint: string,
    options?: {
      method?: 'GET' | 'POST';
      body?: any;
      isFormData?: boolean;
    }
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    try {
      const headers: Record<string, string> = {
        'Accept': 'application/json',
      };

      const token = this.getToken();
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      let requestBody: any = undefined;

      if (options?.isFormData) {
        requestBody = options.body;
      } else if (options?.body) {
        headers['Content-Type'] = 'application/json';
        requestBody = JSON.stringify(options.body);
      }

      const response = await fetch(url, {
        method: options?.method || 'GET',
        headers,
        body: requestBody,
      });

      if (!response.ok) {
        let errorMsg = `HTTP Error ${response.status}: ${response.statusText}`;
        try {
          const errorJson = await response.json();
          if (errorJson.detail) {
            errorMsg = typeof errorJson.detail === 'string'
              ? errorJson.detail
              : JSON.stringify(errorJson.detail);
          }
        } catch {
          // ignore json parse error
        }
        throw new Error(errorMsg);
      }

      return (await response.json()) as T;
    } catch (err: unknown) {
      if (err instanceof Error) {
        throw err;
      }
      throw new Error('Unknown network error occurred while connecting to research API.');
    }
  }

  // ---------------------------------------------------------------------------
  // System Health
  // ---------------------------------------------------------------------------
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/health');
  }

  // ---------------------------------------------------------------------------
  // Research Benchmarks & Experiments
  // ---------------------------------------------------------------------------
  async getCleanBenchmarks(): Promise<CleanBenchmarkResponse> {
    return this.request<CleanBenchmarkResponse>('/benchmarks/clean');
  }

  async getRobustnessMatrix(params?: {
    model?: string;
    experiment?: string;
    condition?: string;
  }): Promise<RobustnessResponse> {
    const query = new URLSearchParams();
    if (params?.model) query.append('model', params.model);
    if (params?.experiment) query.append('experiment', params.experiment);
    if (params?.condition) query.append('condition', params.condition);
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return this.request<RobustnessResponse>(`/robustness/matrix${queryString}`);
  }

  async getStatisticalSummary(): Promise<StatisticalSummaryResponse> {
    return this.request<StatisticalSummaryResponse>('/statistics/summary');
  }

  async getFamilyDetail(familyKey: string): Promise<FamilyDetailResponse> {
    return this.request<FamilyDetailResponse>(`/statistics/family/${encodeURIComponent(familyKey)}`);
  }

  async getStatisticalFamilyDetail(familyKey: string | number): Promise<FamilyDetailResponse> {
    return this.request<FamilyDetailResponse>(`/statistics/family/${encodeURIComponent(familyKey.toString())}`);
  }

  async getXAISummary(): Promise<XAISummaryResponse> {
    return this.request<XAISummaryResponse>('/xai/summary');
  }

  async getXAITemporalLags(params?: {
    model?: string;
    condition?: string;
    experiment?: string;
  }): Promise<TemporalLagRecord[]> {
    const query = new URLSearchParams();
    if (params?.model) query.append('model', params.model);
    if (params?.condition) query.append('condition', params.condition);
    if (params?.experiment) query.append('experiment', params.experiment);
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return this.request<TemporalLagRecord[]>(`/xai/temporal-lags${queryString}`);
  }

  async getXAITemporal(params?: {
    model?: string;
    condition?: string;
    experiment?: string;
  }): Promise<TemporalLagRecord[]> {
    return this.getXAITemporalLags(params);
  }

  async getXAIWeatherFeatures(params?: {
    model?: string;
    condition?: string;
    experiment?: string;
  }): Promise<WeatherFeatureRecord[]> {
    const query = new URLSearchParams();
    if (params?.model) query.append('model', params.model);
    if (params?.condition) query.append('condition', params.condition);
    if (params?.experiment) query.append('experiment', params.experiment);
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return this.request<WeatherFeatureRecord[]>(`/xai/weather-features${queryString}`);
  }

  async getConformalSummary(): Promise<ConformalSummaryResponse> {
    return this.request<ConformalSummaryResponse>('/conformal/summary');
  }

  async getConformalIntervals(
    model: string,
    condition: string,
    windowIdx: number = 0
  ): Promise<ConformalWindowIntervalResponse> {
    return this.request<ConformalWindowIntervalResponse>(
      `/conformal/intervals/${encodeURIComponent(model)}/${encodeURIComponent(condition)}?window_idx=${windowIdx}`
    );
  }

  async getTestWindowsMetadata(): Promise<TestWindowsMetadataResponse> {
    return this.request<TestWindowsMetadataResponse>('/forecasts/test-windows');
  }

  async getTestWindowForecast(windowIdx: number): Promise<WindowForecastResponse> {
    return this.request<WindowForecastResponse>(`/forecasts/test-window/${windowIdx}`);
  }

  // ---------------------------------------------------------------------------
  // Live Forecast Execution & Persistence
  // ---------------------------------------------------------------------------
  async runForecast(payload: ForecastRunRequest): Promise<ForecastRunResponse> {
    return this.request<ForecastRunResponse>('/forecasts/run', {
      method: 'POST',
      body: payload,
    });
  }

  async uploadForecastCsv(
    file: File,
    model: string = 'auto',
    userId?: string,
    location?: LocationInput
  ): Promise<ForecastRunResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    if (userId) {
      formData.append('user_id', userId);
    }
    if (location && location.latitude !== undefined && location.longitude !== undefined) {
      formData.append('latitude', location.latitude.toString());
      formData.append('longitude', location.longitude.toString());
      if (location.label) {
        formData.append('location_label', location.label);
      }
    }
    return this.request<ForecastRunResponse>('/forecasts/upload-csv', {
      method: 'POST',
      body: formData,
      isFormData: true,
    });
  }

  // ---------------------------------------------------------------------------
  // Monthly Electricity Bill Ingestion & Review (Phase 2 & Task 2.6B)
  // ---------------------------------------------------------------------------
  async parseMonthlyBillPdf(file: File): Promise<ExtractedBillData> {
    return this.parseMonthlyBillFile(file);
  }

  async parseMonthlyBillFile(file: File): Promise<ExtractedBillData> {
    const formData = new FormData();
    formData.append('file', file);
    return this.request<ExtractedBillData>('/forecasts/monthly/parse-bill', {
      method: 'POST',
      body: formData,
      isFormData: true,
    });
  }

  async runMonthlyForecast(payload: MonthlyForecastRequest): Promise<MonthlyForecastResponse> {
    return this.request<MonthlyForecastResponse>('/forecasts/monthly/run', {
      method: 'POST',
      body: payload,
    });
  }

  async saveMonthlyForecast(payload: SaveMonthlyForecastRequest): Promise<MonthlyForecastRecordDetail> {
    return this.request<MonthlyForecastRecordDetail>('/forecasts/monthly/save', {
      method: 'POST',
      body: payload,
    });
  }

  async getMonthlyForecastRecords(params?: {
    limit?: number;
    offset?: number;
  }): Promise<MonthlyForecastRecordListResponse> {
    const query = new URLSearchParams();
    if (params?.limit !== undefined) query.append('limit', params.limit.toString());
    if (params?.offset !== undefined) query.append('offset', params.offset.toString());
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return this.request<MonthlyForecastRecordListResponse>(`/forecasts/monthly/records${queryString}`);
  }

  async getMonthlyForecastRecord(runId: string): Promise<MonthlyForecastRecordDetail> {
    return this.request<MonthlyForecastRecordDetail>(`/forecasts/monthly/records/${encodeURIComponent(runId)}`);
  }

  async getForecastRecords(params?: {
    limit?: number;
    offset?: number;
    status?: string;
    model?: string;
  }): Promise<ForecastRecordListResponse> {
    const query = new URLSearchParams();
    if (params?.limit !== undefined) query.append('limit', params.limit.toString());
    if (params?.offset !== undefined) query.append('offset', params.offset.toString());
    if (params?.status) query.append('status', params.status);
    if (params?.model) query.append('model', params.model);
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return this.request<ForecastRecordListResponse>(`/forecasts/records${queryString}`);
  }

  async getDeveloperForecastRecords(params?: {
    limit?: number;
    offset?: number;
    status?: string;
    model?: string;
  }): Promise<ForecastRecordListResponse> {
    const query = new URLSearchParams();
    if (params?.limit !== undefined) query.append('limit', params.limit.toString());
    if (params?.offset !== undefined) query.append('offset', params.offset.toString());
    if (params?.status) query.append('status', params.status);
    if (params?.model) query.append('model', params.model);
    const queryString = query.toString() ? `?${query.toString()}` : '';
    return this.request<ForecastRecordListResponse>(`/forecasts/developer/records${queryString}`);
  }

  async getForecastRecord(runId: string): Promise<ForecastRecordDetail> {
    return this.request<ForecastRecordDetail>(`/forecasts/records/${encodeURIComponent(runId)}`);
  }

  async getForecastStatus(runId: string): Promise<ForecastStatusResponse> {
    return this.request<ForecastStatusResponse>(`/forecasts/records/${encodeURIComponent(runId)}/status`);
  }

  // ---------------------------------------------------------------------------
  // Authentication & Session
  // ---------------------------------------------------------------------------
  async login(username: string, password: string): Promise<AuthResponse> {
    const res = await this.request<AuthResponse>('/auth/login', {
      method: 'POST',
      body: { username, password },
    });
    if (res.access_token) {
      this.setToken(res.access_token);
    }
    return res;
  }

  async register(username: string, password: string): Promise<UserProfile> {
    return this.request<UserProfile>('/auth/register', {
      method: 'POST',
      body: { username, password },
    });
  }

  async getMe(): Promise<UserProfile> {
    return this.request<UserProfile>('/auth/me');
  }

  async logout(): Promise<void> {
    try {
      await this.request<{ message: string }>('/auth/logout', { method: 'POST' });
    } catch {
      // ignore server logout errors
    } finally {
      this.setToken(null);
    }
  }

  // ---------------------------------------------------------------------------
  // PDF Report Download
  // ---------------------------------------------------------------------------
  async downloadForecastPdf(runId: string): Promise<void> {
    const url = `${API_BASE_URL}/forecasts/records/${encodeURIComponent(runId)}/pdf`;
    const headers: Record<string, string> = {};
    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      method: 'GET',
      headers,
    });

    if (!response.ok) {
      let errorMsg = `PDF Download Failed (${response.status}): ${response.statusText}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore json parse error
      }
      throw new Error(errorMsg);
    }

    const blob = await response.blob();
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = `forecast_report_${runId}.pdf`;
    if (contentDisposition) {
      const match = contentDisposition.match(/filename=["']?([^"';]+)["']?/i);
      if (match && match[1]) {
        filename = match[1];
      }
    }

    const objectUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(objectUrl);
  }

  async downloadMonthlyForecastPdf(runId: string): Promise<void> {
    const url = `${API_BASE_URL}/forecasts/monthly/records/${encodeURIComponent(runId)}/report.pdf`;
    const headers: Record<string, string> = {};
    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      method: 'GET',
      headers,
    });

    if (!response.ok) {
      let errorMsg = `Monthly Forecast PDF Download Failed (${response.status}): ${response.statusText}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // ignore json parse error
      }
      throw new Error(errorMsg);
    }

    const blob = await response.blob();
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = `monthly_forecast_report_${runId}.pdf`;
    if (contentDisposition) {
      const match = contentDisposition.match(/filename=["']?([^"';]+)["']?/i);
      if (match && match[1]) {
        filename = match[1];
      }
    }

    const objectUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(objectUrl);
  }

  // ---------------------------------------------------------------------------
  // Household Energy Coach & Appliance Simulator (Task 2.7)
  // ---------------------------------------------------------------------------
  async getCoachOverview(): Promise<EnergyCoachOverviewResponse> {
    return this.request<EnergyCoachOverviewResponse>('/coach/overview');
  }

  async simulateApplianceScenario(
    payload: ApplianceScenarioRequest
  ): Promise<ApplianceScenarioResponse> {
    return this.request<ApplianceScenarioResponse>('/coach/simulate-appliance', {
      method: 'POST',
      body: payload,
    });
  }

  async getAppliancePresets(): Promise<Record<string, AppliancePreset>> {
    return this.request<Record<string, AppliancePreset>>('/coach/presets');
  }

  async compareHouseholdScenario(
    payload: HouseholdScenarioComparisonRequest
  ): Promise<HouseholdScenarioComparisonResponse> {
    try {
      return await this.request<HouseholdScenarioComparisonResponse>('/coach/compare-household-scenario', {
        method: 'POST',
        body: payload,
      });
    } catch (err: unknown) {
      // Graceful fallback if backend server route table has not reloaded the new endpoint
      const is404 =
        err instanceof Error &&
        (err.message.includes('404') || err.message.toLowerCase().includes('not found'));
      if (is404) {
        const applianceRes = await this.simulateApplianceScenario(payload.appliance_scenario);
        const delta = applianceRes.delta_kwh;
        const appBaseline = applianceRes.baseline_kwh;
        let totalBase = 0;
        let totalAdj: number | null = 0;
        let totalDelta: number | null = 0;
        let totalSavings: number | null =
          applianceRes.is_tariff_verified && applianceRes.estimated_savings_inr != null ? 0 : null;

        // Physical plausibility validation:
        // Do not subtract appliance reduction if appliance baseline or reduction exceeds household forecast
        let isSupported = true;
        for (const baseKwh of payload.baseline_forecast_kwh) {
          if (delta > 0) {
            if (appBaseline > baseKwh || delta >= baseKwh || (baseKwh - delta) <= 0) {
              isSupported = false;
              break;
            }
          } else if (delta < 0) {
            if (appBaseline > baseKwh && payload.appliance_scenario.baseline.hours_per_day > 0) {
              isSupported = false;
              break;
            }
          }
        }

        const fallbackTariffStatus = applianceRes.is_tariff_verified
          ? 'verified'
          : payload.provider_id === 'msedcl'
          ? 'expired_schedule'
          : payload.provider_id
          ? 'unverified_schedule'
          : 'no_provider';
        const fallbackTariffLabel = applianceRes.is_tariff_verified
          ? 'Verified'
          : payload.provider_id === 'msedcl'
          ? 'Expired schedule'
          : payload.provider_id
          ? 'Current tariff unverified'
          : 'No provider selected';

        const steps: ScenarioComparisonStep[] = payload.baseline_forecast_kwh.map((baseKwh, idx) => {
          totalBase += baseKwh;
          const monthLabel = payload.month_labels[idx] || `Month ${idx + 1}`;

          if (!isSupported) {
            return {
              month_index: idx + 1,
              month_label: monthLabel,
              baseline_kwh: Number(baseKwh.toFixed(2)),
              adjusted_kwh: null,
              delta_kwh: null,
              baseline_bill_amount: null,
              adjusted_bill_amount: null,
              savings_amount: null,
              tariff_status: fallbackTariffStatus,
              tariff_status_label: fallbackTariffLabel,
              is_adjustment_supported: false,
              unsupported_reason: `Assumed appliance baseline (${appBaseline.toFixed(1)} kWh) or reduction (${delta.toFixed(1)} kWh) exceeds household baseline forecast (${baseKwh.toFixed(1)} kWh).`,
            };
          }

          const adjKwh = Number((baseKwh - delta).toFixed(2));
          const stepDelta = Number((baseKwh - adjKwh).toFixed(2));
          if (totalAdj !== null) totalAdj += adjKwh;
          if (totalDelta !== null) totalDelta += stepDelta;
          const stepSaving =
            totalSavings !== null && applianceRes.estimated_savings_inr != null
              ? applianceRes.estimated_savings_inr
              : null;
          if (totalSavings !== null && stepSaving !== null) {
            totalSavings += stepSaving;
          }
          return {
            month_index: idx + 1,
            month_label: monthLabel,
            baseline_kwh: Number(baseKwh.toFixed(2)),
            adjusted_kwh: adjKwh,
            delta_kwh: stepDelta,
            baseline_bill_amount: applianceRes.estimated_baseline_cost_inr,
            adjusted_bill_amount: applianceRes.estimated_modified_cost_inr,
            savings_amount: stepSaving,
            tariff_status: fallbackTariffStatus,
            tariff_status_label: fallbackTariffLabel,
            is_adjustment_supported: true,
            unsupported_reason: null,
          };
        });

        return {
          steps,
          total_baseline_kwh: Number(totalBase.toFixed(2)),
          total_adjusted_kwh: isSupported && totalAdj !== null ? Number(totalAdj.toFixed(2)) : null,
          total_delta_kwh: isSupported && totalDelta !== null ? Number(totalDelta.toFixed(2)) : null,
          total_savings_inr: isSupported && totalSavings !== null ? Number(totalSavings.toFixed(2)) : null,
          is_verified_tariff: applianceRes.is_tariff_verified && isSupported,
          tariff_status: fallbackTariffStatus,
          tariff_status_label: fallbackTariffLabel,
          is_household_adjustment_supported: isSupported,
          household_adjustment_reason: isSupported
            ? null
            : `Household scenario adjustment withheld: The hypothetical appliance baseline (${appBaseline.toFixed(1)} kWh/month) or reduction (${delta.toFixed(1)} kWh/month) exceeds the total household forecast.`,
          appliance_result: applianceRes,
          recommendations: [],
          disclosures: applianceRes.disclosures,
        };
      }
      throw err;
    }
  }
}

export const api = new ApiClient();


