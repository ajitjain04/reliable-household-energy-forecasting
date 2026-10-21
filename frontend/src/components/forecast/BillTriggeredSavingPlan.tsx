import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  TrendingUp,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  ShieldCheck,
  ShieldAlert,
  Info,
  Zap,
  ArrowRight,
  RefreshCw,
  Plus,
  FileText,
  Clock,
  History,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

import { api } from '../../api/client';
import {
  MonthlyConsumptionRecord,
  ExtractedBillData,
  MonthlyForecastResponse,
  MonthlyForecastStep,
  ApplianceCategory,
  AppliancePreset,
  ApplianceScenarioRequest,
  HouseholdScenarioComparisonResponse,
} from '../../api/types';
import { useTheme } from '../../context/ThemeContext';

export function getStepTariffInfo(
  step: MonthlyForecastStep,
  selectedProvider: string
): {
  status: 'verified' | 'expired_schedule' | 'unverified_schedule' | 'no_provider';
  label: string;
  withheldLabel: string;
  badgeClass: string;
  icon: 'check' | 'clock' | 'alert' | 'none';
  tooltip: string;
} {
  const status = step.tariff_status || step.bill_estimate?.tariff_status;
  const label = step.tariff_status_label || step.bill_estimate?.tariff_status_label;

  if (status === 'verified') {
    return {
      status: 'verified',
      label: label || 'Verified',
      withheldLabel: 'Withheld (Unverified)',
      badgeClass: 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
      icon: 'check',
      tooltip: 'MERC LT-I(B) verified tariff schedule active for this billing cycle.',
    };
  }

  if (status === 'expired_schedule') {
    return {
      status: 'expired_schedule',
      label: label || 'Expired schedule',
      withheldLabel: 'Withheld (Expired Tariff)',
      badgeClass: 'bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
      icon: 'clock',
      tooltip: 'Tariff schedule expired on 2025-03-31; not verified for this forecast period.',
    };
  }

  if (status === 'unverified_schedule') {
    return {
      status: 'unverified_schedule',
      label: label || 'Current tariff unverified',
      withheldLabel: 'Withheld (Unverified)',
      badgeClass: 'bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
      icon: 'alert',
      tooltip: 'No verified tariff schedule available for this provider.',
    };
  }

  if (status === 'no_provider') {
    return {
      status: 'no_provider',
      label: label || 'No provider selected',
      withheldLabel: 'Withheld (No Provider)',
      badgeClass: 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700',
      icon: 'none',
      tooltip: 'Select an electricity provider to evaluate applicable tariffs.',
    };
  }

  // Fallback if status was not provided directly on the step
  if (step.estimated_bill_amount !== null && step.estimated_bill_amount !== undefined) {
    return {
      status: 'verified',
      label: 'Verified',
      withheldLabel: 'Withheld (Unverified)',
      badgeClass: 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
      icon: 'check',
      tooltip: 'Verified tariff schedule active for this billing cycle.',
    };
  }

  const normId = (selectedProvider || '').trim().toLowerCase();
  if (normId === 'msedcl') {
    const pDate = step.period_start ? new Date(step.period_start) : null;
    if (pDate && !isNaN(pDate.getTime()) && pDate <= new Date('2025-03-31T23:59:59Z')) {
      return {
        status: 'verified',
        label: 'Verified',
        withheldLabel: 'Withheld (Unverified)',
        badgeClass: 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
        icon: 'check',
        tooltip: 'MERC LT-I(B) verified tariff schedule active for this billing cycle.',
      };
    }
    return {
      status: 'expired_schedule',
      label: 'Expired schedule',
      withheldLabel: 'Withheld (Expired Tariff)',
      badgeClass: 'bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
      icon: 'clock',
      tooltip: 'MSEDCL LT-I(B) schedule expired on 2025-03-31; not verified for this forecast period.',
    };
  }

  if (normId) {
    return {
      status: 'unverified_schedule',
      label: 'Current tariff unverified',
      withheldLabel: 'Withheld (Unverified)',
      badgeClass: 'bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
      icon: 'alert',
      tooltip: 'No verified tariff schedule available for this provider.',
    };
  }

  return {
    status: 'no_provider',
    label: 'No provider selected',
    withheldLabel: 'Withheld (No Provider)',
    badgeClass: 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-700',
    icon: 'none',
    tooltip: 'Select an electricity provider to evaluate applicable tariffs.',
  };
}

interface BillTriggeredSavingPlanProps {
  isConfirmed: boolean;
  confirmedRecords: MonthlyConsumptionRecord[];
  extractedBillData: ExtractedBillData | null;
  selectedProviderId: string;
  phaseType: 'single_phase' | 'three_phase';
  sanctionedLoadKw: string;
  forecastHorizon: number;
  onForecastHorizonChange: (h: number) => void;
  monthlyForecastResult: MonthlyForecastResponse | null;
  isForecasting: boolean;
  onGenerateForecast: () => Promise<void>;
  onSaveForecast: () => Promise<void>;
  isSaving: boolean;
  savedRunId: string | null;
}

export const BillTriggeredSavingPlan: React.FC<BillTriggeredSavingPlanProps> = ({
  isConfirmed,
  confirmedRecords,
  extractedBillData,
  selectedProviderId,
  phaseType,
  sanctionedLoadKw,
  forecastHorizon,
  onForecastHorizonChange,
  monthlyForecastResult,
  isForecasting,
  onGenerateForecast,
  onSaveForecast,
  isSaving,
  savedRunId,
}) => {
  const { isDark } = useTheme();

  // Appliance What-If Simulator State
  const [presets, setPresets] = useState<Record<string, AppliancePreset>>({});
  const [selectedCategory, setSelectedCategory] = useState<ApplianceCategory>('air_conditioner');
  const [applianceName, setApplianceName] = useState<string>('1.5 Ton Split AC');
  const [powerWatts, setPowerWatts] = useState<number>(1450);
  const [baselineHours, setBaselineHours] = useState<number>(8.0);
  const [modifiedHours, setModifiedHours] = useState<number>(7.0);
  const [daysPerMonth, setDaysPerMonth] = useState<number>(30);
  const [quantity, setQuantity] = useState<number>(1);

  // Scenario Comparison API Results
  const [comparisonResult, setComparisonResult] = useState<HouseholdScenarioComparisonResponse | null>(null);
  const [isEvaluatingScenario, setIsEvaluatingScenario] = useState<boolean>(false);
  const [scenarioError, setScenarioError] = useState<string | null>(null);

  // Load Appliance Presets
  useEffect(() => {
    let isMounted = true;
    api
      .getAppliancePresets()
      .then((data) => {
        if (isMounted) setPresets(data);
      })
      .catch(() => {});
    return () => {
      isMounted = false;
    };
  }, []);

  // When category changes, apply standard BEE preset values
  const handleCategoryChange = (cat: ApplianceCategory) => {
    setSelectedCategory(cat);
    const p = presets[cat];
    if (p) {
      setApplianceName(p.name);
      setPowerWatts(p.default_power_watts);
      setBaselineHours(p.default_hours_per_day);
      setModifiedHours(Math.max(0, p.default_hours_per_day - 1.0));
      setDaysPerMonth(p.default_days_per_month);
      setQuantity(p.default_quantity);
    } else {
      setApplianceName(cat === 'other' ? 'Custom Appliance' : cat.replace('_', ' '));
    }
    setComparisonResult(null);
  };

  // Evaluate household scenario comparison against the active forecast steps
  const handleEvaluateScenario = async () => {
    if (!monthlyForecastResult || monthlyForecastResult.forecast_steps.length === 0) {
      setScenarioError('Please generate an upcoming monthly forecast before running scenario comparisons.');
      return;
    }

    setIsEvaluatingScenario(true);
    setScenarioError(null);

    const baselineSteps = monthlyForecastResult.forecast_steps;
    const baselineKwhSeries = baselineSteps.map((s) => s.forecast_kwh);
    const monthLabels = baselineSteps.map((s) => s.month_label);
    const periodStarts = baselineSteps.map((s) => s.period_start);
    const periodEnds = baselineSteps.map((s) => s.period_end);

    const latestConfirmedKwh = confirmedRecords.length > 0 ? confirmedRecords[confirmedRecords.length - 1].consumption_kwh : undefined;

    const payload: ApplianceScenarioRequest = {
      baseline: {
        category: selectedCategory,
        name: applianceName,
        power_watts: powerWatts,
        hours_per_day: baselineHours,
        days_per_month: daysPerMonth,
        quantity: quantity,
      },
      modified: {
        category: selectedCategory,
        name: `${applianceName} (Proposed)`,
        power_watts: powerWatts,
        hours_per_day: modifiedHours,
        days_per_month: daysPerMonth,
        quantity: quantity,
      },
      provider_id: selectedProviderId.trim() || undefined,
      phase_type: phaseType,
      current_monthly_total_kwh: latestConfirmedKwh,
    };

    try {
      const res = await api.compareHouseholdScenario({
        baseline_forecast_kwh: baselineKwhSeries,
        month_labels: monthLabels,
        appliance_scenario: payload,
        provider_id: selectedProviderId.trim() || undefined,
        phase_type: phaseType,
        sanctioned_load_kw: sanctionedLoadKw.trim() ? parseFloat(sanctionedLoadKw) : undefined,
        period_starts: periodStarts,
        period_ends: periodEnds,
      });
      setComparisonResult(res);
    } catch (err: unknown) {
      setScenarioError(err instanceof Error ? err.message : 'Failed to evaluate household scenario comparison.');
    } finally {
      setIsEvaluatingScenario(false);
    }
  };

  // Trigger scenario evaluation automatically once a forecast is ready
  useEffect(() => {
    if (isConfirmed && monthlyForecastResult && monthlyForecastResult.forecast_steps.length > 0) {
      handleEvaluateScenario();
    }
  }, [monthlyForecastResult?.forecast_id]);

  if (!isConfirmed) {
    return null;
  }

  const latestRecord = confirmedRecords.length > 0 ? confirmedRecords[confirmedRecords.length - 1] : null;

  // Chart data preparing baseline vs scenario (adjustedKwh omitted if household adjustment is withheld)
  const isAdjustmentSupported = comparisonResult?.is_household_adjustment_supported !== false;
  const chartData = (monthlyForecastResult?.forecast_steps || []).map((step, idx) => {
    const compStep = comparisonResult?.steps[idx];
    return {
      name: step.month_label,
      baselineKwh: step.forecast_kwh,
      adjustedKwh: isAdjustmentSupported && compStep && compStep.adjusted_kwh != null ? compStep.adjusted_kwh : undefined,
      baselineBill: step.estimated_bill_amount,
      adjustedBill: isAdjustmentSupported && compStep ? compStep.adjusted_bill_amount : undefined,
    };
  });

  return (
    <section
      id="electricity-saving-plan"
      data-testid="saving-plan-section"
      className="mt-8 bg-gradient-to-b from-white to-slate-50/50 dark:from-slate-900 dark:to-slate-950 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 sm:p-8 space-y-8 shadow-sm transition-colors"
    >
      {/* ===================================================================== */}
      {/* SECTION HEADER                                                        */}
      {/* ===================================================================== */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-6 border-b border-slate-200 dark:border-slate-800">
        <div className="space-y-1">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-950/60 border border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
            Task 2.7B • Personalized Energy Plan
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white tracking-tight">
            Your Upcoming Electricity Plan & Savings Opportunities
          </h2>
          <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400">
            Ground truth analysis and actionable what-if appliance simulations anchored to your confirmed bill records.
          </p>
        </div>

        {/* Confirmed Anchor Status Pill */}
        <div className="flex items-center gap-2 self-start lg:self-auto bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800 px-3.5 py-2 rounded-xl text-emerald-800 dark:text-emerald-300 text-xs font-medium">
          <ShieldCheck className="w-4 h-4 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
          <span>Locked to Confirmed Household History ({confirmedRecords.length} cycle{confirmedRecords.length === 1 ? '' : 's'})</span>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* A2. CONFIRMED BASELINE SNAPSHOT                                       */}
      {/* ===================================================================== */}
      <div className="bg-white dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 flex items-center gap-2">
            <Clock className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            Confirmed Historical Baseline Snapshot
          </h3>
          <span className="text-[11px] text-slate-500 dark:text-slate-400">
            Verified by user • Never synthetic
          </span>
        </div>

        {latestRecord ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
            {/* Billing Period */}
            <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400">Confirmed Billing Period</span>
              <div className="font-bold text-slate-900 dark:text-white text-sm">
                {latestRecord.period_label || `${latestRecord.billing_period_start} – ${latestRecord.billing_period_end}`}
              </div>
              <span className="text-[10px] text-slate-400 font-mono">
                {latestRecord.billing_period_start} to {latestRecord.billing_period_end}
              </span>
            </div>

            {/* Consumption */}
            <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400">Confirmed Consumption</span>
              <div className="font-bold text-blue-600 dark:text-blue-400 text-lg">
                {latestRecord.consumption_kwh.toFixed(1)} <span className="text-xs font-normal text-slate-500 dark:text-slate-400">kWh</span>
              </div>
              <span className="text-[10px] text-slate-400">Physical electricity meter reading</span>
            </div>

            {/* Bill Amount */}
            <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400">Confirmed Bill Amount</span>
              <div className="font-bold text-slate-900 dark:text-white text-lg">
                {latestRecord.bill_amount !== null && latestRecord.bill_amount !== undefined ? (
                  `₹${latestRecord.bill_amount.toFixed(2)}`
                ) : (
                  <span className="text-xs font-normal text-slate-400 italic">Not extracted / entered</span>
                )}
              </div>
              <span className="text-[10px] text-slate-400">Total payable on electricity bill</span>
            </div>

            {/* Utility Provider */}
            <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 space-y-1">
              <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400">Electricity Provider</span>
              <div className="font-bold text-slate-900 dark:text-white text-sm truncate" title={selectedProviderId}>
                {selectedProviderId === 'msedcl' ? 'MSEDCL (Maharashtra)' : selectedProviderId ? selectedProviderId.toUpperCase() : 'Not specified'}
              </div>
              <div className="flex flex-col gap-0.5 pt-0.5">
                {selectedProviderId === 'msedcl' ? (
                  <>
                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-blue-700 dark:text-blue-300">
                      <ShieldCheck className="w-3 h-3 text-blue-600 dark:text-blue-400" /> MERC LT-I(B) (FY 2024-25)
                    </span>
                    <span className="text-[9px] text-slate-500 dark:text-slate-400">
                      Evaluated per cycle dates (expired past 2025-03-31)
                    </span>
                  </>
                ) : selectedProviderId ? (
                  <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-amber-600 dark:text-amber-400">
                    <ShieldAlert className="w-3 h-3" /> Current tariff unverified
                  </span>
                ) : (
                  <span className="text-[10px] text-slate-400 italic">
                    No provider selected
                  </span>
                )}
              </div>
            </div>
          </div>
        ) : null}

        {/* Extraction Quality Warnings */}
        {extractedBillData && extractedBillData.warnings && extractedBillData.warnings.length > 0 && (
          <div className="p-3 rounded-lg bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-800 dark:text-amber-300 text-xs flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-semibold">Extraction and Freshness Disclosures:</span>
              <ul className="list-disc list-inside text-[11px] space-y-0.5 text-amber-700 dark:text-amber-300">
                {extractedBillData.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        <div className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-1.5 font-mono">
          <Info className="w-3.5 h-3.5 text-blue-500 flex-shrink-0" />
          <span>
            Upcoming forecasts are anchored directly to your verified {latestRecord?.consumption_kwh.toFixed(1)} kWh cycle. Unconfirmed OCR scans are never treated as baseline history.
          </span>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* A3. UPCOMING MONTHLY PROJECTIONS (1, 3, or 6 MONTHS)                  */}
      {/* ===================================================================== */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              Upcoming Monthly Electricity Demand Projections
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Conservative demand forecasting based on your verified consumption history.
            </p>
          </div>

          {/* Horizon Selector */}
          <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl self-start sm:self-auto border border-slate-200 dark:border-slate-700">
            {[1, 3, 6].map((h) => (
              <button
                key={h}
                type="button"
                onClick={() => {
                  onForecastHorizonChange(h);
                }}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  forecastHorizon === h
                    ? 'bg-blue-600 text-white shadow-xs'
                    : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                {h} {h === 1 ? 'Month' : 'Months'}
              </button>
            ))}
          </div>
        </div>

        {/* Generate / Refresh Projection Action */}
        {!monthlyForecastResult ? (
          <div className="p-8 text-center bg-white dark:bg-slate-900 rounded-xl border border-dashed border-slate-300 dark:border-slate-700 space-y-3">
            <Zap className="w-8 h-8 text-blue-500 mx-auto animate-pulse" />
            <h4 className="text-sm font-bold text-slate-800 dark:text-slate-200">
              Ready to Generate Your {forecastHorizon}-Month Energy Plan
            </h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto">
              Calculate projected kilowatt-hour requirements and applicable slab tariffs for upcoming months based on your confirmed records.
            </p>
            <button
              type="button"
              onClick={onGenerateForecast}
              disabled={isForecasting}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-xs transition-all"
            >
              {isForecasting ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing Forecast...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Generate {forecastHorizon}-Month Forecast</span>
                </>
              )}
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Disclaimers & Methodology Notice */}
            <div className="p-3.5 rounded-xl bg-blue-50/80 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900 text-xs text-blue-900 dark:text-blue-200 space-y-1">
              <div className="font-bold flex items-center gap-1.5">
                <Info className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                <span>Forecasting Methodology & Confidence Constraints:</span>
              </div>
              <p className="text-[11px] leading-relaxed text-blue-800 dark:text-blue-300">
                {confirmedRecords.length === 1 ? (
                  <>
                    <strong>Baseline Persistence Model (N=1):</strong> A single bill reading is not enough to establish seasonality or a reliable multi-year trend. This projection represents a baseline persistence model rather than a complex predictive ML model. Consecutive bill uploads will unlock moving average and seasonal trend forecasting.
                  </>
                ) : (
                  <>
                    <strong>History-Tiered Model (N={confirmedRecords.length}):</strong> Computed using conservative chronological backtesting across your confirmed {confirmedRecords.length} billing cycles.
                  </>
                )}
              </p>
            </div>

            {/* Projection Table */}
            <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl bg-white dark:bg-slate-900 shadow-2xs">
              <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
                <thead className="bg-slate-50 dark:bg-slate-800/80 text-[11px] uppercase font-bold text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                  <tr>
                    <th className="py-3 px-4">Cycle</th>
                    <th className="py-3 px-4">Billing Month</th>
                    <th className="py-3 px-4 text-right">Projected (kWh)</th>
                    <th className="py-3 px-4 text-right">Confidence Bounds</th>
                    <th className="py-3 px-4 text-right">Estimated Bill</th>
                    <th className="py-3 px-4 text-center">Tariff Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                  {monthlyForecastResult.forecast_steps.map((step) => {
                    const tariffInfo = getStepTariffInfo(step, selectedProviderId);
                    return (
                      <tr key={step.month_index} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40 transition-colors">
                        <td className="py-2.5 px-4 font-sans font-medium text-slate-500">#{step.month_index}</td>
                        <td className="py-2.5 px-4 font-sans font-bold text-slate-900 dark:text-white">{step.month_label}</td>
                        <td className="py-2.5 px-4 text-right font-bold text-blue-600 dark:text-blue-400">
                          {step.forecast_kwh.toFixed(1)} kWh
                        </td>
                        <td className="py-2.5 px-4 text-right text-slate-500 dark:text-slate-400 text-[11px]">
                          {step.uncertainty_lower_kwh != null && step.uncertainty_upper_kwh != null
                            ? `[${step.uncertainty_lower_kwh.toFixed(1)} – ${step.uncertainty_upper_kwh.toFixed(1)}]`
                            : '—'}
                        </td>
                        <td className="py-2.5 px-4 text-right font-bold text-slate-900 dark:text-white font-sans">
                          {step.estimated_bill_amount !== null && step.estimated_bill_amount !== undefined ? (
                            `₹${step.estimated_bill_amount.toFixed(2)}`
                          ) : (
                            <span
                              className="text-[11px] text-amber-600 dark:text-amber-400 font-sans italic font-normal"
                              title={tariffInfo.tooltip}
                            >
                              {tariffInfo.withheldLabel}
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-4 text-center font-sans">
                          <span
                            data-testid={`tariff-status-${step.month_index}`}
                            className={`inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full border ${tariffInfo.badgeClass}`}
                            title={tariffInfo.tooltip}
                          >
                            {tariffInfo.icon === 'check' && (
                              <ShieldCheck className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                            )}
                            {tariffInfo.icon === 'clock' && (
                              <Clock className="w-3 h-3 text-amber-600 dark:text-amber-400" />
                            )}
                            {tariffInfo.icon === 'alert' && (
                              <ShieldAlert className="w-3 h-3 text-amber-600 dark:text-amber-400" />
                            )}
                            {tariffInfo.label}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* ===================================================================== */}
      {/* A4. INTEGRATED APPLIANCE-SAVING SIMULATOR                             */}
      {/* ===================================================================== */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-6 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-200 dark:border-slate-800">
          <div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300 text-xs font-semibold mb-1">
              <Sliders className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
              Appliance What-If Simulation
            </div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white">
              Hypothetical Appliance Usage & Schedule Scenarios
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Simulate operational modifications (e.g. reducing daily AC or geyser runtime) and calculate projected monthly reductions.
            </p>
          </div>

          <div className="text-[11px] font-mono px-3 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 self-start sm:self-auto">
            kWh = (Watts × Hours/Day × Days/Mo × Qty) / 1000
          </div>
        </div>

        {/* Appliance Category Selector */}
        <div className="space-y-2">
          <label className="text-xs font-bold text-slate-700 dark:text-slate-300">
            Select Appliance Category:
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-1.5">
            {[
              { id: 'air_conditioner', label: 'AC' },
              { id: 'refrigerator', label: 'Refrigerator' },
              { id: 'water_heater', label: 'Geyser' },
              { id: 'washing_machine', label: 'Washer' },
              { id: 'ceiling_fan', label: 'Fan' },
              { id: 'lighting', label: 'Lighting' },
              { id: 'television', label: 'TV' },
              { id: 'other', label: 'Custom' },
            ].map((cat) => (
              <button
                key={cat.id}
                type="button"
                onClick={() => handleCategoryChange(cat.id as ApplianceCategory)}
                className={`py-2 px-2 text-xs font-semibold rounded-lg border text-center transition-all truncate ${
                  selectedCategory === cat.id
                    ? 'bg-blue-600 text-white border-blue-600 shadow-2xs'
                    : 'bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700'
                }`}
              >
                {cat.label}
              </button>
            ))}
          </div>
        </div>

        {/* Refrigerator Safety Warning (if selected) */}
        {selectedCategory === 'refrigerator' && (
          <div className="p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-300 text-xs flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-bold">Essential Health & Food Safety Warning:</span>
              <p className="leading-relaxed text-[11px]">
                Refrigerators must remain powered 24 hours a day to prevent bacterial growth and food spoilage. Never shut down refrigeration units for energy savings. Focus instead on optimal thermostat temperature regulation (maintain 3°C to 4°C).
              </p>
            </div>
          </div>
        )}

        {/* Interactive Parameter Controls */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5 text-xs">
          {/* Appliance Name */}
          <div className="space-y-1">
            <label className="font-semibold text-slate-700 dark:text-slate-300">Appliance Label</label>
            <input
              type="text"
              value={applianceName}
              onChange={(e) => setApplianceName(e.target.value)}
              className="w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>

          {/* Rated Power (Watts) */}
          <div className="space-y-1">
            <label className="font-semibold text-slate-700 dark:text-slate-300">Rated Power (Watts)</label>
            <input
              type="number"
              step="10"
              min="1"
              max="50000"
              value={powerWatts}
              onChange={(e) => setPowerWatts(parseFloat(e.target.value) || 0)}
              className="w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
            />
            <span className="text-[10px] text-slate-400">Nameplate wattage</span>
          </div>

          {/* Baseline Operating Hours */}
          <div className="space-y-1">
            <label className="font-semibold text-slate-700 dark:text-slate-300">Current Hours/Day</label>
            <input
              type="number"
              step="0.5"
              min="0"
              max="24"
              value={baselineHours}
              onChange={(e) => setBaselineHours(parseFloat(e.target.value) || 0)}
              className="w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono"
            />
            <span className="text-[10px] text-slate-400">Baseline daily usage</span>
          </div>

          {/* Proposed Operating Hours */}
          <div className="space-y-1">
            <label className="font-semibold text-blue-600 dark:text-blue-400">Proposed Hours/Day</label>
            <input
              type="number"
              step="0.5"
              min="0"
              max="24"
              value={modifiedHours}
              onChange={(e) => setModifiedHours(parseFloat(e.target.value) || 0)}
              className="w-full bg-white dark:bg-slate-800 border border-blue-300 dark:border-blue-700 rounded-lg px-2.5 py-1.5 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-blue-500 font-mono font-bold"
            />
            <span className="text-[10px] text-blue-500">Hypothetical scenario</span>
          </div>

          {/* Days / Month & Quantity */}
          <div className="space-y-1">
            <div className="flex gap-2">
              <div className="flex-1">
                <label className="font-semibold text-slate-700 dark:text-slate-300">Days/Mo</label>
                <input
                  type="number"
                  min="0"
                  max="31"
                  value={daysPerMonth}
                  onChange={(e) => setDaysPerMonth(parseInt(e.target.value) || 0)}
                  className="w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1.5 text-slate-800 dark:text-slate-100 font-mono text-xs"
                />
              </div>
              <div className="w-16">
                <label className="font-semibold text-slate-700 dark:text-slate-300">Qty</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={quantity}
                  onChange={(e) => setQuantity(parseInt(e.target.value) || 1)}
                  className="w-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1.5 text-slate-800 dark:text-slate-100 font-mono text-xs"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Quick Scenario Preset Buttons */}
        <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
          <span className="text-slate-500 dark:text-slate-400 text-[11px] font-semibold">Quick Presets:</span>
          <button
            type="button"
            onClick={() => {
              handleCategoryChange('air_conditioner');
              setBaselineHours(8);
              setModifiedHours(7);
            }}
            className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-[11px] transition-colors"
          >
            Reduce AC by 1 hr/day
          </button>
          <button
            type="button"
            onClick={() => {
              handleCategoryChange('water_heater');
              setBaselineHours(1.5);
              setModifiedHours(1.0);
            }}
            className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-[11px] transition-colors"
          >
            Reduce Geyser runtime (-30m)
          </button>
          <button
            type="button"
            onClick={() => {
              handleCategoryChange('ceiling_fan');
              setPowerWatts(35); // BLDC upgrade
              setBaselineHours(12);
              setModifiedHours(12);
            }}
            className="px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-[11px] transition-colors"
          >
            Upgrade to BLDC Fans (35W)
          </button>
        </div>

        {/* Evaluate Scenario Button */}
        <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3 border-t border-slate-100 dark:border-slate-800">
          <div className="text-[11px] text-slate-500 dark:text-slate-400">
            Simulations evaluate marginal energy on top of your verified household baseline.
          </div>
          <button
            type="button"
            id="simulate-appliance-btn"
            data-testid="simulate-appliance-btn"
            onClick={handleEvaluateScenario}
            disabled={isEvaluatingScenario || !monthlyForecastResult}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white font-semibold text-xs shadow-xs transition-all disabled:bg-slate-300 dark:disabled:bg-slate-800 disabled:cursor-not-allowed"
          >
            {isEvaluatingScenario ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Simulating Scenario...</span>
              </>
            ) : (
              <>
                <Sliders className="w-3.5 h-3.5" />
                <span>Simulate Appliance Impact on Household Plan</span>
              </>
            )}
          </button>
        </div>

        {scenarioError && (
          <div className="p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-300 text-xs flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 flex-shrink-0" />
              <span>{scenarioError}</span>
            </div>
            <button
              type="button"
              onClick={handleEvaluateScenario}
              className="text-[11px] font-semibold text-rose-700 dark:text-rose-300 underline hover:no-underline"
            >
              Retry
            </button>
          </div>
        )}
      </div>

      {/* ===================================================================== */}
      {/* A5. SCENARIO COMPARISON RESULTS: BASELINE VS ADJUSTED                 */}
      {/* ===================================================================== */}
      {comparisonResult && (
        <div className="space-y-6">
          {/* Consistency Alert when household adjustment cannot be calculated defensibly */}
          {comparisonResult.is_household_adjustment_supported === false && (
            <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-900 dark:text-amber-200 text-xs space-y-1.5">
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <h4 className="font-bold text-sm text-amber-950 dark:text-amber-100">
                    Household Scenario Adjustment Withheld — Baseline Inconsistency
                  </h4>
                  <p className="leading-relaxed">
                    {comparisonResult.household_adjustment_reason ||
                      'The hypothetical appliance baseline or reduction exceeds your total household forecast. Total household consumption cannot be reduced below physical limits, and unmetered appliance ratings cannot be reconciled with this household baseline. Appliance-only savings are shown below, while adjusted household projections and bill savings are withheld.'}
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Monthly Energy Reduction (Appliance-Only Isolated) */}
            <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1 shadow-2xs">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 block">
                Monthly Appliance Reduction (Hypothetical)
              </span>
              <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                {comparisonResult.appliance_result.delta_kwh > 0 ? '-' : '+'}
                {Math.abs(comparisonResult.appliance_result.delta_kwh).toFixed(1)}{' '}
                <span className="text-xs font-normal text-slate-500">kWh/mo</span>
              </div>
              <span className="text-[11px] text-slate-400 block">
                Baseline: {comparisonResult.appliance_result.baseline_kwh.toFixed(1)} kWh → Proposed: {comparisonResult.appliance_result.modified_kwh.toFixed(1)} kWh
              </span>
              <span className="inline-block mt-1 px-1.5 py-0.5 rounded text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-sans font-medium">
                Nameplate Rated Power Estimate
              </span>
            </div>

            {/* Total Horizon Reduction */}
            <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1 shadow-2xs">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 block">
                {comparisonResult.steps.length}-Month Household Energy Delta
              </span>
              {comparisonResult.total_delta_kwh !== null && comparisonResult.total_delta_kwh !== undefined ? (
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 font-mono">
                  {comparisonResult.total_delta_kwh > 0 ? '-' : '+'}
                  {Math.abs(comparisonResult.total_delta_kwh).toFixed(1)}{' '}
                  <span className="text-xs font-normal text-slate-500">kWh</span>
                </div>
              ) : (
                <div className="text-sm font-semibold text-amber-700 dark:text-amber-400 py-1.5 font-sans">
                  Withheld (Exceeds Household Baseline)
                </div>
              )}
              <span className="text-[11px] text-slate-400 block">
                {comparisonResult.total_adjusted_kwh !== null && comparisonResult.total_adjusted_kwh !== undefined
                  ? `Forecast total drops from ${comparisonResult.total_baseline_kwh.toFixed(1)} to ${comparisonResult.total_adjusted_kwh.toFixed(1)} kWh`
                  : 'Appliance demand cannot be reconciled with household baseline'}
              </span>
            </div>

            {/* Total Monetary Savings */}
            <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1 shadow-2xs">
              <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 block">
                {comparisonResult.steps.length}-Month Bill Savings (INR)
              </span>
              <div className="text-2xl font-bold text-slate-900 dark:text-white font-mono">
                {comparisonResult.total_savings_inr !== null && comparisonResult.total_savings_inr !== undefined ? (
                  `₹${comparisonResult.total_savings_inr.toFixed(2)}`
                ) : (
                  <span className="text-xs font-normal text-amber-600 dark:text-amber-400 italic font-sans">
                    Withheld {comparisonResult.is_household_adjustment_supported === false ? '(Unsupported Scenario)' : comparisonResult.tariff_status === 'expired_schedule' ? '(Expired Tariff)' : '(Unverified Tariff)'}
                  </span>
                )}
              </div>
              <span className="text-[11px] text-slate-400 block">
                {comparisonResult.is_verified_tariff && comparisonResult.total_savings_inr !== null
                  ? 'Official MERC LT-I(B) progressive slabs'
                  : comparisonResult.is_household_adjustment_supported === false
                  ? 'Withheld to avoid fabricating impossible savings'
                  : comparisonResult.tariff_status === 'expired_schedule'
                  ? 'Tariff schedule expired on 2025-03-31; withheld to prevent fabricated savings'
                  : 'Tariff withheld to prevent fabricated savings'}
              </span>
            </div>
          </div>

          {/* Interactive Chart: Baseline Forecast vs Scenario */}
          {chartData.length > 0 && (
            <div className="bg-white dark:bg-slate-900 p-5 rounded-xl border border-slate-200 dark:border-slate-800 space-y-3 shadow-2xs">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300">
                  Forecast Comparison: Baseline vs Scenario
                </h4>
                <span className="text-[11px] text-slate-400">Values in Kilowatt-Hours (kWh)</span>
              </div>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} />
                    <XAxis dataKey="name" tick={{ fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }} />
                    <YAxis tick={{ fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: isDark ? '#0f172a' : '#ffffff',
                        borderColor: isDark ? '#334155' : '#e2e8f0',
                        color: isDark ? '#f8fafc' : '#0f172a',
                        fontSize: 12,
                        borderRadius: 8,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
                    <Bar dataKey="baselineKwh" name="Baseline Forecast (kWh)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    {isAdjustmentSupported && (
                      <Bar dataKey="adjustedKwh" name="Scenario Adjusted (kWh)" fill="#10b981" radius={[4, 4, 0, 0]} />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Step-by-Step Scenario Table */}
          <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-xl bg-white dark:bg-slate-900 shadow-2xs">
            <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
              <thead className="bg-slate-50 dark:bg-slate-800/80 text-[11px] uppercase font-bold text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Month</th>
                  <th className="py-2.5 px-3 text-right">Baseline (kWh)</th>
                  <th className="py-2.5 px-3 text-right text-emerald-600 dark:text-emerald-400">Scenario (kWh)</th>
                  <th className="py-2.5 px-3 text-right font-bold">Delta (kWh)</th>
                  <th className="py-2.5 px-3 text-right">Baseline Bill</th>
                  <th className="py-2.5 px-3 text-right">Scenario Bill</th>
                  <th className="py-2.5 px-3 text-right text-emerald-600 dark:text-emerald-400 font-bold">Savings</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                {comparisonResult.steps.map((st) => (
                  <tr key={st.month_index} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                    <td className="py-2 px-3 font-sans font-bold text-slate-900 dark:text-white">{st.month_label}</td>
                    <td className="py-2 px-3 text-right">{st.baseline_kwh.toFixed(1)}</td>
                    <td className="py-2 px-3 text-right text-emerald-600 dark:text-emerald-400 font-bold">
                      {st.adjusted_kwh !== null && st.adjusted_kwh !== undefined ? st.adjusted_kwh.toFixed(1) : '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-bold text-blue-600 dark:text-blue-400">
                      {st.delta_kwh !== null && st.delta_kwh !== undefined ? `-${st.delta_kwh.toFixed(1)}` : '—'}
                    </td>
                    <td className="py-2 px-3 text-right font-sans">
                      {st.baseline_bill_amount !== null && st.baseline_bill_amount !== undefined ? (
                        `₹${st.baseline_bill_amount.toFixed(2)}`
                      ) : (
                        <span className="text-[10px] text-amber-600 dark:text-amber-400 italic">
                          {st.tariff_status === 'expired_schedule' ? 'Expired' : 'Withheld'}
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right font-sans">
                      {st.adjusted_bill_amount !== null && st.adjusted_bill_amount !== undefined ? (
                        `₹${st.adjusted_bill_amount.toFixed(2)}`
                      ) : (
                        <span className="text-[10px] text-amber-600 dark:text-amber-400 italic">
                          {st.tariff_status === 'expired_schedule' ? 'Expired' : 'Withheld'}
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-3 text-right font-sans font-bold text-emerald-600 dark:text-emerald-400">
                      {st.savings_amount !== null && st.savings_amount !== undefined ? (
                        `+₹${st.savings_amount.toFixed(2)}`
                      ) : (
                        <span className="text-[10px] text-amber-600 dark:text-amber-400 italic font-normal">
                          Withheld
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Disclosures from Scenario Evaluation */}
          <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 text-xs text-slate-600 dark:text-slate-400 space-y-1">
            <span className="font-semibold text-slate-800 dark:text-slate-200">Regulatory & Computational Disclosures:</span>
            <ul className="list-disc list-inside text-[11px] space-y-0.5">
              {comparisonResult.disclosures.map((d, i) => (
                <li key={i}>{d}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* A6. ACTIONABLE RECOMMENDATIONS                                        */}
      {/* ===================================================================== */}
      {comparisonResult && comparisonResult.recommendations.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-500" />
            Practical Energy-Saving Recommendations
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {comparisonResult.recommendations.map((rec) => (
              <div
                key={rec.id}
                className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-2 shadow-2xs hover:border-blue-300 dark:hover:border-blue-800 transition-colors"
              >
                <div className="flex items-start justify-between gap-2">
                  <h4 className="text-xs font-bold text-slate-900 dark:text-white leading-tight">{rec.title}</h4>
                  <span
                    className={`text-[9px] px-2 py-0.5 rounded-full font-semibold uppercase flex-shrink-0 ${
                      rec.is_hypothetical
                        ? 'bg-purple-50 dark:bg-purple-950/50 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800'
                        : 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800'
                    }`}
                  >
                    {rec.is_hypothetical ? 'Simulation' : 'Measured Data'}
                  </span>
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">{rec.description}</p>
                <div className="pt-1 flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 border-t border-slate-100 dark:border-slate-800">
                  <span className="italic">{rec.rationale}</span>
                  {rec.estimated_monthly_savings_inr ? (
                    <span className="font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                      ~₹{rec.estimated_monthly_savings_inr.toFixed(0)}/mo
                    </span>
                  ) : rec.estimated_monthly_kwh_reduction ? (
                    <span className="font-bold text-blue-600 dark:text-blue-400 font-mono">
                      ~{rec.estimated_monthly_kwh_reduction.toFixed(1)} kWh/mo
                    </span>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ===================================================================== */}
      {/* PERSISTENT STORAGE & PDF ACTIONS                                      */}
      {/* ===================================================================== */}
      {monthlyForecastResult && (
        <div className="pt-4 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-slate-200 dark:border-slate-800">
          <div className="space-y-0.5">
            <span className="text-xs font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
              <History className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              Persist Plan & Download Certified Report
            </span>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Save this {forecastHorizon}-month plan into your persistent account history to monitor future variance.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            {savedRunId ? (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300 text-xs font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                  Saved to Account
                </span>
                <Link
                  to={`/forecast/monthly/${savedRunId}`}
                  className="inline-flex items-center gap-1 px-3.5 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 text-xs font-medium transition-colors shadow-2xs"
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>View Snapshot & PDF</span>
                </Link>
              </div>
            ) : (
              <button
                type="button"
                id="save-saving-plan-btn"
                onClick={onSaveForecast}
                disabled={isSaving}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-xs transition-all disabled:bg-blue-400"
              >
                {isSaving ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Saving Plan...</span>
                  </>
                ) : (
                  <>
                    <Plus className="w-3.5 h-3.5" />
                    <span>Save Upcoming Plan to History</span>
                  </>
                )}
              </button>
            )}

            <Link
              to="/coach"
              className="inline-flex items-center gap-1 px-3.5 py-2 rounded-xl bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 text-xs font-semibold transition-colors"
            >
              <span>Energy Coach Dashboard</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      )}
    </section>
  );
};
