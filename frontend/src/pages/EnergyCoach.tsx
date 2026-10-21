import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Compass,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Info,
  Zap,
  Sliders,
  Calendar,
  FileText,
  Download,
  ShieldCheck,
  ShieldAlert,
  ArrowRight,
  RefreshCw,
  Lightbulb,
} from 'lucide-react';
import {
  ResponsiveContainer,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Line,
  ComposedChart,
} from 'recharts';

import { api } from '../api/client';
import {
  EnergyCoachOverviewResponse,
  ApplianceCategory,
  ApplianceSimulationInput,
  ApplianceScenarioResponse,
  AppliancePreset,
} from '../api/types';
import { MetricCard } from '../components/common/MetricCard';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import { useTheme } from '../context/ThemeContext';

export const EnergyCoach: React.FC = () => {
  const { isDark } = useTheme();
  const [overview, setOverview] = useState<EnergyCoachOverviewResponse | null>(null);
  const [presets, setPresets] = useState<Record<string, AppliancePreset>>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Appliance Simulator State
  const [selectedCategory, setSelectedCategory] = useState<ApplianceCategory>('air_conditioner');
  const [applianceName, setApplianceName] = useState<string>('1.5 Ton Split AC');
  const [powerWatts, setPowerWatts] = useState<number>(1450);
  const [baselineHours, setBaselineHours] = useState<number>(8.0);
  const [modifiedHours, setModifiedHours] = useState<number>(7.0);
  const [daysPerMonth, setDaysPerMonth] = useState<number>(30);
  const [quantity, setQuantity] = useState<number>(1);
  const [modifiedPowerWatts, setModifiedPowerWatts] = useState<number>(1450);

  const [simResult, setSimResult] = useState<ApplianceScenarioResponse | null>(null);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [simError, setSimError] = useState<string | null>(null);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [overviewData, presetsData] = await Promise.all([
        api.getCoachOverview(),
        api.getAppliancePresets(),
      ]);
      setOverview(overviewData);
      setPresets(presetsData);

      // Initialize simulator with Air Conditioner preset
      if (presetsData['air_conditioner']) {
        const ac = presetsData['air_conditioner'];
        setApplianceName(ac.name);
        setPowerWatts(ac.default_power_watts);
        setModifiedPowerWatts(ac.default_power_watts);
        setBaselineHours(ac.default_hours_per_day);
        setModifiedHours(Math.max(0, ac.default_hours_per_day - 1.0));
        setDaysPerMonth(ac.default_days_per_month);
        setQuantity(ac.default_quantity);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to load household energy coaching data.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Trigger initial simulation whenever data is loaded
  useEffect(() => {
    if (overview) {
      handleSimulate();
    }
  }, [overview?.provider_id]);

  const handleCategorySelect = (cat: ApplianceCategory) => {
    setSelectedCategory(cat);
    const p = presets[cat];
    if (p) {
      setApplianceName(p.name);
      setPowerWatts(p.default_power_watts);
      setModifiedPowerWatts(p.default_power_watts);
      setBaselineHours(p.default_hours_per_day);
      setModifiedHours(Math.max(0, p.default_hours_per_day - 1.0));
      setDaysPerMonth(p.default_days_per_month);
      setQuantity(p.default_quantity);
    } else {
      setApplianceName('Custom Appliance');
    }
  };

  const handleSimulate = async () => {
    setIsSimulating(true);
    setSimError(null);
    try {
      const baseline: ApplianceSimulationInput = {
        category: selectedCategory,
        name: applianceName,
        power_watts: Number(powerWatts),
        hours_per_day: Number(baselineHours),
        days_per_month: Number(daysPerMonth),
        quantity: Number(quantity),
      };

      const modified: ApplianceSimulationInput = {
        category: selectedCategory,
        name: applianceName,
        power_watts: Number(modifiedPowerWatts),
        hours_per_day: Number(modifiedHours),
        days_per_month: Number(daysPerMonth),
        quantity: Number(quantity),
      };

      const res = await api.simulateApplianceScenario({
        baseline,
        modified,
        provider_id: overview?.provider_id || undefined,
        current_monthly_total_kwh: overview?.latest_consumption_kwh || undefined,
      });
      setSimResult(res);
    } catch (err: any) {
      setSimError(err?.message || 'Failed to run appliance simulation.');
    } finally {
      setIsSimulating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="p-8">
        <LoadingState message="Compiling your personalized Household Energy Coach..." />
      </div>
    );
  }

  if (error || !overview) {
    return (
      <div className="p-8">
        <ErrorState
          title="Energy Coach Unavailable"
          message={error || 'Could not retrieve household profile.'}
          onRetry={loadData}
        />
      </div>
    );
  }

  // Prepare combined historical + forecast chart data
  const chartData = [
    ...overview.historical_records.map((r) => ({
      label: r.period_label || r.billing_period_start.slice(0, 7),
      actual_kwh: r.consumption_kwh,
      forecast_kwh: null as number | null,
      bill_amount: r.bill_amount,
      is_forecast: false,
    })),
    ...overview.forecast_steps.map((s) => ({
      label: s.month_label,
      actual_kwh: null as number | null,
      forecast_kwh: s.forecast_kwh,
      lower_kwh: s.uncertainty_lower_kwh,
      upper_kwh: s.uncertainty_upper_kwh,
      bill_amount: s.estimated_bill_amount,
      is_forecast: true,
    })),
  ];

  return (
    <div className="p-6 md:p-8 space-y-8 max-w-7xl mx-auto transition-colors">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-200 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
              <Compass className="w-5 h-5" />
            </div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Household Energy Coach</h1>
            <span className="text-xs px-2 py-0.5 rounded bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300 font-medium">
              Task 2.7
            </span>
          </div>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Personalized energy efficiency insights, empirical trend analysis, and appliance what-if simulations.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadData}
            className="flex items-center gap-2 px-3 py-2 text-xs font-medium rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 transition"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh Data
          </button>
          <Link
            to="/forecast/new"
            className="flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-blue-600 hover:bg-blue-500 text-white shadow-xs transition"
          >
            <Zap className="w-3.5 h-3.5" />
            New Forecast / Bill
          </Link>
        </div>
      </div>

      {/* Freshness / Quality Warnings Banner */}
      {overview.warnings.length > 0 && (
        <div className="bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 rounded-xl p-4 text-amber-800 dark:text-amber-300 text-sm space-y-1">
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            Data Quality & Freshness Notice
          </div>
          {overview.warnings.map((w, idx) => (
            <p key={idx} className="text-xs text-amber-700 dark:text-amber-400 pl-6">
              • {w}
            </p>
          ))}
        </div>
      )}

      {/* SECTION A: Household Energy Overview */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
            <span>A. Household Energy Overview</span>
          </h2>
          {overview.data_freshness_status === 'current' && (
            <span className="text-xs px-2.5 py-1 rounded bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 flex items-center gap-1">
              <CheckCircle2 className="w-3 h-3" /> Data Fresh (Up to date)
            </span>
          )}
          {overview.data_freshness_status === 'stale' && (
            <span className="text-xs px-2.5 py-1 rounded bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800 flex items-center gap-1">
              <AlertTriangle className="w-3 h-3" /> Stale History ({'>'}60 days)
            </span>
          )}
          {overview.data_freshness_status === 'insufficient_history' && (
            <span className="text-xs px-2.5 py-1 rounded bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 flex items-center gap-1">
              <Info className="w-3 h-3" /> Limited History
            </span>
          )}
        </div>

        {overview.has_data ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Latest Monthly Consumption"
              value={`${overview.latest_consumption_kwh?.toFixed(1) || '—'} kWh`}
              subtitle={overview.latest_period_label ? `Period: ${overview.latest_period_label}` : undefined}
            />

            <MetricCard
              title="Month-over-Month Change"
              value={
                overview.consumption_change_kwh !== null && overview.consumption_change_kwh !== undefined
                  ? `${overview.consumption_change_kwh > 0 ? '+' : ''}${overview.consumption_change_kwh.toFixed(1)} kWh`
                  : 'N/A'
              }
              subtitle={
                overview.consumption_change_percent !== null && overview.consumption_change_percent !== undefined
                  ? `${overview.consumption_change_percent > 0 ? '+' : ''}${overview.consumption_change_percent.toFixed(1)}% vs ${overview.previous_period_label || 'preceding month'}`
                  : 'Requires ≥2 billing cycles'
              }
            />

            <MetricCard
              title="Latest Confirmed Bill"
              value={
                overview.latest_bill_amount_inr !== null && overview.latest_bill_amount_inr !== undefined
                  ? `₹${overview.latest_bill_amount_inr.toFixed(2)}`
                  : 'Not Extracted'
              }
              subtitle={
                overview.latest_bill_amount_inr !== null && overview.latest_bill_amount_inr !== undefined
                  ? 'Confirmed from bill document'
                  : 'Amount not present on record'
              }
            />

            <MetricCard
              title="Latest Forecast (Horizon Avg)"
              value={
                overview.latest_forecast_average_kwh !== null && overview.latest_forecast_average_kwh !== undefined
                  ? `${overview.latest_forecast_average_kwh.toFixed(1)} kWh`
                  : 'No Active Forecast'
              }
              subtitle={
                overview.latest_forecast_period_range
                  ? `${overview.latest_forecast_horizon_months}M ahead (${overview.latest_forecast_period_range})`
                  : 'Run forecast to project'
              }
            />
          </div>
        ) : (
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-8 text-center space-y-3 shadow-xs">
            <div className="w-12 h-12 rounded-full bg-blue-50 dark:bg-blue-950/50 text-blue-600 dark:text-blue-400 mx-auto flex items-center justify-center border border-blue-200 dark:border-blue-800">
              <Zap className="w-6 h-6" />
            </div>
            <h3 className="text-base font-semibold text-slate-800 dark:text-slate-200">No Household History Recorded</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">
              Energy Coach requires confirmed electricity bills to analyze baseline trends and compute accurate appliance savings.
            </p>
            <Link
              to="/forecast/new"
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition shadow-xs"
            >
              Upload or Enter Bill Readings <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        )}
      </section>

      {/* SECTION B: Personalized Energy Insights */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
          <span>B. Personalized Energy Insights</span>
          <span className="text-xs text-slate-500 dark:text-slate-400 font-normal">(Deterministic & Grounded in Real Data)</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {overview.insights.map((insight) => (
            <div
              key={insight.id}
              className={`p-5 rounded-xl border transition-all ${
                insight.severity === 'positive'
                  ? 'bg-emerald-50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-500/30'
                  : insight.severity === 'warning'
                  ? 'bg-amber-50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-500/30'
                  : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5">
                  {insight.severity === 'positive' && <TrendingDown className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />}
                  {insight.severity === 'warning' && <TrendingUp className="w-5 h-5 text-amber-600 dark:text-amber-400" />}
                  {insight.severity === 'info' && <Info className="w-5 h-5 text-blue-600 dark:text-blue-400" />}
                  {insight.severity === 'neutral' && <Minus className="w-5 h-5 text-slate-400" />}
                </div>
                <div className="space-y-1.5 flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{insight.title}</h3>
                    <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400">
                      {insight.category}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">{insight.description}</p>
                  <div className="mt-2 pt-2 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400 font-mono">
                    <strong className="text-slate-700 dark:text-slate-300">Evidence: </strong>
                    {insight.evidence}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* SECTION C: Appliance What-If Simulator */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <span>C. Appliance What-If Simulator</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Estimate incremental monthly energy (kWh) and cost impact using physics formulas:{' '}
              <span className="font-mono text-slate-700 dark:text-slate-300">kWh = (Watts × Hours × Days × Quantity) / 1000</span>.
            </p>
          </div>
          {overview.is_tariff_verified ? (
            <span className="text-xs px-2.5 py-1 rounded bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800 font-medium flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              Verified Tariff ({overview.provider_id?.toUpperCase()})
            </span>
          ) : (
            <span className="text-xs px-2.5 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700 font-medium flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-amber-500" />
              Tariff Unverified (Energy Only)
            </span>
          )}
        </div>

        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-6 shadow-xs">
          {/* Appliance Preset Category Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-2">
              Select Appliance Category Preset:
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
              {[
                { key: 'air_conditioner', label: 'Air Conditioner' },
                { key: 'refrigerator', label: 'Refrigerator' },
                { key: 'water_heater', label: 'Water Heater' },
                { key: 'washing_machine', label: 'Washing Machine' },
                { key: 'ceiling_fan', label: 'Ceiling Fan' },
                { key: 'lighting', label: 'Lighting' },
                { key: 'television', label: 'Television' },
                { key: 'other', label: 'Custom' },
              ].map((cat) => (
                <button
                  key={cat.key}
                  type="button"
                  onClick={() => handleCategorySelect(cat.key as ApplianceCategory)}
                  className={`px-3 py-2 text-xs font-medium rounded-lg border text-center transition ${
                    selectedCategory === cat.key
                      ? 'bg-blue-600 border-blue-500 text-white font-semibold shadow-xs'
                      : 'bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                  }`}
                >
                  {cat.label}
                </button>
              ))}
            </div>
          </div>

          {/* Side-by-side Scenario Input Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-200 dark:border-slate-800">
            {/* Baseline Configuration */}
            <div className="space-y-4 bg-slate-50 dark:bg-slate-800/60 p-5 rounded-xl border border-slate-200 dark:border-slate-700">
              <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-700">
                <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">Current / Baseline Scenario</span>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-400 font-medium">Baseline</span>
              </div>

              <div>
                <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Appliance Label</label>
                <input
                  type="text"
                  value={applianceName}
                  onChange={(e) => setApplianceName(e.target.value)}
                  className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Rated Power (Watts)</label>
                  <input
                    type="number"
                    min={1}
                    max={50000}
                    value={powerWatts}
                    onChange={(e) => setPowerWatts(Number(e.target.value))}
                    className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Quantity</label>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={quantity}
                    onChange={(e) => setQuantity(Number(e.target.value))}
                    className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Operating Hours / Day</label>
                  <input
                    type="number"
                    step={0.5}
                    min={0}
                    max={24}
                    value={baselineHours}
                    onChange={(e) => setBaselineHours(Number(e.target.value))}
                    className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Days Used / Month</label>
                  <input
                    type="number"
                    min={0}
                    max={31}
                    value={daysPerMonth}
                    onChange={(e) => setDaysPerMonth(Number(e.target.value))}
                    className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Modified / Target Scenario */}
            <div className="space-y-4 bg-blue-50/50 dark:bg-blue-950/20 p-5 rounded-xl border border-blue-200 dark:border-blue-500/30">
              <div className="flex items-center justify-between pb-2 border-b border-blue-200 dark:border-blue-500/20">
                <span className="text-sm font-semibold text-blue-900 dark:text-blue-200">Hypothetical / Modified Scenario</span>
                <span className="text-xs px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-300 font-medium">Scenario</span>
              </div>

              <div>
                <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Target Hours / Day</label>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    step={0.5}
                    min={0}
                    max={24}
                    value={modifiedHours}
                    onChange={(e) => setModifiedHours(Number(e.target.value))}
                    className="w-full bg-white dark:bg-slate-900 border border-blue-300 dark:border-blue-500/40 rounded-lg px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-400 font-semibold"
                  />
                  <div className="flex gap-1">
                    <button
                      type="button"
                      onClick={() => setModifiedHours((h) => Math.max(0, Number((h - 1).toFixed(1))))}
                      className="px-2.5 py-2 text-[11px] bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 whitespace-nowrap"
                    >
                      -1h
                    </button>
                    <button
                      type="button"
                      onClick={() => setModifiedHours((h) => Math.max(0, Number((h - 2).toFixed(1))))}
                      className="px-2.5 py-2 text-[11px] bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 whitespace-nowrap"
                    >
                      -2h
                    </button>
                  </div>
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">
                  Target Wattage (e.g. upgrade to energy-efficient BLDC / Inverter)
                </label>
                <input
                  type="number"
                  min={1}
                  max={50000}
                  value={modifiedPowerWatts}
                  onChange={(e) => setModifiedPowerWatts(Number(e.target.value))}
                  className="w-full bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-blue-400"
                />
              </div>

              <div className="pt-2">
                <button
                  type="button"
                  onClick={handleSimulate}
                  disabled={isSimulating}
                  className="w-full py-2.5 px-4 text-xs font-semibold rounded-lg bg-blue-600 hover:bg-blue-500 active:scale-95 text-white transition flex items-center justify-center gap-2 shadow-xs"
                >
                  {isSimulating ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Computing physics & tariff impact...
                    </>
                  ) : (
                    <>
                      <Sliders className="w-3.5 h-3.5" />
                      Recalculate Energy & Cost Impact
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Simulation Output Card */}
          {simError && (
            <div className="p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-300 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{simError}</span>
            </div>
          )}

          {simResult && (
            <div className="bg-slate-50 dark:bg-slate-900/80 rounded-xl p-5 border border-slate-200 dark:border-slate-700 space-y-4 shadow-2xs">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
                <div className="p-3.5 rounded-lg bg-white dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 shadow-2xs">
                  <span className="text-xs text-slate-500 dark:text-slate-400 block mb-1">Baseline Consumption</span>
                  <span className="text-xl font-bold text-slate-800 dark:text-slate-200">{simResult.baseline_kwh.toFixed(1)} kWh</span>
                  <span className="text-[11px] text-slate-400 block">/ month</span>
                </div>

                <div className="p-3.5 rounded-lg bg-white dark:bg-slate-950/60 border border-slate-200 dark:border-slate-800 shadow-2xs">
                  <span className="text-xs text-slate-500 dark:text-slate-400 block mb-1">Scenario Consumption</span>
                  <span className="text-xl font-bold text-blue-600 dark:text-blue-400">{simResult.modified_kwh.toFixed(1)} kWh</span>
                  <span className="text-[11px] text-slate-400 block">/ month</span>
                </div>

                <div className="p-3.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-500/30 shadow-2xs">
                  <span className="text-xs text-emerald-700 dark:text-emerald-400 block mb-1 font-semibold">Potential Monthly Savings</span>
                  <span className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
                    {simResult.delta_kwh > 0 ? `+${simResult.delta_kwh.toFixed(1)}` : simResult.delta_kwh.toFixed(1)} kWh
                  </span>
                  <span className="text-[11px] text-emerald-600 dark:text-emerald-300 block">
                    {simResult.estimated_savings_inr !== null && simResult.estimated_savings_inr !== undefined
                      ? `≈ ₹${simResult.estimated_savings_inr.toFixed(2)} / month`
                      : 'Cost withheld (unverified tariff)'}
                  </span>
                </div>
              </div>

              {/* Duty cycle context & Safety notices */}
              <div className="space-y-2 text-xs">
                {simResult.appliance_notes.map((note, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg flex items-start gap-2 ${
                      note.includes('SAFETY CAUTION')
                        ? 'bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-500/30 text-rose-800 dark:text-rose-300'
                        : 'bg-white dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300'
                    }`}
                  >
                    <Info className="w-4 h-4 flex-shrink-0 mt-0.5 text-blue-500" />
                    <span>{note}</span>
                  </div>
                ))}

                {simResult.disclosures.map((d, idx) => (
                  <p key={idx} className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed pl-1">
                    • {d}
                  </p>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* SECTION D: Monthly Bill and Consumption Insights */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
              <Calendar className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              <span>D. Monthly Bill & Consumption Insights</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Confirmed bill history alongside seasonal forecast projections and uncertainty intervals.
            </p>
          </div>
          {overview.latest_forecast_run_id && (
            <div className="flex items-center gap-2">
              <Link
                to={`/forecast/monthly/${overview.latest_forecast_run_id}`}
                className="text-xs px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-blue-600 dark:text-blue-400 border border-slate-200 dark:border-slate-700 flex items-center gap-1.5 transition"
              >
                <FileText className="w-3.5 h-3.5" />
                View Full Forecast Detail
              </Link>
              <button
                type="button"
                onClick={() => api.downloadMonthlyForecastPdf(overview.latest_forecast_run_id!)}
                className="text-xs px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 flex items-center gap-1.5 transition"
              >
                <Download className="w-3.5 h-3.5" />
                PDF Report
              </button>
            </div>
          )}
        </div>

        {/* Recharts Historical + Forecast Visualization */}
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4 shadow-xs">
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#334155' : '#e2e8f0'} />
                <XAxis dataKey="label" stroke={isDark ? '#94a3b8' : '#64748b'} tick={{ fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }} />
                <YAxis
                  stroke={isDark ? '#94a3b8' : '#64748b'}
                  tick={{ fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }}
                  label={{ value: 'Energy (kWh)', angle: -90, position: 'insideLeft', fill: isDark ? '#94a3b8' : '#64748b', fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? '#0f172a' : '#ffffff',
                    borderColor: isDark ? '#334155' : '#e2e8f0',
                    color: isDark ? '#f8fafc' : '#0f172a',
                    borderRadius: '8px',
                    fontSize: 12,
                  }}
                  labelStyle={{ color: isDark ? '#f8fafc' : '#0f172a', fontWeight: 600, fontSize: 12 }}
                  formatter={(value: any, name: string) => {
                    if (name === 'Actual Billed Energy') return [`${Number(value).toFixed(1)} kWh`, name];
                    if (name === 'Forecasted Energy') return [`${Number(value).toFixed(1)} kWh`, name];
                    return [value, name];
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Bar dataKey="actual_kwh" name="Actual Billed Energy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Line
                  type="monotone"
                  dataKey="forecast_kwh"
                  name="Forecasted Energy"
                  stroke="#10b981"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ r: 4, fill: '#10b981' }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="text-xs text-slate-500 dark:text-slate-400 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pt-2 border-t border-slate-100 dark:border-slate-800">
            <span>
              <strong>Solid Bars: </strong>Confirmed historical monthly consumption extracted from your bills.
            </span>
            <span>
              <strong>Dashed Green Line: </strong>Upcoming cycle projections from the monthly forecasting service.
            </span>
          </div>
        </div>
      </section>

      {/* SECTION E: Actionable Energy-Saving Recommendations */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-amber-500" />
          <span>E. Actionable Energy-Saving Recommendations</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {overview.recommendations.map((rec) => (
            <div key={rec.id} className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-3 shadow-2xs hover:border-slate-300 dark:hover:border-slate-700 transition-colors">
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{rec.title}</h3>
                <span
                  className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded border ${
                    rec.action_type === 'appliance'
                      ? 'bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800'
                      : rec.action_type === 'behavioral'
                      ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800'
                      : 'bg-purple-50 dark:bg-purple-950/40 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-800'
                  }`}
                >
                  {rec.action_type}
                </span>
              </div>

              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">{rec.description}</p>

              <div className="text-[11px] text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800/60 p-2.5 rounded-lg border border-slate-200 dark:border-slate-700/60">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Why this matters: </span>
                {rec.rationale}
              </div>

              {rec.estimated_monthly_kwh_reduction && (
                <div className="pt-2 flex items-center justify-between text-xs border-t border-slate-100 dark:border-slate-800">
                  <span className="text-slate-500 dark:text-slate-400">Potential Reduction:</span>
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                    ≈ {rec.estimated_monthly_kwh_reduction.toFixed(0)} kWh / month
                    {rec.estimated_monthly_savings_inr ? ` (₹${rec.estimated_monthly_savings_inr.toFixed(0)})` : ''}
                  </span>
                </div>
              )}

              {rec.savings_disclaimer && (
                <p className="text-[10px] text-slate-400 dark:text-slate-500 italic">*{rec.savings_disclaimer}</p>
              )}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};
