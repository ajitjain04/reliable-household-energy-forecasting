import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import {
  ConformalSummaryResponse,
  ConformalSummaryRecord,
  ConformalWindowIntervalResponse,
} from '../api/types';
import { MetricCard } from '../components/common/MetricCard';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import {
  Target,
  Database,
  ShieldCheck,
  AlertTriangle,
  SlidersHorizontal,
  ChevronLeft,
  ChevronRight,
  TrendingDown,
  Info,
} from 'lucide-react';

export const UncertaintyAnalysis: React.FC = () => {
  const [summaryData, setSummaryData] = useState<ConformalSummaryResponse | null>(null);
  const [intervalData, setIntervalData] = useState<ConformalWindowIntervalResponse | null>(null);
  const [loadingSummary, setLoadingSummary] = useState<boolean>(true);
  const [loadingInterval, setLoadingInterval] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [intervalError, setIntervalError] = useState<string | null>(null);

  // Interval Viewer Selectors
  const [selectedModel, setSelectedModel] = useState<string>('early_fusion');
  const [selectedCondition, setSelectedCondition] = useState<string>('clean_NN');
  const [windowIdx, setWindowIdx] = useState<number>(0);

  // Load Conformal Summary
  const loadSummary = async () => {
    setLoadingSummary(true);
    setError(null);
    try {
      const summary = await api.getConformalSummary();
      setSummaryData(summary);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to fetch conformal uncertainty calibration summary.');
      }
    } finally {
      setLoadingSummary(false);
    }
  };

  // Load Window Intervals
  const loadIntervals = async (model: string, condition: string, win: number) => {
    setLoadingInterval(true);
    setIntervalError(null);
    try {
      const intervalRes = await api.getConformalIntervals(model, condition, win);
      setIntervalData(intervalRes);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setIntervalError(err.message);
      } else {
        setIntervalError('Failed to fetch interval forecasts for selected window.');
      }
    } finally {
      setLoadingInterval(false);
    }
  };

  useEffect(() => {
    loadSummary();
  }, []);

  useEffect(() => {
    loadIntervals(selectedModel, selectedCondition, windowIdx);
  }, [selectedModel, selectedCondition, windowIdx]);

  // Chart data for Coverage-vs-Width Tradeoff across all 4 models
  const coverageComparisonChartData = useMemo(() => {
    if (!summaryData?.records) return [];
    return summaryData.records.map((r: ConformalSummaryRecord) => ({
      name: r.model,
      'Clean PICP (%)': r.clean_picp_pct,
      'Miss AA PICP (%)': r.miss_AA_picp_pct,
      'Noise AA PICP (%)': r.noise_AA_picp_pct,
      'Target (90%)': 90.0,
      'MPIW (W)': r.mpiw_watts,
    }));
  }, [summaryData]);

  // Window Interval Chart Data (combining lower, upper, y_hat, y_true)
  const windowChartData = useMemo(() => {
    if (!intervalData) return [];
    return intervalData.horizon_hours.map((h, i) => {
      const lower = intervalData.L_raw[i];
      const upper = intervalData.U_raw[i];
      const yHat = intervalData.y_hat[i];
      const yTrue = intervalData.y_true[i];
      const covered = intervalData.is_covered[i];
      const timeLabel = intervalData.target_timestamps[i]
        ? intervalData.target_timestamps[i].split(' ')[1].slice(0, 5)
        : `t+${h}`;

      return {
        step: `t+${h}`,
        time: timeLabel,
        fullTimestamp: intervalData.target_timestamps[i] || `Horizon Step t+${h}`,
        LowerBound: lower,
        UpperBound: upper,
        // Band height for stacked area
        BandRange: upper - lower,
        PointForecast: yHat,
        ActualLoad: yTrue,
        isCovered: covered,
      };
    });
  }, [intervalData]);

  if (loadingSummary) {
    return <LoadingState message="Loading conformal uncertainty calibration artifacts..." />;
  }

  if (error || !summaryData) {
    return (
      <ErrorState
        title="Unable to load conformal uncertainty artifacts"
        message={error || 'No conformal calibration summary returned.'}
        onRetry={loadSummary}
      />
    );
  }

  const earlyRecord = summaryData.records.find((r) => r.model === 'Early Fusion') || summaryData.records[0];

  return (
    <div className="space-y-8">
      {/* Title & Research Mandate */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-emerald-600 mb-1">
          <Database className="w-3.5 h-3.5" />
          Conformal Prediction Lab • Source: {summaryData.source_artifact}
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Conformal Uncertainty & Prediction Interval Calibration
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-4xl leading-relaxed">
          Distribution-free uncertainty quantification using <strong className="text-slate-800">Horizon-Wise Split Conformal Prediction</strong> (nominal{' '}
          <strong className="text-slate-800">90% coverage level</strong>, significance α = 0.10) calibrated on{' '}
          <strong className="text-slate-800">587 rolling calibration windows</strong>. Quantile thresholds q̂_h are frozen from the nominal clean
          partition to rigorously test coverage erosion under sensor missingness and additive Gaussian noise.
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Nominal Target Coverage"
          value={`${summaryData.confidence_level_pct.toFixed(1)}%`}
          unit=""
          subtitle="Pre-registered confidence level (1 - α)"
          badge={{ text: `α = ${summaryData.alpha.toFixed(2)}`, variant: 'info' }}
          icon={<Target className="w-4 h-4 text-blue-600" />}
        />
        <MetricCard
          title="Clean Test Coverage (PICP)"
          value={`${earlyRecord.clean_picp_pct.toFixed(2)}%`}
          unit=""
          subtitle="Early Fusion nominal test coverage"
          badge={{ text: `ACE = ${earlyRecord.clean_ace_pct.toFixed(2)}%`, variant: 'success' }}
          icon={<ShieldCheck className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="Mean Interval Width (MPIW)"
          value={earlyRecord.mpiw_watts.toFixed(1)}
          unit="W"
          subtitle="2 × average horizon quantile q̂_h"
          badge={{ text: 'Frozen Width', variant: 'neutral' }}
          icon={<TrendingDown className="w-4 h-4 text-indigo-600" />}
        />
        <MetricCard
          title="Degraded Joint Coverage"
          value={`${earlyRecord.miss_AA_picp_pct.toFixed(1)}%`}
          unit=""
          subtitle="Early Fusion under 30% Missingness (A-A)"
          badge={{ text: `Erosion: ${(90 - earlyRecord.miss_AA_picp_pct).toFixed(1)}%`, variant: 'warning' }}
          icon={<AlertTriangle className="w-4 h-4 text-amber-600" />}
        />
      </div>

      {/* Coverage vs Width Trade-off Visual Chart */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Coverage Erosion & Width Comparison Across Models
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Empirical PICP (%) under Nominal Clean (NN), Dual Missingness (AA), and Dual Noise (AA) relative to 90% target
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded w-fit">
            Calibrated on N = {summaryData.calibration_samples} Windows
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={coverageComparisonChartData}
              margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={12} tickLine={false} unit="%" domain={[70, 100]} />
              <Tooltip
                formatter={(val: number) => [`${val.toFixed(2)}%`, '']}
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderColor: '#e2e8f0',
                  borderRadius: '0.375rem',
                  fontSize: '0.75rem',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
              <Bar dataKey="Clean PICP (%)" fill="#10b981" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Miss AA PICP (%)" fill="#f59e0b" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Noise AA PICP (%)" fill="#6366f1" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Conformal Calibration Summary Table */}
      <div className="bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-200">
          <h2 className="text-base font-semibold text-slate-900">
            Conformal Calibration Metrics & Oracle Width Expansion Table
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Detailed calibration thresholds, empirical PICP coverage, Winkler interval penalty scores, and required oracle expansion
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-200">
              <tr>
                <th className="px-5 py-3">Model Architecture</th>
                <th className="px-5 py-3 text-right">Cal MAE</th>
                <th className="px-5 py-3 text-right">Mean q̂_h</th>
                <th className="px-5 py-3 text-right font-bold text-slate-800">MPIW Width</th>
                <th className="px-5 py-3 text-right font-bold text-emerald-800">Clean PICP</th>
                <th className="px-5 py-3 text-right">Clean Winkler</th>
                <th className="px-5 py-3 text-right text-amber-800">Miss AA PICP</th>
                <th className="px-5 py-3 text-right">Noise AA PICP</th>
                <th className="px-5 py-3 text-right">Oracle Width Req</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-xs">
              {summaryData.records.map((r: ConformalSummaryRecord) => (
                <tr key={r.model} className="hover:bg-slate-50/80 transition-colors">
                  <td className="px-5 py-3.5 font-medium text-slate-900">{r.model}</td>
                  <td className="px-5 py-3.5 text-right font-mono text-slate-600">{r.calibration_mae_watts.toFixed(2)} W</td>
                  <td className="px-5 py-3.5 text-right font-mono text-slate-600">{r.conformal_quantile_qh_mean_watts.toFixed(2)} W</td>
                  <td className="px-5 py-3.5 text-right font-mono font-bold text-slate-800">{r.mpiw_watts.toFixed(1)} W</td>
                  <td className="px-5 py-3.5 text-right font-mono font-bold text-emerald-700">{r.clean_picp_pct.toFixed(2)}%</td>
                  <td className="px-5 py-3.5 text-right font-mono text-slate-600">{r.clean_winkler_interval_score.toFixed(1)}</td>
                  <td className="px-5 py-3.5 text-right font-mono font-bold text-amber-700">{r.miss_AA_picp_pct.toFixed(2)}%</td>
                  <td className="px-5 py-3.5 text-right font-mono text-slate-700">{r.noise_AA_picp_pct.toFixed(2)}%</td>
                  <td className="px-5 py-3.5 text-right font-mono text-indigo-700">
                    +{r.oracle_width_expansion_req_pct.toFixed(1)}% ({r.oracle_mpiw_miss_AA_watts.toFixed(1)} W)
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Window Prediction Interval Viewer */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-6">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Interactive Test Window Prediction Interval Rollout Viewer
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Inspect precomputed 90% prediction intervals [L, U], point forecasts, and ground truth across the 24-hour horizon
          </p>
        </div>

        {/* Controls Toolbar */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-700 uppercase">
            <SlidersHorizontal className="w-3.5 h-3.5 text-blue-600" />
            Interval Inspection Parameters
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Architecture</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="early_fusion">Early Fusion</option>
                <option value="energy_only">Energy-Only</option>
                <option value="late_fusion">Late Fusion</option>
                <option value="hybrid_fusion">Hybrid Fusion</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">Condition</label>
              <select
                value={selectedCondition}
                onChange={(e) => setSelectedCondition(e.target.value)}
                className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="clean_NN">Clean Nominal (N-N)</option>
                <option value="missingness_NA">Weather Missingness (N-A)</option>
                <option value="missingness_AN">Electricity Missingness (A-N)</option>
                <option value="missingness_AA">Joint Missingness (A-A)</option>
                <option value="noise_NA">Weather Noise (N-A)</option>
                <option value="noise_AN">Electricity Noise (A-N)</option>
                <option value="noise_AA">Joint Noise (A-A)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-700 mb-1">
                Test Window (Index 0 to 586)
              </label>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  disabled={windowIdx <= 0}
                  onClick={() => setWindowIdx((w) => Math.max(0, w - 1))}
                  className="p-2 border border-slate-300 rounded-md bg-white hover:bg-slate-100 disabled:opacity-40 disabled:hover:bg-white"
                  title="Previous Window"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <input
                  type="number"
                  min={0}
                  max={586}
                  value={windowIdx}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    if (!isNaN(v) && v >= 0 && v <= 586) {
                      setWindowIdx(v);
                    }
                  }}
                  className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-center font-mono text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
                <button
                  type="button"
                  disabled={windowIdx >= 586}
                  onClick={() => setWindowIdx((w) => Math.min(586, w + 1))}
                  className="p-2 border border-slate-300 rounded-md bg-white hover:bg-slate-100 disabled:opacity-40 disabled:hover:bg-white"
                  title="Next Window"
                >
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Interval Viewer Content */}
        {loadingInterval ? (
          <LoadingState message={`Fetching prediction intervals for Window #${windowIdx}...`} />
        ) : intervalError ? (
          <ErrorState
            title="Failed to load intervals"
            message={intervalError}
            onRetry={() => loadIntervals(selectedModel, selectedCondition, windowIdx)}
          />
        ) : intervalData ? (
          <div className="space-y-4">
            {/* Window Statistics Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-50/70 border border-slate-200 p-3 rounded-lg text-xs">
              <div className="space-x-2">
                <span className="font-semibold text-slate-800">Forecast Origin:</span>
                <span className="font-mono text-slate-600">{intervalData.forecast_origin_timestamp}</span>
              </div>
              <div className="flex items-center gap-4">
                <div>
                  Window Coverage:{' '}
                  <strong className={intervalData.window_picp_pct >= 90 ? 'text-emerald-700' : 'text-amber-700'}>
                    {intervalData.window_picp_pct.toFixed(1)}%
                  </strong>{' '}
                  ({intervalData.is_covered.filter(Boolean).length}/24 steps)
                </div>
                <div>
                  Window Winkler Score: <strong className="font-mono text-slate-800">{intervalData.window_interval_score.toFixed(1)}</strong>
                </div>
                <div>
                  Dataset MPIW: <strong className="font-mono text-slate-800">{intervalData.dataset_mean_mpiw_watts.toFixed(1)} W</strong>
                </div>
              </div>
            </div>

            {/* Area Chart with Shaded 90% Prediction Interval */}
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={windowChartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis dataKey="step" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit=" W" />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-sm text-xs space-y-1">
                            <div className="font-semibold text-slate-900">{d.step} ({d.fullTimestamp})</div>
                            <div className="text-slate-900 font-bold">
                              Actual Ground Truth: <span className="font-mono">{d.ActualLoad.toFixed(1)} W</span>
                            </div>
                            <div className="text-blue-700">
                              Point Forecast: <span className="font-mono">{d.PointForecast.toFixed(1)} W</span>
                            </div>
                            <div className="text-emerald-700">
                              90% Conformal Band: <span className="font-mono">[{d.LowerBound.toFixed(1)} W, {d.UpperBound.toFixed(1)} W]</span>
                            </div>
                            <div className="pt-1 text-[11px]">
                              Status:{' '}
                              <span className={d.isCovered ? 'text-emerald-700 font-bold' : 'text-rose-700 font-bold'}>
                                {d.isCovered ? 'Covered (Inside Band)' : 'Violation (Outside Band)'}
                              </span>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                  {/* Invisible baseline for band stack */}
                  <Area
                    type="monotone"
                    dataKey="LowerBound"
                    stackId="1"
                    stroke="none"
                    fill="transparent"
                  />
                  {/* 90% Shaded Interval Band */}
                  <Area
                    type="monotone"
                    dataKey="BandRange"
                    name="90% Conformal Prediction Interval [L, U]"
                    stackId="1"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.18}
                  />
                  {/* Point Forecast Line */}
                  <Line
                    type="monotone"
                    dataKey="PointForecast"
                    name="Model Point Forecast ŷ"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                  {/* Actual Ground Truth Line */}
                  <Line
                    type="monotone"
                    dataKey="ActualLoad"
                    name="Actual Ground Truth y"
                    stroke="#0f172a"
                    strokeWidth={2}
                    dot={{ r: 3, fill: '#0f172a' }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : null}
      </div>

      {/* Epistemic Boundaries and Conformal Guarantees Explanation */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 space-y-3 text-xs text-slate-600 leading-relaxed">
        <div className="flex items-center gap-2 font-semibold text-slate-800">
          <Info className="w-4 h-4 text-blue-600" />
          Conformal Prediction Guarantees & Theoretical Boundaries
        </div>
        <p>
          Split conformal prediction provides <strong>finite-sample, distribution-free marginal coverage guarantees</strong>:
          under the exchangeability assumption between the calibration and test splits, the probability that the true multi-step load
          falls within the precomputed interval satisfies P(Y_t+h ∈ [L_t+h, U_t+h]) ≥ 1 - α = 90%.
        </p>
        <p>
          <strong>Critical Theoretical Nuance:</strong> Marginal coverage guarantees do not imply conditional coverage for every
          subpopulation or isolated horizon hour. Furthermore, because the calibration quantiles q̂_h are frozen from the nominal
          clean regime, the interval width (MPIW) remains fixed by design under test missingness or noise. Consequently, when test inputs undergo
          severe distribution shift (e.g. 30% missingness), empirical coverage erodes from 88.6% down to 85.8% (Early Fusion) or 79.9% (Energy-Only),
          demonstrating that conformal guarantees are inherently vulnerable to uncorrected test-time covariate shifts.
        </p>
      </div>
    </div>
  );
};
