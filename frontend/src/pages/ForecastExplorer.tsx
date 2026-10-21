import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import {
  TestWindowsMetadataResponse,
  WindowForecastResponse,
  WindowModelMetrics,
} from '../api/types';
import { MetricCard } from '../components/common/MetricCard';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  TrendingUp,
  Database,
  Calendar,
  ChevronLeft,
  ChevronRight,
  Shuffle,
  Award,
  Layers,
  AlertCircle,
} from 'lucide-react';

const MODEL_COLORS: Record<string, string> = {
  'Early Fusion': '#10b981', // Emerald (Leader)
  'Energy-Only': '#3b82f6',  // Blue (Baseline)
  'Late Fusion': '#8b5cf6',  // Purple
  'Hybrid Fusion': '#f59e0b', // Amber
};

export const ForecastExplorer: React.FC = () => {
  const [meta, setMeta] = useState<TestWindowsMetadataResponse | null>(null);
  const [windowForecast, setWindowForecast] = useState<WindowForecastResponse | null>(null);
  const [loadingMeta, setLoadingMeta] = useState<boolean>(true);
  const [loadingWindow, setLoadingWindow] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [windowError, setWindowError] = useState<string | null>(null);

  // Active window index
  const [windowIdx, setWindowIdx] = useState<number>(0);

  // Model Visibility Toggles
  const [visibleModels, setVisibleModels] = useState<Record<string, boolean>>({
    'Early Fusion': true,
    'Energy-Only': true,
    'Late Fusion': true,
    'Hybrid Fusion': true,
  });

  // Load Window Metadata
  const loadMetadata = async () => {
    setLoadingMeta(true);
    setError(null);
    try {
      const metadata = await api.getTestWindowsMetadata();
      setMeta(metadata);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to load test evaluation window metadata.');
      }
    } finally {
      setLoadingMeta(false);
    }
  };

  // Load Window Forecast
  const loadWindowData = async (idx: number) => {
    setLoadingWindow(true);
    setWindowError(null);
    try {
      const forecast = await api.getTestWindowForecast(idx);
      setWindowForecast(forecast);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setWindowError(err.message);
      } else {
        setWindowError(`Failed to load forecast rollout for window #${idx}.`);
      }
    } finally {
      setLoadingWindow(false);
    }
  };

  useEffect(() => {
    loadMetadata();
  }, []);

  useEffect(() => {
    loadWindowData(windowIdx);
  }, [windowIdx]);

  const toggleModel = (modelName: string) => {
    setVisibleModels((prev) => ({
      ...prev,
      [modelName]: !prev[modelName],
    }));
  };

  // Chart data formatting
  const chartData = useMemo(() => {
    if (!windowForecast) return [];
    return windowForecast.horizon_hours.map((h, i) => {
      const actual = windowForecast.actual_watts[i];
      const targetTs = windowForecast.target_timestamps[i] || `t+${h}`;
      const timeLabel = targetTs.includes(' ') ? targetTs.split(' ')[1].slice(0, 5) : `t+${h}`;

      const point: Record<string, any> = {
        step: `t+${h}`,
        time: timeLabel,
        fullTimestamp: targetTs,
        'Actual Load': actual,
      };

      Object.entries(windowForecast.predictions_watts).forEach(([modelName, preds]) => {
        point[modelName] = preds[i];
      });

      return point;
    });
  }, [windowForecast]);

  // Determine window winning model (lowest window MAE)
  const windowWinner = useMemo(() => {
    if (!windowForecast?.metrics) return null;
    let bestModel = '';
    let minMae = Infinity;
    Object.entries(windowForecast.metrics).forEach(([model, m]) => {
      if (m.mae_watts < minMae) {
        minMae = m.mae_watts;
        bestModel = model;
      }
    });
    return { model: bestModel, mae: minMae };
  }, [windowForecast]);

  if (loadingMeta) {
    return <LoadingState message="Loading 587 rolling evaluation windows metadata..." />;
  }

  if (error || !meta) {
    return (
      <ErrorState
        title="Unable to load forecast explorer"
        message={error || 'No test partition metadata returned.'}
        onRetry={loadMetadata}
      />
    );
  }

  return (
    <div className="space-y-8">
      {/* Title & Research Mandate */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-600 mb-1">
          <Database className="w-3.5 h-3.5" />
          Test Partition Rollout Explorer • Total Windows: {meta.total_windows} (14,088 multi-step points)
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          24-Hour Multi-Step Forecast Rollout Explorer
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-4xl leading-relaxed">
          Interactive trajectory explorer across all <strong className="text-slate-800">587 withheld out-of-sample test windows</strong> covering{' '}
          <strong className="text-slate-800">{meta.first_forecast_origin}</strong> to <strong className="text-slate-800">{meta.last_forecast_origin}</strong>.
          Inspect multi-step predicted load curves vs. ground truth active power and instantaneous error residuals across all four models.
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Test Window"
          value={`#${windowIdx}`}
          unit={`/ ${meta.total_windows - 1}`}
          subtitle={windowForecast ? windowForecast.forecast_origin_timestamp : 'Loading...'}
          badge={{ text: 'Chronological', variant: 'info' }}
          icon={<Calendar className="w-4 h-4 text-blue-600" />}
        />
        <MetricCard
          title="Window Best Architecture"
          value={windowWinner ? windowWinner.model : 'Computing...'}
          unit=""
          subtitle={windowWinner ? `Window MAE: ${windowWinner.mae.toFixed(2)} W` : ''}
          badge={{ text: 'Lowest Window Error', variant: 'success' }}
          icon={<Award className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="Forecast Horizon"
          value={meta.forecast_horizon_hours}
          unit="Hours Forward"
          subtitle="Direct multi-step vector"
          badge={{ text: 't+1 to t+24', variant: 'neutral' }}
          icon={<TrendingUp className="w-4 h-4 text-indigo-600" />}
        />
        <MetricCard
          title="Historical Lookback"
          value={meta.history_length_hours}
          unit="Hours (7 Days)"
          subtitle="Load history + 7 weather channels"
          badge={{ text: 'Conditioning Window', variant: 'neutral' }}
          icon={<Layers className="w-4 h-4 text-slate-600" />}
        />
      </div>

      {/* Navigation & Controls Toolbar */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          {/* Window Navigation Controls */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={windowIdx <= 0}
              onClick={() => setWindowIdx(0)}
              className="px-2.5 py-1.5 text-xs font-medium border border-slate-300 rounded bg-white hover:bg-slate-50 disabled:opacity-40"
              title="Jump to First Window"
            >
              First
            </button>
            <button
              type="button"
              disabled={windowIdx <= 0}
              onClick={() => setWindowIdx((w) => Math.max(0, w - 1))}
              className="p-1.5 border border-slate-300 rounded bg-white hover:bg-slate-50 disabled:opacity-40"
              title="Previous Window"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-1.5 text-xs">
              <span className="font-semibold text-slate-700">Window:</span>
              <input
                type="number"
                min={0}
                max={meta.total_windows - 1}
                value={windowIdx}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v) && v >= 0 && v < meta.total_windows) {
                    setWindowIdx(v);
                  }
                }}
                className="w-16 px-2 py-1 text-center font-mono font-bold text-xs bg-slate-50 border border-slate-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-slate-500 font-mono">/ {meta.total_windows - 1}</span>
            </div>
            <button
              type="button"
              disabled={windowIdx >= meta.total_windows - 1}
              onClick={() => setWindowIdx((w) => Math.min(meta.total_windows - 1, w + 1))}
              className="p-1.5 border border-slate-300 rounded bg-white hover:bg-slate-50 disabled:opacity-40"
              title="Next Window"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              type="button"
              disabled={windowIdx >= meta.total_windows - 1}
              onClick={() => setWindowIdx(meta.total_windows - 1)}
              className="px-2.5 py-1.5 text-xs font-medium border border-slate-300 rounded bg-white hover:bg-slate-50 disabled:opacity-40"
              title="Jump to Last Window"
            >
              Last
            </button>
            <button
              type="button"
              onClick={() => setWindowIdx(Math.floor(Math.random() * meta.total_windows))}
              className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium border border-slate-300 rounded bg-white hover:bg-slate-50 text-slate-700 ml-1"
              title="Random Window"
            >
              <Shuffle className="w-3.5 h-3.5" />
              <span>Random</span>
            </button>
          </div>

          {/* Model Toggle Checkboxes */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold text-slate-500 uppercase mr-1">Models:</span>
            {meta.available_models.map((modelName) => {
              const isVisible = visibleModels[modelName];
              const color = MODEL_COLORS[modelName] || '#64748b';
              return (
                <button
                  key={modelName}
                  type="button"
                  onClick={() => toggleModel(modelName)}
                  className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium border transition-colors ${
                    isVisible
                      ? 'bg-slate-50 border-slate-300 text-slate-900 shadow-sm'
                      : 'bg-white border-dashed border-slate-200 text-slate-400 opacity-60 hover:opacity-100'
                  }`}
                >
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: isVisible ? color : '#cbd5e1' }}
                  />
                  <span>{modelName}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Trajectory Forecast Chart */}
      {loadingWindow ? (
        <LoadingState message={`Fetching trajectory rollout for Window #${windowIdx}...`} />
      ) : windowError ? (
        <ErrorState
          title="Failed to load window forecast"
          message={windowError}
          onRetry={() => loadWindowData(windowIdx)}
        />
      ) : windowForecast ? (
        <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                Window #{windowIdx} Forecast Trajectory vs Ground Truth
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Forecast Origin: <strong className="font-mono text-slate-800">{windowForecast.forecast_origin_timestamp}</strong> • Target Horizon: 24 Hours Forward
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="inline-flex items-center gap-1 text-slate-700 font-semibold bg-slate-100 px-2.5 py-1 rounded">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-950 inline-block" />
                Actual Observed Load
              </span>
            </div>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="step" stroke="#64748b" fontSize={11} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit=" W" />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-sm text-xs space-y-1.5 min-w-56">
                          <div className="font-bold text-slate-900 border-b border-slate-100 pb-1">
                            {d.step} ({d.fullTimestamp})
                          </div>
                          <div className="text-slate-950 font-bold flex justify-between">
                            <span>Actual Ground Truth:</span>
                            <span className="font-mono">{d['Actual Load']?.toFixed(1)} W</span>
                          </div>
                          {Object.entries(visibleModels).map(([mName, isVis]) => {
                            if (!isVis || d[mName] === undefined) return null;
                            const err = d[mName] - d['Actual Load'];
                            return (
                              <div
                                key={mName}
                                className="flex justify-between items-center text-[11px]"
                                style={{ color: MODEL_COLORS[mName] || '#64748b' }}
                              >
                                <span className="font-medium">{mName}:</span>
                                <span className="font-mono font-bold">
                                  {d[mName]?.toFixed(1)} W{' '}
                                  <span className="text-slate-400 font-normal">
                                    ({err > 0 ? '+' : ''}{err.toFixed(1)})
                                  </span>
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                {/* Ground Truth Line */}
                <Line
                  type="monotone"
                  dataKey="Actual Load"
                  stroke="#0f172a"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#0f172a' }}
                />
                {/* Multi-Model Forecast Lines */}
                {Object.entries(visibleModels).map(([mName, isVis]) => {
                  if (!isVis) return null;
                  return (
                    <Line
                      key={mName}
                      type="monotone"
                      dataKey={mName}
                      stroke={MODEL_COLORS[mName] || '#64748b'}
                      strokeWidth={1.75}
                      strokeDasharray={mName === 'Early Fusion' ? undefined : '4 4'}
                      dot={false}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : null}

      {/* Window-Level Model Metrics Comparison */}
      {windowForecast?.metrics && (
        <div className="bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
          <div className="px-6 py-4 border-b border-slate-200">
            <h2 className="text-base font-semibold text-slate-900">
              Window #{windowIdx} Error Metrics Across Architectures
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Instantaneous multi-step error statistics computed over this specific 24-hour horizon
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-600">
              <thead className="bg-slate-50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-200">
                <tr>
                  <th className="px-6 py-3.5">Model Architecture</th>
                  <th className="px-6 py-3.5 text-right font-bold text-slate-800">Window MAE</th>
                  <th className="px-6 py-3.5 text-right">Window RMSE</th>
                  <th className="px-6 py-3.5 text-right">Max Peak Error</th>
                  <th className="px-6 py-3.5 text-center">Window Performance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 text-xs">
                {Object.entries(windowForecast.metrics).map(([mName, met]: [string, WindowModelMetrics]) => {
                  const isBest = windowWinner?.model === mName;
                  return (
                    <tr
                      key={mName}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        isBest ? 'bg-emerald-50/30' : ''
                      }`}
                    >
                      <td className="px-6 py-3.5 font-medium text-slate-900 flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: MODEL_COLORS[mName] || '#64748b' }}
                        />
                        <span>{mName}</span>
                        {isBest && (
                          <span className="inline-flex items-center px-1.5 py-0.2 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800">
                            Window Winner
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono font-bold text-slate-900">
                        {met.mae_watts.toFixed(2)} W
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-slate-600">
                        {met.rmse_watts.toFixed(2)} W
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-slate-600">
                        {met.max_error_watts.toFixed(2)} W
                      </td>
                      <td className="px-6 py-3.5 text-center">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[10px] font-semibold ${
                            isBest ? 'bg-emerald-100 text-emerald-800' : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {isBest ? 'Lowest MAE' : 'Sub-optimal'}
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

      {/* Scope and API Boundary Documentation */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 space-y-2 text-xs text-slate-600 leading-relaxed">
        <div className="flex items-center gap-2 font-semibold text-slate-800">
          <AlertCircle className="w-4 h-4 text-blue-600" />
          Evaluation Partition Scope & Boundary Documentation
        </div>
        <p>
          The forecast explorer displays verified out-of-sample test rollouts across all 587 rolling evaluation windows
          under nominal operating conditions for the four approved neural architectures. Synthetic sensor degradation rollouts
          (30% MCAR missingness and 20% Gaussian noise) are evaluated in aggregate in the Operational Reliability and Conformal Uncertainty
          modules to maintain architectural isolation and prevent non-conforming synthetic trajectory fabrication.
        </p>
      </div>
    </div>
  );
};
