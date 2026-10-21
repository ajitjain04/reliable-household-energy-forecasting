import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import { CleanBenchmarkResponse, CleanBenchmarkItem } from '../api/types';
import { MetricCard } from '../components/common/MetricCard';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Award,
  Zap,
  TrendingDown,
  Database,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Clock,
  Layers,
  Activity,
} from 'lucide-react';

type SortField =
  | 'clean_test_rank'
  | 'model'
  | 'trainable_parameters'
  | 'val_mae_watts'
  | 'val_rmse_watts'
  | 'val_r2'
  | 'test_mae_watts'
  | 'test_rmse_watts'
  | 'test_r2'
  | 'test_mape_pct'
  | 'test_smape_pct'
  | 'delta_mae_vs_energy_only_watts';

type MetricView = 'error' | 'r2' | 'mape';

export const ModelBenchmarks: React.FC = () => {
  const [data, setData] = useState<CleanBenchmarkResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>('clean_test_rank');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [metricView, setMetricView] = useState<MetricView>('error');

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.getCleanBenchmarks();
      setData(result);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to fetch benchmark comparison data from backend.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const sortedRecords = useMemo(() => {
    if (!data?.records) return [];
    return [...data.records].sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      if (typeof valA === 'string') {
        valA = (valA as string).toLowerCase();
        valB = (valB as string).toLowerCase();
      }
      if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
      if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  }, [data, sortField, sortDirection]);

  // Chart data sorted by rank ascending for clean presentation
  const chartData = useMemo(() => {
    if (!data?.records) return [];
    return [...data.records]
      .sort((a, b) => a.clean_test_rank - b.clean_test_rank)
      .map((r) => ({
        name: r.model,
        'Test MAE (W)': r.test_mae_watts,
        'Test RMSE (W)': r.test_rmse_watts,
        'Val MAE (W)': r.val_mae_watts,
        'Test R²': r.test_r2,
        'Val R²': r.val_r2,
        'MAPE (%)': r.test_mape_pct,
        'sMAPE (%)': r.test_smape_pct,
      }));
  }, [data]);

  if (loading) {
    return <LoadingState message="Loading clean model benchmark comparison from API..." />;
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Unable to load benchmark comparisons"
        message={error || 'No benchmark records received from API.'}
        onRetry={loadData}
      />
    );
  }

  const leaderRecord =
    data.records.find((r) => r.model === data.benchmark_leader) || data.records[0];
  const baselineRecord =
    data.records.find((r) => r.model === 'Energy-Only') || data.records[0];

  const renderSortArrow = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-3.5 h-3.5 text-slate-400 opacity-60 group-hover:opacity-100" />;
    }
    return sortDirection === 'asc' ? (
      <ArrowUp className="w-3.5 h-3.5 text-blue-600 font-bold" />
    ) : (
      <ArrowDown className="w-3.5 h-3.5 text-blue-600 font-bold" />
    );
  };

  return (
    <div className="space-y-8">
      {/* Title & Research Mandate */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-600 mb-1">
          <Database className="w-3.5 h-3.5" />
          Audited Research Benchmark • Source: {data.source_artifact}
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Clean Model Benchmarks & Architectural Comparison
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-4xl leading-relaxed">
          Rigorous out-of-sample evaluation across four distinct neural architectures evaluated on{' '}
          <strong className="text-slate-800">587 rolling test windows</strong> (14,088 multi-step hourly
          forecasts) under nominal clean conditions. Models are tasked with forecasting 24 hours of
          household electricity demand given 168 hours of historical load and 7 exogenous weather variables.
        </p>
      </div>

      {/* KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Benchmark Leader"
          value={data.benchmark_leader}
          subtitle="Top Out-of-Sample Accuracy"
          badge={{ text: 'Rank #1', variant: 'success' }}
          icon={<Award className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="Lowest Test MAE"
          value={leaderRecord.test_mae_watts.toFixed(2)}
          unit="W"
          subtitle={`${leaderRecord.model} Point Forecast`}
          badge={{
            text: `${leaderRecord.relative_delta_mae_pct.toFixed(2)}% vs baseline`,
            variant: 'success',
          }}
          icon={<TrendingDown className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="Baseline Test MAE"
          value={baselineRecord.test_mae_watts.toFixed(2)}
          unit="W"
          subtitle="Energy-Only Recurrent Model"
          badge={{ text: 'Unimodal Reference', variant: 'neutral' }}
          icon={<Zap className="w-4 h-4 text-blue-600" />}
        />
        <MetricCard
          title="Evaluated Architectures"
          value={data.total_models}
          unit="Models"
          subtitle="From 51.9k to 145.5k parameters"
          badge={{ text: 'Clean Nominal Test', variant: 'info' }}
          icon={<Layers className="w-4 h-4 text-indigo-600" />}
        />
      </div>

      {/* Interactive Visualization Panel */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Comparative Performance Visualization
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Out-of-sample test metrics across all four evaluated architectures
            </p>
          </div>

          {/* Metric View Toggle */}
          <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg text-xs font-medium text-slate-600">
            <button
              onClick={() => setMetricView('error')}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                metricView === 'error'
                  ? 'bg-white text-blue-700 shadow-sm font-semibold'
                  : 'hover:text-slate-900'
              }`}
            >
              MAE & RMSE (Watts)
            </button>
            <button
              onClick={() => setMetricView('r2')}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                metricView === 'r2'
                  ? 'bg-white text-blue-700 shadow-sm font-semibold'
                  : 'hover:text-slate-900'
              }`}
            >
              R² Score
            </button>
            <button
              onClick={() => setMetricView('mape')}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                metricView === 'mape'
                  ? 'bg-white text-blue-700 shadow-sm font-semibold'
                  : 'hover:text-slate-900'
              }`}
            >
              MAPE & sMAPE (%)
            </button>
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            {metricView === 'error' ? (
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} unit=" W" domain={[60, 105]} />
                <Tooltip
                  formatter={(value: number) => [`${value.toFixed(2)} W`, '']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.375rem',
                    fontSize: '0.75rem',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                <Bar dataKey="Test MAE (W)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Test RMSE (W)" fill="#94a3b8" radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : metricView === 'r2' ? (
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} domain={[0, 0.4]} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(4), '']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.375rem',
                    fontSize: '0.75rem',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                <Bar dataKey="Test R²" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Val R²" fill="#cbd5e1" radius={[4, 4, 0, 0]} />
              </BarChart>
            ) : (
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} unit="%" domain={[30, 60]} />
                <Tooltip
                  formatter={(value: number) => [`${value.toFixed(2)}%`, '']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.375rem',
                    fontSize: '0.75rem',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                <Bar dataKey="MAPE (%)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                <Bar dataKey="sMAPE (%)" fill="#64748b" radius={[4, 4, 0, 0]} />
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sortable Benchmark Comparison Table */}
      <div className="bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Clean Benchmark Performance Matrix
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Click any column header to sort. Evaluated on 587 rolling test windows (168h $\to$ 24h).
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded w-fit">
            N = {data.total_models} Models
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-200 select-none">
              <tr>
                <th
                  onClick={() => handleSort('clean_test_rank')}
                  className="px-6 py-3.5 cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center gap-1.5">
                    <span>Rank</span>
                    {renderSortArrow('clean_test_rank')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('model')}
                  className="px-6 py-3.5 cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center gap-1.5">
                    <span>Model Architecture</span>
                    {renderSortArrow('model')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('trainable_parameters')}
                  className="px-6 py-3.5 text-right cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>Params</span>
                    {renderSortArrow('trainable_parameters')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('val_mae_watts')}
                  className="px-6 py-3.5 text-right cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>Val MAE</span>
                    {renderSortArrow('val_mae_watts')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('test_mae_watts')}
                  className="px-6 py-3.5 text-right font-bold text-slate-800 cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>Test MAE</span>
                    {renderSortArrow('test_mae_watts')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('test_rmse_watts')}
                  className="px-6 py-3.5 text-right cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>Test RMSE</span>
                    {renderSortArrow('test_rmse_watts')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('test_r2')}
                  className="px-6 py-3.5 text-right cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>Test R²</span>
                    {renderSortArrow('test_r2')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('test_mape_pct')}
                  className="px-6 py-3.5 text-right cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>MAPE</span>
                    {renderSortArrow('test_mape_pct')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('test_smape_pct')}
                  className="px-6 py-3.5 text-right cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>sMAPE</span>
                    {renderSortArrow('test_smape_pct')}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('delta_mae_vs_energy_only_watts')}
                  className="px-6 py-3.5 text-right cursor-pointer hover:bg-slate-100 transition-colors group"
                >
                  <div className="flex items-center justify-end gap-1.5">
                    <span>Δ MAE (vs Base)</span>
                    {renderSortArrow('delta_mae_vs_energy_only_watts')}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {sortedRecords.map((r: CleanBenchmarkItem) => {
                const isLeader = r.model === data.benchmark_leader;
                return (
                  <tr
                    key={r.model}
                    className={`hover:bg-slate-50/80 transition-colors ${
                      isLeader ? 'bg-emerald-50/30' : ''
                    }`}
                  >
                    <td className="px-6 py-4 font-mono font-medium text-slate-500">
                      #{r.clean_test_rank}
                    </td>
                    <td className="px-6 py-4 font-medium text-slate-900">
                      <div className="flex items-center gap-2">
                        {r.model}
                        {isLeader && (
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-800">
                            Leader
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-slate-400 font-normal mt-0.5">
                        {r.architecture_type}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-xs text-slate-500">
                      {r.trainable_parameters.toLocaleString()}
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-slate-600">
                      {r.val_mae_watts.toFixed(2)} W
                    </td>
                    <td className="px-6 py-4 text-right font-mono font-bold text-slate-900">
                      {r.test_mae_watts.toFixed(2)} W
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-slate-600">
                      {r.test_rmse_watts.toFixed(2)} W
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-slate-600">
                      {r.test_r2.toFixed(4)}
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-slate-600">
                      {r.test_mape_pct.toFixed(2)}%
                    </td>
                    <td className="px-6 py-4 text-right font-mono text-slate-600">
                      {r.test_smape_pct.toFixed(2)}%
                    </td>
                    <td className="px-6 py-4 text-right font-mono">
                      <span
                        className={
                          r.delta_mae_vs_energy_only_watts < 0
                            ? 'text-emerald-700 font-medium'
                            : r.delta_mae_vs_energy_only_watts > 0
                            ? 'text-rose-700 font-medium'
                            : 'text-slate-500'
                        }
                      >
                        {r.delta_mae_vs_energy_only_watts > 0 ? '+' : ''}
                        {r.delta_mae_vs_energy_only_watts.toFixed(2)} W (
                        {r.relative_delta_mae_pct > 0 ? '+' : ''}
                        {r.relative_delta_mae_pct.toFixed(2)}%)
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Experimental Protocol & Metric Definitions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <Clock className="w-4 h-4 text-blue-600" />
            Evaluation Setup & Horizon Specifications
          </div>
          <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-28">History Window:</span>
              <span>
                168 hours (7 full calendar days) of historical hourly load and exogenous weather observations.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-28">Forecast Horizon:</span>
              <span>
                24 hours forward (direct multi-step day-ahead rolling forecast vector).
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-28">Test Sample Size:</span>
              <span>
                587 rolling origin test windows, yielding 14,088 point forecast evaluations with strictly zero lookahead leakage.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-28">Exogenous Channels:</span>
              <span>
                7 weather channels (temperature, humidity, atmospheric pressure, wind speed, wind direction, rain, cloud cover).
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
            <Activity className="w-4 h-4 text-emerald-600" />
            Metric Definitions & Units
          </div>
          <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-24">MAE (Watts):</span>
              <span>
                Mean Absolute Error: linear penalization across the 24-hour horizon. Primary ranking metric.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-24">RMSE (Watts):</span>
              <span>
                Root Mean Squared Error: quadratic penalization, penalizing severe localized peak forecast errors.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-24">R² Score:</span>
              <span>
                Coefficient of determination measuring variance explained relative to sample mean baseline.
              </span>
            </div>
            <div className="flex items-start gap-2">
              <span className="font-semibold text-slate-800 min-w-24">MAPE / sMAPE:</span>
              <span>
                Mean Absolute Percentage Error and Symmetric MAPE evaluating relative proportional forecasting accuracy.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
