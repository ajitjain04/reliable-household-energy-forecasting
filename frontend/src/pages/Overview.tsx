import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { CleanBenchmarkResponse } from '../api/types';
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
import { Award, Zap, TrendingDown, CheckCircle2, Database } from 'lucide-react';

export const Overview: React.FC = () => {
  const [data, setData] = useState<CleanBenchmarkResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

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
        setError('Failed to fetch benchmark data from backend.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) {
    return <LoadingState message="Fetching clean benchmark comparisons from API..." />;
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Unable to load benchmark data"
        message={error || 'No benchmark data returned from API.'}
        onRetry={loadData}
      />
    );
  }

  if (!data.records || data.records.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-slate-200 p-8 text-center text-slate-500">
        No model benchmark records were returned by the backend.
      </div>
    );
  }

  const leaderRecord = data.records.find((r) => r.model === data.benchmark_leader) || data.records[0];
  const baselineRecord = data.records.find((r) => r.model === 'Energy-Only') || data.records[0];

  const sortedRecords = [...data.records].sort((a, b) => a.clean_test_rank - b.clean_test_rank);

  // Chart data formatted directly from API records sorted by rank
  const chartData = sortedRecords.map((r) => ({
    name: r.model,
    MAE: r.test_mae_watts,
    RMSE: r.test_rmse_watts,
    R2: r.test_r2,
  }));

  return (
    <div className="space-y-8">
      {/* Title & Research Mandate */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-blue-600 mb-1">
          <Database className="w-3.5 h-3.5" />
          Audited Research Benchmark • Source: {data.source_artifact}
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Clean Model Benchmark & Evaluation Overview
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-4xl leading-relaxed">
          Comparing four distinct neural architectures evaluated on 587 rolling test windows
          (14,088 multi-step forecast observations) under nominal clean conditions. Features 168 hours
          of historical household active power consumption fused with 7 exogenous weather variables.
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
          subtitle={`${data.benchmark_leader} Point Forecast`}
          badge={{ text: `${leaderRecord.relative_delta_mae_pct.toFixed(2)}% vs baseline`, variant: 'success' }}
          icon={<TrendingDown className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="Lowest Test RMSE"
          value={leaderRecord.test_rmse_watts.toFixed(2)}
          unit="W"
          subtitle={`Baseline: ${baselineRecord.test_rmse_watts.toFixed(2)} W`}
          badge={{ text: 'Lower Variance', variant: 'info' }}
          icon={<Zap className="w-4 h-4 text-blue-600" />}
        />
        <MetricCard
          title="Best Test R²"
          value={leaderRecord.test_r2.toFixed(4)}
          subtitle={`Baseline R²: ${baselineRecord.test_r2.toFixed(4)}`}
          badge={{ text: 'Variance Explained', variant: 'info' }}
          icon={<CheckCircle2 className="w-4 h-4 text-blue-600" />}
        />
      </div>

      {/* Model Comparison Table */}
      <div className="bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Clean Benchmark Performance Comparison
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Evaluated on 587 rolling test windows (168h lookback → 24h horizon)
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded">
            N = {data.total_models} Models
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-200">
              <tr>
                <th className="px-6 py-3">Rank</th>
                <th className="px-6 py-3">Model Architecture</th>
                <th className="px-6 py-3">Parameters</th>
                <th className="px-6 py-3 text-right">Val MAE</th>
                <th className="px-6 py-3 text-right font-bold text-slate-700">Test MAE</th>
                <th className="px-6 py-3 text-right">Test RMSE</th>
                <th className="px-6 py-3 text-right">Test R²</th>
                <th className="px-6 py-3 text-right">MAPE</th>
                <th className="px-6 py-3 text-right">Δ MAE (vs Base)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {sortedRecords.map((r) => {
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
                    <td className="px-6 py-4 font-mono text-xs text-slate-500">
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
                        {r.delta_mae_vs_energy_only_watts.toFixed(2)} W ({r.relative_delta_mae_pct > 0 ? '+' : ''}
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

      {/* Visual Chart Comparison */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm">
        <h2 className="text-base font-semibold text-slate-900 mb-1">
          Forecasting Error Comparison (Test MAE vs Test RMSE)
        </h2>
        <p className="text-xs text-slate-500 mb-6">
          Direct API metrics comparison across the 4 evaluated neural architectures (lower is better)
        </p>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="name"
                stroke="#64748b"
                fontSize={12}
                tickLine={false}
              />
              <YAxis
                stroke="#64748b"
                fontSize={12}
                tickLine={false}
                unit=" W"
                domain={[60, 110]}
              />
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
              <Legend
                wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }}
              />
              <Bar dataKey="MAE" name="Test MAE (Watts)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="RMSE" name="Test RMSE (Watts)" fill="#94a3b8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
