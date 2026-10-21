import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import { RobustnessResponse, RobustnessRecord } from '../api/types';
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
  ShieldAlert,
  Database,
  SlidersHorizontal,
  AlertTriangle,
  Zap,
  Layers,
  TrendingUp,
} from 'lucide-react';

export const ReliabilityAnalysis: React.FC = () => {
  const [data, setData] = useState<RobustnessResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [selectedExperiment, setSelectedExperiment] = useState<string>('missingness_30');
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [selectedCondition, setSelectedCondition] = useState<string>('all');

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch without query params to obtain full dataset and filter client-side for immediate responsiveness
      const result = await api.getRobustnessMatrix();
      setData(result);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to fetch robustness evaluation data from backend.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Filtered records
  const filteredRecords = useMemo(() => {
    if (!data?.records) return [];
    return data.records.filter((r) => {
      const matchExp = selectedExperiment === 'all' || r.experiment === selectedExperiment;
      const matchModel = selectedModel === 'all' || r.model.toLowerCase() === selectedModel.toLowerCase();
      const matchCond = selectedCondition === 'all' || r.condition.toUpperCase() === selectedCondition.toUpperCase();
      return matchExp && matchModel && matchCond;
    });
  }, [data, selectedExperiment, selectedModel, selectedCondition]);

  // Grouped data for the 2x2 Factorial Bar Chart across models under current experiment
  const factorialChartData = useMemo(() => {
    if (!data?.records) return [];
    const expRecords = data.records.filter((r) => r.experiment === selectedExperiment);

    const models = ['Early Fusion', 'Energy-Only', 'Late Fusion', 'Hybrid Fusion'];
    return models.map((m) => {
      const nn = expRecords.find((r) => r.model === m && r.condition === 'NN')?.test_mae_watts || 0;
      const na = expRecords.find((r) => r.model === m && r.condition === 'NA')?.test_mae_watts || 0;
      const an = expRecords.find((r) => r.model === m && r.condition === 'AN')?.test_mae_watts || 0;
      const aa = expRecords.find((r) => r.model === m && r.condition === 'AA')?.test_mae_watts || 0;
      return {
        name: m,
        'N-N (Clean)': nn,
        'N-A (Degraded Weather)': na,
        'A-N (Degraded Power)': an,
        'A-A (Joint Degraded)': aa,
      };
    });
  }, [data, selectedExperiment]);

  // Relative degradation comparison (% change vs Clean NN)
  const degradationChartData = useMemo(() => {
    if (!data?.records) return [];
    const expRecords = data.records.filter(
      (r) => r.experiment === selectedExperiment && r.condition !== 'NN'
    );
    const models = ['Early Fusion', 'Energy-Only', 'Late Fusion', 'Hybrid Fusion'];
    return models.map((m) => {
      const na = expRecords.find((r) => r.model === m && r.condition === 'NA')?.relative_mae_change_pct || 0;
      const an = expRecords.find((r) => r.model === m && r.condition === 'AN')?.relative_mae_change_pct || 0;
      const aa = expRecords.find((r) => r.model === m && r.condition === 'AA')?.relative_mae_change_pct || 0;
      return {
        name: m,
        'N-A (% Δ)': na,
        'A-N (% Δ)': an,
        'A-A (% Δ)': aa,
      };
    });
  }, [data, selectedExperiment]);

  if (loading) {
    return <LoadingState message="Loading 2x2 factorial reliability evaluations from API..." />;
  }

  if (error || !data) {
    return (
      <ErrorState
        title="Unable to load reliability matrix"
        message={error || 'No robustness data returned by the backend.'}
        onRetry={loadData}
      />
    );
  }

  // Summary figures for KPI cards under currently selected experiment
  const expRecords = data.records.filter((r) => r.experiment === selectedExperiment);
  const maxDegradationRecord = [...expRecords].sort(
    (a, b) => b.relative_mae_change_pct - a.relative_mae_change_pct
  )[0];
  const minDegradationRecord = [...expRecords]
    .filter((r) => r.condition === 'AA')
    .sort((a, b) => a.relative_mae_change_pct - b.relative_mae_change_pct)[0];

  return (
    <div className="space-y-8">
      {/* Title & Research Mandate */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-rose-600 mb-1">
          <Database className="w-3.5 h-3.5" />
          Sensor Degradation Lab • Source: {data.source_artifact}
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Operational Reliability & Sensor Degradation Analysis
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-4xl leading-relaxed">
          Evaluating model fault tolerance and error resilience across the <strong className="text-slate-800">2×2 Factorial Modality Matrix</strong>.
          We systematically contrast <strong className="text-slate-800">30% MCAR Missingness</strong> (forward-fill imputation) against{' '}
          <strong className="text-slate-800">20% Additive Gaussian Noise</strong> across electricity and exogenous weather channels.
        </p>
      </div>

      {/* Critical Scientific Disclaimer Notice */}
      <div className="bg-amber-50/70 border border-amber-200/80 rounded-lg p-4 flex items-start gap-3 text-amber-900 text-xs leading-relaxed">
        <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-amber-950">Scientific Boundary & Interpretation Caution:</span>{' '}
          Degraded inputs represent controlled synthetic perturbations designed to measure architectural stability and sensitivity boundaries.
          They do not simulate genuine household electrical short-circuits or physical appliance anomalies. Missingness and noise mechanisms
          evaluate fundamentally distinct failure modes and are not interchangeable.
        </div>
      </div>

      {/* KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Active Experiment"
          value={selectedExperiment === 'missingness_30' ? '30% Missingness' : '20% Noise'}
          subtitle={
            selectedExperiment === 'missingness_30'
              ? 'MCAR Forward-Fill Imputation'
              : 'Zero-Mean Gaussian Jitter'
          }
          badge={{ text: 'Synthetic Stress', variant: 'warning' }}
          icon={<ShieldAlert className="w-4 h-4 text-rose-600" />}
        />
        <MetricCard
          title="Max Observed Degradation"
          value={maxDegradationRecord ? `+${maxDegradationRecord.relative_mae_change_pct.toFixed(2)}%` : 'N/A'}
          unit=""
          subtitle={
            maxDegradationRecord
              ? `${maxDegradationRecord.model} (${maxDegradationRecord.condition})`
              : 'Error Surge'
          }
          badge={{ text: 'Peak Sensitivity', variant: 'warning' }}
          icon={<TrendingUp className="w-4 h-4 text-rose-600" />}
        />
        <MetricCard
          title="Most Resilient Joint Model"
          value={minDegradationRecord ? minDegradationRecord.model : 'N/A'}
          unit=""
          subtitle={
            minDegradationRecord
              ? `Only +${minDegradationRecord.relative_mae_change_pct.toFixed(2)}% under AA`
              : 'Lowest AA relative delta'
          }
          badge={{ text: 'Dual-Failure Robust', variant: 'success' }}
          icon={<Zap className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="Evaluated Factorial Cells"
          value={data.total_evaluations}
          unit="Configurations"
          subtitle="4 Models × 4 Conditions × 2 Experiments"
          badge={{ text: 'Exhaustive 2x2', variant: 'info' }}
          icon={<Layers className="w-4 h-4 text-indigo-600" />}
        />
      </div>

      {/* Filter Toolbar */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
          <SlidersHorizontal className="w-3.5 h-3.5 text-blue-600" />
          Matrix Filtering & Experiment Selector
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Experiment Selector */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">
              Degradation Mechanism
            </label>
            <div className="flex rounded-md shadow-sm">
              <button
                type="button"
                onClick={() => setSelectedExperiment('missingness_30')}
                className={`flex-1 px-3 py-2 text-xs font-medium rounded-l-md border transition-colors ${
                  selectedExperiment === 'missingness_30'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                }`}
              >
                30% MCAR Missingness
              </button>
              <button
                type="button"
                onClick={() => setSelectedExperiment('noise_20')}
                className={`flex-1 px-3 py-2 text-xs font-medium rounded-r-md border-t border-b border-r transition-colors ${
                  selectedExperiment === 'noise_20'
                    ? 'bg-blue-600 text-white border-blue-600'
                    : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
                }`}
              >
                20% Gaussian Noise
              </button>
            </div>
          </div>

          {/* Model Filter */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">
              Model Filter
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="all">All Models (4 Architectures)</option>
              {data.available_models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Condition Filter */}
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">
              Factorial Condition Filter
            </label>
            <select
              value={selectedCondition}
              onChange={(e) => setSelectedCondition(e.target.value)}
              className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="all">All Conditions (NN, NA, AN, AA)</option>
              <option value="NN">N-N: Clean Electricity, Clean Weather</option>
              <option value="NA">N-A: Clean Electricity, Degraded Weather</option>
              <option value="AN">A-N: Degraded Electricity, Clean Weather</option>
              <option value="AA">A-A: Joint Degraded (Both Channels)</option>
            </select>
          </div>
        </div>
      </div>

      {/* 2x2 Factorial Matrix Legend & Explanations */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-xs font-bold text-slate-800 bg-slate-100 px-2 py-0.5 rounded">
              Cell N–N
            </span>
            <span className="text-[10px] uppercase font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded">
              Nominal Baseline
            </span>
          </div>
          <h3 className="text-xs font-semibold text-slate-800">Clean Electricity • Clean Weather</h3>
          <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
            Reference operating state with zero sensor degradation. Used as denominator for relative degradation percentage.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-xs font-bold text-slate-800 bg-slate-100 px-2 py-0.5 rounded">
              Cell N–A
            </span>
            <span className="text-[10px] uppercase font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
              Weather Isolated
            </span>
          </div>
          <h3 className="text-xs font-semibold text-slate-800">Clean Electricity • Degraded Weather</h3>
          <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
            Isolates weather telemetry breakdown. Models demonstrate negligible error degradation (&lt;1%), revealing load anchoring.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-xs font-bold text-slate-800 bg-slate-100 px-2 py-0.5 rounded">
              Cell A–N
            </span>
            <span className="text-[10px] uppercase font-semibold text-amber-600 bg-amber-50 px-2 py-0.5 rounded">
              Load Isolated
            </span>
          </div>
          <h3 className="text-xs font-semibold text-slate-800">Degraded Electricity • Clean Weather</h3>
          <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
            Isolates smart meter corruption. Causes major error increases (+4.9% to +11.7%), demonstrating primary modality sensitivity.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="font-mono text-xs font-bold text-slate-800 bg-slate-100 px-2 py-0.5 rounded">
              Cell A–A
            </span>
            <span className="text-[10px] uppercase font-semibold text-rose-600 bg-rose-50 px-2 py-0.5 rounded">
              Joint Failure
            </span>
          </div>
          <h3 className="text-xs font-semibold text-slate-800">Degraded Electricity • Degraded Weather</h3>
          <p className="text-[11px] text-slate-500 mt-1 leading-relaxed">
            Simultaneous failure across both load and exogenous weather channels. Represents worst-case operational stress.
          </p>
        </div>
      </div>

      {/* Visualizations Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Factorial Grouped MAE Chart */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm">
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-slate-900">
              Factorial Test MAE Across Conditions (Watts)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Experiment: {selectedExperiment === 'missingness_30' ? '30% MCAR Missingness' : '20% Gaussian Noise'} (lower is better)
            </p>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={factorialChartData}
                margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit=" W" domain={[60, 95]} />
                <Tooltip
                  formatter={(val: number) => [`${val.toFixed(2)} W`, '']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.375rem',
                    fontSize: '0.75rem',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                <Bar dataKey="N-N (Clean)" fill="#10b981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="N-A (Degraded Weather)" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="A-N (Degraded Power)" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="A-A (Joint Degraded)" fill="#ef4444" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Relative Degradation Percentage Chart */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm">
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-slate-900">
              Relative Degradation vs Clean Baseline (% Δ MAE)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Percentage error increase relative to nominal N-N clean state
            </p>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={degradationChartData}
                margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit="%" domain={[0, 14]} />
                <Tooltip
                  formatter={(val: number) => [`+${val.toFixed(2)}%`, '']}
                  contentStyle={{
                    backgroundColor: '#ffffff',
                    borderColor: '#e2e8f0',
                    borderRadius: '0.375rem',
                    fontSize: '0.75rem',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                <Bar dataKey="N-A (% Δ)" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                <Bar dataKey="A-N (% Δ)" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="A-A (% Δ)" fill="#ef4444" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Granular Master Robustness Table */}
      <div className="bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Factorial Robustness Evaluations Table
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Filtered records: {filteredRecords.length} of {data.total_evaluations} total evaluations
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2.5 py-1 rounded w-fit">
            Artifact: {data.source_artifact}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-200">
              <tr>
                <th className="px-6 py-3.5">Experiment</th>
                <th className="px-6 py-3.5">Model</th>
                <th className="px-6 py-3.5">Cell</th>
                <th className="px-6 py-3.5">Condition Status</th>
                <th className="px-6 py-3.5 text-right font-bold text-slate-800">Test MAE</th>
                <th className="px-6 py-3.5 text-right">Δ vs Clean</th>
                <th className="px-6 py-3.5 text-right font-bold">Relative Change</th>
                <th className="px-6 py-3.5 text-right">Test RMSE</th>
                <th className="px-6 py-3.5 text-right">Test R²</th>
                <th className="px-6 py-3.5 text-right">PICP (90% Target)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {filteredRecords.length === 0 ? (
                <tr>
                  <td colSpan={10} className="px-6 py-8 text-center text-slate-400 text-xs">
                    No evaluations match the active filter criteria.
                  </td>
                </tr>
              ) : (
                filteredRecords.map((r: RobustnessRecord, idx: number) => {
                  const isClean = r.condition === 'NN';
                  const isSevere = r.relative_mae_change_pct > 8;
                  const isModerate = r.relative_mae_change_pct > 2 && !isSevere;

                  return (
                    <tr
                      key={`${r.experiment}-${r.model}-${r.condition}-${idx}`}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        isClean ? 'bg-emerald-50/20' : ''
                      }`}
                    >
                      <td className="px-6 py-3.5 font-mono text-xs text-slate-600">
                        {r.experiment_label}
                      </td>
                      <td className="px-6 py-3.5 font-medium text-slate-900">
                        {r.model}
                      </td>
                      <td className="px-6 py-3.5">
                        <span
                          className={`font-mono text-xs font-semibold px-2 py-0.5 rounded ${
                            r.condition === 'NN'
                              ? 'bg-emerald-100 text-emerald-800'
                              : r.condition === 'NA'
                              ? 'bg-blue-100 text-blue-800'
                              : r.condition === 'AN'
                              ? 'bg-amber-100 text-amber-800'
                              : 'bg-rose-100 text-rose-800'
                          }`}
                        >
                          {r.condition}
                        </span>
                      </td>
                      <td className="px-6 py-3.5 text-xs text-slate-500">
                        {r.condition_description}
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono font-bold text-slate-900">
                        {r.test_mae_watts.toFixed(2)} W
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-slate-600">
                        {r.delta_mae_vs_clean_watts > 0 ? '+' : ''}
                        {r.delta_mae_vs_clean_watts.toFixed(2)} W
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono font-medium">
                        <span
                          className={
                            isClean
                              ? 'text-slate-400'
                              : isSevere
                              ? 'text-rose-700 font-bold'
                              : isModerate
                              ? 'text-amber-700'
                              : 'text-blue-700'
                          }
                        >
                          {r.relative_mae_change_pct > 0 ? '+' : ''}
                          {r.relative_mae_change_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-slate-600">
                        {r.test_rmse_watts.toFixed(2)} W
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-slate-600">
                        {r.test_r2.toFixed(4)}
                      </td>
                      <td className="px-6 py-3.5 text-right font-mono text-xs text-slate-600">
                        {r.conformal_picp_pct.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
