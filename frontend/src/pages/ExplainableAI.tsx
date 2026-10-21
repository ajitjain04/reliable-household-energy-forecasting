import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import {
  XAISummaryResponse,
  XAISummaryRecord,
  TemporalLagRecord,
  WeatherFeatureRecord,
} from '../api/types';
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
  SearchCode,
  Database,
  Layers,
  Clock,
  CloudSun,
  AlertCircle,
  HelpCircle,
  Cpu,
} from 'lucide-react';

export const ExplainableAI: React.FC = () => {
  const [summaryData, setSummaryData] = useState<XAISummaryResponse | null>(null);
  const [temporalData, setTemporalData] = useState<TemporalLagRecord[]>([]);
  const [weatherData, setWeatherData] = useState<WeatherFeatureRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters for granular sections
  const [selectedModel, setSelectedModel] = useState<string>('Early Fusion');
  const [selectedCondition, setSelectedCondition] = useState<string>('NN');
  const [selectedExperiment, setSelectedExperiment] = useState<string>('missingness');

  const loadAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [summaryRes, temporalRes, weatherRes] = await Promise.all([
        api.getXAISummary(),
        api.getXAITemporal(),
        api.getXAIWeatherFeatures(),
      ]);
      setSummaryData(summaryRes);
      setTemporalData(temporalRes);
      setWeatherData(weatherRes);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to fetch Explainable AI attribution artifacts from API.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllData();
  }, []);

  // Modality Share Chart Data (Total vs Per-Channel Density)
  const modalityChartData = useMemo(() => {
    if (!summaryData?.records) return [];
    return summaryData.records.map((r: XAISummaryRecord) => ({
      name: r.model,
      'Electricity Total (%)': r.clean_electricity_share_pct,
      'Weather Total (%)': r.clean_weather_share_pct,
      'Electricity Density (%)': r.clean_electricity_density_share_pct,
      'Weather Density (%)': r.clean_weather_density_share_pct,
    }));
  }, [summaryData]);

  // Filtered Temporal Data for selected model, experiment, and condition
  const filteredTemporalData = useMemo(() => {
    return temporalData.filter(
      (r) =>
        r.model.toLowerCase() === selectedModel.toLowerCase() &&
        r.experiment.toLowerCase() === selectedExperiment.toLowerCase() &&
        r.condition.toUpperCase() === selectedCondition.toUpperCase()
    );
  }, [temporalData, selectedModel, selectedExperiment, selectedCondition]);

  // Filtered Weather Features for selected model, experiment, and condition
  const filteredWeatherData = useMemo(() => {
    return weatherData
      .filter(
        (r) =>
          r.model.toLowerCase() === selectedModel.toLowerCase() &&
          r.experiment.toLowerCase() === selectedExperiment.toLowerCase() &&
          r.condition.toUpperCase() === selectedCondition.toUpperCase()
      )
      .sort((a, b) => a.feature_rank - b.feature_rank);
  }, [weatherData, selectedModel, selectedExperiment, selectedCondition]);

  if (loading) {
    return <LoadingState message="Loading Integrated Gradients modality attribution artifacts..." />;
  }

  if (error || !summaryData) {
    return (
      <ErrorState
        title="Unable to load Explainable AI artifacts"
        message={error || 'No XAI data returned by backend.'}
        onRetry={loadAllData}
      />
    );
  }

  const earlyRecord = summaryData.records.find((r) => r.model === 'Early Fusion') || summaryData.records[0];

  return (
    <div className="space-y-8">
      {/* Title & Research Mandate */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-purple-600 mb-1">
          <Database className="w-3.5 h-3.5" />
          Explainable AI • Method: {summaryData.method} • Source: {summaryData.source_artifact}
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Integrated Gradients Modality & Temporal Attribution
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-4xl leading-relaxed">
          Uncovering architectural feature sensitivity using path-integrated gradients ($m=20$ steps, zero-mean baseline).
          We quantify the macro balance between electricity load history and exogenous weather, single-channel density concentration,
          temporal memory recency, and specific weather feature hierarchies across nominal clean and degraded operating states.
        </p>
      </div>

      {/* KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Top Architecture Balance"
          value={`${earlyRecord.clean_electricity_share_pct.toFixed(1)}% / ${earlyRecord.clean_weather_share_pct.toFixed(1)}%`}
          unit=""
          subtitle="Early Fusion Load vs Weather"
          badge={{ text: 'Multimodal Co-Attendance', variant: 'success' }}
          icon={<SearchCode className="w-4 h-4 text-purple-600" />}
        />
        <MetricCard
          title="Single-Channel Density"
          value={`${earlyRecord.clean_electricity_density_share_pct.toFixed(1)}%`}
          unit=""
          subtitle="Per-channel active power concentration"
          badge={{ text: '3.2× Weather Density', variant: 'info' }}
          icon={<Cpu className="w-4 h-4 text-blue-600" />}
        />
        <MetricCard
          title="Recent 24h Memory Share"
          value={`${earlyRecord.electricity_temporal_recent_24h_pct.toFixed(1)}%`}
          unit=""
          subtitle="Load attribution in hours t-1 to t-24"
          badge={{ text: 'Recency Anchored', variant: 'info' }}
          icon={<Clock className="w-4 h-4 text-indigo-600" />}
        />
        <MetricCard
          title="Lead Weather Signal"
          value={earlyRecord.top_weather_features_clean.split(',')[0] || 'Temperature'}
          unit=""
          subtitle="Primary exogenous meteorological predictor"
          badge={{ text: 'Consensus Rank #1', variant: 'neutral' }}
          icon={<CloudSun className="w-4 h-4 text-amber-600" />}
        />
      </div>

      {/* Modality Attribution Overview Section */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-6">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Modality Attribution Shares & Channel Density Concentration
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Comparing aggregate modality attribution shares against normalized per-channel density shares under clean nominal conditions
          </p>
        </div>

        {/* Informational Callout on Density vs Total Share */}
        <div className="bg-purple-50/70 border border-purple-200/80 rounded-lg p-4 text-xs text-purple-900 leading-relaxed flex items-start gap-3">
          <HelpCircle className="w-4 h-4 text-purple-600 flex-shrink-0 mt-0.5" />
          <div>
            <strong className="text-purple-950">Crucial Distinction: Total Modality Share vs Single-Channel Density Share:</strong>
            <p className="mt-1">
              While the weather modality collectively accounts for 66.9% of total attribution in Early Fusion, this signal is distributed across{' '}
              <strong>7 distinct channels</strong>. On a normalized per-channel density basis, the single electricity load channel accounts for{' '}
              <strong>76.2% of attribution density</strong> (3.2× higher than any single weather channel). In Late and Hybrid Fusion, electricity
              density exceeds 84%–88%, demonstrating heavy load-channel anchoring.
            </p>
          </div>
        </div>

        {/* Visual Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Total Modality Share Chart */}
          <div className="border border-slate-100 rounded-lg p-4 bg-slate-50/50">
            <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Aggregate Modality Share (%)
            </h3>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modalityChartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit="%" domain={[0, 100]} />
                  <Tooltip
                    formatter={(val: number) => [`${val.toFixed(1)}%`, '']}
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', fontSize: '0.75rem' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                  <Bar dataKey="Electricity Total (%)" fill="#3b82f6" stackId="total" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="Weather Total (%)" fill="#f59e0b" stackId="total" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Per-Channel Density Share Chart */}
          <div className="border border-slate-100 rounded-lg p-4 bg-slate-50/50">
            <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
              Normalized Per-Channel Density Share (%)
            </h3>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modalityChartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit="%" domain={[0, 100]} />
                  <Tooltip
                    formatter={(val: number) => [`${val.toFixed(1)}%`, '']}
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', fontSize: '0.75rem' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
                  <Bar dataKey="Electricity Density (%)" fill="#2563eb" stackId="density" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="Weather Density (%)" fill="#fbbf24" stackId="density" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Dynamic Attribution Shifts Under Sensor Degradation */}
        <div className="border-t border-slate-200 pt-6">
          <h3 className="text-sm font-semibold text-slate-900 mb-2">
            Gradient Reallocation Shifts Under Sensor Masking & Noise (Percentage Points)
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Measures how electricity attribution reallocates when weather is masked (miss NA), electricity is masked (miss AN), or both channels fail (AA).
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-600">
              <thead className="bg-slate-50 uppercase font-semibold text-slate-500 border-b border-slate-200">
                <tr>
                  <th className="px-4 py-3">Model Architecture</th>
                  <th className="px-4 py-3 text-right">Weather Masked (miss NA)</th>
                  <th className="px-4 py-3 text-right">Electricity Masked (miss AN)</th>
                  <th className="px-4 py-3 text-right">Dual Missingness (miss AA)</th>
                  <th className="px-4 py-3 text-right">Dual Noise (noise AA)</th>
                  <th className="px-4 py-3">Top Clean Weather Predictors</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {summaryData.records.map((r: XAISummaryRecord) => (
                  <tr key={r.model} className="hover:bg-slate-50/80 transition-colors">
                    <td className="px-4 py-3 font-medium text-slate-900">{r.model}</td>
                    <td className="px-4 py-3 text-right font-mono text-slate-700">
                      {r.miss_NA_elec_shift_pct_pts > 0 ? '+' : ''}
                      {r.miss_NA_elec_shift_pct_pts.toFixed(1)} pp
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-slate-700">
                      {r.miss_AN_elec_shift_pct_pts > 0 ? '+' : ''}
                      {r.miss_AN_elec_shift_pct_pts.toFixed(1)} pp
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-slate-700">
                      {r.miss_AA_elec_shift_pct_pts > 0 ? '+' : ''}
                      {r.miss_AA_elec_shift_pct_pts.toFixed(1)} pp
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-slate-700">
                      {r.noise_AA_elec_shift_pct_pts > 0 ? '+' : ''}
                      {r.noise_AA_elec_shift_pct_pts.toFixed(1)} pp
                    </td>
                    <td className="px-4 py-3 font-mono text-[11px] text-slate-600">
                      {r.top_weather_features_clean}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Filter Toolbar for Granular Temporal and Weather Sections */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
          <Layers className="w-3.5 h-3.5 text-blue-600" />
          Interactive Explorer Controls (Temporal Memory & Weather Features)
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="Early Fusion">Early Fusion (Joint-Input Multimodal)</option>
              <option value="Late Fusion">Late Fusion (Dual-Encoder)</option>
              <option value="Hybrid Fusion">Hybrid Fusion (Cross-Modal Projection)</option>
              <option value="Energy-Only">Energy-Only (Unimodal Baseline)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">Experiment</label>
            <select
              value={selectedExperiment}
              onChange={(e) => setSelectedExperiment(e.target.value)}
              className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="missingness">30% MCAR Missingness</option>
              <option value="noise">20% Additive Gaussian Noise</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-700 mb-1.5">Condition Code</label>
            <select
              value={selectedCondition}
              onChange={(e) => setSelectedCondition(e.target.value)}
              className="w-full text-xs bg-white border border-slate-300 rounded-md py-2 px-3 text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="NN">N-N: Clean Nominal Baseline</option>
              <option value="NA">N-A: Weather Degraded</option>
              <option value="AN">A-N: Electricity Degraded</option>
              <option value="AA">A-A: Joint Dual Failure</option>
            </select>
          </div>
        </div>
      </div>

      {/* Two-Column Granular Sections: Temporal Memory and Weather Channels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Temporal Lag Breakdown */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold text-slate-800 uppercase tracking-wider">
              <Clock className="w-4 h-4 text-blue-600" />
              Electricity Temporal Memory Lag Partition
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Active Model: <strong className="text-slate-800">{selectedModel}</strong> ({selectedExperiment}, condition {selectedCondition})
            </p>
          </div>

          {filteredTemporalData.length === 0 ? (
            <div className="bg-slate-50 p-8 rounded-lg text-center text-xs text-slate-500">
              No temporal lag records available for this filter combination.
            </div>
          ) : (
            <>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={filteredTemporalData}
                    margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                    <XAxis dataKey="bin_name" stroke="#64748b" fontSize={11} tickLine={false} />
                    <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit="%" domain={[0, 100]} />
                    <Tooltip
                      formatter={(val: number) => [`${val.toFixed(2)}%`, 'Temporal Share']}
                      contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', fontSize: '0.75rem' }}
                    />
                    <Bar dataKey="temporal_share_pct" name="Share within Electricity (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="space-y-2 pt-2 border-t border-slate-100">
                {filteredTemporalData.map((b) => (
                  <div key={b.bin_name} className="flex items-center justify-between text-xs py-1 px-2 rounded bg-slate-50">
                    <span className="font-medium text-slate-800">{b.bin_name} ({b.n_hours}h)</span>
                    <div className="flex items-center gap-3">
                      <span className="text-slate-500 font-mono">Mean Mag: {b.attribution_magnitude_mean.toFixed(2)}</span>
                      <span className="font-mono font-bold text-blue-700">{b.temporal_share_pct.toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Weather Feature Attribution Hierarchy */}
        <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-4">
          <div>
            <div className="flex items-center gap-2 text-xs font-bold text-slate-800 uppercase tracking-wider">
              <CloudSun className="w-4 h-4 text-amber-600" />
              Exogenous Weather Channel Attribution Hierarchy
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Active Model: <strong className="text-slate-800">{selectedModel}</strong> ({selectedExperiment}, condition {selectedCondition})
            </p>
          </div>

          {selectedModel === 'Energy-Only' ? (
            <div className="bg-slate-50 p-8 rounded-lg text-center text-xs text-slate-500">
              Energy-Only baseline does not ingest exogenous weather channels (unimodal architecture).
            </div>
          ) : filteredWeatherData.length === 0 ? (
            <div className="bg-slate-50 p-8 rounded-lg text-center text-xs text-slate-500">
              No weather feature records match this filter combination.
            </div>
          ) : (
            <>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={filteredWeatherData}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                    <XAxis type="number" stroke="#64748b" fontSize={11} tickLine={false} unit="%" domain={[0, 35]} />
                    <YAxis dataKey="feature_name" type="category" stroke="#64748b" fontSize={11} tickLine={false} />
                    <Tooltip
                      formatter={(val: number) => [`${val.toFixed(2)}%`, 'Share in Weather']}
                      contentStyle={{ backgroundColor: '#ffffff', borderColor: '#e2e8f0', fontSize: '0.75rem' }}
                    />
                    <Bar dataKey="attribution_share_within_weather_pct" name="Share in Weather (%)" fill="#f59e0b" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="overflow-x-auto pt-2 border-t border-slate-100">
                <table className="w-full text-left text-xs text-slate-600">
                  <thead className="bg-slate-50 text-[11px] uppercase font-semibold text-slate-500">
                    <tr>
                      <th className="px-3 py-1.5">Rank</th>
                      <th className="px-3 py-1.5">Feature</th>
                      <th className="px-3 py-1.5 text-right">Mean Mag</th>
                      <th className="px-3 py-1.5 text-right font-bold">Share (%)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {filteredWeatherData.map((w: WeatherFeatureRecord) => (
                      <tr key={w.feature_name} className="hover:bg-slate-50/80">
                        <td className="px-3 py-1.5 font-mono font-medium text-slate-500">#{w.feature_rank}</td>
                        <td className="px-3 py-1.5 font-medium text-slate-800">{w.feature_name}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-slate-600">{w.attribution_magnitude_mean.toFixed(2)}</td>
                        <td className="px-3 py-1.5 text-right font-mono font-bold text-amber-700">
                          {w.attribution_share_within_weather_pct.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Scientific Context & Epistemic Limitations */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 space-y-2 text-xs text-slate-600 leading-relaxed">
        <div className="flex items-center gap-2 font-semibold text-slate-800">
          <AlertCircle className="w-4 h-4 text-slate-500" />
          Interpretability Framework & Epistemic Boundaries
        </div>
        <p>
          Integrated Gradients estimates input feature sensitivity by integrating gradients along the straight line from a baseline
          (zero-mean input) to the actual input across $m=20$ Riemann interpolation steps. These values reflect the model's internal gradient
          allocations under the specific trained weights. Attribution shares demonstrate <em>associational model sensitivity</em> rather than
          physical household causality. The strong density concentration on the historical active power channel confirms that all models
          principally rely on recent autoregressive temporal load signatures, utilizing weather primarily as a secondary thermal modulation.
        </p>
      </div>
    </div>
  );
};
