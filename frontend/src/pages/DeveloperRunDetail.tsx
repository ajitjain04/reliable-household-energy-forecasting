import React, { useEffect, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Terminal,
  ArrowLeft,
  RefreshCw,
  ExternalLink,
  Copy,
  Check,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Cpu,
  Layers,
  FileSpreadsheet,
  Code,
  ShieldCheck,
  ShieldAlert,
  Info,
  BarChart2,
  FileJson,
  ChevronDown,
  ChevronUp,
  Download,
  Loader2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { api } from '../api/client';
import { ForecastRecordDetail, ConformalIntervalStep } from '../api/types';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';
import { MetricCard } from '../components/common/MetricCard';

export const DeveloperRunDetail: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();

  const [record, setRecord] = useState<ForecastRecordDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [showJson, setShowJson] = useState<boolean>(false);
  const [downloadingPdf, setDownloadingPdf] = useState<boolean>(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const fetchRecord = async () => {
    if (!runId) {
      setError('No forecast run identifier provided in route.');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await api.getForecastRecord(runId);
      setRecord(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve diagnostic record.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecord();
  }, [runId]);

  const handleCopyId = () => {
    if (record?.run_id) {
      navigator.clipboard.writeText(record.run_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Format chart data from intervals
  const chartData = useMemo(() => {
    if (!record?.intervals || record.intervals.length === 0) {
      return [];
    }

    return record.intervals.map((step: ConformalIntervalStep) => {
      const lower = Math.max(0, Math.round(step.lower_bound_clipped_watts * 10) / 10);
      const upper = Math.round(step.upper_bound_watts * 10) / 10;
      const pred = Math.round(step.prediction_watts * 10) / 10;
      const bandRange = Math.max(0, Math.round((upper - lower) * 10) / 10);

      let timeLabel = `+${step.horizon_hour}h`;
      let fullTime = `Hour +${step.horizon_hour}`;
      if (step.target_timestamp) {
        try {
          const d = new Date(step.target_timestamp);
          timeLabel = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          fullTime = `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${timeLabel}`;
        } catch {
          // fallback
        }
      }

      return {
        step: `+${step.horizon_hour}h`,
        timeLabel,
        fullTime,
        horizon_hour: step.horizon_hour,
        target_timestamp: step.target_timestamp,
        prediction: pred,
        lower: lower,
        upper: upper,
        LowerBound: lower,
        BandRange: bandRange,
        width: Math.round(step.interval_width_watts * 10) / 10,
        rawLower: Math.round(step.lower_bound_watts * 10) / 10,
      };
    });
  }, [record]);

  // Derived statistics
  const summaryStats = useMemo(() => {
    if (!chartData || chartData.length === 0) return null;

    const predictions = chartData.map((d) => d.prediction);
    const sumWatts = predictions.reduce((acc, curr) => acc + curr, 0);
    const totalKwh = sumWatts / 1000;
    const avgWatts = sumWatts / chartData.length;
    const peakStep = chartData.reduce((prev, curr) => (curr.prediction > prev.prediction ? curr : prev), chartData[0]);
    const avgWidth = chartData.reduce((acc, curr) => acc + curr.width, 0) / chartData.length;

    return {
      totalKwh: totalKwh.toFixed(2),
      avgWatts: Math.round(avgWatts).toLocaleString(),
      peakWatts: Math.round(peakStep.prediction).toLocaleString(),
      peakTime: peakStep.timeLabel,
      avgWidth: Math.round(avgWidth).toLocaleString(),
      halfWidth: Math.round(avgWidth / 2).toLocaleString(),
    };
  }, [chartData]);

  if (loading) {
    return <LoadingState message={`Loading diagnostic record #${runId}...`} />;
  }

  if (error || !record) {
    return (
      <div className="space-y-6 max-w-4xl mx-auto">
        <Link
          to="/developer/forecast-records"
          className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-800"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Run Inspector
        </Link>
        <ErrorState
          title="Run Record Not Found"
          message={error || 'The requested forecast run record was not found in the database.'}
          onRetry={fetchRecord}
        />
      </div>
    );
  }

  const isSuccess = record.status === 'succeeded';
  const isFailed = record.status === 'failed';

  const handleDownloadPdf = async () => {
    if (!record || record.status !== 'succeeded') return;
    try {
      setDownloadingPdf(true);
      setPdfError(null);
      await api.downloadForecastPdf(record.run_id);
    } catch (err: unknown) {
      setPdfError(err instanceof Error ? err.message : 'Failed to download PDF report.');
    } finally {
      setDownloadingPdf(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Developer Diagnostic & RBAC Authorized Banner */}
      <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-200 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2.5">
          <ShieldAlert className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span>
            <strong>Developer Run Diagnostic:</strong> Showing internal persistent record fields, model parameter counts, and sanitized telemetry summaries.
          </span>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/30">
          RBAC Authorized
        </span>
      </div>

      {/* Top Breadcrumb & Action Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <Link
            to="/developer/forecast-records"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 bg-white border border-slate-300 px-3 py-1.5 rounded-lg shadow-xs transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Run Inspector
          </Link>
          <span className="text-slate-300">/</span>
          <span className="text-xs font-mono text-slate-500 truncate max-w-xs sm:max-w-md">
            {record.run_id}
          </span>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={handleCopyId}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-xs transition-colors"
            title="Copy UUID to clipboard"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5 text-slate-500" />}
            {copied ? 'Copied' : 'Copy ID'}
          </button>
          <button
            type="button"
            onClick={fetchRecord}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-xs transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
            Refresh
          </button>
          {isSuccess && (
            <>
              <button
                type="button"
                onClick={handleDownloadPdf}
                disabled={downloadingPdf}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 shadow-xs transition-colors disabled:opacity-50"
                title="Download genuine vector PDF report"
              >
                {downloadingPdf ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
                Download PDF
              </button>
              <Link
                to={`/forecast/${record.run_id}`}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-sm transition-colors"
                title="Open Household View"
              >
                <ExternalLink className="w-3.5 h-3.5" />
                Open Household View
              </Link>
            </>
          )}
        </div>
      </div>

      {/* PDF Download Error Banner */}
      {pdfError && (
        <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{pdfError}</span>
          </div>
          <button
            type="button"
            onClick={() => setPdfError(null)}
            className="text-[11px] font-semibold underline text-rose-700 hover:text-rose-900"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Diagnostic Header Card */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <div className="flex flex-wrap items-center gap-2.5 mb-2">
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                  isSuccess
                    ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                    : isFailed
                    ? 'bg-rose-50 text-rose-700 border border-rose-200'
                    : 'bg-amber-50 text-amber-700 border border-amber-200'
                }`}
              >
                {isSuccess ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                ) : isFailed ? (
                  <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
                ) : (
                  <Clock className="w-3.5 h-3.5 text-amber-600" />
                )}
                Status: {record.status.toUpperCase()}
              </span>

              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-slate-100 text-slate-700 border border-slate-200">
                {record.input_method === 'csv' ? (
                  <FileSpreadsheet className="w-3 h-3 text-emerald-600" />
                ) : (
                  <Code className="w-3 h-3 text-blue-600" />
                )}
                Ingestion: {record.input_method.toUpperCase()}
              </span>

              <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700 border border-blue-200">
                Architecture: {record.selected_model || 'None'}
              </span>

              {record.user_id && (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-purple-50 text-purple-700 border border-purple-200">
                  User ID: {record.user_id}
                </span>
              )}
            </div>

            <h1 className="text-xl font-bold text-slate-900 tracking-tight font-mono">
              {record.run_id}
            </h1>
            <p className="text-xs text-slate-500 mt-1">
              Durable audit record recorded in SQLite database at {new Date(record.created_at).toISOString()}
            </p>
          </div>

          <div className="flex flex-col text-xs text-slate-500 space-y-1.5 font-mono">
            <div>
              <span className="text-slate-400 font-sans">Created At:</span>{' '}
              <strong className="text-slate-700">{new Date(record.created_at).toLocaleString()}</strong>
            </div>
            {record.forecast_origin_timestamp && (
              <div>
                <span className="text-slate-400 font-sans">Forecast Origin:</span>{' '}
                <strong className="text-slate-700">{record.forecast_origin_timestamp}</strong>
              </div>
            )}
            <div>
              <span className="text-slate-400 font-sans">Mean Uncertainty:</span>{' '}
              <strong className="text-slate-700">
                {record.mean_interval_width_watts
                  ? `±${Math.round(record.mean_interval_width_watts / 2)} W`
                  : 'N/A'}
              </strong>
            </div>
          </div>
        </div>

        {/* Failure Details Card if Failed */}
        {isFailed && (
          <div className="p-5 rounded-xl bg-rose-50 border border-rose-200 space-y-3">
            <div className="flex items-center gap-2 text-xs font-bold text-rose-900 uppercase tracking-wider">
              <AlertTriangle className="w-4 h-4 text-rose-600" />
              Inference Failure Diagnostics
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-3 bg-white rounded-lg border border-rose-200 space-y-1">
                <span className="text-slate-500 text-[11px] font-sans">Sanitized Error Code:</span>
                <div className="font-bold text-rose-800 text-sm">{record.error_code || 'UNKNOWN_ERROR'}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-rose-200 space-y-1">
                <span className="text-slate-500 text-[11px] font-sans">Sanitized Error Message:</span>
                <div className="text-rose-900 whitespace-pre-wrap">{record.error_message || 'No detailed error message logged.'}</div>
              </div>
            </div>
            <p className="text-xs text-rose-700 leading-relaxed font-sans">
              Failure records store only non-sensitive failure categories and stack traces to ensure user privacy and maintain durability in the audit log.
            </p>
          </div>
        )}
      </div>

      {/* Succeeded Run Diagnostics */}
      {isSuccess && (
        <>
          {/* Quick Metrics */}
          {summaryStats && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="Projected 24h Energy"
                value={summaryStats.totalKwh}
                unit="kWh"
                subtitle="Calculated active sum / 1000"
                badge={{ text: '24h Total', variant: 'info' }}
                icon={<Cpu className="w-4 h-4 text-blue-600" />}
              />
              <MetricCard
                title="Mean Active Demand"
                value={summaryStats.avgWatts}
                unit="Watts"
                subtitle="Unscaled physical prediction"
                badge={{ text: 'Mean Power', variant: 'neutral' }}
                icon={<Clock className="w-4 h-4 text-slate-500" />}
              />
              <MetricCard
                title="Peak Horizon Power"
                value={summaryStats.peakWatts}
                unit="Watts"
                subtitle={`Peak step: ${summaryStats.peakTime}`}
                badge={{ text: 'Max Point', variant: 'warning' }}
                icon={<AlertTriangle className="w-4 h-4 text-amber-600" />}
              />
              <MetricCard
                title="Conformal Uncertainty"
                value={`±${summaryStats.halfWidth}`}
                unit="Watts"
                subtitle={`Mean width: ${summaryStats.avgWidth} W`}
                badge={{ text: '90% PICP Target', variant: 'success' }}
                icon={<ShieldCheck className="w-4 h-4 text-emerald-600" />}
              />
            </div>
          )}

          {/* Model Architecture & Selection Metadata Card */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-700">
                <Cpu className="w-4 h-4 text-blue-600" />
                Model & Execution Parameters
              </div>
              <div className="space-y-2 text-xs text-slate-600">
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Selected Architecture:</span>
                  <span className="font-semibold text-slate-900">{record.selected_model}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Internal Model Key:</span>
                  <span className="font-mono text-slate-800">{record.model_key || '—'}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Model PyTorch Class:</span>
                  <span className="font-mono text-slate-800">
                    {record.model_metadata?.model_class || '—'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Trainable Parameters:</span>
                  <span className="font-mono font-semibold text-slate-800">
                    {record.model_metadata?.trainable_parameters?.toLocaleString() || '—'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Context Lookback Window:</span>
                  <span className="font-mono text-slate-800">168 hours (7 days)</span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-slate-500">Forecast Horizon:</span>
                  <span className="font-mono text-slate-800">24 hours multi-step</span>
                </div>
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-700">
                <Layers className="w-4 h-4 text-emerald-600" />
                Ingestion & Selection Rationale
              </div>
              <div className="space-y-2 text-xs text-slate-600">
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Selection Mode:</span>
                  <span className="font-semibold text-slate-900">
                    {record.selection_mode === 'auto' ? 'Automated Weather-Adaptive' : 'Manual Override'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Selection Rationale:</span>
                  <span className="text-slate-800 text-right max-w-xs truncate" title={record.selection_reason || ''}>
                    {record.selection_reason || '—'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Energy Scaler Configuration:</span>
                  <span className="font-mono text-slate-800">
                    {record.model_metadata?.energy_scaler_type || 'StandardScaler'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-100">
                  <span className="text-slate-500">Weather Scaler Status:</span>
                  <span className="font-mono text-slate-800">
                    {record.model_metadata?.weather_scaler_type || 'None (Univariate)'}
                  </span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span className="text-slate-500">Input Data Privacy Guarantee:</span>
                  <span className="text-emerald-700 font-medium">Raw sequence discarded after inference</span>
                </div>
              </div>
            </div>
          </div>

          {/* Input Telemetry & Metadata Audit */}
          {record.input_metadata && (
            <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-700">
                <FileSpreadsheet className="w-4 h-4 text-blue-600" />
                Sanitized Input Telemetry Audit (No Raw Series Stored)
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                {record.input_metadata.source_filename && (
                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="text-slate-500 text-[11px]">Filename</span>
                    <div className="font-mono font-bold text-slate-800 truncate" title={record.input_metadata.source_filename}>
                      {record.input_metadata.source_filename}
                    </div>
                  </div>
                )}
                {record.input_metadata.file_size_bytes !== undefined && (
                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <span className="text-slate-500 text-[11px]">File Size</span>
                    <div className="font-mono font-bold text-slate-800">
                      {(record.input_metadata.file_size_bytes / 1024).toFixed(1)} KB
                    </div>
                  </div>
                )}
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-slate-500 text-[11px]">Observations Validated</span>
                  <div className="font-mono font-bold text-slate-800">168 hourly steps</div>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                  <span className="text-slate-500 text-[11px]">Imputation Policy</span>
                  <div className="font-bold text-emerald-700">Zero imputation (Strict)</div>
                </div>
              </div>
            </div>
          )}

          {/* Interactive Recharts 24-Hour Horizon Visualizer */}
          <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  24-Hour Horizon Multi-Step Power Demand & Conformal Bounds
                </h2>
                <p className="text-xs text-slate-500">
                  Point forecast (Watts) plotted with 90% distribution-free split conformal uncertainty envelope.
                </p>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 bg-blue-600 rounded-full" />
                  <span className="text-slate-600 font-medium">Point Forecast</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 bg-blue-200 border border-blue-400 rounded-xs" />
                  <span className="text-slate-600 font-medium">90% Conformal Range</span>
                </div>
              </div>
            </div>

            <div className="h-80 w-full pt-2">
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                  <XAxis
                    dataKey="step"
                    stroke="#64748b"
                    fontSize={11}
                    tickLine={false}
                    tickFormatter={(val, idx) => {
                      const item = chartData[idx];
                      return item && item.timeLabel ? item.timeLabel : val;
                    }}
                  />
                  <YAxis stroke="#64748b" fontSize={11} tickLine={false} unit=" W" domain={[0, 'auto']} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-md text-xs space-y-1">
                            <div className="font-bold text-slate-900 pb-1 border-b border-slate-100 flex justify-between">
                              <span>{d.step}</span>
                              <span className="font-normal text-slate-500">{d.fullTime}</span>
                            </div>
                            <div className="text-blue-700 font-bold flex justify-between gap-4">
                              <span>Point Forecast:</span>
                              <span className="font-mono">{d.prediction.toLocaleString()} W</span>
                            </div>
                            <div className="text-emerald-700 font-medium flex justify-between gap-4">
                              <span>90% Range:</span>
                              <span className="font-mono">[{d.lower.toLocaleString()} W, {d.upper.toLocaleString()} W]</span>
                            </div>
                            <div className="text-slate-500 text-[11px] flex justify-between gap-4 pt-1 border-t border-slate-100">
                              <span>Interval Width:</span>
                              <span className="font-mono">{d.width.toLocaleString()} W</span>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />

                  {/* Baseline for stacked interval */}
                  <Area type="monotone" dataKey="LowerBound" stackId="dev-band" stroke="none" fill="transparent" isAnimationActive={false} />
                  {/* Shaded Conformal Interval */}
                  <Area
                    type="monotone"
                    dataKey="BandRange"
                    name="90% Conformal Interval [L, U]"
                    stackId="dev-band"
                    stroke="#60a5fa"
                    strokeWidth={1}
                    fill="#93c5fd"
                    fillOpacity={0.32}
                  />
                  {/* Point Forecast Line */}
                  <Line
                    type="monotone"
                    dataKey="prediction"
                    name="Point Forecast (Watts)"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: '#1d4ed8' }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 24-Hour Horizon Schedule Table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
            <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/50 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900">24-Hour Horizon Forecast Steps & Conformal Bounds</h3>
                <p className="text-xs text-slate-500">Complete multi-step predictions saved for run #{record.run_id}</p>
              </div>
              <span className="text-xs font-mono text-slate-400">Unit: Watts (W)</span>
            </div>

            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase tracking-wider font-semibold sticky top-0">
                  <tr>
                    <th className="py-2.5 px-4">Horizon Step</th>
                    <th className="py-2.5 px-4">Target Timestamp</th>
                    <th className="py-2.5 px-4 text-right">Point Forecast (W)</th>
                    <th className="py-2.5 px-4 text-right">Raw Lower Bound</th>
                    <th className="py-2.5 px-4 text-right">Clipped Lower (≥0W)</th>
                    <th className="py-2.5 px-4 text-right">Upper Bound (W)</th>
                    <th className="py-2.5 px-4 text-right">Interval Width</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  {chartData.map((row) => (
                    <tr key={row.horizon_hour} className="hover:bg-slate-50 transition-colors">
                      <td className="py-2 px-4 font-bold text-slate-700">{row.step}</td>
                      <td className="py-2 px-4 text-slate-600 font-sans">{row.fullTime}</td>
                      <td className="py-2 px-4 text-right font-bold text-blue-700">{row.prediction.toLocaleString()} W</td>
                      <td className="py-2 px-4 text-right text-slate-400">{row.rawLower.toLocaleString()} W</td>
                      <td className="py-2 px-4 text-right text-slate-700">{row.lower.toLocaleString()} W</td>
                      <td className="py-2 px-4 text-right text-slate-700">{row.upper.toLocaleString()} W</td>
                      <td className="py-2 px-4 text-right text-slate-500">{row.width.toLocaleString()} W</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Scientific Research Integrity Notice & Links */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center flex-shrink-0 text-blue-600">
            <Info className="w-5 h-5" />
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-bold text-slate-900">
              Scientific Integrity Notice: Run-Specific Data vs. Aggregate Research Benchmarks
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              The metrics and predictions displayed above are specific to run <strong className="font-mono text-slate-800">{record.run_id}</strong>.
              In contrast, the research links below navigate to population-level evaluations calibrated across the 587 rolling test windows from the Step 10 study.
              <strong className="text-slate-800"> Individual-run explainability (e.g. Integrated Gradients attribution) and ground-truth error scoring are not computed at live inference time</strong> because true ground-truth power readings are unobserved in real time.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-2 text-xs">
          <Link
            to="/benchmarks"
            className="p-3.5 rounded-lg border border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-white transition-all space-y-1"
          >
            <div className="font-bold text-slate-800 flex items-center justify-between">
              <span>Clean Model Benchmarks</span>
              <BarChart2 className="w-3.5 h-3.5 text-blue-600" />
            </div>
            <p className="text-[11px] text-slate-500">
              Compare {record.selected_model || 'this model'}&apos;s test MAE (71.64 W) against other architectures.
            </p>
          </Link>

          <Link
            to="/reliability"
            className="p-3.5 rounded-lg border border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-white transition-all space-y-1"
          >
            <div className="font-bold text-slate-800 flex items-center justify-between">
              <span>Reliability Degradation Matrix</span>
              <ShieldAlert className="w-3.5 h-3.5 text-blue-600" />
            </div>
            <p className="text-[11px] text-slate-500">
              Inspect performance under sensor missingness and noise conditions (N-N, N-A, A-N, A-A).
            </p>
          </Link>

          <Link
            to="/uncertainty"
            className="p-3.5 rounded-lg border border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-white transition-all space-y-1"
          >
            <div className="font-bold text-slate-800 flex items-center justify-between">
              <span>Conformal Calibration Lab</span>
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            </div>
            <p className="text-[11px] text-slate-500">
              Review empirical coverage (PICP) and Winkler scores from 587 calibration windows.
            </p>
          </Link>

          <Link
            to="/statistical-tests"
            className="p-3.5 rounded-lg border border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-white transition-all space-y-1"
          >
            <div className="font-bold text-slate-800 flex items-center justify-between">
              <span>Hypothesis Tests (H1–H4)</span>
              <Terminal className="w-3.5 h-3.5 text-purple-600" />
            </div>
            <p className="text-[11px] text-slate-500">
              Pairwise Wilcoxon signed-rank tests and Holm-Bonferroni significance checks.
            </p>
          </Link>

          <Link
            to="/xai"
            className="p-3.5 rounded-lg border border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-white transition-all space-y-1"
          >
            <div className="font-bold text-slate-800 flex items-center justify-between">
              <span>Integrated Gradients (XAI)</span>
              <Cpu className="w-3.5 h-3.5 text-amber-600" />
            </div>
            <p className="text-[11px] text-slate-500">
              Population-level temporal lag shares and weather feature importance ranking.
            </p>
          </Link>

          <Link
            to="/explorer"
            className="p-3.5 rounded-lg border border-slate-200 hover:border-blue-400 bg-slate-50/50 hover:bg-white transition-all space-y-1"
          >
            <div className="font-bold text-slate-800 flex items-center justify-between">
              <span>Rolling Window Explorer</span>
              <ExternalLink className="w-3.5 h-3.5 text-blue-600" />
            </div>
            <p className="text-[11px] text-slate-500">
              Inspect multi-model forecasts against ground truth across all 587 historical windows.
            </p>
          </Link>
        </div>
      </div>

      {/* Raw Record JSON Inspection Drawer */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <button
          type="button"
          onClick={() => setShowJson(!showJson)}
          className="w-full px-6 py-4 flex items-center justify-between bg-slate-50/60 hover:bg-slate-100/60 transition-colors text-left"
        >
          <div className="flex items-center gap-2">
            <FileJson className="w-4 h-4 text-slate-600" />
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Raw Record JSON Inspection
            </span>
            <span className="text-[11px] text-slate-500 font-normal">
              (Direct payload from SQLite database)
            </span>
          </div>
          {showJson ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
        </button>

        {showJson && (
          <div className="p-4 bg-slate-950 border-t border-slate-200">
            <pre className="text-[11px] font-mono text-emerald-400 overflow-x-auto p-4 rounded-lg bg-black/60 max-h-96 leading-relaxed">
              {JSON.stringify(record, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default DeveloperRunDetail;
