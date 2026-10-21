import React, { useEffect, useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Zap,
  Calendar,
  Clock,
  Printer,
  ArrowLeft,
  PlusCircle,
  AlertTriangle,
  Info,
  CheckCircle2,
  ShieldCheck,
  Cpu,
  Layers,
  RefreshCw,
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

export const ForecastResult: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();

  const [record, setRecord] = useState<ForecastRecordDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingPdf, setDownloadingPdf] = useState<boolean>(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const fetchRecord = async () => {
    if (!runId) {
      setError('No forecast run identifier was provided in the URL.');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await api.getForecastRecord(runId);
      setRecord(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve forecast record.');
    } finally {
      setLoading(false);
    }
  };

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

  useEffect(() => {
    fetchRecord();
  }, [runId]);

  // Derived metrics and chart data
  const chartData = useMemo(() => {
    if (!record || !record.intervals || record.intervals.length === 0) {
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
          // fallback to +H
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
      };
    });
  }, [record]);

  // Summary statistics computed from the 24-hour horizon
  const summaryStats = useMemo(() => {
    if (!chartData || chartData.length === 0) {
      return null;
    }

    const predictions = chartData.map((d) => d.prediction);
    const sumWatts = predictions.reduce((acc, curr) => acc + curr, 0);
    // Total energy in kWh (sum of hourly active power in Watts / 1000)
    const totalKwh = sumWatts / 1000;
    const avgWatts = sumWatts / chartData.length;
    const peakStep = chartData.reduce((prev, curr) => (curr.prediction > prev.prediction ? curr : prev), chartData[0]);
    const minStep = chartData.reduce((prev, curr) => (curr.prediction < prev.prediction ? curr : prev), chartData[0]);
    const avgWidth = chartData.reduce((acc, curr) => acc + curr.width, 0) / chartData.length;

    return {
      totalKwh: totalKwh.toFixed(2),
      avgWatts: Math.round(avgWatts).toLocaleString(),
      peakWatts: Math.round(peakStep.prediction).toLocaleString(),
      peakTime: peakStep.timeLabel,
      minWatts: Math.round(minStep.prediction).toLocaleString(),
      minTime: minStep.timeLabel,
      avgWidth: Math.round(avgWidth).toLocaleString(),
      halfWidth: Math.round(avgWidth / 2).toLocaleString(),
    };
  }, [chartData]);

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return <LoadingState message={`Retrieving forecast record #${runId}...`} />;
  }

  if (error || !record) {
    return (
      <div className="space-y-6 max-w-4xl mx-auto">
        <Link
          to="/forecast/history"
          className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-800"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Forecast History
        </Link>
        <ErrorState
          title="Forecast Record Not Found"
          message={error || 'The requested forecast record could not be loaded.'}
          onRetry={fetchRecord}
        />
      </div>
    );
  }

  // Handle failed forecast run
  if (record.status === 'failed') {
    return (
      <div className="space-y-6 max-w-4xl mx-auto">
        <div className="flex items-center justify-between">
          <Link
            to="/forecast/history"
            className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-800"
          >
            <ArrowLeft className="w-4 h-4" /> Back to Forecast History
          </Link>
          <Link
            to="/forecast/new"
            className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700"
          >
            <PlusCircle className="w-4 h-4" /> Try Another Forecast
          </Link>
        </div>

        <div className="bg-white rounded-xl border border-rose-200 p-8 shadow-sm">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-full bg-rose-100 flex items-center justify-center flex-shrink-0 text-rose-600">
              <AlertTriangle className="w-6 h-6" />
            </div>
            <div className="space-y-3 flex-1">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-xs font-semibold bg-rose-100 text-rose-800 uppercase tracking-wider">
                  Forecast Execution Failed
                </span>
                <span className="text-xs font-mono text-slate-500">Run ID: {record.run_id}</span>
              </div>
              <h1 className="text-xl font-bold text-slate-900">
                Inference Was Unable to Complete
              </h1>
              <p className="text-sm text-slate-600">
                The forecasting pipeline encountered an issue during processing and logged this failure.
              </p>

              <div className="bg-rose-50 border border-rose-200 rounded-lg p-4 font-mono text-xs text-rose-900 space-y-1">
                <div>
                  <span className="font-semibold text-rose-950">Error Code:</span>{' '}
                  {record.error_code || 'INFERENCE_ERROR'}
                </div>
                <div>
                  <span className="font-semibold text-rose-950">Details:</span>{' '}
                  {record.error_message || 'An unknown error occurred during model prediction.'}
                </div>
              </div>

              <div className="text-xs text-slate-500 pt-2 space-y-1">
                <p>• Ensure your CSV input file contains exactly 168 rows of hourly historical electricity active power.</p>
                <p>• Check that active power values are non-negative and formatted as valid numbers.</p>
                <p>• If using multimodal weather fusion, ensure temperature, humidity, pressure, and wind speed columns are present.</p>
              </div>

              <div className="pt-4 flex gap-3">
                <Link
                  to="/forecast/new"
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
                >
                  <PlusCircle className="w-4 h-4" /> Upload Corrected CSV
                </Link>
                <button
                  type="button"
                  onClick={fetchRecord}
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-300"
                >
                  <RefreshCw className="w-4 h-4" /> Check Status
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Handle pending forecast run
  if (record.status === 'pending') {
    return (
      <div className="space-y-6 max-w-4xl mx-auto">
        <Link
          to="/forecast/history"
          className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-slate-800"
        >
          <ArrowLeft className="w-4 h-4" /> Back to History
        </Link>
        <div className="bg-white rounded-xl border border-amber-200 p-8 shadow-sm text-center space-y-4">
          <div className="w-12 h-12 border-3 border-amber-200 border-t-amber-600 rounded-full animate-spin mx-auto" />
          <h2 className="text-lg font-bold text-slate-900">Forecast Run In Progress</h2>
          <p className="text-sm text-slate-600 max-w-md mx-auto">
            This forecast record is currently being processed. Click refresh below to check for completed results.
          </p>
          <button
            type="button"
            onClick={fetchRecord}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-amber-600 text-white hover:bg-amber-700"
          >
            <RefreshCw className="w-4 h-4" /> Refresh Status
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Top Navigation & Action Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 print:hidden">
        <div className="flex items-center gap-3">
          <Link
            to="/forecast/history"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 bg-white border border-slate-200 px-3 py-1.5 rounded-lg shadow-sm"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Past Runs
          </Link>
          <span className="text-slate-300">/</span>
          <span className="text-xs font-mono text-slate-500">Run {record.run_id}</span>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={handleDownloadPdf}
            disabled={downloadingPdf || record.status !== 'succeeded'}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm transition-colors disabled:opacity-50"
            title="Download genuine vector PDF report"
          >
            {downloadingPdf ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            Download PDF Report
          </button>
          <button
            type="button"
            onClick={handlePrint}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-sm transition-colors"
            title="Print or Save as PDF"
          >
            <Printer className="w-3.5 h-3.5 text-slate-500" />
            Print
          </button>
          <Link
            to="/forecast/new"
            className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-sm transition-colors"
          >
            <PlusCircle className="w-3.5 h-3.5" />
            New Forecast
          </Link>
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

      {/* Printable Report Header */}
      <div className="hidden print:block border-b border-slate-300 pb-4 mb-4">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-xl font-bold text-slate-900">Household Electricity Forecast Report</h1>
            <p className="text-xs text-slate-600">Reliable Multimodal Energy Forecasting System</p>
          </div>
          <div className="text-right text-xs font-mono text-slate-500">
            <div>Run ID: {record.run_id}</div>
            <div>Generated: {new Date(record.created_at).toLocaleString()}</div>
          </div>
        </div>
      </div>

      {/* Main Title & Status Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2.5 mb-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                Forecast Generated Successfully
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-50 text-blue-700 border border-blue-200">
                {record.selected_model || 'Early Fusion'}
              </span>
              {record.selection_mode && (
                <span
                  className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                    record.selection_mode === 'fallback'
                      ? 'bg-amber-50 text-amber-700 border-amber-200'
                      : record.selection_mode === 'auto'
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                      : 'bg-purple-50 text-purple-700 border-purple-200'
                  }`}
                >
                  {record.selection_mode === 'fallback'
                    ? 'Weather Fallback (Energy-Only)'
                    : record.selection_mode === 'auto'
                    ? 'Auto-Adaptive Selection'
                    : 'Explicit Override'}
                </span>
              )}
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-slate-100 text-slate-600 border border-slate-200">
                Method: {record.input_method.toUpperCase()}
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              24-Hour Electricity Load Forecast
            </h1>
            <p className="text-sm text-slate-600 mt-1">
              Hourly active power demand forecast (in Watts) for your household over the upcoming 24-hour horizon.
            </p>
          </div>

          {/* Timestamp Badges */}
          <div className="flex flex-col sm:flex-row lg:flex-col text-xs text-slate-500 gap-2 border-t lg:border-t-0 pt-3 lg:pt-0 border-slate-100">
            <div className="flex items-center gap-2">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>Created:</span>
              <span className="font-mono text-slate-700">
                {new Date(record.created_at).toLocaleString()}
              </span>
            </div>
            {record.forecast_origin_timestamp && (
              <div className="flex items-center gap-2">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                <span>Forecast Origin:</span>
                <span className="font-mono text-slate-700">
                  {record.forecast_origin_timestamp}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Model Selection & Automated Weather Telemetry Banner */}
      {record.selection_reason && (
        <div
          className={`border rounded-xl p-4 shadow-sm text-xs ${
            record.selection_mode === 'fallback'
              ? 'bg-amber-50/80 border-amber-200/80 text-amber-900'
              : record.selection_reason.toLowerCase().includes('automatically retrieved') || record.input_metadata?.location
              ? 'bg-emerald-50/80 border-emerald-200/80 text-emerald-900'
              : 'bg-slate-50 border-slate-200 text-slate-800'
          }`}
        >
          <div className="flex items-start gap-3">
            <div
              className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 ${
                record.selection_mode === 'fallback'
                  ? 'bg-amber-100 text-amber-700'
                  : record.selection_reason.toLowerCase().includes('automatically retrieved') || record.input_metadata?.location
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-blue-100 text-blue-700'
              }`}
            >
              {record.selection_mode === 'fallback' ? (
                <AlertTriangle className="w-4 h-4" />
              ) : record.selection_reason.toLowerCase().includes('automatically retrieved') || record.input_metadata?.location ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : (
                <Info className="w-4 h-4" />
              )}
            </div>
            <div className="space-y-1">
              <div className="font-bold text-sm">
                {record.selection_mode === 'fallback'
                  ? 'Weather Telemetry Fallback: Energy-Only Model Applied'
                  : record.selection_reason.toLowerCase().includes('automatically retrieved')
                  ? `Automated Weather Integration Active: ${record.selected_model}`
                  : `Model Selection Rationale: ${record.selected_model}`}
              </div>
              <p className="leading-relaxed opacity-90">{record.selection_reason}</p>
              {record.input_metadata?.location && (
                <div className="text-[11px] font-mono mt-1 opacity-80">
                  Household Location:{' '}
                  {record.input_metadata.location.label
                    ? `${record.input_metadata.location.label} (${Number(record.input_metadata.location.latitude).toFixed(4)}, ${Number(record.input_metadata.location.longitude).toFixed(4)})`
                    : `(${Number(record.input_metadata.location.latitude).toFixed(4)}, ${Number(record.input_metadata.location.longitude).toFixed(4)})`}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Uncertainty Notice Alert */}
      <div className="bg-blue-50/70 border border-blue-200/80 rounded-xl p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0 text-blue-600">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div className="text-xs text-slate-700 space-y-1">
            <div className="font-bold text-slate-900 flex items-center gap-2 text-sm">
              Conformal Uncertainty Quantification (90% Nominal Confidence Level)
            </div>
            <p className="leading-relaxed text-slate-600">
              The shaded confidence range on the chart represents a 90% prediction interval derived using rigorous
              Horizon-Wise Split Conformal Prediction.
              <strong className="text-slate-800"> These intervals are empirical estimates calibrated across rolling evaluation windows, not an absolute operational guarantee.</strong>{' '}
              Actual household demand may vary outside this envelope during unpredicted large appliance cycles or extreme weather fluctuations.
            </p>
          </div>
        </div>
      </div>

      {/* Key Household Metrics */}
      {summaryStats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Estimated 24h Energy"
            value={summaryStats.totalKwh}
            unit="kWh"
            subtitle="Cumulative consumption"
            badge={{ text: '24-hour total', variant: 'info' }}
            icon={<Zap className="w-4 h-4 text-amber-500" />}
          />
          <MetricCard
            title="Average Active Power"
            value={summaryStats.avgWatts}
            unit="Watts"
            subtitle="Mean hourly electricity draw"
            badge={{ text: 'Baseline demand', variant: 'neutral' }}
            icon={<Clock className="w-4 h-4 text-blue-500" />}
          />
          <MetricCard
            title="Projected Peak Load"
            value={summaryStats.peakWatts}
            unit="Watts"
            subtitle={`Expected peak at ${summaryStats.peakTime}`}
            badge={{ text: 'Max demand step', variant: 'warning' }}
            icon={<AlertTriangle className="w-4 h-4 text-amber-600" />}
          />
          <MetricCard
            title="Prediction Band Margin"
            value={`±${summaryStats.halfWidth}`}
            unit="Watts"
            subtitle={`Average width: ${summaryStats.avgWidth} W`}
            badge={{ text: '90% Conformal Range', variant: 'success' }}
            icon={<ShieldCheck className="w-4 h-4 text-emerald-600" />}
          />
        </div>
      )}

      {/* Chart Section */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              Hourly Power Demand & Prediction Intervals
            </h2>
            <p className="text-xs text-slate-500">
              Solid blue line depicts expected active power (Watts). Shaded band represents 90% conformal uncertainty range.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-blue-600 rounded-full" />
              <span className="text-slate-600 font-medium">Point Forecast</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 bg-blue-200 border border-blue-400 rounded-sm" />
              <span className="text-slate-600 font-medium">90% Confidence Interval</span>
            </div>
          </div>
        </div>

        {/* 24-Hour Horizon Chart */}
        <div className="h-88 w-full pt-2">
          <ResponsiveContainer width="100%" height={360}>
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
              <YAxis
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                unit=" W"
                domain={[0, 'auto']}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-white border border-slate-200 p-3.5 rounded-lg shadow-lg text-xs space-y-1.5 max-w-xs">
                        <div className="font-bold text-slate-900 border-b border-slate-100 pb-1 flex justify-between">
                          <span>{d.step}</span>
                          <span className="font-normal text-slate-500">{d.fullTime}</span>
                        </div>
                        <div className="text-blue-700 font-bold flex justify-between items-center">
                          <span>Expected Load:</span>
                          <span className="font-mono text-sm">{d.prediction.toLocaleString()} W</span>
                        </div>
                        <div className="text-emerald-700 font-medium flex justify-between items-center">
                          <span>90% Uncertainty Band:</span>
                          <span className="font-mono">
                            [{d.lower.toLocaleString()} W, {d.upper.toLocaleString()} W]
                          </span>
                        </div>
                        <div className="text-slate-500 text-[11px] flex justify-between items-center pt-1 border-t border-slate-100">
                          <span>Interval Width:</span>
                          <span className="font-mono">{d.width.toLocaleString()} W</span>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '12px' }} />

              {/* Invisible baseline for band stack */}
              <Area
                type="monotone"
                dataKey="LowerBound"
                stackId="interval-band"
                stroke="none"
                fill="transparent"
                isAnimationActive={false}
              />
              {/* 90% Shaded Uncertainty Range */}
              <Area
                type="monotone"
                dataKey="BandRange"
                name="90% Conformal Interval [Lower, Upper]"
                stackId="interval-band"
                stroke="#60a5fa"
                strokeWidth={1}
                fill="#93c5fd"
                fillOpacity={0.32}
              />
              {/* Point Forecast Line */}
              <Line
                type="monotone"
                dataKey="prediction"
                name="Expected Demand (Watts)"
                stroke="#2563eb"
                strokeWidth={2.5}
                dot={{ r: 3, fill: '#1d4ed8' }}
                activeDot={{ r: 6, fill: '#1d4ed8' }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model & Execution Provenance */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
            <Cpu className="w-4 h-4 text-blue-600" />
            Model Execution Details
          </div>
          <div className="space-y-2 text-xs text-slate-600">
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Forecasting Architecture:</span>
              <span className="font-semibold text-slate-800">{record.selected_model}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Selection Policy:</span>
              <span className="font-medium text-slate-800">
                {record.selection_mode === 'auto'
                  ? 'Automated Weather-Adaptive Selection'
                  : record.selection_mode === 'fallback'
                  ? 'Automated Weather Fallback (Energy-Only)'
                  : record.selection_mode === 'override'
                  ? 'Explicit User Override'
                  : record.selection_mode || 'Default operational policy'}
              </span>
            </div>
            {record.input_metadata?.location && (
              <div className="flex justify-between py-1 border-b border-slate-100">
                <span className="text-slate-500">Household Location:</span>
                <span className="font-medium text-slate-800">
                  {record.input_metadata.location.label
                    ? `${record.input_metadata.location.label} (${Number(record.input_metadata.location.latitude).toFixed(4)}, ${Number(record.input_metadata.location.longitude).toFixed(4)})`
                    : `(${Number(record.input_metadata.location.latitude).toFixed(4)}, ${Number(record.input_metadata.location.longitude).toFixed(4)})`}
                </span>
              </div>
            )}
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Input Lookback Window:</span>
              <span className="font-mono text-slate-800">168 consecutive hours (7 days)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Forecast Horizon:</span>
              <span className="font-mono text-slate-800">24 hours multi-step</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-500">Execution Hardware:</span>
              <span className="font-mono text-slate-800">PyTorch CPU Runtime</span>
            </div>
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
            <Layers className="w-4 h-4 text-emerald-600" />
            Reliability & Calibration Provenance
          </div>
          <div className="space-y-2 text-xs text-slate-600">
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Calibration Method:</span>
              <span className="font-medium text-slate-800">Horizon-Wise Split Conformal Prediction</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Nominal Confidence Level:</span>
              <span className="font-mono font-semibold text-emerald-700">90.0% (α = 0.10)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Average Interval Margin:</span>
              <span className="font-mono text-slate-800">
                {record.mean_interval_width_watts
                  ? `${Math.round(record.mean_interval_width_watts).toLocaleString()} Watts`
                  : summaryStats ? `${summaryStats.avgWidth} Watts` : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-100">
              <span className="text-slate-500">Energy Feature Scaling:</span>
              <span className="font-mono text-slate-800">StandardScaler (Inverse-transformed to Watts)</span>
            </div>
            <div className="flex flex-col py-1 gap-1">
              <span className="text-slate-500">Reason for Selection:</span>
              <span className="text-slate-700 text-xs leading-relaxed" title={record.selection_reason || ''}>
                {record.selection_reason || 'Default operational model'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 24-Hour Detailed Schedule Table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-50/50 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-bold text-slate-900">24-Hour Hourly Forecast Schedule</h2>
            <p className="text-xs text-slate-500">Detailed hourly breakdown of expected power draw and uncertainty intervals.</p>
          </div>
          <div className="text-xs text-slate-400 font-mono">Units: Watts (W)</div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-100/70 border-b border-slate-200 text-slate-600 uppercase tracking-wider font-semibold">
                <th className="py-3 px-4">Step</th>
                <th className="py-3 px-4">Target Time</th>
                <th className="py-3 px-4 text-right">Expected Load (W)</th>
                <th className="py-3 px-4 text-right">Lower Bound (90%)</th>
                <th className="py-3 px-4 text-right">Upper Bound (90%)</th>
                <th className="py-3 px-4 text-right">Interval Width</th>
                <th className="py-3 px-4 text-center">Load Category</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {chartData.map((row) => {
                let badgeClass = 'bg-slate-100 text-slate-700';
                let badgeText = 'Moderate';
                if (row.prediction < 500) {
                  badgeClass = 'bg-emerald-50 text-emerald-700 border border-emerald-200';
                  badgeText = 'Low';
                } else if (row.prediction > 1500) {
                  badgeClass = 'bg-amber-50 text-amber-700 border border-amber-200';
                  badgeText = 'High Demand';
                }

                return (
                  <tr key={row.horizon_hour} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-2.5 px-4 font-bold text-slate-700">{row.step}</td>
                    <td className="py-2.5 px-4 text-slate-600 font-sans">{row.fullTime}</td>
                    <td className="py-2.5 px-4 text-right font-bold text-blue-700">
                      {row.prediction.toLocaleString()} W
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-600">
                      {row.lower.toLocaleString()} W
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-600">
                      {row.upper.toLocaleString()} W
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-400">
                      {row.width.toLocaleString()} W
                    </td>
                    <td className="py-2.5 px-4 text-center font-sans">
                      <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-medium ${badgeClass}`}>
                        {badgeText}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-600 print:hidden">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-blue-600 flex-shrink-0" />
          <span>
            This record is permanently saved in the local forecast database and can be reviewed anytime via the history dashboard.
          </span>
        </div>
        <div className="flex items-center gap-3">
          <Link
            to="/forecast/history"
            className="px-3.5 py-2 rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium"
          >
            View All Past Runs
          </Link>
          <Link
            to="/forecast/new"
            className="px-3.5 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 font-medium shadow-sm"
          >
            Generate Another Forecast
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ForecastResult;
