import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  Calendar,
  ArrowLeft,
  PlusCircle,
  AlertTriangle,
  Info,
  ShieldCheck,
  ChevronDown,
  ChevronUp,
  FileSpreadsheet,
  Lock,
  Download,
  Loader2,
  CheckCircle2,
  FileDown,
} from 'lucide-react';
import { api } from '../api/client';
import { MonthlyForecastRecordDetail, MonthlyForecastStep } from '../api/types';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';

export const MonthlyForecastDetail: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();

  const [record, setRecord] = useState<MonthlyForecastRecordDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isForbidden, setIsForbidden] = useState<boolean>(false);
  const [expandedBreakdownIndex, setExpandedBreakdownIndex] = useState<number | null>(null);
  const [showInputSnapshot, setShowInputSnapshot] = useState<boolean>(false);

  // PDF report download state
  const [downloadingPdf, setDownloadingPdf] = useState<boolean>(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const [pdfSuccess, setPdfSuccess] = useState<boolean>(false);

  const handleDownloadPdf = async () => {
    if (!record || downloadingPdf) return;
    try {
      setDownloadingPdf(true);
      setPdfError(null);
      setPdfSuccess(false);
      await api.downloadMonthlyForecastPdf(record.run_id);
      setPdfSuccess(true);
      setTimeout(() => setPdfSuccess(false), 4000);
    } catch (err: unknown) {
      setPdfError(err instanceof Error ? err.message : 'Failed to download monthly forecast PDF report.');
    } finally {
      setDownloadingPdf(false);
    }
  };

  const fetchRecord = async () => {
    if (!runId) {
      setError('No forecast run identifier was provided.');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      setIsForbidden(false);
      const data = await api.getMonthlyForecastRecord(runId);
      setRecord(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to retrieve monthly forecast record.';
      if (msg.includes('403') || msg.toLowerCase().includes('forbidden') || msg.toLowerCase().includes('permission')) {
        setIsForbidden(true);
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecord();
  }, [runId]);

  if (loading) {
    return (
      <div className="max-w-5xl mx-auto py-12">
        <LoadingState message="Loading saved monthly forecast snapshot..." />
      </div>
    );
  }

  if (isForbidden) {
    return (
      <div className="max-w-2xl mx-auto py-12 space-y-4">
        <div className="p-6 rounded-xl bg-amber-50 border border-amber-300 text-amber-900 space-y-3">
          <div className="flex items-center gap-2 text-amber-700 font-bold">
            <Lock className="w-5 h-5 text-amber-600" />
            <h3>Access Forbidden</h3>
          </div>
          <p className="text-sm text-amber-800 leading-relaxed">
            You do not have authorization to view this monthly forecast record. Forecast runs are strictly scoped to their authenticated owners.
          </p>
          <div className="pt-2 flex items-center gap-3">
            <Link
              to="/forecast/history?tab=monthly"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-lg bg-amber-600 text-white hover:bg-amber-700 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Return to Your Forecast History
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (error || !record) {
    return (
      <div className="max-w-3xl mx-auto py-12">
        <ErrorState
          title="Monthly Forecast Not Found"
          message={error || `Monthly forecast record with ID '${runId}' could not be located.`}
          onRetry={fetchRecord}
        />
        <div className="mt-4 text-center">
          <Link
            to="/forecast/history?tab=monthly"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 hover:text-blue-800 underline"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back to Monthly Forecast History
          </Link>
        </div>
      </div>
    );
  }

  const createdDate = new Date(record.created_at);
  const formattedCreated = `${createdDate.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })} at ${createdDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-12">
      {/* Top Navigation */}
      <div className="flex items-center justify-between text-xs">
        <Link
          to="/forecast/history?tab=monthly"
          className="inline-flex items-center gap-1.5 text-slate-600 hover:text-blue-600 font-medium transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Forecast History
        </Link>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleDownloadPdf}
            disabled={downloadingPdf}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-600 text-white font-semibold hover:bg-emerald-700 shadow-xs transition-colors disabled:opacity-50"
            title="Download authoritative monthly forecast vector PDF report"
          >
            {downloadingPdf ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Download className="w-3.5 h-3.5" />
            )}
            <span>{downloadingPdf ? 'Generating PDF...' : 'Download PDF Report'}</span>
          </button>
          <Link
            to="/forecast/new"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 text-white font-semibold hover:bg-blue-700 shadow-xs transition-colors"
          >
            <PlusCircle className="w-3.5 h-3.5" />
            New Forecast
          </Link>
        </div>
      </div>

      {/* PDF Status / Error Notifications */}
      {pdfError && (
        <div className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0" />
            <span>{pdfError}</span>
          </div>
          <button
            type="button"
            onClick={() => setPdfError(null)}
            className="text-rose-600 hover:text-rose-800 font-bold ml-2"
          >
            ×
          </button>
        </div>
      )}
      {pdfSuccess && (
        <div className="p-3.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
          <span>PDF report successfully generated and downloaded.</span>
        </div>
      )}

      {/* Header Snapshot Card */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-200 text-indigo-700 text-xs font-semibold uppercase tracking-wider mb-2">
              <ShieldCheck className="w-3.5 h-3.5" />
              Saved Forecast Snapshot
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              {record.horizon_months}-Month Household Electricity Projection
            </h1>
            <p className="text-xs text-slate-500 mt-1 font-mono">
              Snapshot ID: <span className="text-slate-800 font-semibold">{record.run_id}</span> • Recorded: {formattedCreated}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold border ${
                record.confidence_rating === 'high'
                  ? 'bg-emerald-50 border-emerald-300 text-emerald-800'
                  : record.confidence_rating === 'moderate'
                  ? 'bg-blue-50 border-blue-300 text-blue-800'
                  : 'bg-amber-50 border-amber-300 text-amber-800'
              }`}
            >
              Confidence: {record.confidence_rating.toUpperCase()}
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 border border-slate-200 text-slate-700">
              Provider: {record.provider_id ? record.provider_id.toUpperCase() : 'None'}
            </span>
            <button
              type="button"
              onClick={handleDownloadPdf}
              disabled={downloadingPdf}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 border border-emerald-300 text-emerald-800 hover:bg-emerald-100 transition-colors disabled:opacity-50"
              title="Download vector PDF report"
            >
              {downloadingPdf ? (
                <Loader2 className="w-3 h-3 animate-spin text-emerald-700" />
              ) : (
                <FileDown className="w-3 h-3 text-emerald-700" />
              )}
              <span>{downloadingPdf ? 'Downloading...' : 'PDF Report'}</span>
            </button>
          </div>
        </div>

        {/* Snapshot Integrity Notice */}
        <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-600 text-xs flex items-start gap-2.5">
          <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong>Durable Audit Snapshot:</strong> This is a persistent historical forecast record. It preserves the exact input billing data, model methodology, predicted consumption, and tariff estimations evaluated at creation time. Newly calculated forecasts will not modify this snapshot.
          </p>
        </div>
      </div>

      {/* Summary Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1">
          <span className="text-slate-500 font-medium block">Total Forecast Energy</span>
          <div className="text-2xl font-extrabold text-slate-900 font-mono">
            {record.total_forecast_kwh.toFixed(1)} <span className="text-sm font-sans font-bold text-slate-600">kWh</span>
          </div>
          <span className="text-[11px] text-slate-500 block">Across {record.horizon_months} billing cycle{record.horizon_months === 1 ? '' : 's'}</span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1">
          <span className="text-slate-500 font-medium block">Average Monthly Demand</span>
          <div className="text-2xl font-extrabold text-blue-600 font-mono">
            {record.average_monthly_kwh.toFixed(1)} <span className="text-sm font-sans font-bold text-slate-600">kWh/mo</span>
          </div>
          <span className="text-[11px] text-slate-500 block">Mean projected monthly consumption</span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1">
          <span className="text-slate-500 font-medium block">Estimated Cumulative Bill</span>
          {record.total_estimated_bill_inr !== null && record.total_estimated_bill_inr !== undefined ? (
            <div className="text-2xl font-extrabold text-slate-900 font-mono">
              ₹{record.total_estimated_bill_inr.toFixed(2)}
            </div>
          ) : (
            <div className="text-base font-bold text-amber-700 flex items-center gap-1.5 py-1">
              <AlertTriangle className="w-4 h-4 text-amber-600" />
              Unavailable
            </div>
          )}
          <span className="text-[11px] text-slate-500 block truncate" title={record.bill_estimate_notes || ''}>
            {record.bill_estimate_notes || 'Tariff rate notice'}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-xs space-y-1">
          <span className="text-slate-500 font-medium block">Forecasting Method</span>
          <div className="text-sm font-bold text-slate-900 capitalize truncate">
            {record.selected_method.replace(/_/g, ' ')}
          </div>
          <span className="text-[11px] text-slate-500 block">
            {record.history_length_months} historical cycle{record.history_length_months === 1 ? '' : 's'} observed
          </span>
        </div>
      </div>

      {/* Warnings & Notices */}
      {record.warnings && record.warnings.length > 0 && (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs space-y-2">
          <div className="font-bold flex items-center gap-2 text-amber-950">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            Notices & Model Disclosures
          </div>
          <ul className="list-disc pl-5 space-y-1 text-amber-800 leading-relaxed">
            {record.warnings.map((w, i) => (
              <li key={`w-${i}`}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Projected Monthly Steps Breakdown Table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Calendar className="w-4 h-4 text-blue-600" />
            Projected Monthly Billing Cycles
          </h2>
          <span className="text-xs text-slate-500 font-mono">
            {record.forecast_steps.length} Step{record.forecast_steps.length === 1 ? '' : 's'} Projected
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="py-3 px-4">Cycle / Month</th>
                <th className="py-3 px-4">Billing Interval</th>
                <th className="py-3 px-4 text-right">Forecast (kWh)</th>
                <th className="py-3 px-4 text-right">Uncertainty Bound</th>
                <th className="py-3 px-4 text-right">Bill Estimate (INR)</th>
                <th className="py-3 px-4 text-center">Tariff Breakdown</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {record.forecast_steps.map((step: MonthlyForecastStep, idx: number) => {
                const hasBreakdown =
                  step.bill_estimate &&
                  step.bill_estimate.charge_breakdown &&
                  Object.keys(step.bill_estimate.charge_breakdown).length > 0;
                const isExpanded = expandedBreakdownIndex === idx;

                return (
                  <React.Fragment key={`step-${step.month_index}`}>
                    <tr className="hover:bg-slate-50/70 transition-colors">
                      <td className="py-3 px-4 font-sans font-bold text-slate-900">
                        {step.month_label}
                        <span className="ml-1.5 text-[10px] font-normal text-slate-500">
                          (Month #{step.month_index})
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-600 font-sans">
                        {step.period_start} to {step.period_end}
                      </td>
                      <td className="py-3 px-4 text-right font-bold text-blue-600">
                        {step.forecast_kwh.toFixed(1)} kWh
                      </td>
                      <td className="py-3 px-4 text-right text-slate-500 font-sans">
                        {step.uncertainty_lower_kwh !== null &&
                        step.uncertainty_upper_kwh !== null &&
                        step.uncertainty_lower_kwh !== undefined &&
                        step.uncertainty_upper_kwh !== undefined ? (
                          <span className="font-mono text-slate-700">
                            [{step.uncertainty_lower_kwh.toFixed(1)} – {step.uncertainty_upper_kwh.toFixed(1)}]
                          </span>
                        ) : (
                          <span className="text-[11px] text-slate-400 italic">Static persistence</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right font-sans">
                        {step.estimated_bill_amount !== null && step.estimated_bill_amount !== undefined ? (
                          <span className="font-bold text-slate-900 font-mono">
                            ₹{step.estimated_bill_amount.toFixed(2)}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded">
                            Unavailable
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-center font-sans">
                        {hasBreakdown ? (
                          <button
                            type="button"
                            onClick={() => setExpandedBreakdownIndex(isExpanded ? null : idx)}
                            className="inline-flex items-center gap-1 text-[11px] text-blue-600 hover:text-blue-800 font-semibold transition-colors"
                          >
                            <span>{isExpanded ? 'Hide' : 'Inspect'}</span>
                            {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                          </button>
                        ) : (
                          <span className="text-[11px] text-slate-400 italic">No breakdown</span>
                        )}
                      </td>
                    </tr>

                    {/* Expandable Tariff Breakdown Drawer */}
                    {isExpanded && step.bill_estimate && (
                      <tr className="bg-slate-50/80">
                        <td colSpan={6} className="py-3 px-6 font-sans">
                          <div className="p-4 rounded-lg bg-white border border-slate-200 space-y-3 shadow-2xs">
                            <div className="flex items-center justify-between text-xs pb-2 border-b border-slate-100">
                              <span className="font-bold text-slate-900">
                                MSEDCL LT-I(B) Residential Breakdown — {step.month_label}
                              </span>
                              <span className="text-slate-500 font-mono text-[11px]">
                                Base Tariff: ₹{step.estimated_bill_amount?.toFixed(2)}
                              </span>
                            </div>

                            {step.bill_estimate.charge_breakdown && (
                              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 text-xs">
                                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 space-y-0.5">
                                  <span className="text-slate-500 text-[10px] block">Fixed Charges</span>
                                  <span className="font-bold text-slate-800 font-mono">
                                    ₹{step.bill_estimate.charge_breakdown.fixed_charges_inr ?? step.bill_estimate.charge_breakdown.fixed_charge ?? 0}
                                  </span>
                                </div>
                                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 space-y-0.5">
                                  <span className="text-slate-500 text-[10px] block">Energy Charges</span>
                                  <span className="font-bold text-slate-800 font-mono">
                                    ₹{step.bill_estimate.charge_breakdown.energy_charges_inr ?? step.bill_estimate.charge_breakdown.energy_charge ?? 0}
                                  </span>
                                </div>
                                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 space-y-0.5">
                                  <span className="text-slate-500 text-[10px] block">Wheeling Charges</span>
                                  <span className="font-bold text-slate-800 font-mono">
                                    ₹{step.bill_estimate.charge_breakdown.wheeling_charges_inr ?? step.bill_estimate.charge_breakdown.wheeling_charge ?? 0}
                                  </span>
                                </div>
                                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 space-y-0.5">
                                  <span className="text-slate-500 text-[10px] block">Electricity Duty (16%)</span>
                                  <span className="font-bold text-slate-800 font-mono">
                                    ₹{step.bill_estimate.charge_breakdown.electricity_duty_inr ?? step.bill_estimate.charge_breakdown.electricity_duty ?? 0}
                                  </span>
                                </div>
                                <div className="p-2.5 rounded bg-slate-50 border border-slate-100 space-y-0.5">
                                  <span className="text-slate-500 text-[10px] block">FAC Charges</span>
                                  <span className="font-bold text-slate-800 font-mono">
                                    ₹{step.bill_estimate.charge_breakdown.fac_charges_inr ?? step.bill_estimate.charge_breakdown.fac_amount ?? 0}
                                  </span>
                                </div>
                              </div>
                            )}

                            {step.bill_estimate.disclosures && step.bill_estimate.disclosures.length > 0 && (
                              <div className="text-[11px] text-slate-500 space-y-0.5 pt-1">
                                <span className="font-semibold text-slate-700 block">Regulatory Notes:</span>
                                {step.bill_estimate.disclosures.map((d, di) => (
                                  <p key={`d-${di}`} className="leading-relaxed text-slate-600">
                                    • {d}
                                  </p>
                                ))}
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Input Records Snapshot Collapsible Card */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <button
          type="button"
          onClick={() => setShowInputSnapshot(!showInputSnapshot)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-slate-50/80 transition-colors"
        >
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="w-4 h-4 text-emerald-600" />
            <h2 className="text-sm font-bold text-slate-900">
              Historical Billing Input Snapshot ({record.input_records.length} Billing Record{record.input_records.length === 1 ? '' : 's'})
            </h2>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span>{showInputSnapshot ? 'Collapse' : 'Inspect Input Data'}</span>
            {showInputSnapshot ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        </button>

        {showInputSnapshot && (
          <div className="p-4 border-t border-slate-200 bg-slate-50/50 space-y-3">
            <p className="text-xs text-slate-600">
              The exact confirmed billing records that were used to generate this forecast:
            </p>
            <div className="overflow-x-auto bg-white rounded-lg border border-slate-200">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-100/70 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-2.5 px-3">Period Label</th>
                    <th className="py-2.5 px-3">Billing Start</th>
                    <th className="py-2.5 px-3">Billing End</th>
                    <th className="py-2.5 px-3 text-right">Consumption (kWh)</th>
                    <th className="py-2.5 px-3 text-right">Billed Amount</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  {record.input_records.map((r, i) => (
                    <tr key={`rec-${i}`} className="hover:bg-slate-50">
                      <td className="py-2.5 px-3 font-sans font-semibold text-slate-900">{r.period_label || `Cycle #${i + 1}`}</td>
                      <td className="py-2.5 px-3 text-slate-600 font-sans">{r.billing_period_start}</td>
                      <td className="py-2.5 px-3 text-slate-600 font-sans">{r.billing_period_end}</td>
                      <td className="py-2.5 px-3 text-right font-bold text-slate-900">{r.consumption_kwh.toFixed(1)} kWh</td>
                      <td className="py-2.5 px-3 text-right text-slate-600">
                        {r.bill_amount !== null && r.bill_amount !== undefined ? `₹${r.bill_amount.toFixed(2)}` : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Model & Methodology Details */}
      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs text-slate-600">
        <h3 className="font-bold text-slate-800 text-xs">Methodology & Verification Details</h3>
        <p className="leading-relaxed">{record.method_description}</p>
        {record.limitations && record.limitations.length > 0 && (
          <div className="pt-1">
            <span className="font-semibold text-slate-700 block">Limitations:</span>
            <ul className="list-disc pl-5 space-y-0.5 mt-1 text-slate-500">
              {record.limitations.map((l, i) => (
                <li key={`lim-${i}`}>{l}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default MonthlyForecastDetail;
