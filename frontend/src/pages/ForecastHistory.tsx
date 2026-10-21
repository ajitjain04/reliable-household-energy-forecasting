import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import {
  History,
  PlusCircle,
  Filter,
  CheckCircle2,
  AlertTriangle,
  Clock,
  ArrowRight,
  RefreshCw,
  Search,
  FileSpreadsheet,
  Code,
  ShieldAlert,
  Calendar,
  FileDown,
  Loader2,
} from 'lucide-react';
import { api } from '../api/client';
import { ForecastRecordSummary, MonthlyForecastRecordSummary } from '../api/types';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';

const PAGE_SIZE = 10;

export const ForecastHistory: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get('tab') === 'monthly' ? 'monthly' : 'hourly';

  // --- 24-Hour Hourly Forecast Records State ---
  const [records, setRecords] = useState<ForecastRecordSummary[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(0);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // --- Monthly Forecast Records State (Task 2.4) ---
  const [monthlyRecords, setMonthlyRecords] = useState<MonthlyForecastRecordSummary[]>([]);
  const [monthlyTotalCount, setMonthlyTotalCount] = useState<number>(0);
  const [monthlyCurrentPage, setMonthlyCurrentPage] = useState<number>(0);
  const [monthlyLoading, setMonthlyLoading] = useState<boolean>(true);
  const [monthlyError, setMonthlyError] = useState<string | null>(null);

  // Monthly PDF Report Download State (Task 2.5)
  const [downloadingPdfRunId, setDownloadingPdfRunId] = useState<string | null>(null);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const handleDownloadMonthlyPdf = async (runId: string) => {
    if (downloadingPdfRunId) return;
    try {
      setDownloadingPdfRunId(runId);
      setPdfError(null);
      await api.downloadMonthlyForecastPdf(runId);
    } catch (err: unknown) {
      setPdfError(err instanceof Error ? err.message : 'Failed to download monthly forecast PDF report.');
    } finally {
      setDownloadingPdfRunId(null);
    }
  };

  const fetchHourlyRecords = async () => {
    try {
      setLoading(true);
      setError(null);
      const params: {
        limit: number;
        offset: number;
        status?: string;
        model?: string;
      } = {
        limit: PAGE_SIZE,
        offset: currentPage * PAGE_SIZE,
      };

      if (statusFilter !== 'all') {
        params.status = statusFilter;
      }
      if (modelFilter !== 'all') {
        params.model = modelFilter;
      }

      const res = await api.getForecastRecords(params);
      setRecords(res.records);
      setTotalCount(res.total_count);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch hourly forecast records.');
    } finally {
      setLoading(false);
    }
  };

  const fetchMonthlyRecords = async () => {
    try {
      setMonthlyLoading(true);
      setMonthlyError(null);
      const res = await api.getMonthlyForecastRecords({
        limit: PAGE_SIZE,
        offset: monthlyCurrentPage * PAGE_SIZE,
      });
      setMonthlyRecords(res.records);
      setMonthlyTotalCount(res.total_count);
    } catch (err: unknown) {
      setMonthlyError(err instanceof Error ? err.message : 'Failed to fetch monthly forecast records.');
    } finally {
      setMonthlyLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'hourly') {
      fetchHourlyRecords();
    } else {
      fetchMonthlyRecords();
    }
  }, [activeTab, currentPage, statusFilter, modelFilter, monthlyCurrentPage]);

  // Initial load to get counts for both tabs
  useEffect(() => {
    if (activeTab === 'hourly') {
      api.getMonthlyForecastRecords({ limit: 1, offset: 0 }).then((r) => setMonthlyTotalCount(r.total_count)).catch(() => {});
    } else {
      api.getForecastRecords({ limit: 1, offset: 0 }).then((r) => setTotalCount(r.total_count)).catch(() => {});
    }
  }, []);

  const totalPages = Math.ceil(totalCount / PAGE_SIZE);
  const monthlyTotalPages = Math.ceil(monthlyTotalCount / PAGE_SIZE);

  const handleTabChange = (tab: 'hourly' | 'monthly') => {
    setSearchParams({ tab });
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider mb-2">
              <History className="w-3.5 h-3.5" />
              Persistent Audit Trail
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Forecast History & Saved Runs
            </h1>
            <p className="text-sm text-slate-600 mt-1">
              Review, inspect, and reload all past 24-hour smart meter runs and saved monthly household projections.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={activeTab === 'hourly' ? fetchHourlyRecords : fetchMonthlyRecords}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-sm transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
              Refresh
            </button>
            <Link
              to="/forecast/new"
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-sm transition-colors"
            >
              <PlusCircle className="w-4 h-4" />
              New Forecast
            </Link>
          </div>
        </div>

        {/* Tab Switcher: Hourly 24h vs Monthly Bill Forecasts */}
        <div className="flex items-center gap-2 pt-6 border-t border-slate-100 mt-6">
          <button
            type="button"
            id="tab-hourly"
            onClick={() => handleTabChange('hourly')}
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'hourly'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>24-Hour Smart Meter Runs</span>
            {totalCount > 0 && (
              <span
                className={`ml-1 px-1.5 py-0.5 rounded-full text-[10px] font-mono ${
                  activeTab === 'hourly' ? 'bg-blue-700 text-white' : 'bg-slate-200 text-slate-700'
                }`}
              >
                {totalCount}
              </span>
            )}
          </button>

          <button
            type="button"
            id="tab-monthly"
            onClick={() => handleTabChange('monthly')}
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'monthly'
                ? 'bg-blue-600 text-white shadow-xs'
                : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
            }`}
          >
            <Calendar className="w-3.5 h-3.5" />
            <span>Monthly Bill Forecasts</span>
            {monthlyTotalCount > 0 && (
              <span
                className={`ml-1 px-1.5 py-0.5 rounded-full text-[10px] font-mono ${
                  activeTab === 'monthly' ? 'bg-blue-700 text-white' : 'bg-slate-200 text-slate-700'
                }`}
              >
                {monthlyTotalCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* ===================================================================== */}
      {/* TAB 1: MONTHLY BILL FORECASTS (TASK 2.4)                              */}
      {/* ===================================================================== */}
      {activeTab === 'monthly' && (
        <div className="space-y-6">
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
          {monthlyLoading ? (
            <div className="bg-white border border-slate-200 rounded-xl p-12 shadow-sm">
              <LoadingState message="Loading saved monthly forecast history..." />
            </div>
          ) : monthlyError ? (
            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
              <ErrorState title="Failed to Load Monthly History" message={monthlyError} onRetry={fetchMonthlyRecords} />
            </div>
          ) : monthlyRecords.length === 0 ? (
            /* Empty State */
            <div className="bg-white border border-slate-200 rounded-xl p-12 text-center shadow-sm space-y-4">
              <div className="w-12 h-12 rounded-full bg-blue-50 text-blue-600 mx-auto flex items-center justify-center">
                <Calendar className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-slate-900">No Saved Monthly Forecasts Found</h3>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  You haven't saved any monthly electricity forecasts yet. Upload your monthly electricity bills, generate a forecast, and save the snapshot to track it here.
                </p>
              </div>
              <Link
                to="/forecast/new"
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 text-white text-xs font-semibold hover:bg-blue-700 shadow-xs transition-colors"
              >
                <PlusCircle className="w-4 h-4" />
                Create Monthly Forecast
              </Link>
            </div>
          ) : (
            /* Monthly Table */
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[11px]">
                    <tr>
                      <th className="py-3 px-4">Snapshot ID</th>
                      <th className="py-3 px-4">Saved Date</th>
                      <th className="py-3 px-4">Horizon</th>
                      <th className="py-3 px-4">Forecast Period</th>
                      <th className="py-3 px-4">Provider</th>
                      <th className="py-3 px-4">Method & Confidence</th>
                      <th className="py-3 px-4 text-right">Projected (kWh)</th>
                      <th className="py-3 px-4 text-right">Estimated Bill</th>
                      <th className="py-3 px-4 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {monthlyRecords.map((r) => {
                      const createdDate = new Date(r.created_at);
                      const formattedDate = `${createdDate.toLocaleDateString([], {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                      })} ${createdDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

                      return (
                        <tr key={r.run_id} className="hover:bg-slate-50/80 transition-colors">
                          <td className="py-3 px-4 font-sans font-bold">
                            <Link
                              to={`/forecast/monthly/${r.run_id}`}
                              className="text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1"
                              title="Inspect saved monthly forecast snapshot"
                            >
                              <Search className="w-3 h-3 text-slate-400" />
                              {r.run_id}
                            </Link>
                          </td>
                          <td className="py-3 px-4 text-slate-600 font-sans">{formattedDate}</td>
                          <td className="py-3 px-4 font-sans">
                            <span className="inline-block px-2 py-0.5 rounded text-[11px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                              {r.horizon_months} {r.horizon_months === 1 ? 'Mo' : 'Mos'}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-sans text-slate-800 font-medium">
                            {r.forecast_period_range || '-'}
                          </td>
                          <td className="py-3 px-4 font-sans">
                            <span className="inline-block px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-800 uppercase">
                              {r.provider_id || 'None'}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-sans">
                            <div className="space-y-0.5">
                              <span className="block font-semibold text-slate-800 capitalize">
                                {r.selected_method.replace(/_/g, ' ')}
                              </span>
                              <span
                                className={`inline-block px-1.5 py-0.2 rounded text-[10px] font-bold uppercase border ${
                                  r.confidence_rating === 'high'
                                    ? 'bg-emerald-50 border-emerald-300 text-emerald-800'
                                    : r.confidence_rating === 'moderate'
                                    ? 'bg-blue-50 border-blue-300 text-blue-800'
                                    : 'bg-amber-50 border-amber-300 text-amber-800'
                                }`}
                              >
                                {r.confidence_rating}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4 text-right font-bold text-blue-600">
                            {r.total_forecast_kwh.toFixed(1)} kWh
                          </td>
                          <td className="py-3 px-4 text-right font-sans">
                            {r.total_estimated_bill_inr !== null && r.total_estimated_bill_inr !== undefined ? (
                              <span className="font-bold text-slate-900 font-mono">
                                ₹{r.total_estimated_bill_inr.toFixed(2)}
                              </span>
                            ) : (
                              <span className="inline-block px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200">
                                Unavailable
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-right font-sans">
                            <div className="flex items-center justify-end gap-1.5">
                              <button
                                type="button"
                                onClick={() => handleDownloadMonthlyPdf(r.run_id)}
                                disabled={downloadingPdfRunId === r.run_id}
                                className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-semibold rounded bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100 transition-colors disabled:opacity-50"
                                title="Download authoritative vector PDF report"
                              >
                                {downloadingPdfRunId === r.run_id ? (
                                  <Loader2 className="w-3 h-3 animate-spin text-emerald-600" />
                                ) : (
                                  <FileDown className="w-3 h-3 text-emerald-600" />
                                )}
                                <span>{downloadingPdfRunId === r.run_id ? 'Downloading...' : 'PDF'}</span>
                              </button>
                              <Link
                                to={`/forecast/monthly/${r.run_id}`}
                                className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-semibold rounded bg-slate-100 text-slate-700 hover:bg-blue-50 hover:text-blue-700 transition-colors"
                              >
                                <span>View Details</span>
                                <ArrowRight className="w-3 h-3" />
                              </Link>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Monthly Pagination Controls */}
              {monthlyTotalPages > 1 && (
                <div className="px-6 py-4 border-t border-slate-200 bg-slate-50/50 flex items-center justify-between text-xs">
                  <span className="text-slate-500">
                    Showing{' '}
                    <strong className="text-slate-700">{monthlyCurrentPage * PAGE_SIZE + 1}</strong> to{' '}
                    <strong className="text-slate-700">
                      {Math.min((monthlyCurrentPage + 1) * PAGE_SIZE, monthlyTotalCount)}
                    </strong>{' '}
                    of <strong className="text-slate-700">{monthlyTotalCount}</strong> saved runs
                  </span>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={monthlyCurrentPage === 0}
                      onClick={() => setMonthlyCurrentPage((p) => Math.max(0, p - 1))}
                      className="px-3 py-1.5 rounded border border-slate-300 bg-white text-slate-700 font-medium hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-white"
                    >
                      Previous
                    </button>
                    <span className="px-2 text-slate-500">
                      Page {monthlyCurrentPage + 1} of {monthlyTotalPages}
                    </span>
                    <button
                      type="button"
                      disabled={monthlyCurrentPage >= monthlyTotalPages - 1}
                      onClick={() => setMonthlyCurrentPage((p) => Math.min(monthlyTotalPages - 1, p + 1))}
                      className="px-3 py-1.5 rounded border border-slate-300 bg-white text-slate-700 font-medium hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-white"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ===================================================================== */}
      {/* TAB 2: HOURLY 24-HOUR SMART METER RUNS                                */}
      {/* ===================================================================== */}
      {activeTab === 'hourly' && (
        <div className="space-y-6">
          {/* Filter and Control Bar */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-500">
                <Filter className="w-3.5 h-3.5" /> Filter by:
              </div>

              {/* Status Filter */}
              <select
                value={statusFilter}
                onChange={(e) => {
                  setStatusFilter(e.target.value);
                  setCurrentPage(0);
                }}
                className="text-xs bg-slate-50 border border-slate-300 rounded-lg py-1.5 px-3 text-slate-700 font-medium focus:ring-1 focus:ring-blue-500 focus:outline-none"
              >
                <option value="all">All Statuses</option>
                <option value="succeeded">Succeeded Only</option>
                <option value="failed">Failed Only</option>
                <option value="pending">Pending Only</option>
              </select>

              {/* Model Filter */}
              <select
                value={modelFilter}
                onChange={(e) => {
                  setModelFilter(e.target.value);
                  setCurrentPage(0);
                }}
                className="text-xs bg-slate-50 border border-slate-300 rounded-lg py-1.5 px-3 text-slate-700 font-medium focus:ring-1 focus:ring-blue-500 focus:outline-none"
              >
                <option value="all">All Models</option>
                <option value="early_fusion">Early Fusion</option>
                <option value="energy_only">Energy-only</option>
                <option value="late_fusion">Late Fusion</option>
                <option value="hybrid_fusion">Hybrid Fusion</option>
              </select>
            </div>

            <span className="text-xs text-slate-500 font-medium">
              Found <strong className="text-slate-800">{totalCount}</strong> recorded forecast run{totalCount === 1 ? '' : 's'}
            </span>
          </div>

          {/* Hourly State Container */}
          {loading ? (
            <div className="bg-white border border-slate-200 rounded-xl p-12 shadow-sm">
              <LoadingState message="Loading persistent forecast records..." />
            </div>
          ) : error ? (
            <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
              <ErrorState title="Failed to Load History" message={error} onRetry={fetchHourlyRecords} />
            </div>
          ) : records.length === 0 ? (
            <div className="bg-white border border-slate-200 rounded-xl p-12 text-center shadow-sm space-y-4">
              <div className="w-12 h-12 rounded-full bg-slate-100 text-slate-400 mx-auto flex items-center justify-center">
                <History className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-slate-900">No Forecast Runs Found</h3>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  {statusFilter !== 'all' || modelFilter !== 'all'
                    ? 'No forecast runs match your selected filter criteria. Try resetting filters.'
                    : "You haven't run any 24-hour electricity forecasts yet. Upload a 168-hour CSV file or enter power readings to generate your first forecast."}
                </p>
              </div>
              <Link
                to="/forecast/new"
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 text-white text-xs font-semibold hover:bg-blue-700 shadow-xs transition-colors"
              >
                <PlusCircle className="w-4 h-4" />
                Create Forecast Run
              </Link>
            </div>
          ) : (
            <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-bold uppercase tracking-wider text-[11px]">
                    <tr>
                      <th className="py-3 px-4">Run Identifier</th>
                      <th className="py-3 px-4">Execution Time</th>
                      <th className="py-3 px-4">Architecture</th>
                      <th className="py-3 px-4">Ingestion</th>
                      <th className="py-3 px-4">Status</th>
                      <th className="py-3 px-4 text-right">Mean Bound (±W)</th>
                      <th className="py-3 px-4 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {records.map((r) => {
                      let statusBadge = (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 font-sans">
                          <CheckCircle2 className="w-3 h-3 text-emerald-600" /> Succeeded
                        </span>
                      );
                      if (r.status === 'failed') {
                        statusBadge = (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-200 font-sans">
                            <AlertTriangle className="w-3 h-3 text-rose-600" /> Failed
                          </span>
                        );
                      } else if (r.status === 'pending') {
                        statusBadge = (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200 font-sans">
                            <Clock className="w-3 h-3 text-amber-600" /> Pending
                          </span>
                        );
                      }

                      const createdDate = new Date(r.created_at);
                      const formattedDate = `${createdDate.toLocaleDateString([], {
                        month: 'short',
                        day: 'numeric',
                      })} ${createdDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

                      return (
                        <tr key={r.run_id} className="hover:bg-slate-50/80 transition-colors">
                          <td className="py-3 px-4">
                            <Link
                              to={`/forecast/${r.run_id}`}
                              className="font-bold text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1"
                              title="View this forecast"
                            >
                              <Search className="w-3 h-3 text-slate-400" />
                              {r.run_id}
                            </Link>
                          </td>
                          <td className="py-3 px-4 text-slate-600 font-sans">{formattedDate}</td>
                          <td className="py-3 px-4 font-sans">
                            <span className="inline-block px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-800">
                              {r.selected_model || 'Early Fusion'}
                            </span>
                          </td>
                          <td className="py-3 px-4 font-sans">
                            <span className="inline-flex items-center gap-1 text-[11px] text-slate-600">
                              {r.input_method === 'csv' ? (
                                <FileSpreadsheet className="w-3 h-3 text-emerald-600" />
                              ) : (
                                <Code className="w-3 h-3 text-blue-600" />
                              )}
                              {r.input_method.toUpperCase()}
                            </span>
                          </td>
                          <td className="py-3 px-4">{statusBadge}</td>
                          <td className="py-3 px-4 text-right text-slate-700">
                            {r.mean_interval_width_watts
                              ? `±${Math.round(r.mean_interval_width_watts / 2).toLocaleString()} W`
                              : r.status === 'failed'
                              ? <span className="text-rose-500 font-sans text-[11px]">{r.error_code || 'Error'}</span>
                              : '-'}
                          </td>
                          <td className="py-3 px-4 text-right font-sans">
                            <Link
                              to={`/forecast/${r.run_id}`}
                              className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-semibold rounded bg-slate-100 text-slate-700 hover:bg-blue-50 hover:text-blue-700 transition-colors"
                            >
                              {r.status === 'succeeded' ? 'View Result' : 'Details'}
                              <ArrowRight className="w-3 h-3" />
                            </Link>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Pagination Controls */}
              {totalPages > 1 && (
                <div className="px-6 py-4 border-t border-slate-200 bg-slate-50/50 flex items-center justify-between text-xs">
                  <span className="text-slate-500">
                    Showing{' '}
                    <strong className="text-slate-700">{currentPage * PAGE_SIZE + 1}</strong> to{' '}
                    <strong className="text-slate-700">
                      {Math.min((currentPage + 1) * PAGE_SIZE, totalCount)}
                    </strong>{' '}
                    of <strong className="text-slate-700">{totalCount}</strong> records
                  </span>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={currentPage === 0}
                      onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                      className="px-3 py-1.5 rounded border border-slate-300 bg-white text-slate-700 font-medium hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-white"
                    >
                      Previous
                    </button>
                    <span className="px-2 text-slate-500">
                      Page {currentPage + 1} of {totalPages}
                    </span>
                    <button
                      type="button"
                      disabled={currentPage >= totalPages - 1}
                      onClick={() => setCurrentPage((p) => Math.min(totalPages - 1, p + 1))}
                      className="px-3 py-1.5 rounded border border-slate-300 bg-white text-slate-700 font-medium hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-white"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Persistence Guarantee Note */}
      <div className="p-4 rounded-lg bg-slate-100 border border-slate-200 text-xs text-slate-500 flex items-start gap-2.5">
        <ShieldAlert className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
        <div>
          <strong className="text-slate-700">Privacy & Persistence Design:</strong> Forecast records are durably saved to the local SQLite database. In accordance with household data protection best practices, raw 168-hour input time series are discarded after inference, while monthly bill forecasts retain immutable input snapshot records for auditability.
        </div>
      </div>
    </div>
  );
};

export default ForecastHistory;
