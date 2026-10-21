import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Terminal,
  Eye,
  Filter,
  CheckCircle2,
  AlertTriangle,
  Clock,
  RefreshCw,
  Search,
  FileSpreadsheet,
  Code,
  ShieldCheck,
  Cpu,
  BarChart2,
  PlusCircle,
  ExternalLink,
  Layers,
} from 'lucide-react';
import { api } from '../api/client';
import { ForecastRecordSummary } from '../api/types';
import { useAuth } from '../context/AuthContext';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';

const PAGE_SIZE = 10;

export const DeveloperForecastRecords: React.FC = () => {
  const { user } = useAuth();
  const [records, setRecords] = useState<ForecastRecordSummary[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(0);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [modelFilter, setModelFilter] = useState<string>('all');
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRecords = async () => {
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

      const res = await api.getDeveloperForecastRecords(params);
      setRecords(res.records);
      setTotalCount(res.total_count);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve forecast records from backend.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, [currentPage, statusFilter, modelFilter]);

  const totalPages = Math.ceil(totalCount / PAGE_SIZE);

  // Calculate quick stats from current page records
  const succeededCount = records.filter((r) => r.status === 'succeeded').length;
  const failedCount = records.filter((r) => r.status === 'failed').length;

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Developer Diagnostic Environment & Verified RBAC Access Control Banner */}
      <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-950 flex items-start gap-3 text-xs">
        <ShieldCheck className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
        <div className="space-y-1 flex-1">
          <div className="font-bold text-emerald-950 flex flex-wrap items-center justify-between gap-2">
            <span className="flex items-center gap-2">
              <span>Developer Console — Operational Run Inspector</span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-emerald-200/90 text-emerald-900 font-bold border border-emerald-300">
                RBAC Enforced • Developer Authorized
              </span>
            </span>
            <span className="text-[11px] font-mono text-emerald-800 font-medium">
              Authenticated: <strong>{user?.username || 'developer'}</strong> ({user?.role || 'developer'})
            </span>
          </div>
          <p className="text-emerald-800 leading-relaxed">
            Role-based access control (RBAC) is strictly enforced for this inspection console.
            The backend verifies cryptographic HS256 JWT bearer tokens via the <code className="px-1 py-0.5 rounded bg-emerald-100 text-emerald-900 font-mono text-[11px]">require_developer</code> dependency, restricting platform-wide run inspection, model parameter counts, and persistent database telemetry exclusively to authenticated developer accounts.
          </p>
        </div>
      </div>

      {/* Page Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-100 border border-slate-300 text-slate-700 text-xs font-semibold uppercase tracking-wider mb-2">
              <Terminal className="w-3.5 h-3.5 text-slate-600" />
              Operational Telemetry & Records
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Forecast Run Inspector
            </h1>
            <p className="text-sm text-slate-600 mt-1">
              Audit, inspect, and verify persisted forecast execution runs, model architectures, inputs, and conformal intervals.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={fetchRecords}
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
              Trigger Run
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Diagnostic Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-semibold uppercase tracking-wider">Total Recorded Runs</span>
            <Layers className="w-4 h-4 text-blue-600" />
          </div>
          <div className="text-2xl font-bold text-slate-900 font-mono">{totalCount}</div>
          <div className="text-xs text-slate-500 mt-1">Durable SQLite database entries</div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-semibold uppercase tracking-wider">Successful Runs (Page)</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-2xl font-bold text-emerald-700 font-mono">{succeededCount}</div>
          <div className="text-xs text-slate-500 mt-1">Predictions & intervals persisted</div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-semibold uppercase tracking-wider">Failed Runs (Page)</span>
            <AlertTriangle className="w-4 h-4 text-rose-600" />
          </div>
          <div className="text-2xl font-bold text-rose-700 font-mono">{failedCount}</div>
          <div className="text-xs text-slate-500 mt-1">Captured with sanitized error codes</div>
        </div>

        <div className="bg-white rounded-lg border border-slate-200 p-4 shadow-sm">
          <div className="flex items-center justify-between text-slate-500 mb-1">
            <span className="text-xs font-semibold uppercase tracking-wider">Active Execution Runtime</span>
            <Cpu className="w-4 h-4 text-purple-600" />
          </div>
          <div className="text-sm font-bold text-slate-800 font-mono mt-1">PyTorch CPU</div>
          <div className="text-xs text-slate-500 mt-1">4 Model Architectures Loaded</div>
        </div>
      </div>

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
            <option value="succeeded">Succeeded</option>
            <option value="failed">Failed</option>
            <option value="pending">Pending</option>
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
            <option value="Early Fusion">Early Fusion</option>
            <option value="Energy-only">Energy-only</option>
            <option value="Late Fusion">Late Fusion</option>
            <option value="Hybrid Fusion">Hybrid Fusion</option>
          </select>
        </div>

        <div className="text-xs text-slate-500 font-mono">
          Showing <strong className="text-slate-800">{records.length}</strong> of{' '}
          <strong className="text-slate-800">{totalCount}</strong> matching runs
        </div>
      </div>

      {/* Content Area */}
      {loading ? (
        <LoadingState message="Querying persistent forecast records from SQLite database..." />
      ) : error ? (
        <ErrorState
          title="Failed to Load Forecast Records"
          message={error}
          onRetry={fetchRecords}
        />
      ) : records.length === 0 ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center space-y-4 shadow-sm">
          <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center mx-auto text-slate-400">
            <Terminal className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-800">No Forecast Records in Database</h3>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            {statusFilter !== 'all' || modelFilter !== 'all'
              ? 'No persisted records match the active filter criteria. Try resetting status or model filters.'
              : 'The database currently contains zero forecast executions. Execute an inference run to generate inspection telemetry.'}
          </p>
          <div className="pt-2">
            <Link
              to="/forecast/new"
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
            >
              <PlusCircle className="w-4 h-4" /> Trigger New Run
            </Link>
          </div>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-100/75 border-b border-slate-200 text-slate-600 uppercase tracking-wider font-semibold">
                  <th className="py-3 px-4">Run Identifier (UUID)</th>
                  <th className="py-3 px-4">Timestamp (UTC)</th>
                  <th className="py-3 px-4">Pseudonymous User ID</th>
                  <th className="py-3 px-4">Ingestion</th>
                  <th className="py-3 px-4">Architecture</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4 text-right">Avg Uncertainty Margin</th>
                  <th className="py-3 px-4 text-right">Inspection Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {records.map((r) => {
                  let statusBadge = (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 font-sans">
                      <CheckCircle2 className="w-3 h-3 text-emerald-600" /> succeeded
                    </span>
                  );
                  if (r.status === 'failed') {
                    statusBadge = (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-rose-50 text-rose-700 border border-rose-200 font-sans">
                        <AlertTriangle className="w-3 h-3 text-rose-600" /> failed
                      </span>
                    );
                  } else if (r.status === 'pending') {
                    statusBadge = (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-amber-50 text-amber-700 border border-amber-200 font-sans">
                        <Clock className="w-3 h-3 text-amber-600" /> pending
                      </span>
                    );
                  }

                  const createdDate = new Date(r.created_at);
                  const formattedDate = `${createdDate.toLocaleDateString([], {
                    month: 'short',
                    day: 'numeric',
                  })} ${createdDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;

                  return (
                    <tr key={r.run_id} className="hover:bg-slate-50/80 transition-colors">
                      <td className="py-3 px-4">
                        <Link
                          to={`/developer/forecast-records/${r.run_id}`}
                          className="font-bold text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1.5"
                          title="Inspect run telemetry"
                        >
                          <Search className="w-3.5 h-3.5 text-slate-400" />
                          <span>{r.run_id}</span>
                        </Link>
                      </td>
                      <td className="py-3 px-4 text-slate-600 font-sans">{formattedDate}</td>
                      <td className="py-3 px-4 font-sans">
                        {r.user_id ? (
                          <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200">
                            {r.user_id}
                          </span>
                        ) : (
                          <span className="text-slate-400 italic text-[11px]">Anonymous</span>
                        )}
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <span className="inline-flex items-center gap-1 text-[11px] text-slate-700 font-mono font-bold uppercase bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                          {r.input_method === 'csv' ? (
                            <FileSpreadsheet className="w-3 h-3 text-emerald-600" />
                          ) : (
                            <Code className="w-3 h-3 text-blue-600" />
                          )}
                          {r.input_method}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <span className="inline-block px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-800">
                          {r.selected_model || '—'}
                        </span>
                      </td>
                      <td className="py-3 px-4">{statusBadge}</td>
                      <td className="py-3 px-4 text-right text-slate-700">
                        {r.mean_interval_width_watts
                          ? `±${Math.round(r.mean_interval_width_watts / 2).toLocaleString()} W`
                          : r.status === 'failed'
                          ? <span className="text-rose-600 font-sans text-[11px]">{r.error_code || 'Error'}</span>
                          : '-'}
                      </td>
                      <td className="py-3 px-4 text-right font-sans">
                        <Link
                          to={`/developer/forecast-records/${r.run_id}`}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-200 transition-colors shadow-xs"
                          title="Open detailed developer inspection"
                        >
                          <Eye className="w-3.5 h-3.5 text-blue-600" />
                          Inspect Run
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

      {/* Scientific Research Cross-Reference Links */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-700">
            <BarChart2 className="w-4 h-4 text-blue-600" />
            Empirical Benchmark Research Suite (Step 10 Offline Findings)
          </div>
          <span className="text-[11px] text-slate-500 font-medium">Population-Level Evaluations</span>
        </div>
        <p className="text-xs text-slate-600 leading-relaxed">
          The records above represent individual live inference runs. To inspect population-level scientific evaluations calibrated on the 587 rolling test windows from the Step 10 study, access the dedicated research modules:
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 pt-1 text-xs font-medium">
          <Link
            to="/benchmarks"
            className="p-2.5 rounded-lg bg-white border border-slate-200 hover:border-blue-400 text-slate-700 hover:text-blue-700 transition-colors flex items-center justify-between shadow-xs"
          >
            <span>Benchmarks</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </Link>
          <Link
            to="/reliability"
            className="p-2.5 rounded-lg bg-white border border-slate-200 hover:border-blue-400 text-slate-700 hover:text-blue-700 transition-colors flex items-center justify-between shadow-xs"
          >
            <span>Reliability</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </Link>
          <Link
            to="/statistical-tests"
            className="p-2.5 rounded-lg bg-white border border-slate-200 hover:border-blue-400 text-slate-700 hover:text-blue-700 transition-colors flex items-center justify-between shadow-xs"
          >
            <span>Statistics</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </Link>
          <Link
            to="/xai"
            className="p-2.5 rounded-lg bg-white border border-slate-200 hover:border-blue-400 text-slate-700 hover:text-blue-700 transition-colors flex items-center justify-between shadow-xs"
          >
            <span>Explainable AI</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </Link>
          <Link
            to="/uncertainty"
            className="p-2.5 rounded-lg bg-white border border-slate-200 hover:border-blue-400 text-slate-700 hover:text-blue-700 transition-colors flex items-center justify-between shadow-xs"
          >
            <span>Uncertainty</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </Link>
          <Link
            to="/explorer"
            className="p-2.5 rounded-lg bg-white border border-slate-200 hover:border-blue-400 text-slate-700 hover:text-blue-700 transition-colors flex items-center justify-between shadow-xs"
          >
            <span>Explorer</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default DeveloperForecastRecords;
