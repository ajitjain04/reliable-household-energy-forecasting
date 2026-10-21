import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Zap,
  ArrowRight,
  UploadCloud,
  History,
  ShieldCheck,
  Clock,
  TrendingUp,
  AlertCircle,
  FileText,
  BarChart2,
  Calendar,
  CheckCircle2,
  PlusCircle,
} from 'lucide-react';
import { api } from '../api/client';
import { ForecastRecordSummary } from '../api/types';
import { useAuth } from '../context/AuthContext';

export const Home: React.FC = () => {
  const { isAuthenticated, openAuthModal } = useAuth();
  const [recentRecords, setRecentRecords] = useState<ForecastRecordSummary[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    if (!isAuthenticated) {
      setRecentRecords([]);
      setTotalCount(0);
      setLoading(false);
      setError(null);
      return;
    }

    const fetchRecent = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await api.getForecastRecords({ limit: 5, offset: 0 });
        if (isMounted) {
          setRecentRecords(res.records);
          setTotalCount(res.total_count);
        }
      } catch (err: unknown) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch recent forecasts.');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchRecent();
    return () => {
      isMounted = false;
    };
  }, [isAuthenticated]);

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-navy-900 via-navy-800 to-blue-950 border border-navy-700/60 p-8 shadow-xl text-white">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-80 h-80 rounded-full bg-blue-500/10 blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 border border-blue-400/30 text-blue-300 text-xs font-semibold uppercase tracking-wider mb-4">
            <Zap className="w-3.5 h-3.5 text-blue-400" />
            Reliable Multimodal Forecasting
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
            Household Electricity Consumption Forecast
          </h1>
          <p className="mt-3 text-base sm:text-lg text-slate-300 leading-relaxed">
            Generate calibrated 24-hour ahead electricity demand forecasts from your smart meter history.
            Features statistically guaranteed 90% conformal prediction intervals and robust multimodal fusion.
          </p>

          <div className="mt-6 flex flex-wrap gap-4 items-center">
            {isAuthenticated ? (
              <Link
                to="/forecast/new"
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <UploadCloud className="w-4 h-4" />
                Upload & Generate Forecast
                <ArrowRight className="w-4 h-4 ml-1" />
              </Link>
            ) : (
              <button
                type="button"
                onClick={() => openAuthModal('/forecast/new')}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-sm shadow-lg shadow-blue-600/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                <UploadCloud className="w-4 h-4" />
                Upload & Generate Forecast
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>
            )}

            {isAuthenticated ? (
              <Link
                to="/forecast/history"
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-navy-800/80 hover:bg-navy-700 text-slate-200 hover:text-white border border-navy-600/60 font-semibold text-sm transition-all"
              >
                <History className="w-4 h-4 text-slate-400" />
                View Past Runs ({totalCount})
              </Link>
            ) : (
              <button
                type="button"
                onClick={() => openAuthModal('/forecast/history')}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-navy-800/80 hover:bg-navy-700 text-slate-200 hover:text-white border border-navy-600/60 font-semibold text-sm transition-all"
              >
                <History className="w-4 h-4 text-slate-400" />
                View Past Runs
              </button>
            )}

            <Link
              to="/overview"
              className="inline-flex items-center gap-2 px-4 py-3 rounded-xl text-slate-400 hover:text-blue-300 text-sm font-medium transition-colors"
            >
              <BarChart2 className="w-4 h-4" />
              Research & Benchmarks
            </Link>
          </div>
        </div>
      </div>

      {/* Feature Highlights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-slate-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600 mb-3">
            <Clock className="w-5 h-5" />
          </div>
          <h2 className="text-base font-bold text-slate-900">7-Day Lookback</h2>
          <p className="mt-1 text-xs text-slate-500 leading-relaxed">
            Requires exactly 168 consecutive hourly active power observations to model weekly diurnal patterns.
          </p>
        </div>

        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-slate-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600 mb-3">
            <TrendingUp className="w-5 h-5" />
          </div>
          <h2 className="text-base font-bold text-slate-900">24-Hour Horizon</h2>
          <p className="mt-1 text-xs text-slate-500 leading-relaxed">
            Multi-step forecast delivering hourly demand estimates in physical Watts (W) for the full day ahead.
          </p>
        </div>

        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-slate-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-purple-50 border border-purple-200 flex items-center justify-center text-purple-600 mb-3">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h2 className="text-base font-bold text-slate-900">90% Conformal Bounds</h2>
          <p className="mt-1 text-xs text-slate-500 leading-relaxed">
            Split conformal uncertainty intervals calibrated across 587 benchmark windows without distribution assumptions.
          </p>
        </div>

        <div className="p-5 rounded-xl bg-white border border-slate-200 shadow-sm hover:border-slate-300 transition-colors">
          <div className="w-10 h-10 rounded-lg bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-600 mb-3">
            <FileText className="w-5 h-5" />
          </div>
          <h2 className="text-base font-bold text-slate-900">Privacy-Preserving</h2>
          <p className="mt-1 text-xs text-slate-500 leading-relaxed">
            Raw smart meter histories are never logged or stored. Only aggregate metadata and forecast outputs are retained.
          </p>
        </div>
      </div>

      {/* Recent Activity Section */}
      <div className="rounded-xl bg-white border border-slate-200 p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 border-b border-slate-100 pb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <History className="w-4 h-4 text-blue-600" />
              Recent Forecast Runs
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Persistent forecast records retrieved from the local database
            </p>
          </div>
          {totalCount > 5 && (
            <Link
              to="/forecast/history"
              className="text-xs font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1 transition-colors"
            >
              View all ({totalCount}) <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          )}
        </div>

        {/* Unauthenticated Visitor State */}
        {!isAuthenticated && (
          <div className="py-10 text-center text-slate-500 max-w-lg mx-auto">
            <div className="w-12 h-12 rounded-2xl bg-blue-50 border border-blue-200 flex items-center justify-center mx-auto text-blue-600 mb-3">
              <Zap className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-sm font-bold text-slate-800">Household Forecasting Workflow</h3>
            <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">
              Sign in or create a household account to upload 168-hour smart meter readings, generate calibrated 24-hour ahead forecasts, and download verified PDF reports.
            </p>
            <div className="mt-5 flex items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => openAuthModal('/forecast/new')}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs shadow-sm transition-all"
              >
                <UploadCloud className="w-3.5 h-3.5" />
                Sign In to Generate Forecast
              </button>
              <Link
                to="/overview"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold text-xs transition-colors"
              >
                <BarChart2 className="w-3.5 h-3.5" />
                Explore Research Benchmarks
              </Link>
            </div>
          </div>
        )}

        {/* Loading State */}
        {isAuthenticated && loading && (
          <div className="py-12 flex flex-col items-center justify-center text-slate-500 gap-3">
            <div className="w-8 h-8 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
            <p className="text-xs font-medium">Loading saved forecast records...</p>
          </div>
        )}

        {/* Error State */}
        {isAuthenticated && !loading && error && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-center gap-3 text-xs">
            <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0" />
            <div>
              <p className="font-bold text-rose-900">Unable to load recent forecasts</p>
              <p className="text-rose-700 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Empty State */}
        {isAuthenticated && !loading && !error && recentRecords.length === 0 && (
          <div className="py-12 text-center text-slate-500">
            <div className="w-12 h-12 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center mx-auto text-slate-400 mb-3">
              <Calendar className="w-6 h-6" />
            </div>
            <h3 className="text-sm font-bold text-slate-800">No forecasts recorded yet</h3>
            <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
              You haven&apos;t run any forecasts yet. Upload your first 168-hour smart meter CSV to generate 24-hour predictions.
            </p>
            <div className="mt-4">
              <Link
                to="/forecast/new"
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium text-xs shadow-sm transition-colors"
              >
                <PlusCircle className="w-3.5 h-3.5" />
                Upload CSV File
              </Link>
            </div>
          </div>
        )}

        {/* Records Table */}
        {isAuthenticated && !loading && !error && recentRecords.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200 uppercase tracking-wider text-[11px]">
                <tr>
                  <th className="py-3 px-4">Run ID</th>
                  <th className="py-3 px-4">Created Time</th>
                  <th className="py-3 px-4">Method</th>
                  <th className="py-3 px-4">Model Used</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {recentRecords.map((r) => {
                  const isSuccess = r.status === 'succeeded';
                  return (
                    <tr key={r.run_id} className="hover:bg-slate-50 transition-colors">
                      <td className="py-3 px-4 font-bold text-blue-600">
                        <Link to={`/forecast/${r.run_id}`} className="hover:underline">
                          {r.run_id}
                        </Link>
                      </td>
                      <td className="py-3 px-4 text-slate-600 font-sans">
                        {r.created_at ? r.created_at.replace('T', ' ').slice(0, 19) : '—'}
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-slate-100 border border-slate-200 text-slate-700">
                          {r.input_method}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-800 font-medium font-sans">
                        {r.selected_model || '—'}
                      </td>
                      <td className="py-3 px-4 font-sans">
                        <span
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                            isSuccess
                              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                              : 'bg-rose-50 text-rose-700 border border-rose-200'
                          }`}
                        >
                          {isSuccess ? <CheckCircle2 className="w-3 h-3 text-emerald-600" /> : <AlertCircle className="w-3 h-3 text-rose-600" />}
                          {r.status}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right font-sans">
                        {isSuccess ? (
                          <Link
                            to={`/forecast/${r.run_id}`}
                            className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 font-semibold text-[11px] transition-colors"
                          >
                            View Result <ArrowRight className="w-3 h-3" />
                          </Link>
                        ) : (
                          <span
                            className="text-slate-400 text-[11px]"
                            title={r.error_message || 'Execution error'}
                          >
                            Failed
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;
