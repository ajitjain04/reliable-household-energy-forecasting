import React, { useEffect, useState, useMemo } from 'react';
import { api } from '../api/client';
import {
  StatisticalSummaryResponse,
  FamilyDetailResponse,
  HypothesisFamilySummary,
  GranularTestRecord,
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
  Scale,
  Database,
  CheckCircle2,
  AlertCircle,
  BookOpen,
  Filter,
  BarChart2,
  ShieldCheck,
  ChevronRight,
} from 'lucide-react';

export const StatisticalTests: React.FC = () => {
  const [summaryData, setSummaryData] = useState<StatisticalSummaryResponse | null>(null);
  const [activeFamilyId, setActiveFamilyId] = useState<number>(1);
  const [familyDetail, setFamilyDetail] = useState<FamilyDetailResponse | null>(null);
  const [loadingSummary, setLoadingSummary] = useState<boolean>(true);
  const [loadingFamily, setLoadingFamily] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [familyError, setFamilyError] = useState<string | null>(null);

  // Filter within selected family
  const [sigFilter, setSigFilter] = useState<'all' | 'sig' | 'nonsig'>('all');

  // Load high-level summary
  const loadSummary = async () => {
    setLoadingSummary(true);
    setError(null);
    try {
      const summary = await api.getStatisticalSummary();
      setSummaryData(summary);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to fetch statistical testing summary.');
      }
    } finally {
      setLoadingSummary(false);
    }
  };

  // Load selected family details
  const loadFamilyDetail = async (id: number) => {
    setLoadingFamily(true);
    setFamilyError(null);
    try {
      const detail = await api.getStatisticalFamilyDetail(id);
      setFamilyDetail(detail);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setFamilyError(err.message);
      } else {
        setFamilyError(`Failed to load details for Hypothesis Family ${id}.`);
      }
    } finally {
      setLoadingFamily(false);
    }
  };

  useEffect(() => {
    loadSummary();
  }, []);

  useEffect(() => {
    if (activeFamilyId) {
      loadFamilyDetail(activeFamilyId);
    }
  }, [activeFamilyId]);

  // Chart data: 6 Families planned vs Holm-significant
  const familyOverviewChartData = useMemo(() => {
    if (!summaryData?.families) return [];
    return summaryData.families.map((f: HypothesisFamilySummary) => ({
      name: `Fam ${f.family_id}`,
      fullName: f.family_name,
      'Planned Tests': f.total_tests,
      'Holm Significant': f.significant_tests_holm,
      'FDR Significant': f.significant_tests_fdr,
    }));
  }, [summaryData]);

  // Filtered granular tests for active family
  const filteredGranularTests = useMemo(() => {
    if (!familyDetail?.granular_tests) return [];
    if (sigFilter === 'sig') {
      return familyDetail.granular_tests.filter((t) => t.sig_holm);
    }
    if (sigFilter === 'nonsig') {
      return familyDetail.granular_tests.filter((t) => !t.sig_holm);
    }
    return familyDetail.granular_tests;
  }, [familyDetail, sigFilter]);

  // Format very small p-values cleanly without displaying 0.0000
  const formatPValue = (p: number | null | undefined): string => {
    if (p === null || p === undefined) return 'N/A';
    if (p === 0) return '< 1.00e-100';
    if (p < 0.0001) {
      return p.toExponential(3);
    }
    return p.toFixed(4);
  };

  if (loadingSummary) {
    return <LoadingState message="Loading 78-test statistical hypothesis testing battery..." />;
  }

  if (error || !summaryData) {
    return (
      <ErrorState
        title="Unable to load statistical hypothesis data"
        message={error || 'No statistical summary received from API.'}
        onRetry={loadSummary}
      />
    );
  }

  return (
    <div className="space-y-8">
      {/* Title & Research Mandate */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-indigo-600 mb-1">
          <Database className="w-3.5 h-3.5" />
          Formal Statistical Testing Battery • Source: {summaryData.source_artifact}
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          Statistical Hypothesis Tests & Multiple-Testing Control
        </h1>
        <p className="text-sm text-slate-600 mt-2 max-w-4xl leading-relaxed">
          Exhaustive pre-registered inferential battery comprising <strong className="text-slate-800">78 planned pairwise tests</strong> across{' '}
          <strong className="text-slate-800">6 hypothesis families</strong> on 587 rolling test windows. We employ paired two-sided Wilcoxon
          signed-rank tests, rank-biserial effect sizes ($r_{'{rb}'}$), 95% paired bootstrap confidence intervals, and Harvey-Leybourne-Newbery
          (HLN) modified Diebold-Mariano tests with Newey-West HAC adjustments. Family-Wise Error Rate is strictly controlled via{' '}
          <strong className="text-slate-800">Holm-Bonferroni step-down</strong> ($\alpha = 0.05$).
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Planned Hypothesis Tests"
          value={summaryData.total_planned_tests}
          unit="Tests"
          subtitle="Pre-registered comparisons"
          badge={{ text: `${summaryData.total_families} Families`, variant: 'info' }}
          icon={<Scale className="w-4 h-4 text-blue-600" />}
        />
        <MetricCard
          title="Holm-Significant Tests"
          value={summaryData.total_significant_holm}
          unit={`/ ${summaryData.total_planned_tests}`}
          subtitle="Family-Wise Error Rate (α=0.05)"
          badge={{ text: 'FWER Controlled', variant: 'success' }}
          icon={<ShieldCheck className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="Overall Rejection Rate"
          value={`${summaryData.overall_rejection_rate_pct.toFixed(1)}%`}
          subtitle="60 of 78 null hypotheses rejected"
          badge={{ text: 'High Statistical Power', variant: 'success' }}
          icon={<CheckCircle2 className="w-4 h-4 text-emerald-600" />}
        />
        <MetricCard
          title="FDR Significance"
          value={summaryData.families.reduce((acc, f) => acc + f.significant_tests_fdr, 0)}
          unit={`/ ${summaryData.total_planned_tests}`}
          subtitle="Benjamini-Hochberg (q=0.05)"
          badge={{ text: 'FDR Controlled', variant: 'neutral' }}
          icon={<BarChart2 className="w-4 h-4 text-indigo-600" />}
        />
      </div>

      {/* Multiple-Testing Correction Methods Card */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-800 uppercase">
            <span className="w-2 h-2 rounded-full bg-blue-600"></span>
            Holm-Bonferroni Correction
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Strictly controls Family-Wise Error Rate ($FWER \le 0.05$) via sequentially rejective Bonferroni adjustments.
            Serves as the primary definitive criterion for rejecting null hypotheses.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-800 uppercase">
            <span className="w-2 h-2 rounded-full bg-indigo-600"></span>
            Benjamini-Hochberg FDR
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Controls False Discovery Rate ($q \le 0.05$) to balance type I error protection against statistical power.
            Recorded alongside Holm for sensitivity assessment.
          </p>
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-2">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-800 uppercase">
            <span className="w-2 h-2 rounded-full bg-emerald-600"></span>
            Diebold-Mariano (HLN HAC)
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Modified Harvey-Leybourne-Newbery DM test with Newey-West Heteroskedasticity & Autocorrelation Consistent (HAC)
            variance adjustment for multi-step rolling residuals.
          </p>
        </div>
      </div>

      {/* Across-Family Overview Chart */}
      <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm">
        <div className="mb-4">
          <h2 className="text-base font-semibold text-slate-900">
            Planned vs Significant Tests Across the 6 Hypothesis Families
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) rejection counts across families
          </p>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={familyOverviewChartData}
              margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={12} tickLine={false} domain={[0, 20]} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-white border border-slate-200 p-3 rounded-lg shadow-sm text-xs space-y-1">
                        <div className="font-semibold text-slate-900">{d.fullName}</div>
                        <div className="text-slate-600">Planned Tests: <span className="font-mono font-bold">{d['Planned Tests']}</span></div>
                        <div className="text-emerald-700">Holm Significant: <span className="font-mono font-bold">{d['Holm Significant']}</span></div>
                        <div className="text-indigo-700">FDR Significant: <span className="font-mono font-bold">{d['FDR Significant']}</span></div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ fontSize: '0.75rem', paddingTop: '10px' }} />
              <Bar dataKey="Planned Tests" fill="#cbd5e1" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Holm Significant" fill="#10b981" radius={[3, 3, 0, 0]} />
              <Bar dataKey="FDR Significant" fill="#6366f1" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Family Selector & Granular Explorer */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
          <Filter className="w-3.5 h-3.5 text-blue-600" />
          Interactive Hypothesis Family Explorer
        </div>

        {/* 6 Family Tabs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {summaryData.families.map((f: HypothesisFamilySummary) => {
            const isSelected = f.family_id === activeFamilyId;
            return (
              <button
                key={f.family_id}
                type="button"
                onClick={() => setActiveFamilyId(f.family_id)}
                className={`text-left p-4 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-blue-50/70 border-blue-500 shadow-sm ring-1 ring-blue-500'
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/80'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span
                    className={`text-xs font-bold font-mono px-2 py-0.5 rounded ${
                      isSelected ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700'
                    }`}
                  >
                    Family {f.family_id}
                  </span>
                  <span className="text-[11px] font-mono text-slate-500">
                    {f.significant_tests_holm} / {f.total_tests} Sig
                  </span>
                </div>
                <div className="text-xs font-semibold text-slate-900 line-clamp-1">
                  {f.family_name.replace(`Family ${f.family_id}: `, '')}
                </div>
                <div className="text-[11px] text-slate-500 mt-1 flex items-center justify-between">
                  <span>Holm Rejection: {((f.significant_tests_holm / f.total_tests) * 100).toFixed(0)}%</span>
                  <ChevronRight className={`w-3 h-3 ${isSelected ? 'text-blue-600' : 'text-slate-400'}`} />
                </div>
              </button>
            );
          })}
        </div>

        {/* Selected Family Details Container */}
        {loadingFamily ? (
          <LoadingState message={`Fetching granular test battery for Family ${activeFamilyId}...`} />
        ) : familyError ? (
          <ErrorState
            title="Failed to load family tests"
            message={familyError}
            onRetry={() => loadFamilyDetail(activeFamilyId)}
          />
        ) : familyDetail ? (
          <div className="space-y-6">
            {/* Primary Inference Synthesis Card */}
            <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-800 uppercase tracking-wide">
                <BookOpen className="w-4 h-4 text-blue-600" />
                Audited Inferential Synthesis Narrative • {familyDetail.family_name}
              </div>
              <p className="text-xs text-slate-700 leading-relaxed bg-slate-50 p-4 rounded-md border border-slate-200/80 font-mono">
                {familyDetail.primary_inference}
              </p>
              <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500 pt-1">
                <div>Total Tests in Family: <strong className="text-slate-800">{familyDetail.total_tests}</strong></div>
                <div>Significant under Holm (α=0.05): <strong className="text-emerald-700">{familyDetail.significant_tests_holm}</strong></div>
                <div>Rejection Rate: <strong className="text-slate-800">{((familyDetail.significant_tests_holm / familyDetail.total_tests) * 100).toFixed(1)}%</strong></div>
              </div>
            </div>

            {/* Granular Pairwise Tests Table */}
            <div className="bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
              <div className="px-6 py-4 border-b border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                  <h3 className="text-base font-semibold text-slate-900">
                    Granular Pairwise Tests ({familyDetail.granular_tests.length} Planned Comparisons)
                  </h3>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Evaluated across 587 rolling test windows (sample size N = 587)
                  </p>
                </div>

                {/* Filter by Significance */}
                <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg text-xs font-medium text-slate-600">
                  <button
                    onClick={() => setSigFilter('all')}
                    className={`px-2.5 py-1 rounded-md transition-colors ${
                      sigFilter === 'all' ? 'bg-white text-blue-700 shadow-sm font-semibold' : 'hover:text-slate-900'
                    }`}
                  >
                    All ({familyDetail.granular_tests.length})
                  </button>
                  <button
                    onClick={() => setSigFilter('sig')}
                    className={`px-2.5 py-1 rounded-md transition-colors ${
                      sigFilter === 'sig' ? 'bg-white text-emerald-700 shadow-sm font-semibold' : 'hover:text-slate-900'
                    }`}
                  >
                    Significant ({familyDetail.granular_tests.filter((t) => t.sig_holm).length})
                  </button>
                  <button
                    onClick={() => setSigFilter('nonsig')}
                    className={`px-2.5 py-1 rounded-md transition-colors ${
                      sigFilter === 'nonsig' ? 'bg-white text-slate-700 shadow-sm font-semibold' : 'hover:text-slate-900'
                    }`}
                  >
                    Non-Sig ({familyDetail.granular_tests.filter((t) => !t.sig_holm).length})
                  </button>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-slate-600">
                  <thead className="bg-slate-50 text-xs uppercase font-semibold text-slate-500 border-b border-slate-200">
                    <tr>
                      <th className="px-5 py-3">ID</th>
                      <th className="px-5 py-3">Comparison</th>
                      <th className="px-5 py-3 text-right">Mean Diff (W)</th>
                      <th className="px-5 py-3 text-right">95% CI (Bootstrap)</th>
                      <th className="px-5 py-3 text-right">Effect (r_rb)</th>
                      <th className="px-5 py-3 text-right">Wilcoxon W</th>
                      <th className="px-5 py-3 text-right">p_raw</th>
                      <th className="px-5 py-3 text-right font-bold text-slate-800">p_holm</th>
                      <th className="px-5 py-3 text-right">HLN DM Stat</th>
                      <th className="px-5 py-3 text-center">Decision</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 text-xs">
                    {filteredGranularTests.map((t: GranularTestRecord) => {
                      return (
                        <tr
                          key={t.test_id}
                          className={`hover:bg-slate-50/80 transition-colors ${
                            t.sig_holm ? '' : 'bg-slate-50/30'
                          }`}
                        >
                          <td className="px-5 py-3.5 font-mono text-slate-500 font-medium">
                            #{t.test_id}
                          </td>
                          <td className="px-5 py-3.5">
                            <div className="font-medium text-slate-900">{t.comparison_name}</div>
                            <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                              {t.model_1} ({t.cond_1}) vs {t.model_2} ({t.cond_2})
                            </div>
                          </td>
                          <td className="px-5 py-3.5 text-right font-mono font-medium text-slate-800">
                            {t.mean_diff_watts > 0 ? '+' : ''}
                            {t.mean_diff_watts.toFixed(2)} W
                          </td>
                          <td className="px-5 py-3.5 text-right font-mono text-slate-500 text-[11px]">
                            [{t.mean_ci_95_low.toFixed(2)}, {t.mean_ci_95_high.toFixed(2)}]
                          </td>
                          <td className="px-5 py-3.5 text-right font-mono text-slate-700">
                            {t.rank_biserial_r_rb.toFixed(3)}
                          </td>
                          <td className="px-5 py-3.5 text-right font-mono text-slate-500">
                            {t.wilcoxon_stat.toLocaleString()}
                          </td>
                          <td className="px-5 py-3.5 text-right font-mono text-slate-500" title={String(t.wilcoxon_p_raw)}>
                            {formatPValue(t.wilcoxon_p_raw)}
                          </td>
                          <td
                            className={`px-5 py-3.5 text-right font-mono font-bold ${
                              t.sig_holm ? 'text-emerald-700' : 'text-slate-500'
                            }`}
                            title={String(t.wilcoxon_p_holm)}
                          >
                            {formatPValue(t.wilcoxon_p_holm)}
                          </td>
                          <td className="px-5 py-3.5 text-right font-mono text-slate-500">
                            {t.hln_stat !== null ? t.hln_stat.toFixed(3) : '—'}
                          </td>
                          <td className="px-5 py-3.5 text-center">
                            <span
                              className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                                t.sig_holm
                                  ? 'bg-emerald-100 text-emerald-800'
                                  : 'bg-slate-100 text-slate-600'
                              }`}
                            >
                              {t.sig_holm ? 'Reject H0' : 'Fail to Reject'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Scientific Cautionary Interpretation */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-5 space-y-2 text-xs text-slate-600 leading-relaxed">
        <div className="flex items-center gap-2 font-semibold text-slate-800">
          <AlertCircle className="w-4 h-4 text-slate-500" />
          Methodological Interpretation Guidelines
        </div>
        <p>
          Formal rejection of $H_0$ under Holm-Bonferroni correction demonstrates that the paired out-of-sample error difference
          is statistically distinguishable from zero given 587 rolling forecast windows. However, statistical significance must not be
          conflated with practical or operational engineering superiority. For instance, in Family 1, Early Fusion achieves a highly
          significant reduction in MAE vs Energy-Only ($p_{'{holm}'} &lt; 10^{'-39'}$, effect size $r_{'{rb}'} = -0.639$), whereas
          Late Fusion and Hybrid Fusion exhibit significantly higher clean error than Energy-Only, demonstrating the risk of cross-modal parameter dilution.
        </p>
      </div>
    </div>
  );
};
