import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  badge?: {
    text: string;
    variant?: 'success' | 'warning' | 'info' | 'neutral';
  };
  icon?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  badge,
  icon,
}) => {
  const badgeStyles = {
    success: 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
    warning: 'bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
    info: 'bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800',
    neutral: 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700',
  };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 p-5 shadow-xs hover:border-slate-300 dark:hover:border-slate-700 transition-colors">
      <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
        <span className="text-xs font-semibold tracking-wider uppercase text-slate-500 dark:text-slate-400">
          {title}
        </span>
        {icon && <span className="text-slate-400 dark:text-slate-500">{icon}</span>}
      </div>

      <div className="flex items-baseline gap-1.5 mb-1.5">
        <span className="text-2xl font-bold text-slate-900 dark:text-white tracking-tight">
          {value}
        </span>
        {unit && <span className="text-sm font-medium text-slate-500 dark:text-slate-400">{unit}</span>}
      </div>

      <div className="flex items-center justify-between text-xs">
        {subtitle && <span className="text-slate-500 dark:text-slate-400">{subtitle}</span>}
        {badge && (
          <span
            className={`inline-block px-2 py-0.5 rounded text-xs font-medium border ${
              badgeStyles[badge.variant || 'neutral']
            }`}
          >
            {badge.text}
          </span>
        )}
      </div>
    </div>
  );
};
