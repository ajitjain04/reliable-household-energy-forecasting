import React from 'react';

interface StatusBadgeProps {
  status: 'connected' | 'disconnected' | 'checking';
  label?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label }) => {
  if (status === 'connected') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
        {label || 'Backend Online'}
      </span>
    );
  }

  if (status === 'disconnected') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-rose-50 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800">
        <span className="h-1.5 w-1.5 rounded-full bg-rose-500" />
        {label || 'Backend Disconnected'}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800">
      <span className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-ping" />
      {label || 'Connecting...'}
    </span>
  );
};
