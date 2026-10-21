import React from 'react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ message = 'Loading research data...' }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 bg-white dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800">
      <div className="w-8 h-8 border-3 border-slate-200 dark:border-slate-700 border-t-blue-600 dark:border-t-blue-400 rounded-full animate-spin mb-4" />
      <p className="text-sm font-medium text-slate-600 dark:text-slate-300">{message}</p>
      <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">Connecting to FastAPI backend at 127.0.0.1:8000</p>
    </div>
  );
};
