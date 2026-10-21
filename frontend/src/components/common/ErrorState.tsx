import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Failed to load research data',
  message,
  onRetry,
}) => {
  return (
    <div className="bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 rounded-lg p-6 my-4">
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-rose-900 dark:text-rose-200">{title}</h3>
          <p className="text-xs text-rose-700 dark:text-rose-300 mt-1 font-mono break-all">{message}</p>
          <p className="text-xs text-rose-600 dark:text-rose-400 mt-2">
            Verify that the FastAPI backend is running via <code className="bg-rose-100 dark:bg-rose-900 px-1 py-0.5 rounded text-rose-900 dark:text-rose-200">python backend/run_server.py</code> on port 8000.
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-4 inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-rose-700 dark:text-rose-200 bg-white dark:bg-slate-800 border border-rose-300 dark:border-rose-700 rounded hover:bg-rose-100 dark:hover:bg-slate-700 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Retry Connection
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
