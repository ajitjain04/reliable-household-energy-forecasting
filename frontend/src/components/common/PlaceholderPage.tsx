import React from 'react';
import { Layers } from 'lucide-react';

interface PlaceholderPageProps {
  title: string;
  description: string;
  stepNumber: string;
}

export const PlaceholderPage: React.FC<PlaceholderPageProps> = ({
  title,
  description,
  stepNumber,
}) => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">{title}</h1>
        <p className="text-sm text-slate-500 mt-1">{description}</p>
      </div>

      <div className="bg-white rounded-lg border border-slate-200 p-12 text-center">
        <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-500">
          <Layers className="w-6 h-6" />
        </div>
        <h2 className="text-base font-semibold text-slate-800">Module Ready for Implementation</h2>
        <p className="text-sm text-slate-500 max-w-md mx-auto mt-2">
          This section is scaffolded in the router and ready for detailed analytical UI integration in {stepNumber}.
        </p>
        <div className="mt-6 inline-flex items-center gap-2 px-3 py-1 rounded bg-slate-100 text-xs font-mono text-slate-600">
          Status: Router Registered & Backend Endpoint Ready
        </div>
      </div>
    </div>
  );
};
