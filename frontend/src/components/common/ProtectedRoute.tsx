import React from 'react';
import { Navigate, useLocation, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { LoadingState } from './LoadingState';
import { LoginRequiredModal } from './LoginRequiredModal';
import { ShieldAlert, ArrowLeft, LogIn } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: 'developer' | 'household';
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole,
}) => {
  const { user, isAuthenticated, loading } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  if (loading) {
    return <LoadingState message="Verifying authentication session..." />;
  }

  if (!isAuthenticated) {
    if (requiredRole === 'developer') {
      return <Navigate to="/login" state={{ from: location }} replace />;
    }

    return (
      <div className="relative min-h-[50vh] flex items-center justify-center">
        <LoginRequiredModal
          isOpen={true}
          intendedDestination={location.pathname}
          onClose={() => {
            if (window.history.length > 1) {
              navigate(-1);
            } else {
              navigate('/', { replace: true });
            }
          }}
        />
      </div>
    );
  }

  if (requiredRole && user?.role !== requiredRole) {
    return (
      <div className="max-w-xl mx-auto my-12 p-8 bg-white border border-rose-200 rounded-2xl shadow-sm text-center">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-rose-50 text-rose-600 mb-4 border border-rose-200">
          <ShieldAlert className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-bold text-slate-900 mb-2">
          Authorized Role Required
        </h2>
        <p className="text-xs text-slate-600 mb-6 max-w-md mx-auto">
          The requested console page is restricted to accounts with the{' '}
          <strong className="text-slate-900 font-mono">'{requiredRole}'</strong> role.
          Your current account is authenticated as{' '}
          <strong className="text-slate-900 font-mono">'{user?.username}'</strong> with role{' '}
          <span className="font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
            {user?.role}
          </span>.
        </p>
        <div className="flex items-center justify-center gap-3">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-xs"
          >
            <ArrowLeft className="w-3.5 h-3.5" /> Return Home
          </Link>
          <Link
            to="/login"
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
          >
            <LogIn className="w-3.5 h-3.5" /> Sign In as Developer
          </Link>
        </div>
      </div>
    );
  }

  return <>{children}</>;
};
