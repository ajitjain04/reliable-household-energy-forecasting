import React, { useState } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  ShieldCheck,
  Lock,
  User,
  ArrowRight,
  AlertCircle,
  Zap,
  Terminal,
  CheckCircle2,
  Eye,
  EyeOff,
} from 'lucide-react';

export const Login: React.FC = () => {
  const { login, register, isAuthenticated, isDeveloper, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const isRegisterRoute = location.pathname === '/register' || (location.state as any)?.mode === 'register';
  const [mode, setMode] = useState<'login' | 'register'>(isRegisterRoute ? 'register' : 'login');
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [showPassword, setShowPassword] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Sync mode if navigation state changes
  React.useEffect(() => {
    if (location.pathname === '/register' || (location.state as any)?.mode === 'register') {
      setMode('register');
    }
  }, [location.pathname, location.state]);

  // Default redirect destination from location state or role
  const stateFrom = (location.state as any)?.from;
  const targetDestination = typeof stateFrom === 'string'
    ? stateFrom
    : stateFrom?.pathname || (isDeveloper ? '/developer/forecast-records' : '/');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMsg(null);

    if (!username.trim() || !password.trim()) {
      setError('Please provide both username and password.');
      return;
    }

    try {
      setLoading(true);
      if (mode === 'login') {
        await login(username.trim(), password);
        // Navigate to preserved destination
        navigate(targetDestination, { replace: true });
      } else {
        await register(username.trim(), password);
        setSuccessMsg('Account registered successfully! Redirecting...');
        setTimeout(() => {
          navigate(targetDestination, { replace: true });
        }, 1000);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Authentication failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto my-12">
      {/* Brand & Title */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-blue-600/10 text-blue-600 mb-4 border border-blue-200">
          <Zap className="w-6 h-6" />
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
          {mode === 'login' ? 'Sign In to Your Account' : 'Create Household Account'}
        </h1>
        <p className="text-xs text-slate-500 mt-1.5 max-w-sm mx-auto">
          {mode === 'login'
            ? 'Access your household electricity forecasts, download verified PDF reports, and view historical runs.'
            : 'Register your household to generate secure electricity forecasts and track durable runs.'}
        </p>
      </div>

      {/* Auth Card */}
      <div className="bg-white border border-slate-200 rounded-2xl p-7 shadow-sm">
        {/* Mode Switch Tabs */}
        <div className="flex border-b border-slate-200 mb-6 pb-2 gap-4">
          <button
            type="button"
            onClick={() => {
              setMode('login');
              setError(null);
              setShowPassword(false);
            }}
            className={`text-sm font-semibold pb-2 border-b-2 transition-colors ${
              mode === 'login'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => {
              setMode('register');
              setError(null);
              setShowPassword(false);
            }}
            className={`text-sm font-semibold pb-2 border-b-2 transition-colors ${
              mode === 'register'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            Register Household
          </button>
        </div>

        {/* Status Alerts */}
        {error && (
          <div className="mb-5 p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {successMsg && (
          <div className="mb-5 p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Already Logged In Banner */}
        {isAuthenticated && (
          <div className="mb-5 p-3.5 rounded-xl bg-blue-50 border border-blue-200 text-blue-800 text-xs">
            <p className="font-semibold">Currently signed in as:</p>
            <p className="font-mono mt-0.5">{user?.username} ({user?.role})</p>
            <div className="mt-2 flex gap-2">
              <Link
                to="/"
                className="px-2.5 py-1 bg-blue-600 text-white rounded text-[11px] font-medium hover:bg-blue-700"
              >
                Go to Dashboard
              </Link>
            </div>
          </div>
        )}

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Username
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                <User className="w-4 h-4" />
              </div>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. household_user"
                className="w-full pl-9 pr-3.5 py-2 text-xs border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-colors"
                autoComplete="username"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                <Lock className="w-4 h-4" />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-9 pr-10 py-2 text-xs border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-colors"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              />
              <button
                type="button"
                onClick={() => setShowPassword((prev) => !prev)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600 focus:outline-none focus:text-slate-600 transition-colors"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                title={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? (
                  <EyeOff className="w-4 h-4" aria-hidden="true" />
                ) : (
                  <Eye className="w-4 h-4" aria-hidden="true" />
                )}
              </button>
            </div>
            {mode === 'register' && (
              <p className="text-[11px] text-slate-400 mt-1">
                Must be at least 6 characters. Stored securely with Argon2id password hashing.
              </p>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 inline-flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500/20 shadow-sm transition-colors disabled:opacity-50"
          >
            {loading ? (
              <span>Authenticating...</span>
            ) : (
              <>
                <span>{mode === 'login' ? 'Sign In' : 'Create Account'}</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </form>

        {/* Roles & Security Notice */}
        <div className="mt-6 pt-5 border-t border-slate-100 text-[11px] text-slate-500 space-y-2">
          <div className="flex items-start gap-2">
            <ShieldCheck className="w-3.5 h-3.5 text-blue-600 flex-shrink-0 mt-0.5" />
            <span>
              <strong>Record Access Control:</strong> Household users have isolated access to their own forecast runs and downloadable PDF reports.
            </span>
          </div>
          <div className="flex items-start gap-2">
            <Terminal className="w-3.5 h-3.5 text-slate-500 flex-shrink-0 mt-0.5" />
            <span>
              <strong>Developer Accounts:</strong> Developer identities are authorized via the server CLI tool (<code>python -m backend.app.cli.manage_users create</code>) or seed environment variables.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
