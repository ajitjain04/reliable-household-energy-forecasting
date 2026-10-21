import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../../api/client';
import { HealthResponse } from '../../api/types';
import { StatusBadge } from '../common/StatusBadge';
import { Activity, Server, User, LogOut, LogIn, ShieldCheck, Sun, Moon } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';

export const Header: React.FC = () => {
  const navigate = useNavigate();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');
  const { user, isAuthenticated, isDeveloper, logout } = useAuth();
  const { toggleTheme, isDark } = useTheme();

  const handleLogout = async () => {
    await logout();
    navigate('/', { replace: true });
  };

  const checkHealth = async () => {
    try {
      const data = await api.getHealth();
      setHealth(data);
      setStatus(data.status === 'ok' ? 'connected' : 'disconnected');
    } catch {
      setStatus('disconnected');
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // 30s heartbeat
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-16 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 px-8 flex items-center justify-between sticky top-0 z-10 shadow-xs transition-colors">
      <div className="flex items-center gap-3">
        <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        <div>
          <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100 leading-tight">
            Reliable Multimodal Household Energy Forecasting
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Degraded Weather Telemetry Empirical Investigation
          </p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Heartbeat Status */}
        <div className="hidden sm:flex items-center gap-2">
          <Server className="w-4 h-4 text-slate-400 dark:text-slate-500" />
          <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">127.0.0.1:8000</span>
          <StatusBadge
            status={status}
            label={status === 'connected' ? `API v${health?.version || '1.0.0'} Online` : undefined}
          />
        </div>

        {/* Application-Wide Light / Dark Mode Toggle */}
        <button
          type="button"
          id="theme-toggle-header"
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          className="p-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-slate-900 dark:hover:text-white transition-all shadow-2xs"
        >
          {isDark ? (
            <Sun className="w-4 h-4 text-amber-400 animate-in fade-in" />
          ) : (
            <Moon className="w-4 h-4 text-slate-600 animate-in fade-in" />
          )}
        </button>

        {/* User Session & Authentication Status */}
        <div className="flex items-center gap-2 pl-3 border-l border-slate-200 dark:border-slate-800">
          {isAuthenticated && user ? (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 text-xs">
                <div className="w-7 h-7 rounded-full bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 flex items-center justify-center text-slate-600 dark:text-slate-300">
                  <User className="w-3.5 h-3.5" />
                </div>
                <div>
                  <div className="font-semibold text-slate-800 dark:text-slate-100 leading-none flex items-center gap-1.5">
                    {user.username}
                    {isDeveloper && (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[10px] font-semibold bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800">
                        <ShieldCheck className="w-3 h-3" /> Dev
                      </span>
                    )}
                  </div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500 capitalize">{user.role}</div>
                </div>
              </div>
              <button
                type="button"
                onClick={handleLogout}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-rose-50 dark:hover:bg-rose-950/40 hover:text-rose-600 dark:hover:text-rose-400 transition-colors"
                title="Sign out of current account"
              >
                <LogOut className="w-3 h-3" />
                <span className="hidden md:inline">Sign Out</span>
              </button>
            </div>
          ) : (
            <Link
              to="/login"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-blue-600 text-white hover:bg-blue-700 shadow-xs transition-colors"
            >
              <LogIn className="w-3.5 h-3.5" />
              Sign In
            </Link>
          )}
        </div>
      </div>
    </header>
  );
};

