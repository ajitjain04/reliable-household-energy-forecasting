import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Home,
  UploadCloud,
  History,
  Compass,
  Terminal,
  Database,
  LayoutDashboard,
  BarChart3,
  ShieldAlert,
  Scale,
  SearchCode,
  Target,
  TrendingUp,
  Zap,
  FlaskConical,
  Lock,
  Sun,
  Moon,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';

interface NavItem {
  name: string;
  path: string;
  icon: React.ReactNode;
}

export const Sidebar: React.FC = () => {
  const { isDeveloper, isAuthenticated, openAuthModal } = useAuth();
  const { toggleTheme, isDark } = useTheme();

  const householdNavItems: NavItem[] = [
    {
      name: 'Home Dashboard',
      path: '/',
      icon: <Home className="w-4 h-4" />,
    },
    {
      name: 'New Forecast',
      path: '/forecast/new',
      icon: <UploadCloud className="w-4 h-4" />,
    },
    {
      name: 'Forecast History',
      path: '/forecast/history',
      icon: <History className="w-4 h-4" />,
    },
    {
      name: 'Energy Coach',
      path: '/coach',
      icon: <Compass className="w-4 h-4" />,
    },
  ];

  const developerNavItems: NavItem[] = [
    {
      name: 'Run Inspector',
      path: '/developer/forecast-records',
      icon: <Database className="w-4 h-4" />,
    },
  ];

  const researchNavItems: NavItem[] = [
    {
      name: 'Overview',
      path: '/overview',
      icon: <LayoutDashboard className="w-4 h-4" />,
    },
    {
      name: 'Model Benchmarks',
      path: '/benchmarks',
      icon: <BarChart3 className="w-4 h-4" />,
    },
    {
      name: 'Reliability Analysis',
      path: '/reliability',
      icon: <ShieldAlert className="w-4 h-4" />,
    },
    {
      name: 'Statistical Tests',
      path: '/statistical-tests',
      icon: <Scale className="w-4 h-4" />,
    },
    {
      name: 'Explainable AI',
      path: '/xai',
      icon: <SearchCode className="w-4 h-4" />,
    },
    {
      name: 'Uncertainty Analysis',
      path: '/uncertainty',
      icon: <Target className="w-4 h-4" />,
    },
    {
      name: 'Forecast Explorer',
      path: '/explorer',
      icon: <TrendingUp className="w-4 h-4" />,
    },
  ];

  return (
    <aside className="w-64 bg-navy-900 text-slate-300 flex flex-col flex-shrink-0 min-h-screen">
      {/* Brand Header */}
      <div className="p-6 border-b border-navy-800 flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
          <Zap className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-white tracking-wide leading-tight">
            Energy Forecasting
          </h1>
          <p className="text-xs text-slate-400 font-mono">Operations Platform</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-5 space-y-6 overflow-y-auto">
        {/* Household Section */}
        <div>
          <div className="px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-blue-400/90 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5" />
            Household Forecasting
          </div>
          <div className="space-y-1">
            {householdNavItems.map((item) => {
              const isProtected = item.path !== '/';
              if (isProtected && !isAuthenticated) {
                return (
                  <button
                    key={item.path}
                    type="button"
                    onClick={() => openAuthModal(item.path)}
                    className="w-full flex items-center justify-between px-3 py-2.5 rounded-md text-sm font-medium text-slate-300 hover:text-white hover:bg-navy-800 transition-colors text-left"
                  >
                    <span className="flex items-center gap-3">
                      {item.icon}
                      {item.name}
                    </span>
                    <Lock className="w-3.5 h-3.5 text-slate-500" />
                  </button>
                );
              }

              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  end={item.path === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white shadow-sm font-semibold'
                        : 'text-slate-300 hover:text-white hover:bg-navy-800'
                    }`
                  }
                >
                  {item.icon}
                  {item.name}
                </NavLink>
              );
            })}
          </div>
        </div>

        {/* Developer Console (Local Diagnostics) */}
        <div>
          <div className="px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-amber-400/90 flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-amber-400" />
              Developer Console
            </span>
            <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold ${
              isDeveloper
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                : 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
            }`}>
              {isDeveloper ? 'Authorized' : 'Restricted'}
            </span>
          </div>
          <div className="space-y-1">
            {developerNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center justify-between px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-navy-800 text-amber-300 border-l-2 border-amber-400 font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-navy-800/60'
                  }`
                }
              >
                <span className="flex items-center gap-3">
                  {item.icon}
                  {item.name}
                </span>
                {!isDeveloper && <Lock className="w-3 h-3 text-slate-500" />}
              </NavLink>
            ))}
          </div>
        </div>

        {/* Research / Developer Section */}
        <div>
          <div className="px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
            <FlaskConical className="w-3.5 h-3.5 text-slate-400" />
            Research & Benchmarks
          </div>
          <div className="space-y-1">
            {researchNavItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-navy-800 text-white border-l-2 border-blue-500 font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-navy-800/60'
                  }`
                }
              >
                {item.icon}
                {item.name}
              </NavLink>
            ))}
          </div>
        </div>
      </nav>

      {/* Theme Toggle & Production / Benchmark Specs Footer */}
      <div className="p-4 border-t border-navy-800 text-xs text-slate-400 font-mono space-y-2.5 bg-navy-900/50">
        <button
          type="button"
          id="theme-toggle-sidebar"
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-navy-800/80 border border-navy-700 hover:bg-navy-700/80 text-slate-300 hover:text-white transition-all text-xs font-sans font-medium"
        >
          <span className="flex items-center gap-2">
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-blue-400" />}
            <span>Appearance</span>
          </span>
          <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-navy-900 text-slate-300 border border-navy-700">
            {isDark ? 'Dark Mode' : 'Light Mode'}
          </span>
        </button>

        <div className="pt-1 space-y-1">
          <div className="text-slate-500 font-sans font-semibold uppercase tracking-wider text-[10px]">
            Operational Runtime
          </div>
          <div className="text-slate-300">Lookback: 168h (7 days)</div>
          <div className="text-slate-300">Horizon: 24h multi-step</div>
          <div className="text-slate-300">Target: Active Power (Watts)</div>
          <div className="text-slate-300">Coverage: 90% Conformal</div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
