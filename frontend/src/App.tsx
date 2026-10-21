import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import { AuthProvider } from './context/AuthContext';
import { Layout } from './components/layout/Layout';
import { ProtectedRoute } from './components/common/ProtectedRoute';

// Authentication page
import { Login } from './pages/Login';

// Household-facing pages
import { Home } from './pages/Home';
import { NewForecast } from './pages/NewForecast';
import { ForecastResult } from './pages/ForecastResult';
import { ForecastHistory } from './pages/ForecastHistory';
import { MonthlyForecastDetail } from './pages/MonthlyForecastDetail';
import { EnergyCoach } from './pages/EnergyCoach';

// Developer Console pages (Local Dev Inspection - Protected for Developer role)
import { DeveloperForecastRecords } from './pages/DeveloperForecastRecords';
import { DeveloperRunDetail } from './pages/DeveloperRunDetail';

// Research Suite pages
import { Overview } from './pages/Overview';
import { ModelBenchmarks } from './pages/ModelBenchmarks';
import { ReliabilityAnalysis } from './pages/ReliabilityAnalysis';
import { StatisticalTests } from './pages/StatisticalTests';
import { ExplainableAI } from './pages/ExplainableAI';
import { UncertaintyAnalysis } from './pages/UncertaintyAnalysis';
import { ForecastExplorer } from './pages/ForecastExplorer';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<Layout />}>
              {/* Household-Facing Forecasting Workflow (Authentication Protected) */}
              <Route index element={<Home />} />
              <Route
                path="forecast/new"
                element={
                  <ProtectedRoute>
                    <NewForecast />
                  </ProtectedRoute>
                }
              />
              <Route
                path="forecast/history"
                element={
                  <ProtectedRoute>
                    <ForecastHistory />
                  </ProtectedRoute>
                }
              />
              <Route
                path="coach"
                element={
                  <ProtectedRoute>
                    <EnergyCoach />
                  </ProtectedRoute>
                }
              />
              <Route path="energy-coach" element={<Navigate to="/coach" replace />} />
              <Route
                path="forecast/:runId"
                element={
                  <ProtectedRoute>
                    <ForecastResult />
                  </ProtectedRoute>
                }
              />
              <Route
                path="forecast/monthly/:runId"
                element={
                  <ProtectedRoute>
                    <MonthlyForecastDetail />
                  </ProtectedRoute>
                }
              />

              {/* Authentication Route */}
              <Route path="login" element={<Login />} />
              <Route path="register" element={<Login />} />

              {/* Developer Console (Restricted to Developer Role) */}
              <Route
                path="developer/forecast-records"
                element={
                  <ProtectedRoute requiredRole="developer">
                    <DeveloperForecastRecords />
                  </ProtectedRoute>
                }
              />
              <Route
                path="developer/forecast-records/:runId"
                element={
                  <ProtectedRoute requiredRole="developer">
                    <DeveloperRunDetail />
                  </ProtectedRoute>
                }
              />
              <Route path="developer/records" element={<Navigate to="/developer/forecast-records" replace />} />
              <Route path="developer" element={<Navigate to="/developer/forecast-records" replace />} />

              {/* Research & Benchmarks Area (All 7 Research Modules Preserved) */}
              <Route path="overview" element={<Overview />} />
              <Route path="research" element={<Overview />} />
              <Route path="benchmarks" element={<ModelBenchmarks />} />
              <Route path="reliability" element={<ReliabilityAnalysis />} />
              <Route path="statistical-tests" element={<StatisticalTests />} />
              <Route path="statistics" element={<StatisticalTests />} />
              <Route path="xai" element={<ExplainableAI />} />
              <Route path="uncertainty" element={<UncertaintyAnalysis />} />
              <Route path="explorer" element={<ForecastExplorer />} />
              <Route path="forecasts" element={<ForecastExplorer />} />

              {/* Fallback */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
};

export default App;

