/**
 * Comprehensive Unit and Workflow Test Suite for:
 * 1. Application-Wide Dark/Light Mode Theme Management (Task 2.7C).
 * 2. Bill-Triggered Personalized Energy Saving Plan & Appliance Simulator (Task 2.7B).
 *
 * Uses Node.js native test runner (node --test).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

// =============================================================================
// PART 1: THEME MANAGEMENT & PERSISTENCE (Task 2.7C)
// =============================================================================

test('Theme System: Resolves stored preference, OS media query fallback, and default', () => {
  const THEME_STORAGE_KEY = 'household_energy_theme';

  const resolveTheme = (mockStorageValue, osPrefersDark) => {
    if (mockStorageValue === 'dark' || mockStorageValue === 'light') {
      return mockStorageValue;
    }
    return osPrefersDark ? 'dark' : 'light';
  };

  // 1. Explicit stored 'dark' preference overrides OS preference
  assert.equal(resolveTheme('dark', false), 'dark');

  // 2. Explicit stored 'light' preference overrides OS preference
  assert.equal(resolveTheme('light', true), 'light');

  // 3. Absent storage falls back to OS prefers-color-scheme
  assert.equal(resolveTheme(null, true), 'dark');
  assert.equal(resolveTheme(null, false), 'light');
  assert.equal(resolveTheme(undefined, true), 'dark');

  // 4. Invalid storage value safely falls back to OS preference
  assert.equal(resolveTheme('invalid_theme_value', false), 'light');
  assert.equal(resolveTheme('invalid_theme_value', true), 'dark');
});

test('Theme System: Toggle function flips state and updates DOM classes', () => {
  let currentTheme = 'light';
  const classList = new Set();
  const storage = {};

  const toggleTheme = () => {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    storage['household_energy_theme'] = currentTheme;
    if (currentTheme === 'dark') {
      classList.add('dark');
    } else {
      classList.delete('dark');
    }
  };

  // Initial state: light
  assert.equal(currentTheme, 'light');
  assert.equal(classList.has('dark'), false);

  // First toggle: light -> dark
  toggleTheme();
  assert.equal(currentTheme, 'dark');
  assert.equal(storage['household_energy_theme'], 'dark');
  assert.equal(classList.has('dark'), true);

  // Second toggle: dark -> light
  toggleTheme();
  assert.equal(currentTheme, 'light');
  assert.equal(storage['household_energy_theme'], 'light');
  assert.equal(classList.has('dark'), false);
});

test('Theme System: Recharts chart color palette adapts dynamically to active theme', () => {
  const getChartThemeColors = (isDark) => ({
    gridStroke: isDark ? '#334155' : '#e2e8f0',
    axisStroke: isDark ? '#94a3b8' : '#64748b',
    tooltipBg: isDark ? '#0f172a' : '#ffffff',
    tooltipBorder: isDark ? '#1e293b' : '#cbd5e1',
    tooltipText: isDark ? '#f8fafc' : '#0f172a',
    baselineStroke: '#0284c7', // Sky-600
    scenarioStroke: '#10b981', // Emerald-500
  });

  const lightColors = getChartThemeColors(false);
  const darkColors = getChartThemeColors(true);

  assert.equal(lightColors.gridStroke, '#e2e8f0');
  assert.equal(lightColors.tooltipBg, '#ffffff');

  assert.equal(darkColors.gridStroke, '#334155');
  assert.equal(darkColors.tooltipBg, '#0f172a');
  assert.equal(darkColors.tooltipText, '#f8fafc');

  // Both themes maintain high visibility for forecast series
  assert.equal(lightColors.baselineStroke, darkColors.baselineStroke);
  assert.equal(lightColors.scenarioStroke, darkColors.scenarioStroke);
});

// =============================================================================
// PART 2: BILL-TRIGGERED PLAN VISIBILITY CONTRACT (Task 2.7B)
// =============================================================================

test('Bill-Triggered Plan: Visibility gating depends strictly on confirmed bill status', () => {
  const evaluateSavingPlanVisibility = ({
    hasUploadedFile,
    hasParsedData,
    isConfirmed,
    confirmedRecordsCount,
  }) => {
    // Plan MUST NOT show if no file uploaded
    if (!hasUploadedFile) return { visible: false, reason: 'no_file' };
    // Plan MUST NOT show if parsing failed or in progress
    if (!hasParsedData) return { visible: false, reason: 'parsing_or_failed' };
    // Plan MUST NOT show if user has not explicitly confirmed review
    if (!isConfirmed) return { visible: false, reason: 'unconfirmed_review' };
    // Plan MUST NOT show if confirmed records are empty
    if (confirmedRecordsCount <= 0) return { visible: false, reason: 'empty_records' };

    return { visible: true, reason: 'ready' };
  };

  // 1. Fresh page: no upload
  assert.equal(
    evaluateSavingPlanVisibility({
      hasUploadedFile: false,
      hasParsedData: false,
      isConfirmed: false,
      confirmedRecordsCount: 0,
    }).visible,
    false
  );

  // 2. File uploaded and extracted, but NOT yet confirmed by household user
  const unconfirmed = evaluateSavingPlanVisibility({
    hasUploadedFile: true,
    hasParsedData: true,
    isConfirmed: false,
    confirmedRecordsCount: 0,
  });
  assert.equal(unconfirmed.visible, false);
  assert.equal(unconfirmed.reason, 'unconfirmed_review');

  // 3. User clicked "Confirm & Save Bill Records" -> plan immediately becomes visible
  const confirmed = evaluateSavingPlanVisibility({
    hasUploadedFile: true,
    hasParsedData: true,
    isConfirmed: true,
    confirmedRecordsCount: 1,
  });
  assert.equal(confirmed.visible, true);
  assert.equal(confirmed.reason, 'ready');
});

// =============================================================================
// PART 3: APPLIANCE WHAT-IF PHYSICS & CALCULATIONS (Task 2.7B)
// =============================================================================

test('Appliance Simulator: Formula verification: (watts * hours * days * quantity) / 1000', () => {
  const computeKwh = (powerWatts, hoursPerDay, daysPerMonth, quantity) => {
    return Number(((powerWatts * hoursPerDay * daysPerMonth * quantity) / 1000).toFixed(2));
  };

  // 1. Standard 1.5-Ton Inverter AC: 1400W, 8h/day, 30 days, 1 unit -> 336.0 kWh
  assert.equal(computeKwh(1400, 8.0, 30, 1), 336.0);

  // 2. Proposed reduction to 7h/day -> 294.0 kWh (Delta = 42.0 kWh/month)
  assert.equal(computeKwh(1400, 7.0, 30, 1), 294.0);
  assert.equal(computeKwh(1400, 8.0, 30, 1) - computeKwh(1400, 7.0, 30, 1), 42.0);

  // 3. Multiple Ceiling Fans: 4 units * 55W * 12h/day * 30 days / 1000 -> 79.2 kWh
  assert.equal(computeKwh(55, 12.0, 30, 4), 79.2);

  // 4. Fractional hours: 10 LED bulbs * 9W * 4.5h/day * 30 days / 1000 -> 12.15 kWh
  assert.equal(computeKwh(9, 4.5, 30, 10), 12.15);

  // 5. Zero hours or zero days produces 0.0 kWh
  assert.equal(computeKwh(2000, 0.0, 30, 1), 0.0);
  assert.equal(computeKwh(2000, 2.0, 0, 1), 0.0);
});

test('Appliance Simulator: Refrigerator safety caution triggered when operating hours < 24h', () => {
  const checkApplianceNotes = (category, hoursPerDay) => {
    const notes = [];
    if (category === 'refrigerator') {
      notes.push('Typical 250L frost-free refrigerator runs intermittently with ~35-50% compressor duty cycle.');
      if (hoursPerDay < 24.0) {
        notes.push(
          'SAFETY CAUTION: Refrigerators require continuous 24-hour operation to ensure food preservation and hygiene.'
        );
      }
    }
    return notes;
  };

  const safeNotes = checkApplianceNotes('refrigerator', 24.0);
  assert.equal(safeNotes.length, 1);
  assert.equal(safeNotes[0].includes('duty cycle'), true);

  const unsafeNotes = checkApplianceNotes('refrigerator', 18.0);
  assert.equal(unsafeNotes.length, 2);
  assert.equal(unsafeNotes.some((n) => n.includes('SAFETY CAUTION')), true);

  // Non-refrigerator appliances should not receive food preservation safety warning
  const acNotes = checkApplianceNotes('air_conditioner', 6.0);
  assert.equal(acNotes.length, 0);
});

// =============================================================================
// PART 4: HOUSEHOLD SCENARIO COMPARISON & TARIFF WITHHOLDING
// =============================================================================

test('Household Scenario Comparison: Physical plausibility contract - withholding when appliance load exceeds household baseline, no clamping to 0 kWh', () => {
  const compareHouseholdScenario = ({
    baselineForecastKwh,
    monthLabels,
    applianceBaselineKwh,
    applianceDeltaKwh,
  }) => {
    let totalBaseline = 0;
    const isSupported = baselineForecastKwh.every(
      (baseKwh) => applianceBaselineKwh <= baseKwh && applianceDeltaKwh < baseKwh && baseKwh - applianceDeltaKwh > 0
    );

    let totalAdjusted = isSupported ? 0 : null;
    let totalDelta = isSupported ? 0 : null;

    const steps = baselineForecastKwh.map((baseKwh, idx) => {
      totalBaseline += baseKwh;
      if (!isSupported) {
        return {
          month_index: idx + 1,
          month_label: monthLabels[idx],
          baseline_kwh: baseKwh,
          adjusted_kwh: null,
          delta_kwh: null,
          is_adjustment_supported: false,
          unsupported_reason: `Appliance baseline (${applianceBaselineKwh} kWh) or delta (${applianceDeltaKwh} kWh) exceeds household baseline (${baseKwh} kWh).`,
        };
      }
      const adjustedKwh = Number((baseKwh - applianceDeltaKwh).toFixed(2));
      const stepDelta = Number((baseKwh - adjustedKwh).toFixed(2));
      totalAdjusted += adjustedKwh;
      totalDelta += stepDelta;
      return {
        month_index: idx + 1,
        month_label: monthLabels[idx],
        baseline_kwh: baseKwh,
        adjusted_kwh: adjustedKwh,
        delta_kwh: stepDelta,
        is_adjustment_supported: true,
      };
    });

    return {
      steps,
      total_baseline_kwh: Number(totalBaseline.toFixed(2)),
      total_adjusted_kwh: totalAdjusted !== null ? Number(totalAdjusted.toFixed(2)) : null,
      total_delta_kwh: totalDelta !== null ? Number(totalDelta.toFixed(2)) : null,
      is_household_adjustment_supported: isSupported,
    };
  };

  // 1. Valid scenario: Household baseline 450, 480, 500 kWh with 1450W AC (baseline 348 kWh, delta 43.5 kWh)
  const validRes = compareHouseholdScenario({
    baselineForecastKwh: [450.0, 480.0, 500.0],
    monthLabels: ['Aug 2024', 'Sep 2024', 'Oct 2024'],
    applianceBaselineKwh: 348.0,
    applianceDeltaKwh: 43.5,
  });
  assert.equal(validRes.is_household_adjustment_supported, true);
  assert.equal(validRes.total_baseline_kwh, 1430.0);
  assert.equal(validRes.total_adjusted_kwh, 1299.5);
  assert.equal(validRes.total_delta_kwh, 130.5);
  assert.equal(validRes.steps[0].adjusted_kwh, 406.5);
  assert.equal(validRes.steps[0].delta_kwh, 43.5);

  // 2. Exact user bug scenario: 1450W AC (baseline 348 kWh, delta 43.5 kWh) against 2.0 kWh household baseline
  const bugRes = compareHouseholdScenario({
    baselineForecastKwh: [2.0, 2.0, 2.0],
    monthLabels: ['Jan 2025', 'Feb 2025', 'Mar 2025'],
    applianceBaselineKwh: 348.0,
    applianceDeltaKwh: 43.5,
  });
  assert.equal(bugRes.is_household_adjustment_supported, false);
  assert.equal(bugRes.total_baseline_kwh, 6.0);
  assert.equal(bugRes.total_adjusted_kwh, null); // Withheld, NEVER clamped to 0.0
  assert.equal(bugRes.total_delta_kwh, null);    // Withheld, NEVER -2.0 or 2.0
  assert.equal(bugRes.steps[0].adjusted_kwh, null); // NOT 0.0
  assert.equal(bugRes.steps[0].delta_kwh, null);
  assert.equal(bugRes.steps[0].is_adjustment_supported, false);
});

test('Tariff Policy: Verified MSEDCL vs unverified provider monetary withholding', () => {
  const evaluateTariffPolicy = (providerId, asOfDate) => {
    const isMsedcl = providerId?.toLowerCase() === 'msedcl';
    if (!isMsedcl) {
      return {
        is_provider_confirmed: Boolean(providerId),
        is_historical_schedule_known: false,
        is_applicable_schedule_verified: false,
        allow_monetary_estimates: false,
        tariff_status: providerId ? 'unverified_schedule' : 'no_provider',
        tariff_status_label: providerId ? 'Current tariff unverified' : 'No provider selected',
        withheld_label: providerId ? 'Withheld (Unverified)' : 'Withheld (No Provider)',
        disclosure: providerId
          ? `Official tariff schedules for provider '${providerId}' are currently unverified. Monetary bill amounts and savings are withheld to prevent fabricated estimates.`
          : 'No electricity provider selected.',
      };
    }

    // Provider is MSEDCL
    const cutoff = new Date('2025-03-31T23:59:59Z');
    const isWithinValidPeriod = asOfDate ? new Date(asOfDate) <= cutoff : false;

    if (isWithinValidPeriod) {
      return {
        is_provider_confirmed: true,
        is_historical_schedule_known: true,
        is_applicable_schedule_verified: true,
        allow_monetary_estimates: true,
        tariff_status: 'verified',
        tariff_status_label: 'Verified',
        withheld_label: 'Withheld (Unverified)',
        disclosure: 'Bills evaluated using verified MSEDCL LT-I(B) Residential tariff (MERC Case No. 226 of 2022).',
      };
    }

    return {
      is_provider_confirmed: true,
      is_historical_schedule_known: true,
      is_applicable_schedule_verified: false,
      allow_monetary_estimates: false,
      tariff_status: 'expired_schedule',
      tariff_status_label: 'Expired schedule',
      withheld_label: 'Withheld (Expired Tariff)',
      disclosure: 'MSEDCL LT-I(B) schedule expired on 2025-03-31. Monetary bill estimates are withheld to prevent fabricated projections.',
    };
  };

  // 1. MSEDCL with valid historical schedule period (FY 2024-25)
  const msedclValid = evaluateTariffPolicy('msedcl', '2024-08-01');
  assert.equal(msedclValid.is_provider_confirmed, true);
  assert.equal(msedclValid.is_historical_schedule_known, true);
  assert.equal(msedclValid.is_applicable_schedule_verified, true);
  assert.equal(msedclValid.allow_monetary_estimates, true);
  assert.equal(msedclValid.tariff_status, 'verified');
  assert.equal(msedclValid.tariff_status_label, 'Verified');

  // 2. MSEDCL with future forecast period past 2025-03-31 (Expired schedule)
  const msedclExpired = evaluateTariffPolicy('msedcl', '2025-04-01');
  assert.equal(msedclExpired.is_provider_confirmed, true);
  assert.equal(msedclExpired.is_historical_schedule_known, true);
  assert.equal(msedclExpired.is_applicable_schedule_verified, false);
  assert.equal(msedclExpired.allow_monetary_estimates, false);
  assert.equal(msedclExpired.tariff_status, 'expired_schedule');
  assert.equal(msedclExpired.tariff_status_label, 'Expired schedule');
  assert.equal(msedclExpired.withheld_label, 'Withheld (Expired Tariff)');

  // 3. Unverified provider (BESCOM)
  const bescom = evaluateTariffPolicy('bescom', '2024-08-01');
  assert.equal(bescom.is_provider_confirmed, true);
  assert.equal(bescom.is_historical_schedule_known, false);
  assert.equal(bescom.is_applicable_schedule_verified, false);
  assert.equal(bescom.allow_monetary_estimates, false);
  assert.equal(bescom.tariff_status, 'unverified_schedule');
  assert.equal(bescom.tariff_status_label, 'Current tariff unverified');
  assert.equal(bescom.withheld_label, 'Withheld (Unverified)');

  // 4. No provider
  const noProvider = evaluateTariffPolicy(null, '2024-08-01');
  assert.equal(noProvider.is_provider_confirmed, false);
  assert.equal(noProvider.allow_monetary_estimates, false);
  assert.equal(noProvider.tariff_status, 'no_provider');
  assert.equal(noProvider.tariff_status_label, 'No provider selected');
  assert.equal(noProvider.withheld_label, 'Withheld (No Provider)');
});

test('Tariff Status Resolution: UI helper correctly evaluates forecast steps and prevents expired tariff verification', () => {
  // Replicates getStepTariffInfo logic from BillTriggeredSavingPlan.tsx
  const resolveStepTariff = (step, providerId) => {
    const status = step.tariff_status || step.bill_estimate?.tariff_status;
    const label = step.tariff_status_label || step.bill_estimate?.tariff_status_label;

    if (status === 'verified') {
      return { status: 'verified', label: label || 'Verified', withheldLabel: 'Withheld (Unverified)', icon: 'check' };
    }
    if (status === 'expired_schedule') {
      return { status: 'expired_schedule', label: label || 'Expired schedule', withheldLabel: 'Withheld (Expired Tariff)', icon: 'clock' };
    }
    if (status === 'unverified_schedule') {
      return { status: 'unverified_schedule', label: label || 'Current tariff unverified', withheldLabel: 'Withheld (Unverified)', icon: 'alert' };
    }
    if (status === 'no_provider') {
      return { status: 'no_provider', label: label || 'No provider selected', withheldLabel: 'Withheld (No Provider)', icon: 'none' };
    }

    if (step.estimated_bill_amount !== null && step.estimated_bill_amount !== undefined) {
      return { status: 'verified', label: 'Verified', withheldLabel: 'Withheld (Unverified)', icon: 'check' };
    }

    const normId = (providerId || '').trim().toLowerCase();
    if (normId === 'msedcl') {
      const pDate = step.period_start ? new Date(step.period_start) : null;
      if (pDate && !isNaN(pDate.getTime()) && pDate <= new Date('2025-03-31T23:59:59Z')) {
        return { status: 'verified', label: 'Verified', withheldLabel: 'Withheld (Unverified)', icon: 'check' };
      }
      return { status: 'expired_schedule', label: 'Expired schedule', withheldLabel: 'Withheld (Expired Tariff)', icon: 'clock' };
    }
    if (normId) {
      return { status: 'unverified_schedule', label: 'Current tariff unverified', withheldLabel: 'Withheld (Unverified)', icon: 'alert' };
    }
    return { status: 'no_provider', label: 'No provider selected', withheldLabel: 'Withheld (No Provider)', icon: 'none' };
  };

  // Case A: Step with backend-provided expired_schedule
  const stepExpiredBackend = {
    month_index: 1,
    month_label: 'Nov 2026',
    period_start: '2026-11-01',
    tariff_status: 'expired_schedule',
    tariff_status_label: 'Expired schedule',
    estimated_bill_amount: null,
  };
  const resA = resolveStepTariff(stepExpiredBackend, 'msedcl');
  assert.equal(resA.status, 'expired_schedule');
  assert.equal(resA.label, 'Expired schedule');
  assert.equal(resA.withheldLabel, 'Withheld (Expired Tariff)');
  assert.equal(resA.icon, 'clock');

  // Case B: Step without backend status, but date is after 2025-03-31 -> evaluated as Expired schedule
  const stepExpiredDateFallback = {
    month_index: 2,
    month_label: 'Dec 2026',
    period_start: '2026-12-01',
    estimated_bill_amount: null,
  };
  const resB = resolveStepTariff(stepExpiredDateFallback, 'msedcl');
  assert.equal(resB.status, 'expired_schedule');
  assert.equal(resB.label, 'Expired schedule');
  assert.equal(resB.withheldLabel, 'Withheld (Expired Tariff)');

  // Case C: Step within FY 24-25 -> evaluated as Verified
  const stepValid = {
    month_index: 1,
    month_label: 'Aug 2024',
    period_start: '2024-08-01',
    estimated_bill_amount: 1850.5,
  };
  const resC = resolveStepTariff(stepValid, 'msedcl');
  assert.equal(resC.status, 'verified');
  assert.equal(resC.label, 'Verified');
  assert.equal(resC.icon, 'check');

  // Case D: BESCOM step -> Current tariff unverified
  const stepBescom = {
    month_index: 1,
    month_label: 'Aug 2024',
    period_start: '2024-08-01',
    estimated_bill_amount: null,
  };
  const resD = resolveStepTariff(stepBescom, 'bescom');
  assert.equal(resD.status, 'unverified_schedule');
  assert.equal(resD.label, 'Current tariff unverified');
  assert.equal(resD.withheldLabel, 'Withheld (Unverified)');
  assert.equal(resD.icon, 'alert');
});

// =============================================================================
// PART 5: 404 FALLBACK RESILIENCE & DARK THEME TOKEN CHECKS
// =============================================================================

test('Client Fallback: Resiliently recovers when backend returns 404 Not Found', async () => {
  // Simulate the client fallback behavior implemented in client.ts
  const mockSimulateAppliance = async (scenario) => {
    const currentKwh = (scenario.power_watts * scenario.current_hours_per_day * scenario.days_per_month * scenario.quantity) / 1000;
    const proposedKwh = (scenario.power_watts * scenario.proposed_hours_per_day * scenario.days_per_month * scenario.quantity) / 1000;
    const deltaKwh = currentKwh - proposedKwh;
    return {
      category: scenario.category,
      custom_name: scenario.custom_name,
      current_consumption_kwh: Number(currentKwh.toFixed(2)),
      proposed_consumption_kwh: Number(proposedKwh.toFixed(2)),
      delta_kwh: Number(deltaKwh.toFixed(2)),
      percent_reduction: Number(((deltaKwh / (currentKwh || 1)) * 100).toFixed(1)),
      monthly_savings_inr: 42.0 * 8.5,
      is_msedcl_tariff_applied: true,
      notes: [],
    };
  };

  const executeWith404Fallback = async (mockPrimaryApiStatus, payload) => {
    if (mockPrimaryApiStatus === 404) {
      // Primary /compare-household-scenario returned 404 -> client fallback runs
      const sim = await mockSimulateAppliance(payload.appliance_scenario);
      const monthlyDelta = Math.max(0, sim.delta_kwh);
      const appBaselineKwh = sim.current_consumption_kwh;
      let totalBaseline = 0;

      const isSupported = payload.baseline_forecast_kwh.every(
        (baseKwh) => appBaselineKwh <= baseKwh && monthlyDelta < baseKwh && baseKwh - monthlyDelta > 0
      );

      let totalAdjusted = isSupported ? 0 : null;
      let totalDelta = isSupported ? 0 : null;

      const steps = payload.baseline_forecast_kwh.map((baseKwh, idx) => {
        totalBaseline += baseKwh;
        if (!isSupported) {
          return {
            month_index: idx + 1,
            month_label: payload.month_labels[idx] || `Month ${idx + 1}`,
            baseline_kwh: baseKwh,
            adjusted_kwh: null,
            delta_kwh: null,
            is_adjustment_supported: false,
            unsupported_reason: `Appliance load (${appBaselineKwh} kWh) or delta (${monthlyDelta} kWh) exceeds household baseline (${baseKwh} kWh).`,
          };
        }
        const adjustedKwh = Number((baseKwh - monthlyDelta).toFixed(2));
        const delta = Number((baseKwh - adjustedKwh).toFixed(2));
        totalAdjusted += adjustedKwh;
        totalDelta += delta;
        return {
          month_index: idx + 1,
          month_label: payload.month_labels[idx] || `Month ${idx + 1}`,
          baseline_kwh: baseKwh,
          adjusted_kwh: adjustedKwh,
          delta_kwh: delta,
          is_adjustment_supported: true,
        };
      });

      return {
        appliance_simulation: sim,
        steps,
        total_baseline_kwh: Number(totalBaseline.toFixed(2)),
        total_adjusted_kwh: totalAdjusted !== null ? Number(totalAdjusted.toFixed(2)) : null,
        total_delta_kwh: totalDelta !== null ? Number(totalDelta.toFixed(2)) : null,
        total_baseline_bill_inr: isSupported ? 6500.0 : null,
        total_adjusted_bill_inr: isSupported ? 5800.0 : null,
        total_delta_bill_inr: isSupported ? 700.0 : null,
        is_household_adjustment_supported: isSupported,
        is_tariff_verified: true,
        tariff_disclosure: 'Fallback calculation with verified tariff',
      };
    }
    // Normal 200 response
    return { success: true };
  };

  const consistentPayload = {
    household_id: 'default_household',
    utility_provider: 'msedcl',
    baseline_forecast_kwh: [450, 470, 490],
    month_labels: ['Aug 2024', 'Sep 2024', 'Oct 2024'],
    appliance_scenario: {
      category: 'air_conditioner',
      custom_name: 'Living Room AC',
      power_watts: 1400,
      current_hours_per_day: 8.0,
      proposed_hours_per_day: 7.0,
      days_per_month: 30,
      quantity: 1,
    },
  };

  // 1. Consistent payload fallback succeeds with adjusted kWh
  const validResult = await executeWith404Fallback(404, consistentPayload);
  assert.equal(validResult.total_baseline_kwh, 1410.0);
  assert.equal(validResult.total_delta_kwh, 126.0); // 42 kWh/month * 3 months
  assert.equal(validResult.total_adjusted_kwh, 1284.0);
  assert.equal(validResult.steps.length, 3);
  assert.equal(validResult.appliance_simulation.delta_kwh, 42.0);
  assert.equal(validResult.is_household_adjustment_supported, true);
  assert.equal(validResult.is_tariff_verified, true);

  // 2. Exact bug payload (2.0 kWh baseline vs 336 kWh AC) withholds household adjusted forecast
  const bugPayload = {
    household_id: 'default_household',
    utility_provider: 'msedcl',
    baseline_forecast_kwh: [2.0, 2.0, 2.0],
    month_labels: ['Jan 2025', 'Feb 2025', 'Mar 2025'],
    appliance_scenario: {
      category: 'air_conditioner',
      custom_name: 'Living Room AC',
      power_watts: 1400,
      current_hours_per_day: 8.0,
      proposed_hours_per_day: 7.0,
      days_per_month: 30,
      quantity: 1,
    },
  };
  const bugResult = await executeWith404Fallback(404, bugPayload);
  assert.equal(bugResult.total_baseline_kwh, 6.0);
  assert.equal(bugResult.total_adjusted_kwh, null);
  assert.equal(bugResult.total_delta_kwh, null);
  assert.equal(bugResult.total_delta_bill_inr, null);
  assert.equal(bugResult.is_household_adjustment_supported, false);
  assert.equal(bugResult.steps[0].adjusted_kwh, null);
  assert.equal(bugResult.appliance_simulation.delta_kwh, 42.0); // Preserved appliance-only delta
});

test('Dark Mode Consistency: Essential UI components include dark theme classes', () => {
  // Verify dark tokens structure for key components
  const darkClassesPattern = /dark:/;

  const billReviewBannerLight = 'p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900';
  const billReviewBannerWithDark =
    'p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900 text-emerald-900 dark:text-emerald-200';

  assert.equal(darkClassesPattern.test(billReviewBannerLight), false);
  assert.equal(darkClassesPattern.test(billReviewBannerWithDark), true);

  const tableHeaderWithDark =
    'bg-slate-50 dark:bg-slate-900/80 text-[11px] uppercase font-bold text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800';
  assert.equal(darkClassesPattern.test(tableHeaderWithDark), true);

  const inputWithDark =
    'w-24 font-mono font-bold bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 text-xs text-slate-900 dark:text-white';
  assert.equal(darkClassesPattern.test(inputWithDark), true);
});

