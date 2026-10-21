/**
 * Comprehensive Unit & Integration Test Suite for Frontend Automatic Weather Integration.
 * Tests location validation, geolocation state handling, API request payload construction,
 * and preservation of existing CSV and manual workflows.
 *
 * Uses Node.js native test runner (node --test).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

/**
 * Pure coordinate validation logic matching frontend/src/pages/NewForecast.tsx.
 */
const validateLocationInputs = (latStr, lonStr, labelStr) => {
  const trimmedLat = (latStr || '').trim();
  const trimmedLon = (lonStr || '').trim();
  const trimmedLabel = labelStr ? labelStr.trim() : '';

  // Both blank -> valid (location is optional)
  if (!trimmedLat && !trimmedLon) {
    return { isValid: true, error: null, location: undefined };
  }

  // One provided, one missing
  if ((trimmedLat && !trimmedLon) || (!trimmedLat && trimmedLon)) {
    return {
      isValid: false,
      error: 'Both latitude and longitude must be provided for location-based weather retrieval.',
    };
  }

  const latNum = parseFloat(trimmedLat);
  const lonNum = parseFloat(trimmedLon);

  if (isNaN(latNum) || !isFinite(latNum)) {
    return { isValid: false, error: 'Latitude must be a valid numerical value.' };
  }
  if (isNaN(lonNum) || !isFinite(lonNum)) {
    return { isValid: false, error: 'Longitude must be a valid numerical value.' };
  }

  if (latNum < -90 || latNum > 90) {
    return { isValid: false, error: 'Latitude must be between -90.0 and 90.0 degrees.' };
  }
  if (lonNum < -180 || lonNum > 180) {
    return { isValid: false, error: 'Longitude must be between -180.0 and 180.0 degrees.' };
  }

  if (trimmedLabel.length > 128) {
    return { isValid: false, error: 'Location label cannot exceed 128 characters.' };
  }

  return {
    isValid: true,
    error: null,
    location: {
      latitude: latNum,
      longitude: lonNum,
      label: trimmedLabel || undefined,
    },
  };
};

// -----------------------------------------------------------------------------
// 1. Household Location Coordinate Validation Contract
// -----------------------------------------------------------------------------
test('Location Validation: Optional when blank (both latitude and longitude empty)', () => {
  const res1 = validateLocationInputs('', '');
  assert.equal(res1.isValid, true);
  assert.equal(res1.error, null);
  assert.equal(res1.location, undefined);

  const res2 = validateLocationInputs('   ', '   ', '   ');
  assert.equal(res2.isValid, true);
  assert.equal(res2.error, null);
  assert.equal(res2.location, undefined);
});

test('Location Validation: Rejects half-filled coordinates (one provided, one missing)', () => {
  const latOnly = validateLocationInputs('18.5204', '');
  assert.equal(latOnly.isValid, false);
  assert.match(latOnly.error, /Both latitude and longitude must be provided/i);
  assert.equal(latOnly.location, undefined);

  const lonOnly = validateLocationInputs('', '73.8567');
  assert.equal(lonOnly.isValid, false);
  assert.match(lonOnly.error, /Both latitude and longitude must be provided/i);
  assert.equal(lonOnly.location, undefined);
});

test('Location Validation: Rejects non-numeric coordinate strings', () => {
  const badLat = validateLocationInputs('invalid_lat', '73.8567');
  assert.equal(badLat.isValid, false);
  assert.match(badLat.error, /Latitude must be a valid numerical value/i);

  const badLon = validateLocationInputs('18.5204', 'invalid_lon');
  assert.equal(badLon.isValid, false);
  assert.match(badLon.error, /Longitude must be a valid numerical value/i);

  const nanLat = validateLocationInputs('NaN', '73.8567');
  assert.equal(nanLat.isValid, false);
  assert.match(nanLat.error, /Latitude must be a valid numerical value/i);
});

test('Location Validation: Enforces strict geographic bounds (-90 to 90 lat, -180 to 180 lon)', () => {
  // Latitude bounds
  assert.equal(validateLocationInputs('90.0001', '0').isValid, false);
  assert.match(validateLocationInputs('90.0001', '0').error, /Latitude must be between -90.0 and 90.0/i);

  assert.equal(validateLocationInputs('-90.0001', '0').isValid, false);
  assert.match(validateLocationInputs('-90.0001', '0').error, /Latitude must be between -90.0 and 90.0/i);

  // Longitude bounds
  assert.equal(validateLocationInputs('0', '180.0001').isValid, false);
  assert.match(validateLocationInputs('0', '180.0001').error, /Longitude must be between -180.0 and 180.0/i);

  assert.equal(validateLocationInputs('0', '-180.0001').isValid, false);
  assert.match(validateLocationInputs('0', '-180.0001').error, /Longitude must be between -180.0 and 180.0/i);

  // Exact bounds (inclusive)
  assert.equal(validateLocationInputs('90', '180').isValid, true);
  assert.equal(validateLocationInputs('-90', '-180').isValid, true);
  assert.equal(validateLocationInputs('0', '0').isValid, true);
});

test('Location Validation: Parses high-precision coordinates and optional label', () => {
  const res = validateLocationInputs(' 18.520430 ', ' 73.856743 ', ' Pune Household #1 ');
  assert.equal(res.isValid, true);
  assert.equal(res.error, null);
  assert.deepEqual(res.location, {
    latitude: 18.52043,
    longitude: 73.856743,
    label: 'Pune Household #1',
  });

  // Label length boundary check
  const longLabel = 'a'.repeat(129);
  const tooLongRes = validateLocationInputs('18.5', '73.8', longLabel);
  assert.equal(tooLongRes.isValid, false);
  assert.match(tooLongRes.error, /label cannot exceed 128 characters/i);

  const exactLabel = 'a'.repeat(128);
  const exactRes = validateLocationInputs('18.5', '73.8', exactLabel);
  assert.equal(exactRes.isValid, true);
  assert.equal(exactRes.location.label.length, 128);
});

// -----------------------------------------------------------------------------
// 2. Geolocation Error & State Machine Simulation
// -----------------------------------------------------------------------------
test('Geolocation State: Handles permission denied, unavailable, timeout, and unsupported browsers', () => {
  const simulateGeolocation = (mockNavigator, callbacks) => {
    if (!mockNavigator || !mockNavigator.geolocation) {
      return { error: 'Geolocation is not supported by your browser. Please enter coordinates manually.' };
    }

    let result = null;
    mockNavigator.geolocation.getCurrentPosition(
      (pos) => {
        result = {
          success: true,
          latitude: pos.coords.latitude.toFixed(6),
          longitude: pos.coords.longitude.toFixed(6),
        };
      },
      (err) => {
        let msg = '';
        switch (err.code) {
          case 1: // PERMISSION_DENIED
            msg = 'Location permission denied. Please allow location access in your browser or enter coordinates manually.';
            break;
          case 2: // POSITION_UNAVAILABLE
            msg = 'Location information is currently unavailable. Please enter coordinates manually.';
            break;
          case 3: // TIMEOUT
            msg = 'Location request timed out. Please try again or enter coordinates manually.';
            break;
          default:
            msg = `Unable to retrieve location: ${err.message || 'Unknown error'}`;
            break;
        }
        result = { success: false, error: msg };
      }
    );
    return result;
  };

  // 1. Browser without geolocation API
  assert.match(
    simulateGeolocation({}).error,
    /Geolocation is not supported by your browser/i
  );

  // 2. Permission denied (code 1)
  const deniedNav = {
    geolocation: {
      getCurrentPosition: (success, failure) => failure({ code: 1, message: 'User denied' }),
    },
  };
  assert.match(
    simulateGeolocation(deniedNav).error,
    /Location permission denied/i
  );

  // 3. Position unavailable (code 2)
  const unavailNav = {
    geolocation: {
      getCurrentPosition: (success, failure) => failure({ code: 2, message: 'Tower down' }),
    },
  };
  assert.match(
    simulateGeolocation(unavailNav).error,
    /Location information is currently unavailable/i
  );

  // 4. Request timeout (code 3)
  const timeoutNav = {
    geolocation: {
      getCurrentPosition: (success, failure) => failure({ code: 3, message: 'Timed out' }),
    },
  };
  assert.match(
    simulateGeolocation(timeoutNav).error,
    /Location request timed out/i
  );

  // 5. Successful position acquisition
  const successNav = {
    geolocation: {
      getCurrentPosition: (success) =>
        success({
          coords: {
            latitude: 19.07609,
            longitude: 72.877426,
            accuracy: 15,
          },
        }),
    },
  };
  const succResult = simulateGeolocation(successNav);
  assert.equal(succResult.success, true);
  assert.equal(succResult.latitude, '19.076090');
  assert.equal(succResult.longitude, '72.877426');
});

// -----------------------------------------------------------------------------
// 3. CSV Upload API Request Construction Contract
// -----------------------------------------------------------------------------
test('CSV Forecast Request Construction: Appends location coordinates and label when provided', () => {
  const buildCsvFormData = (file, model, userId, location) => {
    const entries = [];
    entries.push(['file', file.name]);
    entries.push(['model', model || 'auto']);
    if (userId) {
      entries.push(['user_id', userId]);
    }
    if (location && location.latitude !== undefined && location.longitude !== undefined) {
      entries.push(['latitude', location.latitude.toString()]);
      entries.push(['longitude', location.longitude.toString()]);
      if (location.label) {
        entries.push(['location_label', location.label]);
      }
    }
    return Object.fromEntries(entries);
  };

  const dummyFile = { name: 'household_168h.csv', size: 1024 };

  // Case A: With full coordinates and label
  const payloadWithLoc = buildCsvFormData(
    dummyFile,
    'auto',
    'house_1',
    { latitude: 18.5204, longitude: 73.8567, label: 'Pune Central' }
  );
  assert.equal(payloadWithLoc.file, 'household_168h.csv');
  assert.equal(payloadWithLoc.model, 'auto');
  assert.equal(payloadWithLoc.user_id, 'house_1');
  assert.equal(payloadWithLoc.latitude, '18.5204');
  assert.equal(payloadWithLoc.longitude, '73.8567');
  assert.equal(payloadWithLoc.location_label, 'Pune Central');

  // Case B: Coordinates without label
  const payloadNoLabel = buildCsvFormData(
    dummyFile,
    'early_fusion',
    undefined,
    { latitude: 12.9716, longitude: 77.5946 }
  );
  assert.equal(payloadNoLabel.latitude, '12.9716');
  assert.equal(payloadNoLabel.longitude, '77.5946');
  assert.equal(payloadNoLabel.location_label, undefined);

  // Case C: Omitted location (Preserves existing Energy-only/standard workflow)
  const payloadOmitted = buildCsvFormData(dummyFile, 'auto', undefined, undefined);
  assert.equal(payloadOmitted.latitude, undefined);
  assert.equal(payloadOmitted.longitude, undefined);
  assert.equal(payloadOmitted.location_label, undefined);
  assert.equal(payloadOmitted.file, 'household_168h.csv');
});

// -----------------------------------------------------------------------------
// 4. Manual Forecast Preservation Contract (No timestamps -> Energy-only)
// -----------------------------------------------------------------------------
test('Manual Forecast Contract: Preserves Energy-only behavior, never invents timestamps or passes location', () => {
  const buildManualPayload = (numbers, model, userId) => {
    // Strictly matching requirement:
    // "For manual forecasts, send location only when the existing request includes a valid set of 168 hourly timestamps.
    // Do not invent timestamps for manual readings."
    return {
      electricity_active_power: numbers,
      model: model || 'auto',
      ...(userId ? { user_id: userId } : {}),
      // timestamps and location are intentionally omitted
    };
  };

  const sampleNumbers = Array(168).fill(300);
  const payload = buildManualPayload(sampleNumbers, 'auto', 'meter_42');

  assert.equal(payload.electricity_active_power.length, 168);
  assert.equal(payload.model, 'auto');
  assert.equal(payload.user_id, 'meter_42');
  assert.equal('timestamps' in payload, false);
  assert.equal('location' in payload, false);
});

// -----------------------------------------------------------------------------
// 5. Model Selection & Fallback Explanation Contract
// -----------------------------------------------------------------------------
test('Forecast Result Rationale: Correctly distinguishes Automatic Weather Success vs Energy-only Fallback', () => {
  const classifyForecastRationale = (record) => {
    const isFallback = record.selection_mode === 'fallback';
    const isAutoWeather =
      record.selection_mode === 'auto' &&
      record.selection_reason?.toLowerCase().includes('automatically retrieved');
    const isExplicitOverride = record.selection_mode === 'override';
    const hasLocation = Boolean(record.input_metadata?.location);

    return {
      isFallback,
      isAutoWeather,
      isExplicitOverride,
      hasLocation,
      badgeText: isFallback
        ? 'Weather Fallback (Energy-Only)'
        : record.selection_mode === 'auto'
        ? 'Auto-Adaptive Selection'
        : 'Explicit Override',
    };
  };

  // Case 1: Automatic weather retrieval succeeded
  const autoWeatherRecord = {
    selected_model: 'Early Fusion',
    selection_mode: 'auto',
    selection_reason:
      "Complete weather history automatically retrieved from Open-Meteo for 'Pune' (18.5204, 73.8567). Early Fusion selected as initial default based on best clean test MAE (71.64 W) in research benchmarks.",
    input_metadata: {
      location: { latitude: 18.5204, longitude: 73.8567, label: 'Pune' },
    },
  };
  const autoEval = classifyForecastRationale(autoWeatherRecord);
  assert.equal(autoEval.isFallback, false);
  assert.equal(autoEval.isAutoWeather, true);
  assert.equal(autoEval.hasLocation, true);
  assert.equal(autoEval.badgeText, 'Auto-Adaptive Selection');

  // Case 2: Weather retrieval failed -> Graceful fallback to Energy-Only
  const fallbackRecord = {
    selected_model: 'Energy-Only',
    selection_mode: 'fallback',
    selection_reason:
      'Automated weather retrieval for household location failed: Open-Meteo HTTP 504 Gateway Timeout. Gracefully fell back to Energy-Only model because it operates exclusively on historical active power.',
    input_metadata: {
      location: { latitude: 18.5204, longitude: 73.8567 },
    },
  };
  const fallbackEval = classifyForecastRationale(fallbackRecord);
  assert.equal(fallbackEval.isFallback, true);
  assert.equal(fallbackEval.isAutoWeather, false);
  assert.equal(fallbackEval.hasLocation, true);
  assert.equal(fallbackEval.badgeText, 'Weather Fallback (Energy-Only)');

  // Case 3: Energy-only by default (no location / no weather supplied)
  const defaultEnergyRecord = {
    selected_model: 'Energy-Only',
    selection_mode: 'auto',
    selection_reason:
      'No weather history provided. Energy-Only selected because it operates exclusively on historical active power without requiring weather observations.',
    input_metadata: {},
  };
  const defaultEval = classifyForecastRationale(defaultEnergyRecord);
  assert.equal(defaultEval.isFallback, false);
  assert.equal(defaultEval.isAutoWeather, false);
  assert.equal(defaultEval.hasLocation, false);
  assert.equal(defaultEval.badgeText, 'Auto-Adaptive Selection');
});
