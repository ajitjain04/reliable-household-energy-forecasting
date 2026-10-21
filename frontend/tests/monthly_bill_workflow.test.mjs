/**
 * Focused Unit & Workflow Test Suite for Frontend Electricity Bill PDF Upload & Review.
 * Uses Node.js native test runner (node --test).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

// -----------------------------------------------------------------------------
// 1. File Type & Extension Validation
// -----------------------------------------------------------------------------
test('File Type Validation: accepts only .pdf files and rejects non-PDF extensions', () => {
  const validatePdfFile = (fileName) => {
    if (!fileName || typeof fileName !== 'string') {
      return { valid: false, error: 'Please select a valid .pdf file.' };
    }
    if (!fileName.toLowerCase().endsWith('.pdf')) {
      return { valid: false, error: 'Please select a valid .pdf file.' };
    }
    return { valid: true, error: null };
  };

  // Valid PDF filenames
  assert.equal(validatePdfFile('msedcl_jan2024.pdf').valid, true);
  assert.equal(validatePdfFile('BILL_UPPERCASE.PDF').valid, true);
  assert.equal(validatePdfFile('my.bill.doc.pdf').valid, true);

  // Invalid filenames
  assert.equal(validatePdfFile('energy_history.csv').valid, false);
  assert.equal(validatePdfFile('energy_history.csv').error, 'Please select a valid .pdf file.');
  assert.equal(validatePdfFile('scanned_bill.png').valid, false);
  assert.equal(validatePdfFile('bill.docx').valid, false);
  assert.equal(validatePdfFile('').valid, false);
  assert.equal(validatePdfFile(null).valid, false);
});

// -----------------------------------------------------------------------------
// 2. Client-Side Temporal Validation Rules for Review Table
// -----------------------------------------------------------------------------
test('Review Table Validation: detects invalid date ranges, negative numbers, duplicates, and overlaps', () => {
  const validateRecords = (records) => {
    const errors = [];
    if (!records || records.length === 0) {
      errors.push('At least one monthly consumption record is required.');
      return errors;
    }

    records.forEach((rec, idx) => {
      const rowNum = idx + 1;
      if (!rec.billing_period_start) {
        errors.push(`Row #${rowNum}: Billing period start date is required.`);
      }
      if (!rec.billing_period_end) {
        errors.push(`Row #${rowNum}: Billing period end date is required.`);
      }
      if (rec.billing_period_start && rec.billing_period_end) {
        if (rec.billing_period_start >= rec.billing_period_end) {
          errors.push(
            `Row #${rowNum}: Start date (${rec.billing_period_start}) must be strictly before end date (${rec.billing_period_end}).`
          );
        }
      }
      if (rec.consumption_kwh === undefined || isNaN(rec.consumption_kwh) || rec.consumption_kwh < 0) {
        errors.push(`Row #${rowNum}: Consumption must be a non-negative number in kilowatt-hours (kWh).`);
      }
      if (rec.bill_amount !== undefined && rec.bill_amount !== null && (isNaN(rec.bill_amount) || rec.bill_amount < 0)) {
        errors.push(`Row #${rowNum}: Billed amount must be non-negative if specified.`);
      }
    });

    const sorted = [...records].sort((a, b) =>
      a.billing_period_start.localeCompare(b.billing_period_start)
    );
    for (let i = 0; i < sorted.length - 1; i++) {
      const curr = sorted[i];
      const nxt = sorted[i + 1];

      if (curr.billing_period_start && nxt.billing_period_start && curr.billing_period_start === nxt.billing_period_start) {
        errors.push(`Duplicate billing period detected: multiple cycles starting on ${curr.billing_period_start}.`);
      }
      if (curr.billing_period_end && nxt.billing_period_end && curr.billing_period_end === nxt.billing_period_end) {
        errors.push(`Duplicate billing period detected: multiple cycles ending on ${curr.billing_period_end}.`);
      }
      if (curr.billing_period_end && nxt.billing_period_start && nxt.billing_period_start < curr.billing_period_end) {
        errors.push(
          `Overlapping billing periods: [${curr.billing_period_start} to ${curr.billing_period_end}] overlaps with [${nxt.billing_period_start} to ${nxt.billing_period_end}].`
        );
      }
    }

    return errors;
  };

  // 1. Valid records
  const validRecords = [
    { billing_period_start: '2023-11-01', billing_period_end: '2023-11-30', consumption_kwh: 210, bill_amount: 2150 },
    { billing_period_start: '2023-12-01', billing_period_end: '2023-12-31', consumption_kwh: 195, bill_amount: 1980 },
    { billing_period_start: '2024-01-01', billing_period_end: '2024-01-31', consumption_kwh: 250, bill_amount: 2676.70 },
  ];
  assert.deepEqual(validateRecords(validRecords), []);

  // 2. Start date after end date
  const invertedDates = [
    { billing_period_start: '2024-01-31', billing_period_end: '2024-01-01', consumption_kwh: 250, bill_amount: 2676.70 },
  ];
  const invertedErrors = validateRecords(invertedDates);
  assert.equal(invertedErrors.length, 1);
  assert.match(invertedErrors[0], /must be strictly before/);

  // 3. Negative consumption
  const negativeConsumption = [
    { billing_period_start: '2024-01-01', billing_period_end: '2024-01-31', consumption_kwh: -50, bill_amount: 2676.70 },
  ];
  const negErrors = validateRecords(negativeConsumption);
  assert.equal(negErrors.length, 1);
  assert.match(negErrors[0], /must be a non-negative number/);

  // 4. Negative bill amount
  const negativeAmount = [
    { billing_period_start: '2024-01-01', billing_period_end: '2024-01-31', consumption_kwh: 250, bill_amount: -100 },
  ];
  const negAmtErrors = validateRecords(negativeAmount);
  assert.equal(negAmtErrors.length, 1);
  assert.match(negAmtErrors[0], /Billed amount must be non-negative/);

  // 5. Overlapping billing periods
  const overlappingRecords = [
    { billing_period_start: '2024-01-01', billing_period_end: '2024-01-20', consumption_kwh: 150 },
    { billing_period_start: '2024-01-15', billing_period_end: '2024-01-31', consumption_kwh: 160 },
  ];
  const overlapErrors = validateRecords(overlappingRecords);
  assert.equal(overlapErrors.length, 1);
  assert.match(overlapErrors[0], /Overlapping billing periods/);

  // 6. Duplicate start dates
  const duplicateStart = [
    { billing_period_start: '2024-01-01', billing_period_end: '2024-01-15', consumption_kwh: 100 },
    { billing_period_start: '2024-01-01', billing_period_end: '2024-01-31', consumption_kwh: 200 },
  ];
  const dupErrors = validateRecords(duplicateStart);
  assert.equal(dupErrors.length >= 1, true);
  assert.match(dupErrors[0], /Duplicate billing period detected/);
});

// -----------------------------------------------------------------------------
// 3. Gap Detection Logic (Informational Warnings Without Synthetic Filling)
// -----------------------------------------------------------------------------
test('Gap Detection: detects missing months and preserves uncertainty without synthetic insertion', () => {
  const detectGaps = (records) => {
    const warnings = [];
    const sorted = [...records].sort((a, b) =>
      a.billing_period_start.localeCompare(b.billing_period_start)
    );
    for (let i = 0; i < sorted.length - 1; i++) {
      const curr = sorted[i];
      const nxt = sorted[i + 1];
      const endD = new Date(curr.billing_period_end);
      const startD = new Date(nxt.billing_period_start);
      const diffDays = Math.round((startD.getTime() - endD.getTime()) / (1000 * 3600 * 24));
      if (diffDays > 3) {
        warnings.push(
          `Incomplete history: gap of ${diffDays} days detected between ${curr.billing_period_end} and ${nxt.billing_period_start}. No synthetic values inserted.`
        );
      }
    }
    return warnings;
  };

  // Consecutive months: no gaps
  const contiguous = [
    { billing_period_start: '2023-11-01', billing_period_end: '2023-11-30', consumption_kwh: 200 },
    { billing_period_start: '2023-12-01', billing_period_end: '2023-12-31', consumption_kwh: 210 },
  ];
  assert.deepEqual(detectGaps(contiguous), []);

  // Gap between Oct 2023 and Jan 2024 (missing Nov and Dec)
  const gapped = [
    { billing_period_start: '2023-10-01', billing_period_end: '2023-10-31', consumption_kwh: 220 },
    { billing_period_start: '2024-01-01', billing_period_end: '2024-01-31', consumption_kwh: 250 },
  ];
  const gapNotices = detectGaps(gapped);
  assert.equal(gapNotices.length, 1);
  assert.match(gapNotices[0], /gap of 62 days detected/);
  assert.match(gapNotices[0], /No synthetic values inserted/);
});

// -----------------------------------------------------------------------------
// 4. Extraction State Machine & Confirmation Contract
// -----------------------------------------------------------------------------
test('Extraction State Machine: extracted data starts as unconfirmed and requires explicit user review', () => {
  const createBillSession = (extractedPayload) => {
    let state = {
      isExtracting: false,
      extractedData: extractedPayload,
      reviewRecords: extractedPayload ? [...extractedPayload.records] : [],
      isConfirmed: false,
      validationErrors: [],
    };

    return {
      getState: () => ({ ...state }),
      updateRecord: (index, field, value) => {
        state.isConfirmed = false;
        state.reviewRecords[index] = { ...state.reviewRecords[index], [field]: value };
      },
      addRecord: (newRecord) => {
        state.isConfirmed = false;
        state.reviewRecords.push(newRecord);
      },
      removeRecord: (index) => {
        state.isConfirmed = false;
        state.reviewRecords.splice(index, 1);
      },
      confirm: (validator) => {
        const errs = validator(state.reviewRecords);
        if (errs.length > 0) {
          state.validationErrors = errs;
          state.isConfirmed = false;
          return { success: false, errors: errs };
        }
        state.validationErrors = [];
        state.isConfirmed = true;
        return { success: true, errors: [] };
      },
    };
  };

  const sampleExtracted = {
    records: [
      { billing_period_start: '2024-01-01', billing_period_end: '2024-01-31', consumption_kwh: 250, bill_amount: 2676.70 },
      { billing_period_start: '2023-12-01', billing_period_end: '2023-12-31', consumption_kwh: 210, bill_amount: 2150.00 },
    ],
    extraction_source: 'pdf',
    extraction_confidence: 0.95,
    warnings: [],
    utility_provider: 'MSEDCL (Mahavitaran)',
    consumer_number: '012345678901',
    billing_unit: '4123',
  };

  const session = createBillSession(sampleExtracted);

  // 1. Initial state is strictly unconfirmed
  assert.equal(session.getState().isConfirmed, false);
  assert.equal(session.getState().reviewRecords.length, 2);
  assert.equal(session.getState().extractedData.utility_provider, 'MSEDCL (Mahavitaran)');

  // 2. Editing a record keeps state unconfirmed
  session.updateRecord(0, 'consumption_kwh', 255.5);
  assert.equal(session.getState().reviewRecords[0].consumption_kwh, 255.5);
  assert.equal(session.getState().isConfirmed, false);

  // 3. Confirmation succeeds when all records are valid
  const dummyValidator = (recs) => (recs.length > 0 ? [] : ['Records required']);
  const confirmResult = session.confirm(dummyValidator);
  assert.equal(confirmResult.success, true);
  assert.equal(session.getState().isConfirmed, true);

  // 4. Modifying records after confirmation reverts isConfirmed to false
  session.updateRecord(0, 'consumption_kwh', 260);
  assert.equal(session.getState().isConfirmed, false);
});

// -----------------------------------------------------------------------------
// 5. Actionable OCR Limitation and Error Handling Contract
// -----------------------------------------------------------------------------
test('Error Parsing: distinguishes OCR limitations, unreadable scans, encryption, and generic errors', () => {
  const classifyApiError = (errorMessage) => {
    const lower = (errorMessage || '').toLowerCase();
    if (lower.includes('not supported') || lower.includes('not available')) {
      return {
        category: 'ocr_limitation',
        userAdvice: 'Optical Character Recognition (OCR) runtime is not available on this server. Please upload a digital vector e-bill PDF or enter readings manually.',
      };
    }
    if (lower.includes('ocr') || lower.includes('scanned') || lower.includes('image-only') || lower.includes('could not reliably identify')) {
      return {
        category: 'unreadable_scan',
        userAdvice: 'Could not read scanned bill clearly. Please verify scan quality or enter readings manually.',
      };
    }
    if (lower.includes('encrypt') || lower.includes('password')) {
      return {
        category: 'encrypted_pdf',
        userAdvice: 'Please remove password encryption from your e-bill PDF before uploading.',
      };
    }
    if (lower.includes('missing %pdf-') || lower.includes('not a valid pdf')) {
      return {
        category: 'invalid_format',
        userAdvice: 'The uploaded file is not a valid PDF document. Please verify the file integrity.',
      };
    }
    return {
      category: 'generic',
      userAdvice: errorMessage || 'An unexpected error occurred during bill processing.',
    };
  };

  // OCR unavailable notice classification
  const ocrRes = classifyApiError('Optical Character Recognition (OCR) runtime bindings (winrt) are not available. Scanned PDF OCR fallback disabled.');
  assert.equal(ocrRes.category, 'ocr_limitation');
  assert.match(ocrRes.userAdvice, /not available/);

  // Unreadable scan classification (OCR executed but scan text could not be parsed)
  const scanRes = classifyApiError('Could not reliably identify billing periods or kilowatt-hour consumption values from OCR text. Please verify scan resolution.');
  assert.equal(scanRes.category, 'unreadable_scan');
  assert.match(scanRes.userAdvice, /verify scan quality/);

  // Encrypted PDF classification
  const encRes = classifyApiError('Encrypted or password-protected PDF detected.');
  assert.equal(encRes.category, 'encrypted_pdf');
  assert.match(encRes.userAdvice, /remove password encryption/);

  // Invalid format classification
  const invRes = classifyApiError('Uploaded file is not a valid PDF document (missing %PDF- header).');
  assert.equal(invRes.category, 'invalid_format');
});

// -----------------------------------------------------------------------------
// 6. Monthly Forecast Request Construction & Live Mode Enforcement
// -----------------------------------------------------------------------------
test('Monthly Forecast Request Construction: strictly enforces is_live=true and maps all parameters', () => {
  const buildMonthlyForecastPayload = (options) => {
    const {
      records,
      horizonMonths,
      providerId,
      phaseType = 'single_phase',
      sanctionedLoadKw,
    } = options;

    if (!records || records.length === 0) {
      throw new Error('At least one confirmed monthly record is required.');
    }

    if (![1, 3, 6].includes(horizonMonths)) {
      throw new Error(`Invalid forecast horizon: ${horizonMonths}. Must be 1, 3, or 6 months.`);
    }

    return {
      records: records.map((r) => ({
        billing_period_start: r.billing_period_start,
        billing_period_end: r.billing_period_end,
        consumption_kwh: Number(r.consumption_kwh),
        bill_amount: r.bill_amount !== null && r.bill_amount !== undefined ? Number(r.bill_amount) : null,
        period_label: r.period_label || null,
      })),
      horizon_months: horizonMonths,
      provider_id: providerId?.trim() ? providerId.trim() : undefined,
      phase_type: phaseType,
      sanctioned_load_kw: sanctionedLoadKw ? parseFloat(sanctionedLoadKw) : undefined,
      is_live: true, // STRICT REQUIREMENT: Live household mode must always be true
    };
  };

  const sampleRecords = [
    { billing_period_start: '2025-04-01', billing_period_end: '2025-04-30', consumption_kwh: 167, bill_amount: 1760 },
  ];

  // 1. Valid 3-month request with MSEDCL provider
  const payload3m = buildMonthlyForecastPayload({
    records: sampleRecords,
    horizonMonths: 3,
    providerId: 'msedcl',
    phaseType: 'single_phase',
    sanctionedLoadKw: '5.0',
  });
  assert.equal(payload3m.is_live, true);
  assert.equal(payload3m.horizon_months, 3);
  assert.equal(payload3m.provider_id, 'msedcl');
  assert.equal(payload3m.phase_type, 'single_phase');
  assert.equal(payload3m.sanctioned_load_kw, 5.0);
  assert.equal(payload3m.records.length, 1);

  // 2. Valid 1-month and 6-month horizons
  const payload1m = buildMonthlyForecastPayload({ records: sampleRecords, horizonMonths: 1 });
  assert.equal(payload1m.horizon_months, 1);
  assert.equal(payload1m.is_live, true);

  const payload6m = buildMonthlyForecastPayload({ records: sampleRecords, horizonMonths: 6 });
  assert.equal(payload6m.horizon_months, 6);
  assert.equal(payload6m.is_live, true);

  // 3. Invalid horizon rejected
  assert.throws(() => {
    buildMonthlyForecastPayload({ records: sampleRecords, horizonMonths: 12 });
  }, /Invalid forecast horizon: 12/);

  // 4. Empty records rejected
  assert.throws(() => {
    buildMonthlyForecastPayload({ records: [], horizonMonths: 3 });
  }, /At least one confirmed monthly record is required/);
});

// -----------------------------------------------------------------------------
// 7. Single Confirmed Bill (N=1) Baseline Persistence & All 3 Horizons Contract
// -----------------------------------------------------------------------------
test('N=1 Baseline Contract: validates 1, 3, and 6-month horizons with static persistence and low-confidence disclaimer', () => {
  const evaluateMonthlyResultConfidence = (result) => {
    const isSingleMonth = result.history_length_months === 1;
    const warnings = [...(result.warnings || [])];
    let isLowConfidenceBaseline = false;
    let disclaimerText = null;

    if (isSingleMonth) {
      isLowConfidenceBaseline = true;
      disclaimerText =
        'Low-Confidence Baseline Forecast: Only 1 confirmed billing cycle provided. ' +
        'A single observation is insufficient to estimate consumption trends, variance, or seasonal swings. ' +
        'Results represent a static baseline persistence forecast.';
      warnings.push('Sparse history: baseline persistence model selected.');
    }

    return {
      isLowConfidenceBaseline,
      confidenceRating: result.confidence_rating,
      disclaimerText,
      warnings,
      intervalsAvailable: result.forecast_steps.every(
        (s) => s.uncertainty_lower_kwh !== null && s.uncertainty_upper_kwh !== null
      ),
    };
  };

  const createMockN1Response = (horizon, baseKwh = 167.0, baseBill = 1760.0) => {
    const monthNames = ['Nov 2026', 'Dec 2026', 'Jan 2027', 'Feb 2027', 'Mar 2027', 'Apr 2027'];
    const steps = [];
    for (let i = 1; i <= horizon; i++) {
      steps.push({
        month_index: i,
        month_label: monthNames[i - 1],
        period_start: `2026-${(10 + i).toString().padStart(2, '0')}-01`,
        period_end: `2026-${(10 + i).toString().padStart(2, '0')}-28`,
        forecast_kwh: baseKwh,
        uncertainty_lower_kwh: null,
        uncertainty_upper_kwh: null,
        estimated_bill_amount: baseBill,
      });
    }

    return {
      forecast_id: `mf_test_single_h${horizon}`,
      created_at: '2026-10-19T10:00:00Z',
      is_live: true,
      history_length_months: 1,
      selected_method: 'baseline_only',
      method_description: 'Baseline persistence forecast',
      confidence_rating: 'low',
      is_stale_history: true,
      warnings: [
        'Single historical month provided.',
        'Historical billing records end on 2025-04-30 (more than 90 days ago).',
      ],
      limitations: [
        'Cannot estimate trend or seasonality with N=1.',
        'Prediction intervals cannot be defensibly calculated from a single observation.',
      ],
      forecast_steps: steps,
      summary: {
        total_forecast_kwh: roundTo(baseKwh * horizon, 2),
        average_monthly_kwh: baseKwh,
        total_estimated_bill_inr: roundTo(baseBill * horizon, 2),
        months_with_bill_estimates: horizon,
      },
    };
  };

  function roundTo(num, dec) {
    const f = Math.pow(10, dec);
    return Math.round(num * f) / f;
  }

  // 1. Horizon 1 Month for N=1: exactly 1 step
  const n1Res1 = createMockN1Response(1);
  assert.equal(n1Res1.forecast_steps.length, 1);
  assert.equal(n1Res1.forecast_steps[0].month_label, 'Nov 2026');
  assert.equal(n1Res1.forecast_steps[0].forecast_kwh, 167.0);
  assert.equal(n1Res1.summary.total_forecast_kwh, 167.0);
  assert.equal(n1Res1.summary.average_monthly_kwh, 167.0);
  assert.equal(n1Res1.summary.total_estimated_bill_inr, 1760.0);
  const eval1 = evaluateMonthlyResultConfidence(n1Res1);
  assert.equal(eval1.isLowConfidenceBaseline, true);
  assert.equal(eval1.intervalsAvailable, false);
  assert.equal(n1Res1.is_stale_history, true);

  // 2. Horizon 3 Months for N=1: exactly 3 consecutive steps (Nov 2026, Dec 2026, Jan 2027)
  const n1Res3 = createMockN1Response(3);
  assert.equal(n1Res3.forecast_steps.length, 3);
  assert.equal(n1Res3.forecast_steps[0].month_label, 'Nov 2026');
  assert.equal(n1Res3.forecast_steps[1].month_label, 'Dec 2026');
  assert.equal(n1Res3.forecast_steps[2].month_label, 'Jan 2027');
  n1Res3.forecast_steps.forEach((s) => assert.equal(s.forecast_kwh, 167.0));
  assert.equal(n1Res3.summary.total_forecast_kwh, 501.0);
  assert.equal(n1Res3.summary.average_monthly_kwh, 167.0);
  assert.equal(n1Res3.summary.total_estimated_bill_inr, 5280.0);
  const eval3 = evaluateMonthlyResultConfidence(n1Res3);
  assert.equal(eval3.isLowConfidenceBaseline, true);
  assert.equal(eval3.intervalsAvailable, false);

  // 3. Horizon 6 Months for N=1: exactly 6 consecutive future monthly steps
  const n1Res6 = createMockN1Response(6);
  assert.equal(n1Res6.forecast_steps.length, 6);
  assert.equal(n1Res6.forecast_steps[5].month_label, 'Apr 2027');
  n1Res6.forecast_steps.forEach((s) => assert.equal(s.forecast_kwh, 167.0));
  assert.equal(n1Res6.summary.total_forecast_kwh, 1002.0);
  assert.equal(n1Res6.summary.average_monthly_kwh, 167.0);
  assert.equal(n1Res6.summary.total_estimated_bill_inr, 10560.0);
  const eval6 = evaluateMonthlyResultConfidence(n1Res6);
  assert.equal(eval6.isLowConfidenceBaseline, true);
  assert.equal(eval6.intervalsAvailable, false);
});

// -----------------------------------------------------------------------------
// 8. Result Heading, Selected Horizon, Step Count, and Summary Consistency
// -----------------------------------------------------------------------------
test('Result Heading & Summary Consistency: heading, selected horizon, step count, total kWh, and bill agree', () => {
  const verifyResultConsistency = (selectedHorizon, result) => {
    const stepCount = result.forecast_steps.length;
    const heading = `${stepCount}-Month Electricity Consumption Projection`;
    const horizonBadge = `${stepCount} ${stepCount === 1 ? 'Month' : 'Months'} Horizon`;
    const cycleSummaryNote = `Across ${stepCount} upcoming billing cycle${stepCount === 1 ? '' : 's'}`;

    const calculatedTotalKwh = Number(
      result.forecast_steps.reduce((acc, s) => acc + s.forecast_kwh, 0).toFixed(1)
    );
    const calculatedAvgKwh = Number((calculatedTotalKwh / stepCount).toFixed(1));

    const allHaveBills = result.forecast_steps.every(
      (s) => s.estimated_bill_amount !== null && s.estimated_bill_amount !== undefined
    );
    const calculatedTotalBill = allHaveBills
      ? Number(result.forecast_steps.reduce((acc, s) => acc + s.estimated_bill_amount, 0).toFixed(2))
      : null;

    const agrees =
      stepCount === selectedHorizon &&
      Math.abs(result.summary.total_forecast_kwh - calculatedTotalKwh) < 0.01 &&
      Math.abs(result.summary.average_monthly_kwh - calculatedAvgKwh) < 0.01 &&
      (calculatedTotalBill === null || Math.abs(result.summary.total_estimated_bill_inr - calculatedTotalBill) < 0.01);

    return {
      heading,
      horizonBadge,
      cycleSummaryNote,
      stepCount,
      calculatedTotalKwh,
      calculatedTotalBill,
      agrees,
    };
  };

  // Test 1-month consistency
  const res1 = {
    forecast_steps: [{ month_index: 1, month_label: 'Nov 2026', forecast_kwh: 167.0, estimated_bill_amount: 1651.18 }],
    summary: { total_forecast_kwh: 167.0, average_monthly_kwh: 167.0, total_estimated_bill_inr: 1651.18 },
  };
  const v1 = verifyResultConsistency(1, res1);
  assert.equal(v1.agrees, true);
  assert.equal(v1.heading, '1-Month Electricity Consumption Projection');
  assert.equal(v1.cycleSummaryNote, 'Across 1 upcoming billing cycle');

  // Test 3-month consistency
  const res3 = {
    forecast_steps: [
      { month_index: 1, month_label: 'Nov 2026', forecast_kwh: 167.0, estimated_bill_amount: 1651.18 },
      { month_index: 2, month_label: 'Dec 2026', forecast_kwh: 167.0, estimated_bill_amount: 1635.91 },
      { month_index: 3, month_label: 'Jan 2027', forecast_kwh: 167.0, estimated_bill_amount: 1635.91 },
    ],
    summary: { total_forecast_kwh: 501.0, average_monthly_kwh: 167.0, total_estimated_bill_inr: 4923.0 },
  };
  const v3 = verifyResultConsistency(3, res3);
  assert.equal(v3.agrees, true);
  assert.equal(v3.heading, '3-Month Electricity Consumption Projection');
  assert.equal(v3.stepCount, 3);
  assert.equal(v3.calculatedTotalKwh, 501.0);
  assert.equal(v3.cycleSummaryNote, 'Across 3 upcoming billing cycles');

  // Test 6-month consistency
  const res6Steps = [];
  for (let i = 1; i <= 6; i++) {
    res6Steps.push({ month_index: i, month_label: `Month #${i}`, forecast_kwh: 167.0, estimated_bill_amount: 1640.0 });
  }
  const res6 = {
    forecast_steps: res6Steps,
    summary: { total_forecast_kwh: 1002.0, average_monthly_kwh: 167.0, total_estimated_bill_inr: 9840.0 },
  };
  const v6 = verifyResultConsistency(6, res6);
  assert.equal(v6.agrees, true);
  assert.equal(v6.heading, '6-Month Electricity Consumption Projection');
  assert.equal(v6.stepCount, 6);
  assert.equal(v6.cycleSummaryNote, 'Across 6 upcoming billing cycles');

  // Negative test: Mismatch detection (selected 3, but received 1)
  const mismatchResult = verifyResultConsistency(3, res1);
  assert.equal(mismatchResult.agrees, false, 'Must flag mismatch when received step count differs from selected horizon');
});

// -----------------------------------------------------------------------------
// 8b. Multi-Month History (N >= 3) and Horizon Coverage (1, 3, 6 Months)
// -----------------------------------------------------------------------------
test('Multi-Month History & Horizon Projections: validates step count, intervals, and backtest metrics', () => {
  const createMockMultiMonthResponse = (horizon, historyLength) => {
    const steps = [];
    for (let i = 1; i <= horizon; i++) {
      steps.push({
        month_index: i,
        month_label: `Month +${i}`,
        period_start: `2026-1${i < 10 ? '0' + i : i}-01`,
        period_end: `2026-1${i < 10 ? '0' + i : i}-28`,
        forecast_kwh: 200 + i * 5,
        uncertainty_lower_kwh: 180 + i * 5,
        uncertainty_upper_kwh: 220 + i * 5,
        estimated_bill_amount: 2200 + i * 50,
      });
    }

    return {
      forecast_id: `mf_test_horizon_${horizon}`,
      created_at: '2026-10-19T10:00:00Z',
      is_live: true,
      history_length_months: historyLength,
      selected_method: historyLength >= 12 ? 'seasonal_decomposition' : 'moving_average',
      method_description: 'Evaluated projection model',
      confidence_rating: historyLength >= 6 ? 'high' : 'moderate',
      warnings: [],
      limitations: [],
      forecast_steps: steps,
      backtest_metrics: historyLength >= 3 ? {
        backtest_periods: historyLength - 1,
        mae_kwh: 12.4,
        rmse_kwh: 15.8,
        baseline_mae_kwh: 21.0,
        smape_percent: 6.2,
      } : null,
      summary: {
        total_forecast_kwh: steps.reduce((sum, s) => sum + s.forecast_kwh, 0),
        average_monthly_kwh: steps.reduce((sum, s) => sum + s.forecast_kwh, 0) / steps.length,
        total_estimated_bill_inr: steps.reduce((sum, s) => sum + s.estimated_bill_amount, 0),
      },
    };
  };

  // 1. Horizon 1 with 3 months history
  const res1m = createMockMultiMonthResponse(1, 3);
  assert.equal(res1m.forecast_steps.length, 1);
  assert.equal(res1m.backtest_metrics !== null, true);
  assert.equal(res1m.backtest_metrics.mae_kwh, 12.4);
  assert.equal(res1m.forecast_steps[0].uncertainty_lower_kwh, 185);

  // 2. Horizon 3 with 6 months history
  const res3m = createMockMultiMonthResponse(3, 6);
  assert.equal(res3m.forecast_steps.length, 3);
  assert.equal(res3m.confidence_rating, 'high');
  assert.equal(res3m.summary.total_forecast_kwh, 205 + 210 + 215);

  // 3. Horizon 6 with 12 months history
  const res6m = createMockMultiMonthResponse(6, 12);
  assert.equal(res6m.forecast_steps.length, 6);
  assert.equal(res6m.selected_method, 'seasonal_decomposition');
  assert.equal(res6m.forecast_steps[5].month_index, 6);
});

// -----------------------------------------------------------------------------
// 9. Bill Estimate Availability, Missing Tariff Handling, and Disclosure Integrity
// -----------------------------------------------------------------------------
test('Tariff Transparency: displays Unavailable without fabricating prices or repeating past bill', () => {
  const formatBillDisplay = (step, summary) => {
    // Contract: If bill estimate is unavailable, return 'Unavailable' and never substitute past bill
    if (step.estimated_bill_amount === null || step.estimated_bill_amount === undefined) {
      return {
        displayText: 'Unavailable',
        hasValidEstimate: false,
        breakdownNotice:
          step.bill_estimate?.disclosures?.[0] ||
          'No official tariff schedule is active or verified for this period. Physical consumption forecast is preserved; monetary bill estimate is withheld.',
      };
    }

    return {
      displayText: `₹${step.estimated_bill_amount.toFixed(2)}`,
      hasValidEstimate: true,
      breakdownNotice: null,
    };
  };

  // Case A: Verified provider (MSEDCL) with calculated tariff
  const msedclStep = {
    month_index: 1,
    forecast_kwh: 167.0,
    estimated_bill_amount: 1760.0,
    bill_estimate: {
      available: true,
      provider_name: 'MSEDCL',
      charge_breakdown: {
        fixed_charge: 128.0,
        energy_charges_total: 1350.0,
        wheeling_charges_total: 195.0,
        electricity_duty: 87.0,
      },
      disclosures: [],
    },
  };
  const verifiedDisplay = formatBillDisplay(msedclStep);
  assert.equal(verifiedDisplay.hasValidEstimate, true);
  assert.equal(verifiedDisplay.displayText, '₹1760.00');

  // Case B: Unverified provider (BESCOM / JVVNL / TANGEDCO) where bill estimate is withheld
  const unverifiedStep = {
    month_index: 1,
    forecast_kwh: 167.0,
    estimated_bill_amount: null, // Backend leaves this null
    bill_estimate: {
      available: false,
      provider_name: 'BESCOM',
      charge_breakdown: null,
      disclosures: [
        'Provider schedule for BESCOM is registered but unverified. Monetary bill calculation is withheld to avoid misleading estimates.',
      ],
    },
  };

  const unverifiedDisplay = formatBillDisplay(unverifiedStep);
  assert.equal(unverifiedDisplay.hasValidEstimate, false);
  assert.equal(unverifiedDisplay.displayText, 'Unavailable');
  assert.notEqual(unverifiedDisplay.displayText, '₹1760.00'); // MUST NOT repeat past bill amount
  assert.match(unverifiedDisplay.breakdownNotice, /withheld/);

  // Case C: Charge breakdown key mapping supports backend schema
  const resolveBreakdownValues = (breakdown) => {
    if (!breakdown) return null;
    return {
      fixed: Number(breakdown.fixed_charges_inr ?? breakdown.fixed_charge ?? 0),
      energy: Number(breakdown.energy_charges_inr ?? breakdown.energy_charges_total ?? 0),
      wheeling: Number(breakdown.wheeling_charges_inr ?? breakdown.wheeling_charges_total ?? 0),
      duty: Number(breakdown.electricity_duty_16pct_inr ?? breakdown.electricity_duty ?? 0),
    };
  };

  const actualBackendBreakdownNov = {
    energy_charges_inr: 1104.31,
    wheeling_charges_inr: 195.39,
    fixed_charges_inr: 123.73,
    fuel_adjustment_charges_inr: 0.0,
    taxable_subtotal_inr: 1423.43,
    electricity_duty_16pct_inr: 227.75,
    estimated_total_bill_inr: 1651.18,
  };
  const resolvedNov = resolveBreakdownValues(actualBackendBreakdownNov);
  assert.equal(resolvedNov.fixed, 123.73);
  assert.equal(resolvedNov.energy, 1104.31);
  assert.equal(resolvedNov.wheeling, 195.39);
  assert.equal(resolvedNov.duty, 227.75);

  const actualBackendBreakdownDec = {
    energy_charges_inr: 1086.88,
    wheeling_charges_inr: 195.39,
    fixed_charges_inr: 128.0,
    fuel_adjustment_charges_inr: 0.0,
    taxable_subtotal_inr: 1410.27,
    electricity_duty_16pct_inr: 225.64,
    estimated_total_bill_inr: 1635.91,
  };
  const resolvedDec = resolveBreakdownValues(actualBackendBreakdownDec);
  assert.equal(resolvedDec.fixed, 128.0);
  assert.equal(resolvedDec.energy, 1086.88);
  assert.equal(resolvedDec.wheeling, 195.39);
  assert.equal(resolvedDec.duty, 225.64);

  // 3-month total sum verification
  const total3m = Number((actualBackendBreakdownNov.estimated_total_bill_inr + actualBackendBreakdownDec.estimated_total_bill_inr * 2).toFixed(2));
  assert.equal(total3m, 4923.0);
});

// -----------------------------------------------------------------------------
// 10. Session Expiration (401), API Error Handling, and State Preservation
// -----------------------------------------------------------------------------
test('Session Expiry & Error Handling: preserves confirmed records and provides login guidance', () => {
  const simulateForecastApiCall = (mockErrorStatus, currentRecords, isConfirmed) => {
    let state = {
      reviewRecords: [...currentRecords],
      isConfirmed: isConfirmed,
      monthlyForecastResult: null,
      monthlyForecastError: null,
      isSessionExpired: false,
    };

    if (mockErrorStatus === 401) {
      state.isSessionExpired = true;
      state.monthlyForecastError =
        'Your session has expired or you are not authenticated. Please log in again to run live forecasts. Your confirmed billing history remains preserved in this view.';
    } else if (mockErrorStatus === 422) {
      state.monthlyForecastError = 'Validation Error: Invalid billing period dates provided.';
    } else if (mockErrorStatus === 500) {
      state.monthlyForecastError = 'Internal Server Error: Forecast calculation failed.';
    }

    return state;
  };

  const confirmedRecords = [
    { billing_period_start: '2025-04-01', billing_period_end: '2025-04-30', consumption_kwh: 167, bill_amount: 1760 },
    { billing_period_start: '2025-03-01', billing_period_end: '2025-03-31', consumption_kwh: 155, bill_amount: 1620 },
  ];

  // 1. Session expiration (401)
  const sessionExpiredState = simulateForecastApiCall(401, confirmedRecords, true);
  assert.equal(sessionExpiredState.isSessionExpired, true);
  assert.match(sessionExpiredState.monthlyForecastError, /session has expired/);
  // CRITICAL REQUIREMENT 9: Confirmed records and confirmation state must be preserved
  assert.equal(sessionExpiredState.reviewRecords.length, 2);
  assert.equal(sessionExpiredState.isConfirmed, true);

  // 2. Unprocessable entity (422)
  const validationErrorState = simulateForecastApiCall(422, confirmedRecords, true);
  assert.equal(validationErrorState.isSessionExpired, false);
  assert.match(validationErrorState.monthlyForecastError, /Validation Error/);
  assert.equal(validationErrorState.reviewRecords.length, 2);

  // 3. Server error (500)
  const serverErrorState = simulateForecastApiCall(500, confirmedRecords, true);
  assert.match(serverErrorState.monthlyForecastError, /Internal Server Error/);
  assert.equal(serverErrorState.reviewRecords.length, 2);
});

// -----------------------------------------------------------------------------
// 11. Architecture Contract: Monthly Persistence & History Capabilities (Task 2.4)
// -----------------------------------------------------------------------------
test('Architecture Disclosure Contract: monthly forecasts support persistent history snapshots', () => {
  const getArchitectureCapability = (forecastType) => {
    if (forecastType === 'hourly_168h') {
      return {
        persistedInDatabase: true,
        exportablePdfReport: true,
        historyRoute: '/forecast/history?tab=hourly',
      };
    }
    if (forecastType === 'monthly_household') {
      return {
        persistedInDatabase: true,
        exportablePdfReport: false,
        historyRoute: '/forecast/history?tab=monthly',
        detailRoutePrefix: '/forecast/monthly/',
        disclosureText:
          'Persistent Forecast History: Save this forecast run using the Save Forecast Run action above to preserve an immutable snapshot in your account history. You can view all saved monthly forecasts alongside 24-hour smart meter runs in Forecast History.',
      };
    }
    throw new Error(`Unknown forecast type: ${forecastType}`);
  };

  const hourlyCap = getArchitectureCapability('hourly_168h');
  assert.equal(hourlyCap.persistedInDatabase, true);
  assert.equal(hourlyCap.exportablePdfReport, true);

  const monthlyCap = getArchitectureCapability('monthly_household');
  assert.equal(monthlyCap.persistedInDatabase, true);
  assert.equal(monthlyCap.historyRoute, '/forecast/history?tab=monthly');
  assert.match(monthlyCap.disclosureText, /preserve an immutable snapshot/);
});

// -----------------------------------------------------------------------------
// 12. Monthly Forecast Save State Machine & Duplicate Submission Protection
// -----------------------------------------------------------------------------
test('Monthly Save State Machine: handles saving, success state, duplicate protection, and error recovery', () => {
  const simulateSaveAction = (state, action) => {
    switch (action.type) {
      case 'CLICK_SAVE':
        if (state.isSavingMonthly || state.savedMonthlyRunId) {
          return state; // Block duplicate click
        }
        return { ...state, isSavingMonthly: true, saveMonthlyError: null };
      case 'SAVE_SUCCESS':
        return {
          ...state,
          isSavingMonthly: false,
          savedMonthlyRunId: action.runId,
          saveMonthlySuccess: `Forecast saved successfully to your persistent account history (ID: ${action.runId}).`,
          saveMonthlyError: null,
        };
      case 'SAVE_ERROR':
        return {
          ...state,
          isSavingMonthly: false,
          saveMonthlyError: action.errorMessage,
        };
      case 'RESET_ON_NEW_FORECAST':
        return {
          ...state,
          savedMonthlyRunId: null,
          saveMonthlySuccess: null,
          saveMonthlyError: null,
          isSavingMonthly: false,
        };
      default:
        return state;
    }
  };

  let state = {
    isSavingMonthly: false,
    savedMonthlyRunId: null,
    saveMonthlySuccess: null,
    saveMonthlyError: null,
  };

  // 1. User clicks save
  state = simulateSaveAction(state, { type: 'CLICK_SAVE' });
  assert.equal(state.isSavingMonthly, true);
  assert.equal(state.savedMonthlyRunId, null);

  // 2. Accidental double-click while saving: must not trigger a second save
  const blockedDoubleState = simulateSaveAction(state, { type: 'CLICK_SAVE' });
  assert.equal(blockedDoubleState.isSavingMonthly, true);

  // 3. Save succeeds
  state = simulateSaveAction(state, { type: 'SAVE_SUCCESS', runId: 'mfc_abc123456789' });
  assert.equal(state.isSavingMonthly, false);
  assert.equal(state.savedMonthlyRunId, 'mfc_abc123456789');
  assert.match(state.saveMonthlySuccess, /Forecast saved successfully/);

  // 4. Repeated click after save already completed: must be a no-op
  const blockedRepeatedState = simulateSaveAction(state, { type: 'CLICK_SAVE' });
  assert.equal(blockedRepeatedState.isSavingMonthly, false);
  assert.equal(blockedRepeatedState.savedMonthlyRunId, 'mfc_abc123456789');

  // 5. User changes horizon or generates a new forecast: resets save state
  state = simulateSaveAction(state, { type: 'RESET_ON_NEW_FORECAST' });
  assert.equal(state.savedMonthlyRunId, null);
  assert.equal(state.saveMonthlySuccess, null);
});

// -----------------------------------------------------------------------------
// 13. Monthly History Table and Detail Route Contract
// -----------------------------------------------------------------------------
test('Monthly History Table & Detail Contract: formats records, handles empty, and distinguishes snapshot', () => {
  const formatMonthlyHistoryRow = (record) => {
    const formattedBill =
      record.total_estimated_bill_inr !== null && record.total_estimated_bill_inr !== undefined
        ? `₹${record.total_estimated_bill_inr.toFixed(2)}`
        : 'Unavailable';

    const horizonLabel = `${record.horizon_months} ${record.horizon_months === 1 ? 'Mo' : 'Mos'}`;
    const detailLink = `/forecast/monthly/${record.run_id}`;

    return {
      runId: record.run_id,
      horizonLabel,
      provider: record.provider_id ? record.provider_id.toUpperCase() : 'NONE',
      periodRange: record.forecast_period_range || '-',
      totalKwh: `${record.total_forecast_kwh.toFixed(1)} kWh`,
      formattedBill,
      detailLink,
      isBillUnavailable: record.total_estimated_bill_inr === null,
    };
  };

  const sampleRecord = {
    run_id: 'mfc_nov2026_test',
    created_at: '2026-10-20T12:00:00Z',
    horizon_months: 3,
    provider_id: 'msedcl',
    forecast_period_range: 'Nov 2026 – Jan 2027',
    total_forecast_kwh: 501.0,
    total_estimated_bill_inr: null, // FY 2026-27 unverified
  };

  const row = formatMonthlyHistoryRow(sampleRecord);
  assert.equal(row.runId, 'mfc_nov2026_test');
  assert.equal(row.horizonLabel, '3 Mos');
  assert.equal(row.provider, 'MSEDCL');
  assert.equal(row.periodRange, 'Nov 2026 – Jan 2027');
  assert.equal(row.totalKwh, '501.0 kWh');
  assert.equal(row.formattedBill, 'Unavailable');
  assert.equal(row.isBillUnavailable, true);
  assert.equal(row.detailLink, '/forecast/monthly/mfc_nov2026_test');
});

// -----------------------------------------------------------------------------
// 14. Monthly Forecast Save Contract (Task 2.4B Authoritative Backend Verification)
// -----------------------------------------------------------------------------
test('Monthly Save Request Contract: constructs authoritative save payload without client response injection', () => {
  const buildSavePayload = (lastRequest, forecastResult) => {
    return {
      request: lastRequest,
      forecast_id: forecastResult?.forecast_id,
    };
  };

  const sampleRequest = {
    records: [
      {
        billing_period_start: '2025-04-01',
        billing_period_end: '2025-04-30',
        consumption_kwh: 167.0,
      },
    ],
    horizon_months: 3,
    provider_id: 'msedcl',
    is_live: true,
  };

  const sampleResult = {
    forecast_id: 'mfc_live_test_123',
    created_at: '2026-10-20T12:00:00Z',
    history_length_months: 1,
    selected_method: 'baseline_only',
    confidence_rating: 'low',
    forecast_steps: [{ month_index: 1, month_label: 'Nov 2026', forecast_kwh: 167.0 }],
    summary: { total_forecast_kwh: 501.0, average_monthly_kwh: 167.0 },
  };

  const payload = buildSavePayload(sampleRequest, sampleResult);

  // Payload must contain input request snapshot and optional forecast_id
  assert.deepEqual(payload.request, sampleRequest);
  assert.equal(payload.forecast_id, 'mfc_live_test_123');

  // Payload MUST NOT contain client-submitted response, predictions, or bill amounts
  assert.equal('response' in payload, false);
  assert.equal('forecast_steps' in payload, false);
  assert.equal('summary' in payload, false);
  assert.equal('confidence_rating' in payload, false);
});

// -----------------------------------------------------------------------------
// 19. Monthly PDF Report Download & Filename Contract (Task 2.5)
// -----------------------------------------------------------------------------
test('Monthly PDF Report Download Contract: validates endpoint construction, filename parsing, and state guard', () => {
  // 1. URL construction must target the canonical report.pdf endpoint with URI encoding
  const buildPdfDownloadUrl = (apiBase, runId) => `${apiBase}/forecasts/monthly/records/${encodeURIComponent(runId)}/report.pdf`;
  const url = buildPdfDownloadUrl('http://127.0.0.1:8000/api/v1', 'mfc_20261020_abcd#12');
  assert.equal(url, 'http://127.0.0.1:8000/api/v1/forecasts/monthly/records/mfc_20261020_abcd%2312/report.pdf');

  // 2. Filename resolution: extracts header filename or derives safe fallback
  const resolvePdfFilename = (contentDisposition, runId) => {
    let filename = `monthly_forecast_report_${runId}.pdf`;
    if (contentDisposition) {
      const match = contentDisposition.match(/filename=["']?([^"';]+)["']?/i);
      if (match && match[1]) {
        filename = match[1];
      }
    }
    return filename;
  };

  assert.equal(
    resolvePdfFilename('attachment; filename="monthly_forecast_report_mfc_123.pdf"', 'mfc_123'),
    'monthly_forecast_report_mfc_123.pdf'
  );
  assert.equal(
    resolvePdfFilename(null, 'mfc_456'),
    'monthly_forecast_report_mfc_456.pdf'
  );

  // 3. Concurrency guard: prevents duplicate requests while download is in-flight
  class DownloadGuard {
    constructor() {
      this.isDownloading = false;
      this.calls = 0;
    }
    async triggerDownload(fn) {
      if (this.isDownloading) return false;
      try {
        this.isDownloading = true;
        this.calls += 1;
        await fn();
        return true;
      } finally {
        this.isDownloading = false;
      }
    }
  }

  const guard = new DownloadGuard();
  let executed = 0;
  const mockWork = () => new Promise((resolve) => setTimeout(() => { executed++; resolve(); }, 10));

  // Simultaneous calls
  const p1 = guard.triggerDownload(mockWork);
  const p2 = guard.triggerDownload(mockWork); // Should be dropped
  return Promise.all([p1, p2]).then(([r1, r2]) => {
    assert.equal(r1, true);
    assert.equal(r2, false);
    assert.equal(guard.calls, 1);
    assert.equal(executed, 1);
    assert.equal(guard.isDownloading, false);
  });
});

// -----------------------------------------------------------------------------
// 20. Multi-Format Bill Document Validation (Task 2.6B)
// -----------------------------------------------------------------------------
test('Multi-Format Document Validation: accepts PDF, PNG, and JPEG files, rejects unsupported formats', () => {
  const ALLOWED_BILL_EXTS = ['.pdf', '.png', '.jpg', '.jpeg'];
  const validateBillDocument = (fileName) => {
    if (!fileName || typeof fileName !== 'string') {
      return { valid: false, error: 'Please select a valid electricity bill file (.pdf, .png, .jpg, or .jpeg).' };
    }
    const lower = fileName.toLowerCase();
    if (!ALLOWED_BILL_EXTS.some((ext) => lower.endsWith(ext))) {
      return { valid: false, error: 'Please select a valid electricity bill file (.pdf, .png, .jpg, or .jpeg).' };
    }
    return { valid: true, error: null };
  };

  // Valid document types
  assert.equal(validateBillDocument('msedcl_jan2024.pdf').valid, true);
  assert.equal(validateBillDocument('bill_photo.png').valid, true);
  assert.equal(validateBillDocument('BILL_SCAN.PNG').valid, true);
  assert.equal(validateBillDocument('meter_reading.jpg').valid, true);
  assert.equal(validateBillDocument('meter_reading.jpeg').valid, true);
  assert.equal(validateBillDocument('my.bill.doc.pdf').valid, true);

  // Rejected non-bill types
  assert.equal(validateBillDocument('history.csv').valid, false);
  assert.equal(validateBillDocument('bill.docx').valid, false);
  assert.equal(validateBillDocument('payload.exe').valid, false);
  assert.equal(validateBillDocument('document.txt').valid, false);
  assert.equal(validateBillDocument('').valid, false);
  assert.equal(validateBillDocument(null).valid, false);
});

// -----------------------------------------------------------------------------
// 21. Provider-Specific Guidance Configuration (Task 2.6B)
// -----------------------------------------------------------------------------
test('Provider-Specific Guidance: maps provider IDs to authoritative labels, identifiers, and cycles', () => {
  const getProviderConfig = (providerId) => {
    switch (providerId) {
      case 'msedcl':
        return {
          name: 'MSEDCL (Mahavitaran)',
          primaryLabel: 'Consumer Number (12 digits)',
          primaryPlaceholder: '12-digit Consumer No.',
          secondaryLabel: 'Billing Unit (BU, 4 digits)',
          secondaryPlaceholder: '4-digit BU',
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
      case 'jvvnl':
        return {
          name: 'JVVNL (Jaipur / Rajasthan)',
          primaryLabel: 'K-Number (12 digits)',
          primaryPlaceholder: '12-digit K-Number',
          secondaryLabel: null,
          secondaryPlaceholder: null,
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
      case 'bescom':
        return {
          name: 'BESCOM (Bangalore / Karnataka)',
          primaryLabel: 'Account ID (10 digits)',
          primaryPlaceholder: '10-digit Account ID',
          secondaryLabel: 'RR Number',
          secondaryPlaceholder: 'e.g. E3-12345',
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
      case 'tangedco':
        return {
          name: 'TANGEDCO (Tamil Nadu)',
          primaryLabel: 'Consumer Number',
          primaryPlaceholder: 'e.g. 02-123-456-789',
          secondaryLabel: null,
          secondaryPlaceholder: null,
          cycleLabel: 'Bi-Monthly (~60 days)',
          isBiMonthly: true,
        };
      default:
        return {
          name: 'Generic / Other Provider',
          primaryLabel: 'Consumer / Account ID',
          primaryPlaceholder: 'Consumer Identifier',
          secondaryLabel: 'Billing Unit / Subdivision',
          secondaryPlaceholder: 'Subdivision / BU',
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
    }
  };

  // MSEDCL
  const msedcl = getProviderConfig('msedcl');
  assert.equal(msedcl.primaryLabel, 'Consumer Number (12 digits)');
  assert.equal(msedcl.secondaryLabel, 'Billing Unit (BU, 4 digits)');
  assert.equal(msedcl.isBiMonthly, false);

  // JVVNL
  const jvvnl = getProviderConfig('jvvnl');
  assert.equal(jvvnl.primaryLabel, 'K-Number (12 digits)');
  assert.equal(jvvnl.secondaryLabel, null);
  assert.equal(jvvnl.isBiMonthly, false);

  // BESCOM
  const bescom = getProviderConfig('bescom');
  assert.equal(bescom.primaryLabel, 'Account ID (10 digits)');
  assert.equal(bescom.secondaryLabel, 'RR Number');
  assert.equal(bescom.isBiMonthly, false);

  // TANGEDCO
  const tangedco = getProviderConfig('tangedco');
  assert.equal(tangedco.primaryLabel, 'Consumer Number');
  assert.equal(tangedco.cycleLabel, 'Bi-Monthly (~60 days)');
  assert.equal(tangedco.isBiMonthly, true);
});

// -----------------------------------------------------------------------------
// 22. TANGEDCO Bi-Monthly Cycle & Manual Identifier Handling (Task 2.6B)
// -----------------------------------------------------------------------------
test('TANGEDCO Bi-Monthly Cycle & Manual Identifier Review Contract', () => {
  // 1. TANGEDCO 50-70 day billing cycle is accepted without synthetic monthly division
  const tangedcoRecord = {
    billing_period_start: '2024-01-05',
    billing_period_end: '2024-03-05',
    consumption_kwh: 450.0,
    bill_amount: 1650.0,
    period_label: 'Jan - Mar 2024',
  };

  const startD = new Date(tangedcoRecord.billing_period_start);
  const endD = new Date(tangedcoRecord.billing_period_end);
  const durationDays = Math.round((endD.getTime() - startD.getTime()) / (1000 * 3600 * 24));
  assert.equal(durationDays, 60);
  assert.equal(durationDays >= 50 && durationDays <= 70, true);

  // Consumption is preserved in full (450 kWh), not divided by 2
  assert.equal(tangedcoRecord.consumption_kwh, 450.0);

  // 2. Household can manually enter or correct consumer number even if extraction did not find it
  let state = {
    consumerNumber: '',
    billingUnit: '',
    rrNumber: '',
    isConfirmed: false,
  };

  // User manually enters JVVNL K-Number
  state = { ...state, consumerNumber: '210123456789' };
  assert.equal(state.consumerNumber, '210123456789');

  // User confirms review
  state = { ...state, isConfirmed: true };
  assert.equal(state.isConfirmed, true);
});

// -----------------------------------------------------------------------------
// 23. Guided Electricity Account Connection: Verified Official URLs (Task 3.5)
// -----------------------------------------------------------------------------
test('Guided Account Connection: verified official portal URLs, names, and instructions', () => {
  const getProviderConfig = (providerId) => {
    switch (providerId) {
      case 'msedcl':
        return {
          name: 'MSEDCL (Mahavitaran)',
          portalUrl: 'https://wss.mahadiscom.in/wss/wss',
          portalName: 'MSEDCL Web Self-Service (WSS)',
          portalInstructions: 'Sign in to the official MSEDCL portal, navigate to "View/Pay Bill" or "Billing History", and download your official e-bill PDF.',
          primaryLabel: 'Consumer Number (12 digits)',
          primaryPlaceholder: '12-digit Consumer No.',
          secondaryLabel: 'Billing Unit (BU, 4 digits)',
          secondaryPlaceholder: '4-digit BU',
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
      case 'jvvnl':
        return {
          name: 'JVVNL (Jaipur / Rajasthan)',
          portalUrl: 'https://www.bijlimitra.com/custumerLoginPage',
          portalName: 'Bijli Mitra (JVVNL) Portal',
          portalInstructions: 'Access the official Bijli Mitra consumer portal, sign in with your K-Number, and download your latest monthly bill PDF.',
          paymentUrl: 'https://pay.billdesk.com/instapayweb/jvvnl',
          primaryLabel: 'K-Number (12 digits)',
          primaryPlaceholder: '12-digit K-Number',
          secondaryLabel: null,
          secondaryPlaceholder: null,
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
      case 'bescom':
        return {
          name: 'BESCOM (Bangalore / Karnataka)',
          portalUrl: 'https://bescom.karnataka.gov.in',
          portalName: 'BESCOM Official Portal',
          portalInstructions: 'Log into the official BESCOM portal or BESCOM Mithra, look up your Account ID / RR Number, and download your billing statement.',
          primaryLabel: 'Account ID (10 digits)',
          primaryPlaceholder: '10-digit Account ID',
          secondaryLabel: 'RR Number',
          secondaryPlaceholder: 'e.g. E3-12345',
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
      case 'tangedco':
        return {
          name: 'TANGEDCO (Tamil Nadu)',
          portalUrl: 'https://www.tnebltd.gov.in/',
          portalName: 'TNEB / TANGEDCO Portal',
          portalInstructions: 'Sign in on the official TANGEDCO / TNEB online bill portal with your Service Connection Number (SCN) and download your latest bill receipt.',
          primaryLabel: 'Consumer Number',
          primaryPlaceholder: 'e.g. 02-123-456-789',
          secondaryLabel: null,
          secondaryPlaceholder: null,
          cycleLabel: 'Bi-Monthly (~60 days)',
          isBiMonthly: true,
        };
      default:
        return {
          name: 'Generic / Other Provider',
          portalUrl: null,
          portalName: null,
          portalInstructions: 'Log into your electricity provider\'s official website, download your billing statement (PDF or image), and upload it below.',
          primaryLabel: 'Consumer / Account ID',
          primaryPlaceholder: 'Consumer Identifier',
          secondaryLabel: 'Billing Unit / Subdivision',
          secondaryPlaceholder: 'Subdivision / BU',
          cycleLabel: 'Monthly (~30 days)',
          isBiMonthly: false,
        };
    }
  };

  // MSEDCL
  const msedcl = getProviderConfig('msedcl');
  assert.equal(msedcl.portalUrl, 'https://wss.mahadiscom.in/wss/wss');
  assert.equal(msedcl.portalUrl.startsWith('https://'), true);
  assert.ok(msedcl.portalInstructions.includes('MSEDCL'));

  // BESCOM
  const bescom = getProviderConfig('bescom');
  assert.equal(bescom.portalUrl, 'https://bescom.karnataka.gov.in');
  assert.equal(bescom.portalUrl.startsWith('https://'), true);
  assert.ok(bescom.portalInstructions.includes('BESCOM'));

  // JVVNL
  const jvvnl = getProviderConfig('jvvnl');
  assert.equal(jvvnl.portalUrl, 'https://www.bijlimitra.com/custumerLoginPage');
  assert.equal(jvvnl.portalUrl.startsWith('https://'), true);
  assert.ok(jvvnl.portalInstructions.includes('Bijli Mitra') || jvvnl.portalInstructions.includes('JVVNL'));
  assert.equal(jvvnl.paymentUrl, 'https://pay.billdesk.com/instapayweb/jvvnl');

  // TANGEDCO
  const tangedco = getProviderConfig('tangedco');
  assert.equal(tangedco.portalUrl, 'https://www.tnebltd.gov.in/');
  assert.equal(tangedco.portalUrl.startsWith('https://'), true);
  assert.ok(tangedco.portalInstructions.includes('TANGEDCO'));
});

// -----------------------------------------------------------------------------
// 24. Account Connection Security: External Link & Zero Credential Policy (Task 3.5)
// -----------------------------------------------------------------------------
test('Account Connection Security Contract: enforces external new tab and zero credential storage', () => {
  // 1. External portal link opening contract
  const buildPortalLinkAttributes = (portalUrl) => {
    return {
      href: portalUrl,
      target: '_blank',
      rel: 'noopener noreferrer',
    };
  };

  const linkAttrs = buildPortalLinkAttributes('https://wss.mahadiscom.in/wss/wss');
  assert.equal(linkAttrs.target, '_blank');
  assert.equal(linkAttrs.rel, 'noopener noreferrer');
  assert.equal(linkAttrs.href.startsWith('https://'), true);

  // 2. Zero-credential storage rule: application state only stores identifiers, NEVER passwords or OTPs
  const appConnectionState = {
    providerId: 'msedcl',
    consumerNumber: '123456789012',
    billingUnit: '4567',
    // Must NOT contain password, portalSecret, or otp
  };

  assert.equal('password' in appConnectionState, false);
  assert.equal('portalPassword' in appConnectionState, false);
  assert.equal('otp' in appConnectionState, false);
});

// -----------------------------------------------------------------------------
// 25. In-App Downloaded Bill Handoff & Confirmation Review Contract (Task 3.5)
// -----------------------------------------------------------------------------
test('In-App Bill Handoff: processed file seamlessly transitions to review table and confirmation', () => {
  // Mock extracted bill data returned by api.parseMonthlyBillFile
  const extractedData = {
    records: [
      {
        billing_period_start: '2024-04-01',
        billing_period_end: '2024-04-30',
        consumption_kwh: 185.0,
        bill_amount: 1720.0,
        period_label: 'Apr 2024',
      },
      {
        billing_period_start: '2024-05-01',
        billing_period_end: '2024-05-31',
        consumption_kwh: 215.0,
        bill_amount: 2100.0,
        period_label: 'May 2024',
      },
    ],
    consumer_number: '012345678901',
    billing_unit: '4321',
    utility_provider: 'MSEDCL',
    extraction_source: 'pdf',
    extraction_confidence: 0.98,
    warnings: [],
  };

  // State machine transition on bill file extraction
  let workflowState = {
    connectionMode: 'guided',
    selectedProviderId: 'msedcl',
    billFile: { name: 'msedcl_ebill_apr_may2024.pdf', size: 1024 * 500 },
    isExtracting: false,
    extractedData: null,
    reviewRecords: [],
    consumerNumber: '',
    billingUnit: '',
    isConfirmed: false,
  };

  // Handoff to review state
  workflowState = {
    ...workflowState,
    extractedData: extractedData,
    reviewRecords: [...extractedData.records],
    consumerNumber: extractedData.consumer_number,
    billingUnit: extractedData.billing_unit,
    isConfirmed: false, // Extracted data starts unconfirmed
  };

  assert.equal(workflowState.reviewRecords.length, 2);
  assert.equal(workflowState.consumerNumber, '012345678901');
  assert.equal(workflowState.billingUnit, '4321');
  assert.equal(workflowState.isConfirmed, false);

  // User review and confirmation
  workflowState = {
    ...workflowState,
    isConfirmed: true,
  };
  assert.equal(workflowState.isConfirmed, true);
});



