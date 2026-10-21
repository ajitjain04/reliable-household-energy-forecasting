import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  AlertCircle,
  HelpCircle,
  Sparkles,
  Zap,
  ClipboardList,
  CheckCircle2,
  ArrowLeft,
  FileText,
  Trash2,
  Plus,
  AlertTriangle,
  ShieldCheck,
  RefreshCw,
  Info,
  Calendar,
  TrendingUp,
  Clock,
  ChevronDown,
  ChevronUp,
  History,
  ExternalLink,
  Building2,
  Lock,
  MapPin,
  Navigation,
} from 'lucide-react';
import { api } from '../api/client';
import {
  ExtractedBillData,
  MonthlyConsumptionRecord,
  MonthlyForecastRequest,
  MonthlyForecastResponse,
  LocationInput,
} from '../api/types';
import { BillTriggeredSavingPlan } from '../components/forecast/BillTriggeredSavingPlan';

/**
 * Validate numeric bounds and formatting for household location coordinates.
 * Validates latitude in [-90, 90] and longitude in [-180, 180].
 * Both must be provided together, or both omitted for optional status.
 */
export const validateLocationInputs = (
  latStr: string,
  lonStr: string,
  labelStr?: string
): { isValid: boolean; error: string | null; location?: LocationInput } => {
  const trimmedLat = latStr.trim();
  const trimmedLon = lonStr.trim();
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

export const NewForecast: React.FC = () => {
  const navigate = useNavigate();

  // Workflow tab: CSV (hourly), Electricity Bill PDF (monthly), or Manual text (hourly)
  const [activeTab, setActiveTab] = useState<'csv' | 'bill' | 'manual'>('csv');

  // --- CSV / Hourly Forecast State ---
  const [file, setFile] = useState<File | null>(null);
  const [manualText, setManualText] = useState<string>('');
  const [modelMode, setModelMode] = useState<string>('auto');
  const [userId, setUserId] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // --- Household Location & Automated Weather State (Phase 3 Step 3) ---
  const [latitude, setLatitude] = useState<string>('');
  const [longitude, setLongitude] = useState<string>('');
  const [locationLabel, setLocationLabel] = useState<string>('');
  const [isLocating, setIsLocating] = useState<boolean>(false);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [locationSuccessMsg, setLocationSuccessMsg] = useState<string | null>(null);

  // --- Electricity Bill & Guided Account Connection State (Phase 2 & Task 3.5) ---
  const [connectionMode, setConnectionMode] = useState<'guided' | 'direct'>('guided');
  const [billFile, setBillFile] = useState<File | null>(null);
  const [isExtractingBill, setIsExtractingBill] = useState<boolean>(false);
  const [billError, setBillError] = useState<string | null>(null);
  const [isOcrLimitation, setIsOcrLimitation] = useState<boolean>(false);
  const [extractedBillData, setExtractedBillData] = useState<ExtractedBillData | null>(null);
  const [consumerNumber, setConsumerNumber] = useState<string>('');
  const [billingUnit, setBillingUnit] = useState<string>('');
  const [rrNumber, setRrNumber] = useState<string>('');
  const [reviewRecords, setReviewRecords] = useState<MonthlyConsumptionRecord[]>([]);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [isConfirmed, setIsConfirmed] = useState<boolean>(false);

  // --- Monthly Forecast Pipeline State (Task 2.1 & 2.2) ---
  const [forecastHorizon, setForecastHorizon] = useState<number>(3); // 1, 3, or 6
  const [selectedProviderId, setSelectedProviderId] = useState<string>('msedcl');
  const [phaseType, setPhaseType] = useState<'single_phase' | 'three_phase'>('single_phase');
  const [sanctionedLoadKw, setSanctionedLoadKw] = useState<string>('');
  const [isForecastingMonthly, setIsForecastingMonthly] = useState<boolean>(false);
  const [monthlyForecastError, setMonthlyForecastError] = useState<string | null>(null);
  const [isSessionExpired, setIsSessionExpired] = useState<boolean>(false);
  const [monthlyForecastResult, setMonthlyForecastResult] = useState<MonthlyForecastResponse | null>(null);
  const [expandedBreakdownIndex, setExpandedBreakdownIndex] = useState<number | null>(null);
  const [lastMonthlyRequest, setLastMonthlyRequest] = useState<MonthlyForecastRequest | null>(null);
  const [isSavingMonthly, setIsSavingMonthly] = useState<boolean>(false);
  const [saveMonthlySuccess, setSaveMonthlySuccess] = useState<string | null>(null);
  const [saveMonthlyError, setSaveMonthlyError] = useState<string | null>(null);
  const [savedMonthlyRunId, setSavedMonthlyRunId] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // CSV Handlers
  // ---------------------------------------------------------------------------
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMsg(null);
    if (e.target.files && e.target.files.length > 0) {
      const selected = e.target.files[0];
      if (!selected.name.toLowerCase().endsWith('.csv')) {
        setErrorMsg('Please select a valid .csv file.');
        return;
      }
      setFile(selected);
    }
  };

  const handleDownloadSample = () => {
    const baseTime = new Date('2024-03-01T00:00:00Z');
    const header = 'date,hourly_active_power,temp,feels_like,humidity,pressure,speed,deg';
    const rows = [header];

    for (let i = 0; i < 168; i++) {
      const t = new Date(baseTime.getTime() + i * 3600 * 1000);
      const dateStr = t.toISOString().replace('T', ' ').slice(0, 19);
      const power = (320 + 85 * Math.sin(i / 12) + (i % 7 === 0 ? 110 : 0)).toFixed(2);
      const temp = (19 + 5 * Math.cos(i / 12)).toFixed(2);
      const feelsLike = (18 + 5 * Math.cos(i / 12)).toFixed(2);
      const humidity = (45 + 15 * Math.sin(i / 24)).toFixed(1);
      const pressure = (1014 + 3 * Math.cos(i / 48)).toFixed(1);
      const speed = (2.5 + 1.2 * Math.sin(i / 6)).toFixed(2);
      const deg = ((i * 15) % 360).toFixed(0);

      rows.push(`${dateStr},${power},${temp},${feelsLike},${humidity},${pressure},${speed},${deg}`);
    }

    const csvContent = rows.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'sample_168h_energy_weather.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleLoadSampleManual = () => {
    const values: number[] = [];
    for (let i = 0; i < 168; i++) {
      const power = Math.round(320 + 85 * Math.sin(i / 12) + (i % 7 === 0 ? 110 : 0));
      values.push(power);
    }
    setManualText(values.join(',\n'));
    setErrorMsg(null);
  };

  const handleUseCurrentLocation = () => {
    setLocationError(null);
    setLocationSuccessMsg(null);

    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      setLocationError('Geolocation is not supported by your browser. Please enter coordinates manually.');
      return;
    }

    setIsLocating(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setIsLocating(false);
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        setLatitude(lat.toFixed(6));
        setLongitude(lon.toFixed(6));
        setLocationSuccessMsg(`Current coordinates acquired (${lat.toFixed(4)}, ${lon.toFixed(4)}).`);
      },
      (error) => {
        setIsLocating(false);
        switch (error.code) {
          case error.PERMISSION_DENIED:
            setLocationError('Location permission denied. Please allow location access in your browser or enter coordinates manually.');
            break;
          case error.POSITION_UNAVAILABLE:
            setLocationError('Location information is currently unavailable. Please enter coordinates manually.');
            break;
          case error.TIMEOUT:
            setLocationError('Location request timed out. Please try again or enter coordinates manually.');
            break;
          default:
            setLocationError(`Unable to retrieve location: ${error.message || 'Unknown error'}`);
            break;
        }
      },
      {
        timeout: 10000,
        enableHighAccuracy: true,
        maximumAge: 60000,
      }
    );
  };

  const handleClearLocation = () => {
    setLatitude('');
    setLongitude('');
    setLocationLabel('');
    setLocationError(null);
    setLocationSuccessMsg(null);
  };

  const handleHourlySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);

    // Validate location coordinates and bounds
    const locValidation = validateLocationInputs(latitude, longitude, locationLabel);
    if (!locValidation.isValid) {
      setErrorMsg(locValidation.error);
      return;
    }

    if (activeTab === 'csv') {
      if (!file) {
        setErrorMsg('Please select a CSV file before submitting.');
        return;
      }

      try {
        setIsSubmitting(true);
        const res = await api.uploadForecastCsv(
          file,
          modelMode,
          userId.trim() || undefined,
          locValidation.location
        );
        navigate(`/forecast/${res.run_id}`);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setErrorMsg(err.message);
        } else {
          setErrorMsg('An unexpected error occurred during CSV forecast submission.');
        }
      } finally {
        setIsSubmitting(false);
      }
    } else if (activeTab === 'manual') {
      const tokens = manualText
        .split(/[\n,;\t]+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

      if (tokens.length === 0) {
        setErrorMsg('Please enter or paste historical hourly active power observations.');
        return;
      }

      if (tokens.length !== 168) {
        setErrorMsg(
          `Manual input requires exactly 168 consecutive hourly power values (found ${tokens.length} values). Please verify your series.`
        );
        return;
      }

      const parsedNumbers: number[] = [];
      for (let i = 0; i < tokens.length; i++) {
        const num = parseFloat(tokens[i]);
        if (isNaN(num)) {
          setErrorMsg(`Value at index ${i + 1} ("${tokens[i]}") is not a valid number.`);
          return;
        }
        if (num < 0) {
          setErrorMsg(`Value at index ${i + 1} (${num} W) cannot be negative.`);
          return;
        }
        parsedNumbers.push(num);
      }

      // Manual workflow operates in Energy-only mode since no timestamps are supplied.
      // We strictly do not invent timestamps and do not send location, preserving Energy-only inference.
      try {
        setIsSubmitting(true);
        const res = await api.runForecast({
          electricity_active_power: parsedNumbers,
          model: modelMode,
          user_id: userId.trim() || undefined,
        });
        navigate(`/forecast/${res.run_id}`);
      } catch (err: unknown) {
        if (err instanceof Error) {
          setErrorMsg(err.message);
        } else {
          setErrorMsg('An unexpected error occurred during manual forecast submission.');
        }
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  const manualTokenCount = manualText
    .split(/[\n,;\t]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0).length;

  // ---------------------------------------------------------------------------
  // Provider-Specific Guidance Configuration (Task 2.6B & Task 3.5)
  // ---------------------------------------------------------------------------
  const getProviderConfig = (providerId: string) => {
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
          cycleDescription: 'Standard monthly billing cycle with verified MERC LT-I(B) slab rates.',
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
          cycleDescription: 'Standard monthly billing cycle. Tariff schedule unverified; physical demand forecast only.',
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
          cycleDescription: 'Standard monthly billing cycle. Tariff schedule unverified; physical demand forecast only.',
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
          cycleDescription: 'Bi-monthly billing cycle (50–70 days). Domestic consumers are billed bi-monthly. Actual kWh is preserved without synthetic monthly division.',
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
          cycleDescription: 'Standard monthly billing cycle.',
          isBiMonthly: false,
        };
    }
  };

  // ---------------------------------------------------------------------------
  // Electricity Bill Document Handlers & Client Validation
  // ---------------------------------------------------------------------------
  const ALLOWED_BILL_EXTS = ['.pdf', '.png', '.jpg', '.jpeg'];

  const handleBillFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setBillError(null);
    setIsOcrLimitation(false);
    if (e.target.files && e.target.files.length > 0) {
      const selected = e.target.files[0];
      const lower = selected.name.toLowerCase();
      if (!ALLOWED_BILL_EXTS.some((ext) => lower.endsWith(ext))) {
        setBillError('Please select a valid electricity bill file (.pdf, .png, .jpg, or .jpeg).');
        return;
      }
      setBillFile(selected);
      // Reset extracted state on new file selection
      setExtractedBillData(null);
      setReviewRecords([]);
      setValidationErrors([]);
      setIsConfirmed(false);
    }
  };

  const handleExtractBill = async () => {
    if (!billFile) {
      setBillError('Please select an electricity bill document before extracting.');
      return;
    }

    setBillError(null);
    setIsOcrLimitation(false);
    setIsExtractingBill(true);

    try {
      const data = await api.parseMonthlyBillFile(billFile);
      setExtractedBillData(data);
      setReviewRecords([...data.records]);
      setConsumerNumber(data.consumer_number || '');
      setBillingUnit(data.billing_unit || '');
      setRrNumber(data.rr_number || '');
      setIsConfirmed(false);
      setValidationErrors([]);
      setMonthlyForecastResult(null);
      setMonthlyForecastError(null);
      setIsSessionExpired(false);

      // Auto-detect known utility provider from bill metadata
      if (data.utility_provider) {
        const provLower = data.utility_provider.toLowerCase();
        if (provLower.includes('msedcl') || provLower.includes('mahavitaran')) {
          setSelectedProviderId('msedcl');
        } else if (provLower.includes('bescom')) {
          setSelectedProviderId('bescom');
        } else if (provLower.includes('jvvnl') || provLower.includes('jaipur')) {
          setSelectedProviderId('jvvnl');
        } else if (provLower.includes('tangedco') || provLower.includes('tamil')) {
          setSelectedProviderId('tangedco');
        } else {
          setSelectedProviderId('');
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to extract bill data from uploaded document.';
      setBillError(msg);
      // OCR limitation notice is only shown if OCR runtime bindings are truly unavailable or disabled
      if (
        msg.toLowerCase().includes('not available') ||
        msg.toLowerCase().includes('not supported') ||
        msg.toLowerCase().includes('could not be initialized')
      ) {
        setIsOcrLimitation(true);
      }
    } finally {
      setIsExtractingBill(false);
    }
  };

  // Client-side record validation strictly conforming to schema principles
  const validateRecords = (records: MonthlyConsumptionRecord[]): string[] => {
    const errors: string[] = [];
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

    // Check duplicate start dates, duplicate end dates, and overlaps
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

  const handleUpdateRecord = (
    index: number,
    field: keyof MonthlyConsumptionRecord,
    value: string | number | null
  ) => {
    setIsConfirmed(false);
    const updated = [...reviewRecords];
    const rec = { ...updated[index], [field]: value };

    // Auto-update human-readable label if start date changed
    if (field === 'billing_period_start' && typeof value === 'string' && value) {
      try {
        const d = new Date(value);
        if (!isNaN(d.getTime())) {
          rec.period_label = d.toLocaleString('en-US', { month: 'short', year: 'numeric' });
        }
      } catch {
        // ignore date parse errors during typing
      }
    }
    updated[index] = rec;
    setReviewRecords(updated);

    // Live validation update
    const errs = validateRecords(updated);
    setValidationErrors(errs);
  };

  const handleRemoveRecord = (index: number) => {
    setIsConfirmed(false);
    const updated = reviewRecords.filter((_, i) => i !== index);
    setReviewRecords(updated);
    setValidationErrors(validateRecords(updated));
  };

  const handleAddRecord = () => {
    setIsConfirmed(false);
    const today = new Date();
    const defaultStart = today.toISOString().slice(0, 8) + '01';
    const defaultEnd = today.toISOString().slice(0, 10);
    const newRec: MonthlyConsumptionRecord = {
      billing_period_start: defaultStart,
      billing_period_end: defaultEnd,
      consumption_kwh: 0,
      bill_amount: null,
      period_label: today.toLocaleString('en-US', { month: 'short', year: 'numeric' }),
    };
    const updated = [...reviewRecords, newRec];
    setReviewRecords(updated);
    setValidationErrors(validateRecords(updated));
  };

  const handleConfirmReview = () => {
    const errs = validateRecords(reviewRecords);
    if (errs.length > 0) {
      setValidationErrors(errs);
      return;
    }
    setValidationErrors([]);
    setIsConfirmed(true);
    setMonthlyForecastError(null);
    setIsSessionExpired(false);
    if (extractedBillData) {
      setExtractedBillData({
        ...extractedBillData,
        consumer_number: consumerNumber.trim() || extractedBillData.consumer_number,
        billing_unit: billingUnit.trim() || extractedBillData.billing_unit,
        rr_number: rrNumber.trim() || extractedBillData.rr_number,
      });
    }

    // Auto-generate upcoming baseline plan upon confirmation
    setTimeout(() => {
      handleGenerateMonthlyForecast(true);
    }, 50);
  };

  const handleGenerateMonthlyForecast = async (skipConfirmCheck?: boolean | React.MouseEvent) => {
    const shouldSkip = skipConfirmCheck === true;
    if (!isConfirmed && !shouldSkip) {
      setMonthlyForecastError('Please verify and confirm your billing records before generating a forecast.');
      return;
    }

    const errs = validateRecords(reviewRecords);
    if (errs.length > 0) {
      setValidationErrors(errs);
      setIsConfirmed(false);
      setMonthlyForecastError('Billing records failed validation. Please fix errors before generating a forecast.');
      return;
    }

    setMonthlyForecastError(null);
    setIsSessionExpired(false);
    setIsForecastingMonthly(true);

    try {
      const validatedHorizon = [1, 3, 6].includes(forecastHorizon) ? forecastHorizon : 3;
      const payload: MonthlyForecastRequest = {
        records: reviewRecords,
        horizon_months: validatedHorizon,
        provider_id: selectedProviderId.trim() || undefined,
        phase_type: phaseType,
        sanctioned_load_kw: sanctionedLoadKw.trim() ? parseFloat(sanctionedLoadKw) : undefined,
        is_live: true, // Live household forecasting mode (anchored to current date)
      };

      const result = await api.runMonthlyForecast(payload);
      setLastMonthlyRequest(payload);
      setMonthlyForecastResult(result);
      setSaveMonthlySuccess(null);
      setSaveMonthlyError(null);
      setSavedMonthlyRunId(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to generate monthly electricity forecast.';
      const lower = msg.toLowerCase();
      if (
        lower.includes('401') ||
        lower.includes('not authenticated') ||
        lower.includes('credential') ||
        lower.includes('unauthorized') ||
        lower.includes('expired')
      ) {
        setIsSessionExpired(true);
        setMonthlyForecastError(
          'Your session has expired or you are not authenticated. Please log in again to run live forecasts. Your confirmed billing history remains preserved in this view.'
        );
      } else {
        setMonthlyForecastError(msg);
      }
    } finally {
      setIsForecastingMonthly(false);
    }
  };

  const handleSaveMonthlyForecast = async () => {
    if (!lastMonthlyRequest || !monthlyForecastResult) {
      setSaveMonthlyError('No completed forecast result is available to save.');
      return;
    }
    if (isSavingMonthly || savedMonthlyRunId) {
      return; // prevent duplicate clicks or double submission
    }

    setIsSavingMonthly(true);
    setSaveMonthlyError(null);

    try {
      const saved = await api.saveMonthlyForecast({
        request: lastMonthlyRequest,
        forecast_id: monthlyForecastResult.forecast_id,
      });
      setSavedMonthlyRunId(saved.run_id);
      setSaveMonthlySuccess(`Forecast saved successfully to your persistent account history (ID: ${saved.run_id}).`);
    } catch (err: unknown) {
      setSaveMonthlyError(err instanceof Error ? err.message : 'Failed to save monthly forecast to history.');
    } finally {
      setIsSavingMonthly(false);
    }
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider mb-2">
            <Zap className="w-3.5 h-3.5 text-blue-600" />
            Household Energy Ingestion
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
            New Energy Forecast & Bill Review
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Choose between 168-hour smart meter data or monthly electricity bill PDF ingestion.
          </p>
        </div>

        <Link
          to="/forecast/history"
          className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 shadow-sm transition-colors self-start sm:self-auto"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Past Forecasts
        </Link>
      </div>

      {/* Main Container */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-6">
        {/* Input Method Switcher */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Choose Input Source
            </label>
            {activeTab === 'manual' && (
              <button
                type="button"
                onClick={handleLoadSampleManual}
                className="text-xs text-blue-600 hover:text-blue-800 font-semibold flex items-center gap-1"
              >
                <FileText className="w-3.5 h-3.5" />
                Load 168-Hour Sample Values
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 border border-slate-200 rounded-lg p-1 bg-slate-50">
            {/* Tab 1: CSV Upload */}
            <button
              type="button"
              onClick={() => {
                setActiveTab('csv');
                setErrorMsg(null);
              }}
              className={`flex items-center justify-center gap-2 py-2 px-3 text-xs font-semibold rounded-md transition-all ${
                activeTab === 'csv'
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <UploadCloud className="w-4 h-4" />
              <span>Upload CSV (168h)</span>
            </button>

            {/* Tab 2: Connect Account & Bill Upload */}
            <button
              type="button"
              onClick={() => {
                setActiveTab('bill');
                setErrorMsg(null);
              }}
              className={`flex items-center justify-center gap-2 py-2 px-3 text-xs font-semibold rounded-md transition-all ${
                activeTab === 'bill'
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Building2 className="w-4 h-4 text-emerald-600" />
              <span>Connect Account & Bill</span>
            </button>

            {/* Tab 3: Manual Text Input */}
            <button
              type="button"
              onClick={() => {
                setActiveTab('manual');
                setErrorMsg(null);
              }}
              className={`flex items-center justify-center gap-2 py-2 px-3 text-xs font-semibold rounded-md transition-all ${
                activeTab === 'manual'
                  ? 'bg-white text-blue-700 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <ClipboardList className="w-4 h-4" />
              <span>Paste 168 Values</span>
            </button>
          </div>
        </div>

        {/* ================================================================= */}
        {/* WORKFLOW 1: CONNECT ELECTRICITY ACCOUNT & BILL INGESTION          */}
        {/* ================================================================= */}
        {activeTab === 'bill' && (
          <div className="space-y-6 pt-2">
            {/* Guidance & Connection Mode Header */}
            <div className="bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-3">
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                    <Building2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                    Connect Electricity Account & Bill Ingestion
                  </h3>
                  <p className="text-xs text-slate-600 dark:text-slate-300 mt-0.5">
                    Connect with your distribution company to download your official e-bill, or upload an existing statement directly.
                  </p>
                </div>
                <div className="flex items-center gap-1 bg-slate-200/80 dark:bg-slate-800 p-1 rounded-lg self-start sm:self-auto text-xs font-semibold">
                  <button
                    type="button"
                    onClick={() => setConnectionMode('guided')}
                    className={`px-3 py-1.5 rounded-md transition-all ${
                      connectionMode === 'guided'
                        ? 'bg-white dark:bg-slate-900 text-blue-700 dark:text-blue-400 shadow-sm'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    Guided Connection
                  </button>
                  <button
                    type="button"
                    onClick={() => setConnectionMode('direct')}
                    className={`px-3 py-1.5 rounded-md transition-all ${
                      connectionMode === 'direct'
                        ? 'bg-white dark:bg-slate-900 text-blue-700 dark:text-blue-400 shadow-sm'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                    }`}
                  >
                    Direct Upload Fallback
                  </button>
                </div>
              </div>

              {connectionMode === 'guided' ? (
                <div className="space-y-4 pt-1">
                  {/* Provider Selection */}
                  <div>
                    <label className="block text-xs font-bold uppercase tracking-wider text-slate-700 dark:text-slate-300 mb-2">
                      1. Select Your Electricity Provider
                    </label>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {[
                        { id: 'msedcl', label: 'MSEDCL', region: 'Maharashtra', desc: 'Mahavitaran WSS' },
                        { id: 'bescom', label: 'BESCOM', region: 'Karnataka', desc: 'Bangalore Electricity' },
                        { id: 'jvvnl', label: 'JVVNL', region: 'Rajasthan', desc: 'Jaipur / Bijli Mitra' },
                        { id: 'tangedco', label: 'TANGEDCO', region: 'Tamil Nadu', desc: 'TNPDCL / TNEB' },
                      ].map((prov) => {
                        const isSelected = selectedProviderId === prov.id;
                        return (
                          <button
                            key={prov.id}
                            type="button"
                            onClick={() => {
                              setSelectedProviderId(prov.id);
                              setIsConfirmed(false);
                            }}
                            className={`p-3 rounded-lg text-left border transition-all ${
                              isSelected
                                ? 'border-emerald-500 bg-emerald-50/60 dark:bg-emerald-950/30 text-emerald-950 dark:text-emerald-100 shadow-sm ring-1 ring-emerald-500'
                                : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-800/60 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="font-bold text-xs">{prov.label}</span>
                              {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />}
                            </div>
                            <span className="text-[10px] text-slate-500 dark:text-slate-400 block font-medium">
                              {prov.region}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Provider Account Details & Portal Link Card */}
                  {(() => {
                    const cfg = getProviderConfig(selectedProviderId);
                    return (
                      <div className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-4 space-y-4 shadow-sm">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 dark:border-slate-700 pb-3">
                          <div>
                            <span className="text-xs font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
                              {cfg.name} Account Guidance
                            </span>
                            <span className="text-[11px] text-slate-500 dark:text-slate-400">
                              Billing Cycle: {cfg.cycleLabel} • {cfg.cycleDescription}
                            </span>
                          </div>

                          <div className="flex flex-wrap items-center gap-2">
                            {cfg.portalUrl && (
                              <a
                                href={cfg.portalUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm transition-all self-start sm:self-auto active:scale-95"
                                id="open-official-portal-btn"
                              >
                                <span>Open Official {cfg.portalName || cfg.name}</span>
                                <ExternalLink className="w-3.5 h-3.5" />
                              </a>
                            )}
                            {cfg.paymentUrl && (
                              <a
                                href={cfg.paymentUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 shadow-sm transition-all self-start sm:self-auto"
                                id="open-official-payment-btn"
                              >
                                <span>Quick Payment (BillDesk)</span>
                                <ExternalLink className="w-3 h-3 text-slate-400" />
                              </a>
                            )}
                          </div>
                        </div>

                        {/* Account Identifiers Input for Reference */}
                        <div>
                          <label className="block text-xs font-bold text-slate-700 dark:text-slate-300 mb-1.5">
                            Account Identifiers (Optional for Reference / Pre-filling)
                          </label>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                            <div>
                              <span className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                                {cfg.primaryLabel}
                              </span>
                              <input
                                type="text"
                                value={consumerNumber}
                                onChange={(e) => {
                                  setConsumerNumber(e.target.value);
                                  setIsConfirmed(false);
                                }}
                                placeholder={cfg.primaryPlaceholder || 'Identifier'}
                                className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-md px-3 py-1.5 text-slate-900 dark:text-slate-100 focus:ring-1 focus:ring-emerald-500 focus:outline-none"
                              />
                            </div>

                            {cfg.secondaryLabel && (
                              <div>
                                <span className="text-[11px] font-medium text-slate-600 dark:text-slate-400 block mb-1">
                                  {cfg.secondaryLabel}
                                </span>
                                <input
                                  type="text"
                                  value={selectedProviderId === 'bescom' ? rrNumber : billingUnit}
                                  onChange={(e) => {
                                    if (selectedProviderId === 'bescom') {
                                      setRrNumber(e.target.value);
                                    } else {
                                      setBillingUnit(e.target.value);
                                    }
                                    setIsConfirmed(false);
                                  }}
                                  placeholder={cfg.secondaryPlaceholder || ''}
                                  className="w-full text-xs font-mono bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-md px-3 py-1.5 text-slate-900 dark:text-slate-100 focus:ring-1 focus:ring-emerald-500 focus:outline-none"
                                />
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Step-by-Step Instructions */}
                        <div className="bg-slate-50 dark:bg-slate-900/50 rounded-lg p-3 text-xs space-y-1.5 text-slate-700 dark:text-slate-300">
                          <span className="font-bold text-slate-900 dark:text-white block">
                            How to Retrieve & Ingest Your Bill:
                          </span>
                          <ol className="list-decimal list-inside space-y-1 text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
                            <li>Click the <strong>Open Official Portal</strong> button above. The verified utility website will open in a new browser tab.</li>
                            <li>Sign in directly with your provider and locate your latest billing statement or e-bill PDF ({cfg.portalInstructions}).</li>
                            <li>Download the official e-bill file (PDF, PNG, or JPEG) to your device.</li>
                            <li>Return to this window (your application session remains open) and upload the downloaded document below.</li>
                          </ol>
                        </div>

                        {/* Zero Credential Guarantee Callout */}
                        <div className="flex items-start gap-2 p-2.5 rounded-lg bg-emerald-50/60 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900 text-emerald-900 dark:text-emerald-200 text-[11px]">
                          <Lock className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
                          <span>
                            <strong>Zero Credential Storage Guarantee:</strong> We never collect, ask for, or store your electricity provider password, credentials, or OTPs. You authenticate directly on the official utility website and manually upload your statement.
                          </span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              ) : (
                <div className="space-y-2 pt-1 text-xs text-slate-600 dark:text-slate-400">
                  <p>
                    <strong>Direct Upload Fallback:</strong> If you already have your official electricity bill file (PDF, PNG, JPG) downloaded on your device, upload it directly below without opening provider portal links.
                  </p>
                </div>
              )}
            </div>

            {/* Bill File Dropzone (In-App on Same Screen) */}
            <div className="space-y-2">
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                {connectionMode === 'guided'
                  ? `Upload Downloaded Bill from ${getProviderConfig(selectedProviderId).name} (PDF, PNG, JPEG)`
                  : 'Select Electricity Bill Document (PDF, PNG, JPEG)'}
              </label>
              <div
                className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                  billFile
                    ? 'border-emerald-500 bg-emerald-50/40 dark:bg-emerald-950/20'
                    : 'border-slate-300 dark:border-slate-700 hover:border-emerald-400 dark:hover:border-emerald-500 bg-slate-50/50 dark:bg-slate-900/40 hover:bg-slate-50 dark:hover:bg-slate-900/60'
                }`}
              >
                <input
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg"
                  onChange={handleBillFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  id="bill-file-input"
                />
                <div className="flex flex-col items-center justify-center pointer-events-none">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center mb-3 ${
                      billFile ? 'bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400'
                    }`}
                  >
                    {billFile ? <CheckCircle2 className="w-6 h-6" /> : <FileText className="w-6 h-6" />}
                  </div>
                  {billFile ? (
                    <div>
                      <p className="text-sm font-bold text-slate-900 dark:text-white">{billFile.name}</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        {(billFile.size / (1024 * 1024)).toFixed(2)} MB • Ready to analyze
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                        {connectionMode === 'guided'
                          ? `Drop your downloaded ${getProviderConfig(selectedProviderId).name} bill here, or `
                          : 'Drop your electricity bill (PDF, PNG, JPG) here, or '}
                        <span className="text-emerald-600 dark:text-emerald-400">browse files</span>
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        Accepts official digital e-bills and scanned documents/images up to 10 MB.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Extract Action Button */}
            <div className="flex items-center justify-between pt-1">
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {billFile ? `Ready: ${billFile.name}` : 'No bill file selected yet'}
              </span>

              <button
                type="button"
                onClick={handleExtractBill}
                disabled={isExtractingBill || !billFile}
                className={`inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-xs text-white shadow-sm transition-all ${
                  isExtractingBill || !billFile
                    ? 'bg-slate-300 dark:bg-slate-700 cursor-not-allowed text-slate-500 dark:text-slate-400'
                    : 'bg-emerald-600 hover:bg-emerald-700 active:scale-95'
                }`}
              >
                {isExtractingBill ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Extracting Bill Data...</span>
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4" />
                    <span>Extract Bill Data</span>
                  </>
                )}
              </button>
            </div>

            {/* Error & OCR State Display */}
            {billError && (
              <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-200 space-y-2 text-xs">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 flex-shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <h4 className="font-bold text-rose-900 dark:text-rose-100 text-sm">
                      {isOcrLimitation
                        ? 'OCR Runtime Unavailable'
                        : billError.toLowerCase().includes('resolution') && billError.toLowerCase().includes('too low')
                        ? 'Image Resolution Too Low'
                        : 'Bill Document Extraction Error'}
                    </h4>
                    <p className="text-rose-700 dark:text-rose-300 leading-relaxed">{billError}</p>
                  </div>
                </div>

                {isOcrLimitation ? (
                  <div className="mt-2 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-900 dark:text-amber-200 text-xs space-y-1">
                    <div className="font-bold flex items-center gap-1.5">
                      <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                      OCR Engine Notice
                    </div>
                    <p className="text-amber-800 dark:text-amber-300 leading-relaxed">
                      Windows Native OCR runtime is not available on this server environment. Please upload a digital text-based e-bill PDF, or manually enter readings below.
                    </p>
                  </div>
                ) : billError.toLowerCase().includes('resolution') && billError.toLowerCase().includes('too low') ? (
                  <div className="mt-2 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-900 dark:text-amber-200 text-xs space-y-1">
                    <div className="font-bold flex items-center gap-1.5">
                      <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
                      Image Resolution Recommendation
                    </div>
                    <p className="text-amber-800 dark:text-amber-300 leading-relaxed">
                      Optical character recognition requires text character heights of at least 15–20 pixels (equivalent to 150–300 DPI). For reliable automated extraction, upload an uncompressed photo or the digital e-bill PDF issued by your power utility, or enter readings manually below.
                    </p>
                  </div>
                ) : (
                  <div className="mt-2 p-3 rounded-lg bg-slate-100 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 text-xs space-y-1">
                    <p className="leading-relaxed">
                      If uploading an image or scanned bill, please ensure the document is right-side up, clearly legible, and displays billing dates and consumption values. You can also enter monthly readings manually.
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Extracted Data Review & Verification Section */}
            {extractedBillData && (
              <>
                <div className="space-y-6 pt-4 border-t border-slate-200 dark:border-slate-800">
                {/* Unconfirmed / Confirmed Status Banner */}
                {!isConfirmed ? (
                  <div className="p-4 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 text-amber-900 dark:text-amber-200 flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                    <div className="space-y-1 text-xs">
                      <h3 className="text-sm font-bold text-amber-900 dark:text-amber-100">
                        Unconfirmed Extracted Bill Data — Please Review & Verify
                      </h3>
                      <p className="text-amber-800 dark:text-amber-300 leading-relaxed">
                        The values below were extracted automatically from your document. Please inspect billing dates, kilowatt-hour consumption, and billed amounts. You can edit any field directly in the table before confirming.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="p-4 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900 text-emerald-900 dark:text-emerald-200 flex items-start gap-3">
                    <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
                    <div className="space-y-1 text-xs">
                      <h3 className="text-sm font-bold text-emerald-900 dark:text-emerald-100">
                        Bill Data Verified & Confirmed by Household
                      </h3>
                      <p className="text-emerald-800 dark:text-emerald-300 leading-relaxed">
                        {reviewRecords.length} consumption cycles have been verified. This confirmed history feeds the monthly forecasting pipeline.
                      </p>
                    </div>
                  </div>
                )}

                {/* Provider-Specific Billing Cycle Guidance Banner */}
                {selectedProviderId === 'tangedco' && (
                  <div className="p-3.5 rounded-xl bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900 text-blue-900 dark:text-blue-200 flex items-start gap-2.5 text-xs">
                    <Info className="w-4 h-4 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="font-bold text-blue-950 dark:text-blue-100">TANGEDCO Bi-Monthly Billing Cycle Guidance (~60 Days)</h4>
                      <p className="text-blue-800 dark:text-blue-300 leading-relaxed mt-0.5">
                        Domestic TANGEDCO bills cover approximately 50 to 70 days. The pipeline evaluates actual bi-monthly kilowatt-hour consumption directly for forecasting, without fabricating synthetic monthly fractions or duplicating readings.
                      </p>
                    </div>
                  </div>
                )}

                {/* Provider-Specific Metadata & Quality Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
                      Utility Provider
                    </span>
                    <span className="font-bold text-slate-900 dark:text-white block truncate" title={extractedBillData.utility_provider || 'Not identified'}>
                      {extractedBillData.utility_provider || getProviderConfig(selectedProviderId).name}
                    </span>
                    <span className="text-[10px] text-slate-500 dark:text-slate-400 block">
                      {getProviderConfig(selectedProviderId).cycleLabel}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <label className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
                      {getProviderConfig(selectedProviderId).primaryLabel}
                    </label>
                    <input
                      type="text"
                      value={consumerNumber}
                      onChange={(e) => {
                        setConsumerNumber(e.target.value);
                        setIsConfirmed(false);
                      }}
                      placeholder={getProviderConfig(selectedProviderId).primaryPlaceholder}
                      className="w-full font-mono text-xs font-bold text-slate-900 dark:text-slate-100 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                    />
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <label className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
                      {selectedProviderId === 'bescom'
                        ? 'RR Number'
                        : getProviderConfig(selectedProviderId).secondaryLabel || 'Billing Unit / Ref'}
                    </label>
                    {selectedProviderId === 'bescom' ? (
                      <input
                        type="text"
                        value={rrNumber}
                        onChange={(e) => {
                          setRrNumber(e.target.value);
                          setIsConfirmed(false);
                        }}
                        placeholder="e.g. E3-12345"
                        className="w-full font-mono text-xs font-bold text-slate-900 dark:text-slate-100 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                      />
                    ) : getProviderConfig(selectedProviderId).secondaryLabel ? (
                      <input
                        type="text"
                        value={billingUnit}
                        onChange={(e) => {
                          setBillingUnit(e.target.value);
                          setIsConfirmed(false);
                        }}
                        placeholder={getProviderConfig(selectedProviderId).secondaryPlaceholder || 'N/A'}
                        className="w-full font-mono text-xs font-bold text-slate-900 dark:text-slate-100 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 focus:ring-1 focus:ring-blue-500 focus:outline-none"
                      />
                    ) : (
                      <span className="font-mono text-xs text-slate-500 dark:text-slate-400 block py-1">
                        {selectedProviderId === 'tangedco' ? 'Bi-Monthly Cycle' : 'N/A'}
                      </span>
                    )}
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 space-y-1">
                    <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
                      Extraction Quality
                    </span>
                    <div className="flex items-center gap-1.5">
                      <span className="font-bold text-emerald-700 dark:text-emerald-400">
                        {extractedBillData.extraction_confidence !== null && extractedBillData.extraction_confidence !== undefined
                          ? `${Math.round(extractedBillData.extraction_confidence * 100)}%`
                          : 'N/A'}
                      </span>
                      <span className="text-[10px] text-slate-500 dark:text-slate-400">Document Parsing</span>
                    </div>
                  </div>
                </div>

                {/* Extraction Warnings Callout (e.g. gaps in history) */}
                {extractedBillData.warnings && extractedBillData.warnings.length > 0 && (
                  <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-xs space-y-1.5">
                    <span className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                      <Info className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                      Parser Notes & Observations
                    </span>
                    <ul className="space-y-1 text-slate-600 dark:text-slate-400 list-disc list-inside leading-relaxed">
                      {extractedBillData.warnings.map((w, idx) => (
                        <li key={idx}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Editable Review Table */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-800 dark:text-slate-200">
                        Monthly Consumption Records ({reviewRecords.length} Cycles)
                      </h4>
                      <p className="text-[11px] text-slate-500 dark:text-slate-400">
                        Click any cell below to correct extracted dates, units, or amounts.
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={handleAddRecord}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-xs font-semibold transition-colors"
                    >
                      <Plus className="w-3.5 h-3.5" />
                      Add Billing Period
                    </button>
                  </div>

                  <div className="overflow-x-auto border border-slate-200 dark:border-slate-800 rounded-lg">
                    <table className="w-full text-left text-xs text-slate-700 dark:text-slate-300">
                      <thead className="bg-slate-50 dark:bg-slate-900/80 text-[11px] uppercase font-bold text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                        <tr>
                          <th className="py-2.5 px-3">#</th>
                          <th className="py-2.5 px-3">Start Date</th>
                          <th className="py-2.5 px-3">End Date</th>
                          <th className="py-2.5 px-3">Label</th>
                          <th className="py-2.5 px-3">Consumption (kWh)</th>
                          <th className="py-2.5 px-3">Bill Amount (₹)</th>
                          <th className="py-2.5 px-3 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                        {reviewRecords.length === 0 ? (
                          <tr>
                            <td colSpan={7} className="text-center py-6 text-slate-400 dark:text-slate-500">
                              No records extracted. Click "Add Billing Period" to add readings manually.
                            </td>
                          </tr>
                        ) : (
                          reviewRecords.map((rec, idx) => (
                            <tr key={idx} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/50 transition-colors">
                              <td className="py-2 px-3 font-semibold text-slate-500 dark:text-slate-400 text-xs">
                                {idx + 1}
                              </td>

                              {/* Start Date */}
                              <td className="py-2 px-3">
                                <input
                                  type="date"
                                  value={rec.billing_period_start}
                                  onChange={(e) =>
                                    handleUpdateRecord(idx, 'billing_period_start', e.target.value)
                                  }
                                  className="w-32 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                                />
                              </td>

                              {/* End Date */}
                              <td className="py-2 px-3">
                                <input
                                  type="date"
                                  value={rec.billing_period_end}
                                  onChange={(e) =>
                                    handleUpdateRecord(idx, 'billing_period_end', e.target.value)
                                  }
                                  className="w-32 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                                />
                              </td>

                              {/* Label */}
                              <td className="py-2 px-3">
                                <input
                                  type="text"
                                  value={rec.period_label || ''}
                                  onChange={(e) =>
                                    handleUpdateRecord(idx, 'period_label', e.target.value)
                                  }
                                  placeholder="e.g. Jan 2024"
                                  className="w-24 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                                />
                              </td>

                              {/* Consumption kWh */}
                              <td className="py-2 px-3">
                                <input
                                  type="number"
                                  step="0.1"
                                  min="0"
                                  value={rec.consumption_kwh}
                                  onChange={(e) =>
                                    handleUpdateRecord(
                                      idx,
                                      'consumption_kwh',
                                      parseFloat(e.target.value) || 0
                                    )
                                  }
                                  className="w-24 font-mono font-bold bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-emerald-500"
                                />
                              </td>

                              {/* Bill Amount */}
                              <td className="py-2 px-3">
                                <input
                                  type="number"
                                  step="0.01"
                                  min="0"
                                  value={rec.bill_amount !== null && rec.bill_amount !== undefined ? rec.bill_amount : ''}
                                  onChange={(e) => {
                                    const val = e.target.value === '' ? null : parseFloat(e.target.value);
                                    handleUpdateRecord(idx, 'bill_amount', val);
                                  }}
                                  placeholder="Optional ₹"
                                  className="w-28 font-mono bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded px-2 py-1 text-xs text-slate-800 dark:text-slate-100 placeholder-slate-300 dark:placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                                />
                              </td>

                              {/* Delete row */}
                              <td className="py-2 px-3 text-right">
                                <button
                                  type="button"
                                  onClick={() => handleRemoveRecord(idx)}
                                  className="p-1 rounded text-slate-400 dark:text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors"
                                  title="Delete this row"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Validation Errors List */}
                {validationErrors.length > 0 && (
                  <div className="p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-200 text-xs space-y-1">
                    <span className="font-bold flex items-center gap-1.5 text-rose-900 dark:text-rose-100">
                      <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400" />
                      Please fix the following validation issues before confirming:
                    </span>
                    <ul className="list-disc list-inside space-y-0.5 text-rose-700 dark:text-rose-300">
                      {validationErrors.map((err, i) => (
                        <li key={i}>{err}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Review Action Controls */}
                <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3 border-t border-slate-100 dark:border-slate-800">
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    {!isConfirmed ? (
                      <span className="text-amber-700 dark:text-amber-400 font-medium flex items-center gap-1">
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
                        Unconfirmed: Click button to verify and lock review records.
                      </span>
                    ) : (
                      <span className="text-emerald-700 dark:text-emerald-400 font-medium flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                        Confirmed by household user.
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setExtractedBillData(null);
                        setBillFile(null);
                        setReviewRecords([]);
                        setIsConfirmed(false);
                      }}
                      className="px-3.5 py-2 text-xs font-semibold rounded-lg border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                    >
                      Upload Different PDF
                    </button>

                    <button
                      type="button"
                      onClick={handleConfirmReview}
                      disabled={validationErrors.length > 0 || reviewRecords.length === 0}
                      className={`inline-flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold text-white transition-all ${
                        validationErrors.length > 0 || reviewRecords.length === 0
                          ? 'bg-slate-300 dark:bg-slate-700 cursor-not-allowed text-slate-500 dark:text-slate-400'
                          : isConfirmed
                          ? 'bg-emerald-700 hover:bg-emerald-800'
                          : 'bg-emerald-600 hover:bg-emerald-700'
                      }`}
                    >
                      <ShieldCheck className="w-4 h-4" />
                      <span>{isConfirmed ? 'Re-Confirm Changes' : 'Confirm Verified Bill History'}</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* ========================================================= */}
              {/* TASK 2.7B: BILL-TRIGGERED PERSONALIZED SAVING PLAN       */}
              {/* ========================================================= */}
              {isConfirmed && (
                <BillTriggeredSavingPlan
                  isConfirmed={isConfirmed}
                  confirmedRecords={reviewRecords}
                  extractedBillData={extractedBillData}
                  selectedProviderId={selectedProviderId}
                  phaseType={phaseType}
                  sanctionedLoadKw={sanctionedLoadKw}
                  forecastHorizon={forecastHorizon}
                  onForecastHorizonChange={(h) => {
                    setForecastHorizon(h);
                    setMonthlyForecastResult(null);
                  }}
                  monthlyForecastResult={monthlyForecastResult}
                  isForecasting={isForecastingMonthly}
                  onGenerateForecast={() => handleGenerateMonthlyForecast(true)}
                  onSaveForecast={handleSaveMonthlyForecast}
                  isSaving={isSavingMonthly}
                  savedRunId={savedMonthlyRunId}
                />
              )}

              {/* ========================================================= */}
              {/* ACTION: GENERATE MONTHLY FORECAST (TASK 2.1 & 2.2)        */}
              {/* ========================================================= */}
              {isConfirmed && (
                <div className="bg-gradient-to-br from-blue-50/70 via-indigo-50/40 to-white border border-blue-200 rounded-xl p-5 space-y-4 shadow-sm">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-blue-200/70">
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-blue-600" />
                        <h3 className="text-sm font-bold text-slate-900">
                          Generate Monthly Household Electricity Forecast
                        </h3>
                      </div>
                      <p className="text-xs text-slate-600">
                        Configure forecast horizon and tariff parameters for upcoming billing cycles.
                      </p>
                    </div>
                    <span className="text-[11px] font-semibold px-2.5 py-1 rounded-full bg-blue-100 text-blue-800 border border-blue-200 self-start sm:self-auto">
                      Live Household Mode
                    </span>
                  </div>

                  {/* Parameter Controls Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                    {/* Horizon Selector */}
                    <div className="p-3 rounded-lg bg-white border border-slate-200 space-y-1.5 shadow-xs">
                      <label className="font-bold text-slate-700 block">Forecast Horizon</label>
                      <div className="grid grid-cols-3 gap-1">
                        {[1, 3, 6].map((h) => (
                          <button
                            key={h}
                            id={`horizon-btn-${h}`}
                            data-testid={`horizon-${h}`}
                            type="button"
                            onClick={() => {
                              setForecastHorizon(h);
                              setMonthlyForecastResult(null);
                            }}
                            className={`py-1.5 px-1.5 rounded font-semibold text-xs transition-all ${
                              forecastHorizon === h
                                ? 'bg-blue-600 text-white shadow-xs'
                                : 'bg-slate-100 hover:bg-slate-200 text-slate-700'
                            }`}
                          >
                            {h} {h === 1 ? 'Mo' : 'Mos'}
                          </button>
                        ))}
                      </div>
                      <span className="text-[10px] text-slate-500 block">
                        {forecastHorizon === 1
                          ? 'Next upcoming billing cycle'
                          : forecastHorizon === 3
                          ? 'Next quarter (recommended)'
                          : 'Next 6 calendar months'}
                      </span>
                    </div>

                    {/* Utility Discom Provider */}
                    <div className="p-3 rounded-lg bg-white border border-slate-200 space-y-1.5 shadow-xs">
                      <label className="font-bold text-slate-700 block">Utility Discom</label>
                      <select
                        value={selectedProviderId}
                        onChange={(e) => {
                          setSelectedProviderId(e.target.value);
                          setMonthlyForecastResult(null);
                        }}
                        className="w-full bg-white border border-slate-200 rounded px-2 py-1.5 text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      >
                        <option value="msedcl">MSEDCL (Maharashtra) — Verified LT-I(B)</option>
                        <option value="bescom">BESCOM (Karnataka) — Rate Schedule Unverified</option>
                        <option value="jvvnl">JVVNL (Rajasthan) — Rate Schedule Unverified</option>
                        <option value="tangedco">TANGEDCO (Tamil Nadu) — Rate Schedule Unverified</option>
                        <option value="">No Provider (Physical Demand Only)</option>
                      </select>
                      <span className="text-[10px] text-slate-500 block truncate" title={selectedProviderId === 'msedcl' ? 'Computes official MERC slab rates' : 'No verified tariff'}>
                        {selectedProviderId === 'msedcl'
                          ? '✓ Official MERC LT-I(B) Residential'
                          : selectedProviderId
                          ? '⚠ Schedule unverified; bill withheld'
                          : 'Physical kWh only; bill withheld'}
                      </span>
                    </div>

                    {/* Phase Type */}
                    <div className="p-3 rounded-lg bg-white border border-slate-200 space-y-1.5 shadow-xs">
                      <label className="font-bold text-slate-700 block">Connection Phase</label>
                      <select
                        value={phaseType}
                        onChange={(e) => {
                          setPhaseType(e.target.value as 'single_phase' | 'three_phase');
                          setMonthlyForecastResult(null);
                        }}
                        className="w-full bg-white border border-slate-200 rounded px-2 py-1.5 text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      >
                        <option value="single_phase">Single Phase (1-Phase)</option>
                        <option value="three_phase">Three Phase (3-Phase)</option>
                      </select>
                      <span className="text-[10px] text-slate-500 block">
                        {phaseType === 'single_phase' ? 'Fixed charge ₹128/mo (MSEDCL)' : 'Fixed charge ₹385/mo (MSEDCL)'}
                      </span>
                    </div>

                    {/* Sanctioned Load kW */}
                    <div className="p-3 rounded-lg bg-white border border-slate-200 space-y-1.5 shadow-xs">
                      <label className="font-bold text-slate-700 block">Sanctioned Load (kW)</label>
                      <input
                        type="number"
                        step="0.5"
                        min="0"
                        value={sanctionedLoadKw}
                        onChange={(e) => {
                          setSanctionedLoadKw(e.target.value);
                          setMonthlyForecastResult(null);
                        }}
                        placeholder="Optional kW (e.g. 5.0)"
                        className="w-full bg-white border border-slate-200 rounded px-2 py-1 text-xs text-slate-800 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      />
                      <span className="text-[10px] text-slate-500 block">
                        Optional contract load in kW
                      </span>
                    </div>
                  </div>

                  {/* Notice */}
                  <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 text-blue-900 text-xs flex items-start gap-2.5">
                    <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
                    <p className="leading-relaxed">
                      <strong>Live Forecasting Anchor:</strong> In live mode, future forecast cycles anchor to upcoming calendar months starting from the current date. Your confirmed historical records are preserved and carried forward as demand history.
                    </p>
                  </div>

                  {/* Action Controls */}
                  <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-1">
                    <div className="text-xs text-slate-500">
                      {reviewRecords.length} confirmed cycle{reviewRecords.length === 1 ? '' : 's'} ready.
                    </div>

                    <button
                      id="generate-monthly-forecast-btn"
                      type="button"
                      onClick={handleGenerateMonthlyForecast}
                      disabled={isForecastingMonthly || !isConfirmed || reviewRecords.length === 0}
                      className={`inline-flex items-center gap-2 px-6 py-2.5 rounded-lg text-xs font-semibold text-white shadow-sm transition-all ${
                        isForecastingMonthly || !isConfirmed || reviewRecords.length === 0
                          ? 'bg-slate-300 cursor-not-allowed text-slate-500'
                          : 'bg-blue-600 hover:bg-blue-700 active:scale-95'
                      }`}
                    >
                      {isForecastingMonthly ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>Generating Monthly Forecast...</span>
                        </>
                      ) : (
                        <>
                          <Zap className="w-4 h-4" />
                          <span>Generate {forecastHorizon}-Month Forecast</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Session Expired / Auth Error Notice */}
                  {isSessionExpired && (
                    <div className="p-4 rounded-lg bg-amber-50 border border-amber-300 text-amber-900 text-xs flex items-start gap-3">
                      <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                      <div className="space-y-1">
                        <h4 className="font-bold text-amber-900">Session Expired or Login Required</h4>
                        <p className="text-amber-800 leading-relaxed">
                          {monthlyForecastError}
                        </p>
                        <div className="pt-1">
                          <Link
                            to="/login"
                            className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-amber-600 text-white font-semibold hover:bg-amber-700 transition-colors"
                          >
                            Log In Again
                          </Link>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* General Forecast Error */}
                  {monthlyForecastError && !isSessionExpired && (
                    <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-3">
                      <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
                      <div className="space-y-1">
                        <h4 className="font-bold text-rose-900">Monthly Forecast Execution Failed</h4>
                        <p className="text-rose-700 leading-relaxed">{monthlyForecastError}</p>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* ========================================================= */}
              {/* MONTHLY FORECAST RESULTS PRESENTATION                     */}
              {/* ========================================================= */}
              {monthlyForecastResult && (
                <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-6 shadow-sm">
                  {/* Results Header */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-200">
                    <div>
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold uppercase tracking-wider mb-1">
                        <TrendingUp className="w-3.5 h-3.5 text-blue-600" />
                        Live Monthly Forecast Results
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-lg font-bold text-slate-900">
                          {monthlyForecastResult.forecast_steps.length}-Month Electricity Consumption Projection
                        </h3>
                        <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 border border-blue-200">
                          {monthlyForecastResult.forecast_steps.length} {monthlyForecastResult.forecast_steps.length === 1 ? 'Month' : 'Months'} Horizon
                        </span>
                      </div>
                      <p className="text-xs text-slate-500">
                        Forecast ID: <span className="font-mono text-slate-700 font-semibold">{monthlyForecastResult.forecast_id}</span> • Executed: {new Date(monthlyForecastResult.created_at).toLocaleString()}
                      </p>
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      {/* Confidence Rating Badge */}
                      <span
                        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold border ${
                          monthlyForecastResult.confidence_rating === 'high'
                            ? 'bg-emerald-50 border-emerald-300 text-emerald-800'
                            : monthlyForecastResult.confidence_rating === 'moderate'
                            ? 'bg-blue-50 border-blue-300 text-blue-800'
                            : 'bg-amber-50 border-amber-300 text-amber-800'
                        }`}
                      >
                        Confidence: {monthlyForecastResult.confidence_rating.toUpperCase()}
                      </span>

                      {/* History Length Badge */}
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 border border-slate-200 text-slate-700">
                        History: {monthlyForecastResult.history_length_months} Cycle{monthlyForecastResult.history_length_months === 1 ? '' : 's'}
                      </span>
                    </div>
                  </div>

                  {/* Persistent Save Action Bar */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-lg bg-slate-50 border border-slate-200 shadow-2xs">
                    <div className="space-y-0.5">
                      <div className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                        <History className="w-3.5 h-3.5 text-blue-600" />
                        Save to Persistent Account History
                      </div>
                      <p className="text-[11px] text-slate-600">
                        Store this forecast run and its input snapshot in your durable audit trail to revisit anytime.
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      {savedMonthlyRunId ? (
                        <div className="inline-flex items-center gap-2">
                          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-300 text-emerald-800 text-xs font-semibold">
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                            Saved to History
                          </span>
                          <Link
                            to={`/forecast/monthly/${savedMonthlyRunId}`}
                            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white border border-slate-300 text-slate-700 hover:bg-slate-100 text-xs font-medium transition-colors shadow-2xs"
                          >
                            View Snapshot →
                          </Link>
                        </div>
                      ) : (
                        <button
                          type="button"
                          id="save-monthly-forecast-btn"
                          onClick={handleSaveMonthlyForecast}
                          disabled={isSavingMonthly}
                          className={`inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold text-white transition-all shadow-xs ${
                            isSavingMonthly
                              ? 'bg-blue-400 cursor-not-allowed'
                              : 'bg-blue-600 hover:bg-blue-700 active:scale-95'
                          }`}
                        >
                          {isSavingMonthly ? (
                            <>
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                              <span>Saving Snapshot...</span>
                            </>
                          ) : (
                            <>
                              <Plus className="w-3.5 h-3.5" />
                              <span>Save Forecast Run</span>
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>

                  {saveMonthlySuccess && (
                    <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                        <span>{saveMonthlySuccess}</span>
                      </div>
                      {savedMonthlyRunId && (
                        <Link
                          to={`/forecast/monthly/${savedMonthlyRunId}`}
                          className="font-semibold underline hover:text-emerald-900 flex-shrink-0"
                        >
                          Open Saved Run
                        </Link>
                      )}
                    </div>
                  )}

                  {saveMonthlyError && (
                    <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
                      <span>{saveMonthlyError}</span>
                    </div>
                  )}

                  {/* REQUIREMENT 8: Prominent Low-Confidence Baseline Warning for N=1 */}
                  {monthlyForecastResult.history_length_months === 1 && (
                    <div className="p-4 rounded-xl bg-amber-50 border border-amber-300 text-amber-950 flex items-start gap-3">
                      <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                      <div className="space-y-1 text-xs">
                        <h4 className="font-bold text-amber-900 text-sm">
                          Low-Confidence Baseline Forecast (Single Month Observed)
                        </h4>
                        <p className="text-amber-800 leading-relaxed">
                          Only <strong>1 confirmed historical billing cycle</strong> ({monthlyForecastResult.forecast_steps[0]?.forecast_kwh} kWh) was provided. A single observation is insufficient to estimate consumption trends, variance, or seasonal swings (such as summer air conditioning or winter heating).
                        </p>
                        <p className="text-amber-800 leading-relaxed">
                          Results represent a <strong>static baseline persistence forecast</strong>. Prediction intervals cannot be defensibly calculated from one point. Uploading additional past electricity bills will unlock moving-average and seasonal forecasting models.
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Staleness Policy Notice */}
                  {monthlyForecastResult.is_stale_history && (
                    <div className="p-3.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-900 flex items-start gap-2.5 text-xs">
                      <Clock className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
                      <p className="leading-relaxed">
                        <strong>Stale Billing History Policy:</strong> Historical records ended on{' '}
                        <strong>{monthlyForecastResult.last_observed_end_date}</strong> (more than 90 days ago). In live mode, future forecast periods are anchored to upcoming calendar months. Historical demand is used as a baseline.
                      </p>
                    </div>
                  )}

                  {/* Summary Metric Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                    {/* Total kWh */}
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                        Total Forecast Demand
                      </span>
                      <div className="text-2xl font-bold font-mono text-slate-900">
                        {monthlyForecastResult.summary.total_forecast_kwh.toFixed(1)} <span className="text-xs font-normal text-slate-500">kWh</span>
                      </div>
                      <span className="text-[11px] text-slate-500 block">
                        Across {monthlyForecastResult.forecast_steps.length} upcoming billing cycle{monthlyForecastResult.forecast_steps.length === 1 ? '' : 's'}
                      </span>
                    </div>

                    {/* Monthly Average kWh */}
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                        Monthly Average Demand
                      </span>
                      <div className="text-2xl font-bold font-mono text-slate-900">
                        {monthlyForecastResult.summary.average_monthly_kwh.toFixed(1)} <span className="text-xs font-normal text-slate-500">kWh/mo</span>
                      </div>
                      <span className="text-[11px] text-slate-500 block">
                        Mean projected monthly consumption
                      </span>
                    </div>

                    {/* Total Bill Amount in ₹ */}
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                        Total Estimated Bill
                      </span>
                      <div className="text-2xl font-bold font-mono text-slate-900">
                        {monthlyForecastResult.summary.total_estimated_bill_inr !== null && monthlyForecastResult.summary.total_estimated_bill_inr !== undefined ? (
                          <>
                            ₹{monthlyForecastResult.summary.total_estimated_bill_inr.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                          </>
                        ) : (
                          <span className="text-lg text-slate-500 font-sans font-medium">Unavailable</span>
                        )}
                      </div>
                      <span className="text-[11px] text-slate-500 block truncate" title={monthlyForecastResult.summary.bill_estimate_notes || ''}>
                        {monthlyForecastResult.summary.bill_estimate_notes || 'Tariff unverified or withheld'}
                      </span>
                    </div>

                    {/* Selected Algorithm */}
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                      <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                        Selected Method
                      </span>
                      <div className="text-sm font-bold text-slate-900 capitalize truncate" title={monthlyForecastResult.selected_method}>
                        {monthlyForecastResult.selected_method.replace(/_/g, ' ')}
                      </div>
                      <span className="text-[11px] text-slate-500 block line-clamp-2" title={monthlyForecastResult.method_description}>
                        {monthlyForecastResult.method_description}
                      </span>
                    </div>
                  </div>

                  {/* Step-by-Step Forecast Table */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5 text-blue-600" />
                        Upcoming Monthly Billing Cycle Projections
                      </h4>
                      <span className="text-[11px] text-slate-500">
                        Physical units (kWh) & Monetary estimates (₹)
                      </span>
                    </div>

                    <div className="overflow-x-auto border border-slate-200 rounded-lg">
                      <table className="w-full text-left text-xs text-slate-700">
                        <thead className="bg-slate-50 text-[11px] uppercase font-bold text-slate-600 border-b border-slate-200">
                          <tr>
                            <th className="py-2.5 px-3">Cycle</th>
                            <th className="py-2.5 px-3">Forecast Period</th>
                            <th className="py-2.5 px-3">Billing Dates</th>
                            <th className="py-2.5 px-3">Projected Demand</th>
                            <th className="py-2.5 px-3">95% Uncertainty Range</th>
                            <th className="py-2.5 px-3">Estimated Bill</th>
                            <th className="py-2.5 px-3 text-right">Tariff Details</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {monthlyForecastResult.forecast_steps.map((step, idx) => (
                            <React.Fragment key={idx}>
                              <tr className="hover:bg-slate-50/70 transition-colors">
                                <td className="py-2.5 px-3 font-semibold text-slate-500">
                                  Month #{step.month_index}
                                </td>
                                <td className="py-2.5 px-3 font-bold text-slate-900">
                                  {step.month_label}
                                </td>
                                <td className="py-2.5 px-3 text-slate-600 font-mono text-[11px]">
                                  {step.period_start} to {step.period_end}
                                </td>
                                <td className="py-2.5 px-3 font-mono font-bold text-blue-700 text-sm">
                                  {step.forecast_kwh.toFixed(1)} kWh
                                </td>
                                <td className="py-2.5 px-3 text-slate-600 font-mono text-[11px]">
                                  {step.uncertainty_lower_kwh !== null && step.uncertainty_upper_kwh !== null && step.uncertainty_lower_kwh !== undefined && step.uncertainty_upper_kwh !== undefined ? (
                                    <span>[{step.uncertainty_lower_kwh.toFixed(1)} – {step.uncertainty_upper_kwh.toFixed(1)} kWh]</span>
                                  ) : (
                                    <span className="text-slate-400 italic">None (sparse history)</span>
                                  )}
                                </td>
                                <td className="py-2.5 px-3">
                                  {step.estimated_bill_amount !== null && step.estimated_bill_amount !== undefined ? (
                                    <span className="font-mono font-bold text-emerald-700 text-sm">
                                      ₹{step.estimated_bill_amount.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                                    </span>
                                  ) : (
                                    <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-600 border border-slate-200">
                                      Unavailable
                                    </span>
                                  )}
                                </td>
                                <td className="py-2.5 px-3 text-right">
                                  <button
                                    type="button"
                                    onClick={() => setExpandedBreakdownIndex(expandedBreakdownIndex === idx ? null : idx)}
                                    className="inline-flex items-center gap-1 text-[11px] text-blue-600 hover:text-blue-800 font-semibold"
                                  >
                                    <span>{expandedBreakdownIndex === idx ? 'Hide' : 'Breakdown'}</span>
                                    {expandedBreakdownIndex === idx ? (
                                      <ChevronUp className="w-3 h-3" />
                                    ) : (
                                      <ChevronDown className="w-3 h-3" />
                                    )}
                                  </button>
                                </td>
                              </tr>

                              {/* Expandable Tariff Breakdown Row */}
                              {expandedBreakdownIndex === idx && (
                                <tr className="bg-slate-50/90">
                                  <td colSpan={7} className="p-3 border-t border-b border-slate-200">
                                    <div className="space-y-2 text-xs">
                                      <div className="font-bold text-slate-800 flex items-center justify-between">
                                        <span>Tariff Calculation Breakdown for {step.month_label} ({step.forecast_kwh} kWh)</span>
                                        <span className="font-normal text-[11px] text-slate-500">
                                          Provider: {step.bill_estimate?.provider_name || selectedProviderId.toUpperCase() || 'Unassigned'}
                                        </span>
                                      </div>

                                      {step.bill_estimate?.charge_breakdown ? (
                                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                                          <div className="p-2 rounded bg-white border border-slate-200">
                                            <span className="text-[10px] text-slate-500 uppercase block">Fixed Charges</span>
                                            <span className="font-mono font-bold text-slate-800">
                                              ₹{Number(step.bill_estimate.charge_breakdown.fixed_charges_inr ?? step.bill_estimate.charge_breakdown.fixed_charge ?? 0).toFixed(2)}
                                            </span>
                                          </div>
                                          <div className="p-2 rounded bg-white border border-slate-200">
                                            <span className="text-[10px] text-slate-500 uppercase block">Energy Slabs</span>
                                            <span className="font-mono font-bold text-slate-800">
                                              ₹{Number(step.bill_estimate.charge_breakdown.energy_charges_inr ?? step.bill_estimate.charge_breakdown.energy_charges_total ?? 0).toFixed(2)}
                                            </span>
                                          </div>
                                          <div className="p-2 rounded bg-white border border-slate-200">
                                            <span className="text-[10px] text-slate-500 uppercase block">Wheeling Charge</span>
                                            <span className="font-mono font-bold text-slate-800">
                                              ₹{Number(step.bill_estimate.charge_breakdown.wheeling_charges_inr ?? step.bill_estimate.charge_breakdown.wheeling_charges_total ?? 0).toFixed(2)}
                                            </span>
                                          </div>
                                          <div className="p-2 rounded bg-white border border-slate-200">
                                            <span className="text-[10px] text-slate-500 uppercase block">Electricity Duty (16%)</span>
                                            <span className="font-mono font-bold text-slate-800">
                                              ₹{Number(step.bill_estimate.charge_breakdown.electricity_duty_16pct_inr ?? step.bill_estimate.charge_breakdown.electricity_duty ?? 0).toFixed(2)}
                                            </span>
                                          </div>
                                        </div>
                                      ) : (
                                        <div className="p-2.5 rounded bg-amber-50 border border-amber-200 text-amber-900 text-xs">
                                          <p className="leading-relaxed">
                                            {step.bill_estimate?.disclosures?.[0] ||
                                              'No official tariff schedule is active or verified for this period. Physical consumption forecast is preserved; monetary bill estimate is withheld.'}
                                          </p>
                                        </div>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </React.Fragment>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Backtest Accuracy Metrics (if N >= 3) */}
                  {monthlyForecastResult.backtest_metrics && (
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs">
                      <div className="flex items-center justify-between">
                        <h4 className="font-bold text-slate-800 flex items-center gap-1.5">
                          <ShieldCheck className="w-4 h-4 text-emerald-600" />
                          Walk-Forward Backtesting Accuracy Metrics
                        </h4>
                        <span className="text-[11px] text-slate-500">
                          Evaluated over {monthlyForecastResult.backtest_metrics.backtest_periods} historical cycle{monthlyForecastResult.backtest_metrics.backtest_periods === 1 ? '' : 's'} without leakage
                        </span>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                        <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                          <span className="text-[10px] text-slate-500 uppercase block">Walk-Forward MAE</span>
                          <span className="font-mono font-bold text-slate-800">
                            {monthlyForecastResult.backtest_metrics.mae_kwh.toFixed(1)} kWh
                          </span>
                        </div>
                        <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                          <span className="text-[10px] text-slate-500 uppercase block">Walk-Forward RMSE</span>
                          <span className="font-mono font-bold text-slate-800">
                            {monthlyForecastResult.backtest_metrics.rmse_kwh.toFixed(1)} kWh
                          </span>
                        </div>
                        <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                          <span className="text-[10px] text-slate-500 uppercase block">Naive Baseline MAE</span>
                          <span className="font-mono font-bold text-slate-800">
                            {monthlyForecastResult.backtest_metrics.baseline_mae_kwh.toFixed(1)} kWh
                          </span>
                        </div>
                        <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                          <span className="text-[10px] text-slate-500 uppercase block">Bounded sMAPE</span>
                          <span className="font-mono font-bold text-slate-800">
                            {monthlyForecastResult.backtest_metrics.smape_percent !== null && monthlyForecastResult.backtest_metrics.smape_percent !== undefined
                              ? `${monthlyForecastResult.backtest_metrics.smape_percent.toFixed(1)}%`
                              : 'N/A'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Warnings and Disclosures */}
                  {(monthlyForecastResult.warnings.length > 0 || monthlyForecastResult.limitations.length > 0) && (
                    <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs">
                      <span className="font-bold text-slate-800 block">
                        Forecast Disclosures & Model Limitations:
                      </span>
                      <ul className="list-disc list-inside space-y-1 text-slate-600">
                        {monthlyForecastResult.warnings.map((w, i) => (
                          <li key={`w-${i}`}>{w}</li>
                        ))}
                        {monthlyForecastResult.limitations.map((l, i) => (
                          <li key={`l-${i}`} className="text-slate-500 italic">{l}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Persistence Architecture Notice */}
                  <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200 text-slate-600 text-xs flex items-start gap-2.5">
                    <Info className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
                    <p className="leading-relaxed">
                      <strong>Persistent Forecast History:</strong> Save this forecast run using the <em>Save Forecast Run</em> action above to preserve an immutable snapshot in your account history. You can view all saved monthly forecasts alongside 24-hour smart meter runs in <Link to="/forecast/history?tab=monthly" className="text-blue-600 font-semibold underline">Forecast History</Link>.
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

        {/* ================================================================= */}
        {/* WORKFLOW 2: 168-HOUR HOURLY FORECAST (CSV OR MANUAL)              */}
        {/* ================================================================= */}
        {activeTab !== 'bill' && (
          <form onSubmit={handleHourlySubmit} className="space-y-6 pt-2">
            {/* Input Requirements & File Specification Card */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 pb-2 border-b border-slate-200">
                <div>
                  <h3 className="text-xs font-bold text-slate-900 flex items-center gap-2">
                    <FileSpreadsheet className="w-4 h-4 text-blue-600" />
                    168-Hour Lookback Requirement & Schema
                  </h3>
                  <p className="text-xs text-slate-600 mt-0.5">
                    The day-ahead forecasting models require exactly 168 consecutive hourly electricity observations to capture diurnal and weekly cycles.
                  </p>
                </div>
                {activeTab === 'csv' && (
                  <button
                    type="button"
                    onClick={handleDownloadSample}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 hover:bg-blue-100 border border-blue-200 text-xs font-semibold text-blue-700 transition-colors flex-shrink-0"
                  >
                    <Download className="w-3.5 h-3.5 text-blue-600" />
                    Download Sample CSV
                  </button>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-white border border-slate-200 space-y-1.5">
                  <div className="font-bold text-slate-900 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-blue-600" />
                    Required Electricity Series
                  </div>
                  <ul className="space-y-1 text-slate-600 text-[11px] leading-relaxed">
                    <li>• Column: <strong className="text-slate-800 font-mono">hourly_active_power</strong> (or <span className="font-mono text-slate-800">active_power</span>)</li>
                    <li>• Sequence Length: Exactly <strong className="text-slate-800">168 hourly rows</strong> (7 full days)</li>
                    <li>• Units: Active electricity demand in <strong className="text-slate-800">Watts (W)</strong></li>
                    <li>• Data Integrity: Non-negative numerical values without missing (NaN) cells</li>
                  </ul>
                </div>

                <div className="p-3 rounded-lg bg-white border border-slate-200 space-y-1.5">
                  <div className="font-bold text-slate-900 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-600" />
                    Optional Weather Telemetry
                  </div>
                  <ul className="space-y-1 text-slate-600 text-[11px] leading-relaxed">
                    <li>• Timestamp: <span className="font-mono text-slate-800">date</span> or <span className="font-mono text-slate-800">timestamp</span> (ISO format)</li>
                    <li>• Weather: <span className="font-mono text-slate-800">temp, feels_like, humidity, pressure, speed, deg</span></li>
                    <li>• Adaptive Fallback: If weather is absent, system uses univariate <strong className="text-slate-800">Energy-only</strong> model</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Tab 1: CSV Upload Dropzone */}
            {activeTab === 'csv' ? (
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-2">
                  Select or Drop 168-Hour CSV File
                </label>
                <div
                  className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all ${
                    file
                      ? 'border-blue-500 bg-blue-50/40'
                      : 'border-slate-300 hover:border-blue-400 bg-slate-50/50 hover:bg-slate-50'
                  }`}
                >
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    id="csv-file-input"
                  />
                  <div className="flex flex-col items-center justify-center pointer-events-none">
                    <div
                      className={`w-12 h-12 rounded-full flex items-center justify-center mb-3 ${
                        file ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-500'
                      }`}
                    >
                      {file ? <CheckCircle2 className="w-6 h-6" /> : <UploadCloud className="w-6 h-6" />}
                    </div>
                    {file ? (
                      <div>
                        <p className="text-sm font-bold text-slate-900">{file.name}</p>
                        <p className="text-xs text-slate-500 mt-1">
                          {(file.size / 1024).toFixed(1)} KB • Ready for calibrated inference
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-sm font-semibold text-slate-800">
                          Drop your 168-hour CSV here, or <span className="text-blue-600">browse files</span>
                        </p>
                        <p className="text-xs text-slate-500 mt-1">
                          Must be UTF-8 encoded text with exactly 168 hourly rows.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Tab 3: Manual 168 Numbers Entry */
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-xs font-semibold text-slate-700">
                    Paste 168 Hourly Active Power Values (Watts)
                  </label>
                  <span
                    className={`text-xs font-mono font-semibold ${
                      manualTokenCount === 168
                        ? 'text-emerald-600'
                        : manualTokenCount > 0
                        ? 'text-amber-600'
                        : 'text-slate-400'
                    }`}
                  >
                    {manualTokenCount} / 168 values entered
                  </span>
                </div>
                <textarea
                  value={manualText}
                  onChange={(e) => {
                    setManualText(e.target.value);
                    setErrorMsg(null);
                  }}
                  rows={6}
                  placeholder="Paste 168 power values separated by commas, tabs, or newlines (e.g. 240, 215, 310, 450, ...)"
                  className="w-full font-mono text-xs p-3.5 border border-slate-300 rounded-lg bg-slate-50 text-slate-800 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                />
                <p className="text-[11px] text-slate-500 mt-1.5">
                  Separate values by commas or new lines. Active power must be positive numerical Watts.
                </p>
              </div>
            )}

            {/* Model Selection Options & Tag */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-100">
              <div>
                <label
                  htmlFor="model-selection-mode"
                  className="block text-xs font-bold uppercase tracking-wider text-slate-700 mb-1.5 flex items-center gap-1.5"
                >
                  Model Architecture
                  <span title="Automatic mode uses Early Fusion when full weather features are present, or falls back to Energy-only if weather is absent.">
                    <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
                  </span>
                </label>
                <select
                  id="model-selection-mode"
                  value={modelMode}
                  onChange={(e) => setModelMode(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3.5 py-2 text-xs font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors"
                >
                  <option value="auto">Automatic (Weather-Adaptive Selection)</option>
                  <option value="early_fusion">Early Fusion (Requires weather telemetry)</option>
                  <option value="energy_only">Energy-only (Univariate baseline)</option>
                  <option value="late_fusion">Late Fusion (Dual branch architecture)</option>
                  <option value="hybrid_fusion">Hybrid Fusion (Shared + Head)</option>
                </select>
                <p className="text-[11px] text-slate-500 mt-1">
                  {modelMode === 'auto'
                    ? 'Recommended: Automatically selects Early Fusion when weather is provided, or Energy-only if omitted.'
                    : `Explicit override: ${modelMode}.`}
                </p>
              </div>

              <div>
                <label
                  htmlFor="user-id-input"
                  className="block text-xs font-bold uppercase tracking-wider text-slate-700 mb-1.5"
                >
                  Household Identifier / Label (Optional)
                </label>
                <input
                  id="user-id-input"
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  placeholder="e.g. main_household_meter"
                  className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3.5 py-2 text-xs font-medium text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors"
                />
                <p className="text-[11px] text-slate-500 mt-1">
                  Optional reference tag to distinguish runs in your audit history.
                </p>
              </div>
            </div>

            {/* Household Location & Automated Weather (Optional) */}
            <div className="pt-3 border-t border-slate-100 space-y-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
                    <MapPin className="w-3.5 h-3.5 text-blue-600" />
                    Household Location & Automated Weather
                    <span className="text-[10px] font-normal normal-case px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                      Optional
                    </span>
                  </h4>
                  <p className="text-[11px] text-slate-500 mt-0.5">
                    Enter geographic coordinates to automatically retrieve 168 hours of localized historical weather telemetry (temperature, humidity, wind, and pressure) from Open-Meteo.
                  </p>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  {(latitude || longitude || locationLabel) && (
                    <button
                      type="button"
                      onClick={handleClearLocation}
                      className="px-2.5 py-1 text-xs text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors border border-slate-200"
                      title="Clear location inputs"
                    >
                      Clear
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleUseCurrentLocation}
                    disabled={isLocating || isSubmitting}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 text-xs font-semibold transition-colors disabled:opacity-50"
                    title="Acquire current coordinates using browser geolocation"
                  >
                    {isLocating ? (
                      <>
                        <div className="w-3 h-3 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
                        <span>Locating...</span>
                      </>
                    ) : (
                      <>
                        <Navigation className="w-3.5 h-3.5 text-blue-600" />
                        <span>Use Current Location</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Workflow Guidance Notice */}
              {activeTab === 'manual' ? (
                <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-start gap-2.5">
                  <Info className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                  <div className="space-y-0.5">
                    <div className="font-semibold text-amber-900">Manual Workflow Limitation: Energy-Only Mode</div>
                    <p className="text-amber-700 leading-relaxed text-[11px]">
                      Manual input accepts 168 raw active power readings without hourly timestamps. Because exact calendar timestamps are required to align historical weather observations from Open-Meteo, manual forecasts operate in univariate <strong>Energy-only</strong> mode. To use automatic weather retrieval, please upload a 168-hour CSV with timestamp columns.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-slate-600 text-xs flex items-start gap-2.5">
                  <Info className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
                  <p className="text-slate-600 leading-relaxed text-[11px]">
                    <strong>Weather Precedence Policy:</strong> If your uploaded CSV already contains complete weather columns (temp, humidity, pressure, etc.), the uploaded data is preferred and used directly. Automated weather retrieval is only triggered when weather columns are missing and timestamps are present.
                  </p>
                </div>
              )}

              {/* Geolocation Feedback Messages */}
              {locationError && (
                <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
                    <span>{locationError}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setLocationError(null)}
                    className="text-[11px] font-semibold underline text-rose-700 hover:text-rose-900"
                  >
                    Dismiss
                  </button>
                </div>
              )}

              {locationSuccessMsg && (
                <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                    <span>{locationSuccessMsg}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setLocationSuccessMsg(null)}
                    className="text-[11px] font-semibold underline text-emerald-700 hover:text-emerald-900"
                  >
                    Dismiss
                  </button>
                </div>
              )}

              {/* Coordinates Inputs */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div>
                  <label htmlFor="lat-input" className="block text-[11px] font-semibold text-slate-700 mb-1">
                    Latitude (°N/°S)
                  </label>
                  <input
                    id="lat-input"
                    type="number"
                    step="any"
                    min="-90"
                    max="90"
                    value={latitude}
                    onChange={(e) => {
                      setLatitude(e.target.value);
                      setLocationError(null);
                      setLocationSuccessMsg(null);
                    }}
                    placeholder="e.g. 18.5204"
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors"
                  />
                  <span className="text-[10px] text-slate-400 mt-0.5 block">Bounds: -90.0 to 90.0</span>
                </div>

                <div>
                  <label htmlFor="lon-input" className="block text-[11px] font-semibold text-slate-700 mb-1">
                    Longitude (°E/°W)
                  </label>
                  <input
                    id="lon-input"
                    type="number"
                    step="any"
                    min="-180"
                    max="180"
                    value={longitude}
                    onChange={(e) => {
                      setLongitude(e.target.value);
                      setLocationError(null);
                      setLocationSuccessMsg(null);
                    }}
                    placeholder="e.g. 73.8567"
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors"
                  />
                  <span className="text-[10px] text-slate-400 mt-0.5 block">Bounds: -180.0 to 180.0</span>
                </div>

                <div>
                  <label htmlFor="loc-label-input" className="block text-[11px] font-semibold text-slate-700 mb-1">
                    Location Label (Optional)
                  </label>
                  <input
                    id="loc-label-input"
                    type="text"
                    maxLength={128}
                    value={locationLabel}
                    onChange={(e) => {
                      setLocationLabel(e.target.value);
                      setLocationError(null);
                    }}
                    placeholder="e.g. Pune, MH or Home"
                    className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-medium text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors"
                  />
                  <span className="text-[10px] text-slate-400 mt-0.5 block">Friendly city or site label</span>
                </div>
              </div>
            </div>

            {/* Validation Error Alert */}
            {errorMsg && (
              <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start gap-3 text-xs">
                <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <h4 className="font-bold text-rose-900 text-sm">Input Validation Error</h4>
                  <p className="text-rose-700 leading-relaxed whitespace-pre-wrap">{errorMsg}</p>
                </div>
              </div>
            )}

            {/* Action Submission */}
            <div className="pt-2 flex items-center justify-between border-t border-slate-100">
              <span className="text-xs text-slate-500">
                {activeTab === 'csv'
                  ? file
                    ? `Selected: ${file.name}`
                    : 'No file selected yet'
                  : `${manualTokenCount} of 168 values entered`}
              </span>

              <button
                type="submit"
                disabled={
                  isSubmitting ||
                  (activeTab === 'csv' && !file) ||
                  (activeTab === 'manual' && manualTokenCount === 0)
                }
                className={`inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-semibold text-xs text-white shadow-sm transition-all ${
                  isSubmitting ||
                  (activeTab === 'csv' && !file) ||
                  (activeTab === 'manual' && manualTokenCount === 0)
                    ? 'bg-slate-300 cursor-not-allowed text-slate-500'
                    : 'bg-blue-600 hover:bg-blue-700 active:scale-95'
                }`}
              >
                {isSubmitting ? (
                  <>
                    <div className="w-3.5 h-3.5 rounded-full border-2 border-white border-t-transparent animate-spin" />
                    <span>Validating & Forecasting...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Generate 24-Hour Forecast</span>
                  </>
                )}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};

export default NewForecast;
