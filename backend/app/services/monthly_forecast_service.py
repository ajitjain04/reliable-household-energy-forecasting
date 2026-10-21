"""
Monthly Household Electricity Consumption Forecasting Service.

Implements a transparent, conservative forecasting pipeline for sparse monthly
billing history:
- Strict temporal validation, duplicate detection, and chronological ordering.
- Preserves missing billing cycles as missing; never invents synthetic readings.
- History-dependent method selection:
    A. N = 1: Baseline persistence only (clearly qualified, no fabricated intervals).
    B. N in [2, 5]: Weighted Moving Average / EWMA with empirical baseline variance.
    C. N in [6, 11]: Chronological rolling backtest comparing Naive vs SMA vs Damped Trend.
    D. N >= 12: Seasonal Naive / Holt-Winters with out-of-sample seasonal validation.
- Chronological backtesting metrics (MAE, RMSE, sMAPE) without future-data leakage.
- Independent, decoupled tariff bill estimation preserving physical kWh predictions.
- Staleness detection for older test or legacy bills.
"""

import calendar
import math
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.app.schemas.bills import MonthlyConsumptionRecord, validate_and_sort_billing_records
from backend.app.schemas.monthly_forecast import (
    BacktestMetrics,
    ConfidenceRating,
    MonthlyForecastMethod,
    MonthlyForecastRequest,
    MonthlyForecastResponse,
    MonthlyForecastStep,
    MonthlyForecastSummary,
)
from backend.app.services.tariff_service import tariff_registry


class MonthlyForecastService:
    """Service executing monthly electricity demand forecasting with strict history tiers."""

    def __init__(self):
        pass

    def _project_future_monthly_periods(
        self,
        last_record: MonthlyConsumptionRecord,
        horizon_months: int,
        is_live: bool = True,
        reference_date: Optional[date] = None,
    ) -> List[Tuple[date, date, str]]:
        """
        Project future billing cycle date intervals and human-readable month labels.

        - If is_live is False (offline / historical test fixture):
          Projects immediately forward from the last observed billing cycle
          (e.g., April 2025 -> May, June, July 2025), preserving historical sequence tests.
        - If is_live is True (live household forecast):
          Anchors future forecast months to upcoming calendar periods relative to reference_date
          (or today's date if omitted).
          If the history ended in April 2025 and reference_date is October 2026, upcoming
          forecast periods start in November 2026, NEVER returning May–July 2025.
        """
        results: List[Tuple[date, date, str]] = []
        is_calendar_aligned = (
            last_record.billing_period_start.day == 1
            and last_record.billing_period_end.day >= 28
        )

        last_start = last_record.billing_period_start
        last_end = last_record.billing_period_end

        ref_date = reference_date or date.today()

        if is_live:
            # Anchor to reference date or last_end, whichever is further in time
            if last_end >= ref_date:
                # Latest bill is current or future relative to ref_date
                base_start_y = last_start.year
                base_start_m = last_start.month
            else:
                # Latest bill is historical / stale relative to ref_date (e.g. Apr 2025 in Oct 2026)
                # Next upcoming cycle starts in the month following ref_date (e.g. Nov 2026)
                base_start_y = ref_date.year
                base_start_m = ref_date.month
        else:
            # Offline / historical mode: project strictly forward from last record
            base_start_y = last_start.year
            base_start_m = last_start.month

        for step in range(1, horizon_months + 1):
            if is_calendar_aligned:
                total_months = base_start_y * 12 + (base_start_m - 1) + step
                proj_year = total_months // 12
                proj_month = (total_months % 12) + 1
                _, num_days = calendar.monthrange(proj_year, proj_month)

                step_start = date(proj_year, proj_month, 1)
                step_end = date(proj_year, proj_month, num_days)
            else:
                tot_start_m = base_start_y * 12 + (base_start_m - 1) + step
                s_year = tot_start_m // 12
                s_month = (tot_start_m % 12) + 1
                _, s_max_days = calendar.monthrange(s_year, s_month)
                step_start = date(s_year, s_month, min(last_start.day, s_max_days))

                day_offset = 1 if last_end.day <= last_start.day else 0
                tot_end_m = base_start_y * 12 + (base_start_m - 1) + step + day_offset
                e_year = tot_end_m // 12
                e_month = (tot_end_m % 12) + 1
                _, e_max_days = calendar.monthrange(e_year, e_month)
                step_end = date(e_year, e_month, min(last_end.day, e_max_days))

            month_label = step_start.strftime("%b %Y")
            results.append((step_start, step_end, month_label))

        return results

    def _calculate_smape(self, actuals: List[float], preds: List[float]) -> Optional[float]:
        """Compute Symmetric Mean Absolute Percentage Error bounded in [0%, 200%]."""
        if not actuals or len(actuals) != len(preds):
            return None
        denom_sum = 0.0
        diff_sum = 0.0
        valid_count = 0
        for y, y_hat in zip(actuals, preds):
            denom = abs(y) + abs(y_hat)
            if denom > 1e-4:
                diff_sum += (2.0 * abs(y - y_hat)) / denom
                valid_count += 1
        if valid_count == 0:
            return 0.0
        return round((diff_sum / valid_count) * 100.0, 2)

    def generate_forecast(
        self,
        request: MonthlyForecastRequest,
    ) -> MonthlyForecastResponse:
        """
        Generate a conservative, validated monthly forecast from confirmed billing history.
        """
        # 1. Temporal sorting and duplicate/overlap validation
        sorted_records, gap_warnings = validate_and_sort_billing_records(request.records)
        warnings: List[str] = list(gap_warnings)
        limitations: List[str] = []

        history_kwh = [r.consumption_kwh for r in sorted_records]
        n_obs = len(history_kwh)
        last_rec = sorted_records[-1]

        # 2. Plausibility sanity checks
        if any(kwh > 50000.0 for kwh in history_kwh):
            warnings.append(
                "One or more billing cycles indicate extremely high consumption (>50,000 kWh). "
                "Verify whether these represent multi-family or commercial loads."
            )

        # Check for bi-monthly records (e.g. TANGEDCO 60-day cycles)
        bimonthly_records = [r for r in sorted_records if r.duration_days > 45]
        if bimonthly_records:
            first_bm = bimonthly_records[0]
            warnings.append(
                f"Input history contains bi-monthly record(s) (e.g. [{first_bm.billing_period_start} to {first_bm.billing_period_end}], {first_bm.duration_days} days). "
                "The actual bi-monthly consumption is evaluated directly without synthetic monthly division. "
                "Downstream future monthly projections are generated on standard calendar month cycles."
            )
            limitations.append(
                f"Bi-monthly billing history: {len(bimonthly_records)} record(s) cover ~60-day billing cycles."
            )

        # 3. Staleness detection & documented policy
        ref_date = request.reference_date or date.today()
        is_stale = False
        days_since_last = (ref_date - last_rec.billing_period_end).days
        if days_since_last > 90:
            is_stale = True
            if request.is_live:
                warnings.append(
                    f"Historical billing records end on {last_rec.billing_period_end} "
                    f"({days_since_last} days prior to reference date {ref_date}, more than 90 days ago, exceeding the 90-day threshold). "
                    "Under live household forecasting policy, future projections are anchored to upcoming calendar "
                    "months. The historical consumption baseline is retained as the demand estimate, but prediction "
                    "confidence is reduced because household occupancy, appliances, or seasonal patterns may have shifted."
                )
                limitations.append(
                    f"Stale billing history: Projections are anchored to upcoming months after {ref_date} "
                    f"using historical baseline ending {last_rec.billing_period_end}."
                )
            else:
                warnings.append(
                    f"Historical billing records end on {last_rec.billing_period_end} "
                    f"({days_since_last} days ago, more than 90 days ago). Projected consumption assumes household "
                    "occupancy and major appliances have remained consistent since that cycle."
                )

        # 4. Project future date intervals
        projected_periods = self._project_future_monthly_periods(
            last_record=last_rec,
            horizon_months=request.horizon_months,
            is_live=request.is_live,
            reference_date=request.reference_date,
        )

        forecast_values: List[float] = []
        lower_bounds: List[Optional[float]] = []
        upper_bounds: List[Optional[float]] = []
        backtest_metrics: Optional[BacktestMetrics] = None

        # ---------------------------------------------------------------------
        # TIER A: N = 1 (Single Observation — e.g. Single Bill Upload)
        # ---------------------------------------------------------------------
        if n_obs == 1:
            selected_method = MonthlyForecastMethod.BASELINE_ONLY
            base_val = round(history_kwh[0], 2)
            method_description = (
                f"Baseline Persistence: Last confirmed monthly reading ({base_val} kWh). "
                "Single observation is insufficient to estimate trend, variance, or seasonality."
            )
            confidence_rating = ConfidenceRating.LOW

            warnings.append(
                f"Single billing period observed ({base_val} kWh). One observation is insufficient "
                "to estimate consumption trends or seasonal variations (such as summer air conditioning "
                "or winter heating). Results represent a static baseline persistence only."
            )
            limitations.append(
                "Prediction intervals cannot be defensibly calculated from a single observation."
            )
            if request.horizon_months > 1:
                limitations.append(
                    f"Multi-month horizon ({request.horizon_months} months) is projected using static "
                    "persistence because no trend or seasonality can be identified from one observation."
                )

            for _ in range(request.horizon_months):
                forecast_values.append(base_val)
                lower_bounds.append(None)
                upper_bounds.append(None)

        # ---------------------------------------------------------------------
        # TIER B: Short History (N = 2 to 5 Observations)
        # ---------------------------------------------------------------------
        elif 2 <= n_obs <= 5:
            selected_method = MonthlyForecastMethod.WEIGHTED_MOVING_AVERAGE
            weights = [float(i + 1) for i in range(n_obs)]
            total_w = sum(weights)
            weighted_avg = round(sum(w * y for w, y in zip(weights, history_kwh)) / total_w, 2)

            method_description = (
                f"Exponentially Weighted Moving Average across {n_obs} cycles ({weighted_avg} kWh). "
                "Weights recent billing cycles higher while avoiding overfitting to short history."
            )
            confidence_rating = ConfidenceRating.MODERATE

            warnings.append(
                f"Short billing history ({n_obs} months). Seasonal fluctuations (monsoon, summer, winter) "
                "cannot be confirmed without at least 12 consecutive months of billing history."
            )
            limitations.append(
                "Forecast assumes household baseline demand without seasonal decomposition."
            )

            # Empirical sample standard deviation if N >= 3
            if n_obs >= 3:
                mean_hist = sum(history_kwh) / n_obs
                variance = sum((y - mean_hist) ** 2 for y in history_kwh) / (n_obs - 1)
                std_dev = math.sqrt(variance)

                # Chronological 1-step backtest on the final observed point
                train_series = history_kwh[:-1]
                train_weights = [float(i + 1) for i in range(len(train_series))]
                pred_last = sum(w * y for w, y in zip(train_weights, train_series)) / sum(train_weights)
                actual_last = history_kwh[-1]
                mae_backtest = round(abs(actual_last - pred_last), 2)
                rmse_backtest = round(abs(actual_last - pred_last), 2)
                base_mae = round(abs(actual_last - history_kwh[-2]), 2)
                smape_val = self._calculate_smape([actual_last], [pred_last])

                backtest_metrics = BacktestMetrics(
                    backtest_periods=1,
                    mae_kwh=mae_backtest,
                    rmse_kwh=rmse_backtest,
                    baseline_mae_kwh=base_mae,
                    smape_percent=smape_val,
                )
            else:
                std_dev = 0.0
                limitations.append(
                    "At least 3 historical observations are required to estimate variance for prediction intervals."
                )

            for step_idx in range(1, request.horizon_months + 1):
                forecast_values.append(weighted_avg)
                if n_obs >= 3 and std_dev > 0.0:
                    # Expanding interval with horizon step
                    horizon_factor = math.sqrt(1.0 + (step_idx - 1) * 0.15)
                    margin = round(1.96 * std_dev * horizon_factor, 2)
                    lower = max(0.0, round(weighted_avg - margin, 2))
                    upper = round(weighted_avg + margin, 2)
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                else:
                    lower_bounds.append(None)
                    upper_bounds.append(None)

        # ---------------------------------------------------------------------
        # TIER C: Medium History (N = 6 to 11 Observations)
        # ---------------------------------------------------------------------
        elif 6 <= n_obs <= 11:
            selected_method = MonthlyForecastMethod.DAMPED_TREND_MOVING_AVERAGE
            confidence_rating = ConfidenceRating.MODERATE

            warnings.append(
                f"History contains {n_obs} months. Local trend and moving average baseline are evaluated, "
                "but full annual seasonality requires at least 12 months."
            )

            # Chronological backtesting on last 3 points
            # Compare: 1) Naive, 2) 3-month SMA, 3) Holt Damped Trend
            k_eval = 3
            naive_errors: List[float] = []
            sma_errors: List[float] = []
            damped_errors: List[float] = []
            eval_actuals: List[float] = []
            eval_preds: List[float] = []

            for split_idx in range(n_obs - k_eval, n_obs):
                train_split = history_kwh[:split_idx]
                actual_val = history_kwh[split_idx]
                eval_actuals.append(actual_val)

                # Cand 1: Naive
                p_naive = train_split[-1]
                naive_errors.append(abs(actual_val - p_naive))

                # Cand 2: SMA(3)
                p_sma = sum(train_split[-3:]) / 3.0
                sma_errors.append(abs(actual_val - p_sma))

                # Cand 3: Damped Trend (alpha=0.4, beta=0.2, phi=0.85)
                level = train_split[0]
                trend = train_split[1] - train_split[0]
                for obs in train_split[1:]:
                    last_level = level
                    level = 0.4 * obs + 0.6 * (level + 0.85 * trend)
                    trend = 0.2 * (level - last_level) + 0.8 * 0.85 * trend
                p_damped = max(0.0, level + 0.85 * trend)
                damped_errors.append(abs(actual_val - p_damped))

            mean_naive_mae = sum(naive_errors) / k_eval
            mean_sma_mae = sum(sma_errors) / k_eval
            mean_damped_mae = sum(damped_errors) / k_eval

            # Select best candidate
            if mean_damped_mae <= mean_sma_mae and mean_damped_mae <= mean_naive_mae:
                method_description = (
                    f"Holt's Damped Linear Trend selected via walk-forward backtest (MAE: {mean_damped_mae:.1f} kWh "
                    f"vs Naive {mean_naive_mae:.1f} kWh)."
                )
                winning_mae = mean_damped_mae
                eval_preds = [
                    train_val - err for train_val, err in zip(eval_actuals, damped_errors)
                ]
                # Generate forecasts with Damped Trend on full history
                level = history_kwh[0]
                trend = history_kwh[1] - history_kwh[0]
                for obs in history_kwh[1:]:
                    last_level = level
                    level = 0.4 * obs + 0.6 * (level + 0.85 * trend)
                    trend = 0.2 * (level - last_level) + 0.8 * 0.85 * trend
                for step_idx in range(1, request.horizon_months + 1):
                    cum_phi = sum((0.85 ** i) for i in range(1, step_idx + 1))
                    f_val = max(0.0, round(level + cum_phi * trend, 2))
                    forecast_values.append(f_val)
            else:
                method_description = (
                    f"3-Month Simple Moving Average selected via walk-forward backtest (MAE: {mean_sma_mae:.1f} kWh "
                    f"vs Naive {mean_naive_mae:.1f} kWh)."
                )
                winning_mae = mean_sma_mae
                f_val = round(sum(history_kwh[-3:]) / 3.0, 2)
                for _ in range(request.horizon_months):
                    forecast_values.append(f_val)

            # Residual standard deviation from backtesting
            resids = [abs(a - p) for a, p in zip(eval_actuals, eval_preds)] if eval_preds else damped_errors
            s_e = math.sqrt(sum(r ** 2 for r in resids) / len(resids)) if resids else 10.0

            backtest_metrics = BacktestMetrics(
                backtest_periods=k_eval,
                mae_kwh=round(winning_mae, 2),
                rmse_kwh=round(s_e, 2),
                baseline_mae_kwh=round(mean_naive_mae, 2),
                smape_percent=self._calculate_smape(eval_actuals, [a - r for a, r in zip(eval_actuals, resids)]),
            )

            for step_idx, f_val in enumerate(forecast_values, start=1):
                horizon_factor = math.sqrt(step_idx)
                margin = round(1.96 * s_e * horizon_factor, 2)
                lower = max(0.0, round(f_val - margin, 2))
                upper = round(f_val + margin, 2)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

        # ---------------------------------------------------------------------
        # TIER D: Long History (N >= 12 Observations)
        # ---------------------------------------------------------------------
        else:
            selected_method = MonthlyForecastMethod.SEASONAL_NAIVE
            confidence_rating = ConfidenceRating.HIGH
            method_description = (
                f"Seasonal Naive Forecasting (lag-12) utilizing {n_obs} confirmed monthly records. "
                "Captures verified recurring seasonal cycles from corresponding months of prior years."
            )

            # Chronological backtest on available lag-12 points (requires N >= 13 for out-of-sample seasonal eval)
            k_eval = min(6, n_obs - 12)
            if k_eval >= 1:
                snaive_errors: List[float] = []
                naive_errors: List[float] = []
                eval_actuals: List[float] = []
                eval_preds: List[float] = []

                for split_idx in range(n_obs - k_eval, n_obs):
                    actual_val = history_kwh[split_idx]
                    eval_actuals.append(actual_val)
                    p_snaive = history_kwh[split_idx - 12]
                    eval_preds.append(p_snaive)
                    snaive_errors.append(abs(actual_val - p_snaive))
                    p_naive = history_kwh[split_idx - 1]
                    naive_errors.append(abs(actual_val - p_naive))

                mean_snaive_mae = sum(snaive_errors) / k_eval
                mean_naive_mae = sum(naive_errors) / k_eval
                rmse_snaive = math.sqrt(sum(e ** 2 for e in snaive_errors) / k_eval)

                backtest_metrics = BacktestMetrics(
                    backtest_periods=k_eval,
                    mae_kwh=round(mean_snaive_mae, 2),
                    rmse_kwh=round(rmse_snaive, 2),
                    baseline_mae_kwh=round(mean_naive_mae, 2),
                    smape_percent=self._calculate_smape(eval_actuals, eval_preds),
                )
            else:
                mean_hist = sum(history_kwh) / n_obs
                rmse_snaive = math.sqrt(sum((y - mean_hist) ** 2 for y in history_kwh) / (n_obs - 1))
                backtest_metrics = None
                limitations.append(
                    "Initial 12-month annual cycle established. Out-of-sample seasonal backtesting "
                    "activates when 13 or more historical billing cycles are available."
                )

            # Generate predictions: project each step from corresponding month 12 steps prior
            for step_idx in range(1, request.horizon_months + 1):
                lag_idx = n_obs - 12 + ((step_idx - 1) % 12)
                f_val = round(history_kwh[lag_idx], 2)
                forecast_values.append(f_val)

                margin = round(1.96 * rmse_snaive * math.sqrt(1.0 + (step_idx - 1) * 0.1), 2)
                lower = max(0.0, round(f_val - margin, 2))
                upper = round(f_val + margin, 2)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

        # ---------------------------------------------------------------------
        # 5. Assemble Forecast Steps and Optional Tariff Estimates
        # ---------------------------------------------------------------------
        forecast_steps: List[MonthlyForecastStep] = []
        valid_bill_estimates: List[float] = []

        for idx, (f_kwh, low, high, (s_start, s_end, label)) in enumerate(
            zip(forecast_values, lower_bounds, upper_bounds, projected_periods),
            start=1,
        ):
            bill_est = None
            step_bill_amt: Optional[float] = None
            if request.provider_id:
                bill_est = tariff_registry.estimate_bill(
                    forecast_kwh=f_kwh,
                    provider_id=request.provider_id,
                    as_of_date=s_start,
                    billing_period_start=s_start,
                    billing_period_end=s_end,
                    connection_params={
                        "phase_type": request.phase_type,
                        "sanctioned_load_kw": request.sanctioned_load_kw,
                    },
                )
                if bill_est and bill_est.is_tariff_verified and bill_est.estimated_bill_amount is not None:
                    step_bill_amt = round(float(bill_est.estimated_bill_amount), 2)
                    valid_bill_estimates.append(step_bill_amt)

            t_status = bill_est.tariff_status if bill_est else ("no_provider" if not request.provider_id else "unverified_schedule")
            t_label = bill_est.tariff_status_label if bill_est else ("No provider selected" if not request.provider_id else "Current tariff unverified")

            forecast_steps.append(
                MonthlyForecastStep(
                    month_index=idx,
                    month_label=label,
                    period_start=s_start,
                    period_end=s_end,
                    forecast_kwh=f_kwh,
                    uncertainty_lower_kwh=low,
                    uncertainty_upper_kwh=high,
                    bill_estimate=bill_est,
                    estimated_bill_amount=step_bill_amt,
                    tariff_status=t_status,
                    tariff_status_label=t_label,
                )
            )

        # 6. Build Summary Metrics
        tot_kwh = round(sum(forecast_values), 2)
        avg_kwh = round(tot_kwh / len(forecast_values), 2) if forecast_values else 0.0

        months_with_estimates = len(valid_bill_estimates)
        total_horizon_steps = len(forecast_steps)

        if not request.provider_id:
            total_bill_inr = None
            bill_notes = "No utility provider was specified. Monetary bill estimates are withheld."
        elif months_with_estimates == total_horizon_steps and total_horizon_steps > 0:
            total_bill_inr = round(sum(valid_bill_estimates), 2)
            bill_notes = f"Verified tariff estimates available for all {months_with_estimates} forecast billing cycles."
        elif months_with_estimates > 0:
            total_bill_inr = None
            bill_notes = (
                f"Cumulative bill estimate withheld: only {months_with_estimates} of {total_horizon_steps} "
                "months have verified tariff rates. Partial summing would be misleading."
            )
        else:
            total_bill_inr = None
            bill_notes = "No verified bill estimates available for this horizon (tariff unverified, unsupported, or inactive for forecast dates)."

        summary = MonthlyForecastSummary(
            total_forecast_kwh=tot_kwh,
            average_monthly_kwh=avg_kwh,
            total_estimated_bill_inr=total_bill_inr,
            months_with_bill_estimates=months_with_estimates,
            bill_estimate_notes=bill_notes,
            currency="INR",
        )

        return MonthlyForecastResponse(
            forecast_id=f"mfc_{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            history_length_months=n_obs,
            last_observed_period=last_rec.period_label or f"{last_rec.billing_period_start} to {last_rec.billing_period_end}",
            last_observed_end_date=last_rec.billing_period_end,
            selected_method=selected_method,
            method_description=method_description,
            confidence_rating=confidence_rating,
            is_stale_history=is_stale,
            warnings=warnings,
            limitations=limitations,
            forecast_steps=forecast_steps,
            backtest_metrics=backtest_metrics,
            summary=summary,
        )


# Global singleton monthly forecast service
monthly_forecast_service = MonthlyForecastService()
