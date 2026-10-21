"""
Household Energy Coach Service Layer.

Provides:
- Data-grounded household energy overviews.
- Deterministic, evidence-based energy insights citing exact historical records.
- Appliance What-If Simulator with verified tariff calculations.
- Actionable, transparent recommendations without synthetic or fabricated data.
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from backend.app.schemas.bills import MonthlyConsumptionRecord
from backend.app.schemas.coach import (
    ApplianceCategory,
    ApplianceScenarioRequest,
    ApplianceScenarioResponse,
    ApplianceSimulationInput,
    CoachInsight,
    CoachRecommendation,
    DEFAULT_APPLIANCE_PRESETS,
    EnergyCoachOverviewResponse,
    HouseholdScenarioComparisonRequest,
    HouseholdScenarioComparisonResponse,
    ScenarioComparisonStep,
)
from backend.app.services.monthly_record_service import MonthlyRecordService, monthly_record_service
from backend.app.services.msedcl_tariff_service import msedcl_calculator
from backend.app.services.tariff_service import tariff_registry


class CoachService:
    """Service generating personalized coaching insights and appliance what-if simulations."""

    def __init__(self, monthly_record_service: Optional[MonthlyRecordService] = None):
        self._monthly_record_service = monthly_record_service

    @property
    def monthly_record_service(self) -> MonthlyRecordService:
        if self._monthly_record_service is not None:
            return self._monthly_record_service
        return monthly_record_service

    def compute_appliance_kwh(self, appliance: ApplianceSimulationInput) -> float:
        """
        Compute monthly energy consumption using standard formula:
        kWh = (power in watts * hours per day * days per month * quantity) / 1000.0
        """
        raw_kwh = (appliance.power_watts * appliance.hours_per_day * appliance.days_per_month * appliance.quantity) / 1000.0
        return round(float(raw_kwh), 2)

    def simulate_appliance_scenario(
        self, request: ApplianceScenarioRequest
    ) -> ApplianceScenarioResponse:
        """
        Compare baseline appliance usage against a modified scenario.

        Computes:
        - Baseline kWh & Modified kWh
        - Energy difference (delta_kwh)
        - Cost difference in INR strictly when an official verified tariff schedule applies
        - Safety warnings and duty cycle operational notes
        """
        baseline_kwh = self.compute_appliance_kwh(request.baseline)
        modified_kwh = self.compute_appliance_kwh(request.modified)
        delta_kwh = round(baseline_kwh - modified_kwh, 2)

        disclosures: List[str] = [
            f"Formula: kWh = ({request.baseline.power_watts:.0f} W × {request.baseline.hours_per_day:.1f} h/day × {request.baseline.days_per_month} days × {request.baseline.quantity}) / 1000 = {baseline_kwh:.2f} kWh/month."
        ]
        appliance_notes: List[str] = []

        # Add preset operational behavior context if available
        cat_key = request.baseline.category.value
        preset = DEFAULT_APPLIANCE_PRESETS.get(cat_key)
        if preset and preset.get("notes"):
            appliance_notes.append(preset["notes"])

        # Safety caution for refrigeration
        if (
            request.baseline.category == ApplianceCategory.REFRIGERATOR
            or request.modified.category == ApplianceCategory.REFRIGERATOR
        ):
            if request.modified.hours_per_day < 24.0:
                appliance_notes.append(
                    "SAFETY CAUTION: Refrigerators must remain powered 24 hours a day to prevent bacterial growth and food spoilage. "
                    "Thermostat optimization (e.g. maintaining 3°C to 4°C) is recommended rather than disconnecting power."
                )

        disclosures.append(
            f"Appliance simulation ('{request.modified.name}') is a hypothetical estimate based on nameplate rated power "
            f"({request.modified.power_watts}W) and user-entered hours, not sub-metered from household lines."
        )

        # Check tariff verification status
        provider_id = (request.provider_id or "").strip().lower()
        is_verified = False
        cost_baseline: Optional[float] = None
        cost_modified: Optional[float] = None
        savings_inr: Optional[float] = None

        if provider_id:
            # Check specific schedule date if as_of_date is provided, otherwise check general support
            if request.as_of_date:
                sched = tariff_registry.get_schedule(provider_id=provider_id, as_of_date=request.as_of_date)
                is_verified = bool(sched and sched.is_verified)
            else:
                is_verified = tariff_registry.is_provider_supported(provider_id)

            # Currently MSEDCL is verified under MERC Case No. 226 of 2022
            if is_verified and provider_id == "msedcl":
                try:
                    phase = request.phase_type or "single_phase"
                    base_monthly = request.current_monthly_total_kwh or 0.0

                    if base_monthly > 0:
                        # Incremental marginal calculation respecting progressive tariff slabs
                        res_with_base = msedcl_calculator.calculate(
                            consumption_kwh=base_monthly + baseline_kwh, phase_type=phase
                        )
                        res_with_mod = msedcl_calculator.calculate(
                            consumption_kwh=base_monthly + modified_kwh, phase_type=phase
                        )
                        res_base_only = msedcl_calculator.calculate(
                            consumption_kwh=base_monthly, phase_type=phase
                        )

                        cost_baseline = round(
                            (res_with_base.estimated_bill_amount or 0.0) - (res_base_only.estimated_bill_amount or 0.0),
                            2,
                        )
                        cost_modified = round(
                            (res_with_mod.estimated_bill_amount or 0.0) - (res_base_only.estimated_bill_amount or 0.0),
                            2,
                        )
                        savings_inr = round(cost_baseline - cost_modified, 2)

                        disclosures.append(
                            f"Cost impact evaluated using MSEDCL LT-I(B) Residential tariff (MERC Case No. 226 of 2022). "
                            f"Appliance usage was evaluated on top of your existing {base_monthly:.1f} kWh baseline to account for progressive energy slabs."
                        )
                    else:
                        # Direct standalone slab cost
                        res_base = msedcl_calculator.calculate(consumption_kwh=baseline_kwh, phase_type=phase)
                        res_mod = msedcl_calculator.calculate(consumption_kwh=modified_kwh, phase_type=phase)

                        cost_baseline = res_base.estimated_bill_amount
                        cost_modified = res_mod.estimated_bill_amount
                        if cost_baseline is not None and cost_modified is not None:
                            savings_inr = round(cost_baseline - cost_modified, 2)

                        disclosures.append(
                            "Cost impact evaluated using MSEDCL LT-I(B) Residential baseline rates (MERC Case No. 226 of 2022). "
                            "Actual bills include fixed charges and duties applicable to the overall household bill."
                        )
                except Exception as exc:
                    is_verified = False
                    disclosures.append(f"Tariff calculation encountered an error: {str(exc)}. Monetary savings withheld.")
            elif not is_verified and provider_id == "msedcl":
                disclosures.append(
                    "Official MSEDCL tariff schedule has expired or is not active for the specified date. "
                    "Monetary bill savings are withheld to prevent using outdated rates. Physical energy differences (kWh) are preserved."
                )
            else:
                disclosures.append(
                    f"Official tariff schedules for provider '{provider_id}' are currently unverified or pending regulatory audit. "
                    "Monetary bill savings are withheld to prevent fabricated estimates. Physical energy differences (kWh) are preserved."
                )
        else:
            disclosures.append(
                "No electricity provider selected. Monetary cost difference cannot be calculated without an applicable tariff schedule. "
                "Physical energy differences (kWh) are preserved."
            )

        return ApplianceScenarioResponse(
            baseline_kwh=baseline_kwh,
            modified_kwh=modified_kwh,
            delta_kwh=delta_kwh,
            is_hypothetical_estimate=True,
            is_tariff_verified=is_verified,
            estimated_baseline_cost_inr=cost_baseline,
            estimated_modified_cost_inr=cost_modified,
            estimated_savings_inr=savings_inr,
            disclosures=disclosures,
            appliance_notes=appliance_notes,
        )

    def get_household_overview(self, user_id: str) -> EnergyCoachOverviewResponse:
        """
        Compile household energy overview, data-grounded insights, and recommendations
        strictly from confirmed historical bill records and saved monthly forecasts.
        """
        # 1. Fetch user's saved monthly forecast runs
        _, summaries = self.monthly_record_service.list_monthly_records(owner_user_id=user_id, limit=10)

        if not summaries:
            return EnergyCoachOverviewResponse(
                has_data=False,
                data_freshness_status="no_records",
                warnings=[
                    "No confirmed monthly electricity bills or forecasts found for this household. "
                    "Upload a bill photo or enter monthly readings under 'New Forecast' to activate personalized coaching."
                ],
                insights=[
                    CoachInsight(
                        id="welcome_setup",
                        category="data_quality",
                        title="Welcome to Energy Coach",
                        description="Energy Coach requires at least one confirmed monthly electricity bill to analyze baseline consumption and identify opportunities.",
                        evidence="0 monthly records found in account database.",
                        severity="info",
                    )
                ],
                recommendations=[
                    CoachRecommendation(
                        id="import_first_bill",
                        title="Upload or Enter Your Latest Electricity Bill",
                        description="Go to 'New Forecast' to scan an MSEDCL/utility bill or input meter readings. Once confirmed, Energy Coach will track your trends and seasonal forecasts.",
                        action_type="data_setup",
                        rationale="Personalized recommendations require actual household consumption history.",
                        is_hypothetical=False,
                    )
                ],
            )

        # 2. Retrieve most recent run detail
        latest_summary = summaries[0]
        latest_run = self.monthly_record_service.get_monthly_record(run_id=latest_summary.run_id)

        if not latest_run or not latest_run.input_records:
            return EnergyCoachOverviewResponse(
                has_data=False,
                data_freshness_status="no_records",
                warnings=["Saved forecast run does not contain confirmed input records."],
            )

        # Sort historical records chronologically by billing period start
        hist_records = sorted(latest_run.input_records, key=lambda r: r.billing_period_start)

        latest_rec = hist_records[-1]
        latest_kwh = latest_rec.consumption_kwh
        latest_label = latest_rec.period_label or latest_rec.billing_period_start.strftime("%b %Y")
        latest_end_date = latest_rec.billing_period_end
        latest_amount = latest_rec.bill_amount

        prev_rec = hist_records[-2] if len(hist_records) >= 2 else None
        prev_kwh = prev_rec.consumption_kwh if prev_rec else None
        prev_label = (prev_rec.period_label or prev_rec.billing_period_start.strftime("%b %Y")) if prev_rec else None

        change_kwh: Optional[float] = None
        change_pct: Optional[float] = None
        if prev_kwh is not None:
            change_kwh = round(latest_kwh - prev_kwh, 2)
            if prev_kwh > 0.0:
                change_pct = round((change_kwh / prev_kwh) * 100.0, 1)

        # Assess data freshness
        today = date.today()
        days_since_end = (today - latest_end_date).days if latest_end_date else 999
        warnings: List[str] = []

        if days_since_end > 75 or latest_run.is_stale_history:
            freshness = "stale"
            warnings.append(
                f"Latest confirmed billing period ended on {latest_end_date} ({days_since_end} days ago). "
                "Energy consumption patterns and seasonal appliance demand may have changed since this bill."
            )
        elif len(hist_records) < 3:
            freshness = "insufficient_history"
            warnings.append(
                f"Limited historical depth ({len(hist_records)} month(s) recorded). "
                "Baseline patterns and seasonal comparisons become significantly more reliable with 6 to 12 months of confirmed bills."
            )
        else:
            freshness = "current"

        # Check tariff verification
        prov_id = latest_run.provider_id or ""
        is_verified_tariff = tariff_registry.is_provider_supported(prov_id)
        prov_meta = tariff_registry.get_provider_metadata(prov_id)
        prov_name = prov_meta.provider_name if prov_meta else prov_id

        # 3. Deterministic Evidence-Based Insights
        insights: List[CoachInsight] = []

        # Insight A: Trajectory / Trend
        if change_kwh is not None and change_pct is not None:
            if abs(change_pct) < 5.0:
                insights.append(
                    CoachInsight(
                        id="stable_trend",
                        category="trend",
                        title="Stable Month-over-Month Consumption",
                        description=f"Your electricity consumption changed by only {change_kwh:+.1f} kWh ({change_pct:+.1f}%) from {prev_label} to {latest_label}, indicating consistent household baseline usage.",
                        evidence=f"{prev_label}: {prev_kwh:.1f} kWh vs {latest_label}: {latest_kwh:.1f} kWh (Difference: {change_kwh:+.1f} kWh).",
                        severity="neutral",
                    )
                )
            elif change_pct >= 5.0:
                insights.append(
                    CoachInsight(
                        id="increasing_trend",
                        category="trend",
                        title=f"Consumption Increased by {abs(change_pct):.1f}%",
                        description=f"Electricity consumption rose from {prev_kwh:.1f} kWh in {prev_label} to {latest_kwh:.1f} kWh in {latest_label} (+{change_kwh:.1f} kWh). Common contributors include seasonal temperature shifts (AC/heating), guest occupancy, or extended appliance operating hours.",
                        evidence=f"Recorded bill values: {prev_label} = {prev_kwh:.1f} kWh; {latest_label} = {latest_kwh:.1f} kWh (Net change: +{change_kwh:.1f} kWh).",
                        severity="warning",
                    )
                )
            else:
                insights.append(
                    CoachInsight(
                        id="decreasing_trend",
                        category="trend",
                        title=f"Consumption Decreased by {abs(change_pct):.1f}%",
                        description=f"Electricity consumption decreased from {prev_kwh:.1f} kWh in {prev_label} to {latest_kwh:.1f} kWh in {latest_label} ({change_kwh:.1f} kWh), reflecting reduced overall demand or milder weather.",
                        evidence=f"Recorded bill values: {prev_label} = {prev_kwh:.1f} kWh; {latest_label} = {latest_kwh:.1f} kWh (Net change: {change_kwh:.1f} kWh).",
                        severity="positive",
                    )
                )
        else:
            insights.append(
                CoachInsight(
                    id="baseline_single_month",
                    category="baseline",
                    title="Single Bill Period On Record",
                    description=f"Currently, only one confirmed billing period ({latest_label}: {latest_kwh:.1f} kWh) is available. Energy Coach will automatically calculate trends once a second period is confirmed.",
                    evidence=f"Confirmed record: {latest_label} with {latest_kwh:.1f} kWh.",
                    severity="info",
                )
            )

        # Insight B: Forecast Comparison
        if latest_run.forecast_steps:
            forecast_avg = latest_run.average_monthly_kwh
            if forecast_avg > latest_kwh * 1.10:
                insights.append(
                    CoachInsight(
                        id="forecast_higher",
                        category="forecast",
                        title="Projected Higher Seasonal Consumption Ahead",
                        description=f"The {latest_run.selected_method.value} forecast model projects an average of {forecast_avg:.1f} kWh/month across {latest_run.forecast_period_range}, higher than your latest month ({latest_kwh:.1f} kWh). Plan for seasonal weather demand.",
                        evidence=f"Forecast Horizon: {latest_run.forecast_period_range} (Average {forecast_avg:.1f} kWh/month vs Latest {latest_kwh:.1f} kWh; Confidence: {latest_run.confidence_rating.value}).",
                        severity="info",
                    )
                )
            elif forecast_avg < latest_kwh * 0.90:
                insights.append(
                    CoachInsight(
                        id="forecast_lower",
                        category="forecast",
                        title="Projected Moderate Consumption Ahead",
                        description=f"The {latest_run.selected_method.value} forecast model projects an average of {forecast_avg:.1f} kWh/month across {latest_run.forecast_period_range}, which is lower than your latest month ({latest_kwh:.1f} kWh).",
                        evidence=f"Forecast Horizon: {latest_run.forecast_period_range} (Average {forecast_avg:.1f} kWh/month vs Latest {latest_kwh:.1f} kWh; Confidence: {latest_run.confidence_rating.value}).",
                        severity="positive",
                    )
                )
            else:
                insights.append(
                    CoachInsight(
                        id="forecast_steady",
                        category="forecast",
                        title="Projected Steady Demand Across Horizon",
                        description=f"The {latest_run.selected_method.value} forecast projects an average of {forecast_avg:.1f} kWh/month across {latest_run.forecast_period_range}, closely matching your latest consumption ({latest_kwh:.1f} kWh).",
                        evidence=f"Forecast Horizon: {latest_run.forecast_period_range} (Projected Average {forecast_avg:.1f} kWh/month; Confidence: {latest_run.confidence_rating.value}).",
                        severity="neutral",
                    )
                )

        # Insight C: Historical Depth & Reliability
        if len(hist_records) >= 6:
            insights.append(
                CoachInsight(
                    id="robust_depth",
                    category="data_quality",
                    title="Comprehensive Historical Baseline",
                    description=f"With {len(hist_records)} months of confirmed bills, the household baseline captures seasonal demand variations and provides reliable comparison points.",
                    evidence=f"{len(hist_records)} billing periods spanning from {hist_records[0].billing_period_start} to {hist_records[-1].billing_period_end}.",
                    severity="positive",
                )
            )

        # 4. Actionable, Transparent Recommendations
        recommendations: List[CoachRecommendation] = []

        # Rec 1: Cooling / Air Conditioning (if applicable)
        if latest_kwh > 150.0:
            ac_savings_kwh = 45.0  # Approx savings for 1 hr less AC or 24°C vs 20°C
            ac_savings_inr = round(ac_savings_kwh * 9.64, 2) if is_verified_tariff and prov_id == "msedcl" else None
            recommendations.append(
                CoachRecommendation(
                    id="ac_thermostat_tuning",
                    title="Optimize Air Conditioner Thermostat to 24°C",
                    description="Setting room air conditioners to 24°C instead of 20°C reduces compressor workload by approximately 6% per degree without compromising thermal comfort.",
                    action_type="appliance",
                    rationale=f"Your monthly consumption is {latest_kwh:.1f} kWh. In summer and humid months, cooling typically accounts for 40-60% of residential electricity use.",
                    is_hypothetical=True,
                    estimated_monthly_kwh_reduction=ac_savings_kwh,
                    estimated_monthly_savings_inr=ac_savings_inr,
                    savings_disclaimer="Estimated assuming one 1.5-ton AC operated 8 hours/day adjusted by 2°C in Slab 2 tariff rates.",
                )
            )

        # Rec 2: Telescopic Slab Awareness (MSEDCL specific)
        if is_verified_tariff and prov_id == "msedcl" and latest_kwh > 100.0:
            recommendations.append(
                CoachRecommendation(
                    id="slab_boundary_awareness",
                    title="Monitor Higher Consumption Slabs (MSEDCL LT-I)",
                    description="Under Maharashtra MERC LT-I(B) rates, electricity above 100 kWh/month costs ₹9.64/kWh, and above 300 kWh costs ₹13.61/kWh (compared to ₹4.41/kWh in Slab 1). Keeping monthly use below slab thresholds prevents exponential bill jumps.",
                    action_type="behavioral",
                    rationale=f"Your latest usage ({latest_kwh:.1f} kWh) falls into progressive slab tier(s).",
                    is_hypothetical=False,
                    savings_disclaimer="Marginal cost per unit jumps from ₹4.41 to ₹9.64 beyond 100 kWh.",
                )
            )

        # Rec 3: Water Heating / Geyser
        recommendations.append(
            CoachRecommendation(
                id="water_heater_timer",
                title="Limit Water Geyser Pre-Heating to 20-30 Minutes",
                description="Storage geysers (2000W) consume ~1 kWh per 30 minutes of heating. Switching off immediately after water reaches bathing temperature avoids standby jacket heat loss.",
                action_type="appliance",
                rationale="Resistive geysers draw full rated wattage continuously when active.",
                is_hypothetical=True,
                estimated_monthly_kwh_reduction=30.0,
                estimated_monthly_savings_inr=round(30.0 * 9.64, 2) if is_verified_tariff and prov_id == "msedcl" else None,
                savings_disclaimer="Estimated based on reducing active heating time by 30 minutes daily for 30 days.",
            )
        )

        # Rec 4: BLDC Ceiling Fans
        recommendations.append(
            CoachRecommendation(
                id="bldc_fan_upgrade",
                title="Consider Energy-Efficient BLDC Ceiling Fans",
                description="Upgrading conventional 75W ceiling fans to 28-35W Brushless DC (BLDC) fans cuts continuous fan consumption by over 50% with identical air delivery.",
                action_type="appliance",
                rationale="Ceiling fans run for long durations in tropical Indian climates, yielding high cumulative energy savings.",
                is_hypothetical=True,
                estimated_monthly_kwh_reduction=15.0,
                estimated_monthly_savings_inr=round(15.0 * 9.64, 2) if is_verified_tariff and prov_id == "msedcl" else None,
                savings_disclaimer="Estimated for 1 fan operating 12 hours/day replacing a 75W model with a 32W model.",
            )
        )

        # Rec 5: Data depth recommendation if history is sparse
        if len(hist_records) < 6:
            recommendations.append(
                CoachRecommendation(
                    id="add_more_bills",
                    title="Upload Additional Past Electricity Bills",
                    description="Adding 6 to 12 months of historical bills enables the forecasting service to detect seasonal summer/winter patterns and calibrate baseline uncertainty intervals.",
                    action_type="data_setup",
                    rationale=f"Currently {len(hist_records)} month(s) are recorded. Seasonal forecasting accuracy improves with 12-month depth.",
                    is_hypothetical=False,
                )
            )

        return EnergyCoachOverviewResponse(
            has_data=True,
            latest_consumption_kwh=latest_kwh,
            latest_period_label=latest_label,
            latest_billing_end_date=latest_end_date,
            latest_bill_amount_inr=latest_amount,
            previous_consumption_kwh=prev_kwh,
            previous_period_label=prev_label,
            consumption_change_kwh=change_kwh,
            consumption_change_percent=change_pct,
            historical_records=hist_records,
            latest_forecast_run_id=latest_run.run_id,
            latest_forecast_created_at=latest_run.created_at,
            latest_forecast_horizon_months=latest_run.horizon_months,
            latest_forecast_period_range=latest_run.forecast_period_range,
            latest_forecast_total_kwh=latest_run.total_forecast_kwh,
            latest_forecast_average_kwh=latest_run.average_monthly_kwh,
            latest_forecast_confidence=latest_run.confidence_rating.value,
            forecast_steps=latest_run.forecast_steps,
            provider_id=latest_run.provider_id,
            provider_name=prov_name,
            is_tariff_verified=is_verified_tariff,
            data_freshness_status=freshness,
            warnings=warnings,
            insights=insights,
            recommendations=recommendations,
        )

    def compare_household_scenario(
        self, request: HouseholdScenarioComparisonRequest
    ) -> HouseholdScenarioComparisonResponse:
        """
        Evaluate a multi-month household forecast series against an appliance what-if scenario.

        Computes:
        - Appliance simulation delta_kwh as an isolated hypothetical estimate.
        - Physical plausibility check: validates whether appliance baseline and reduction
          can be defensibly reconciled with the household baseline forecast.
        - If physically inconsistent (e.g. appliance baseline or reduction exceeds total household
          forecast), WITHHOLDS adjusted household kWh, delta, and monetary savings rather than
          clamping to zero.
        - Month-by-month adjusted forecast: adjusted_kwh = baseline_kwh - delta_kwh (only when supported).
        - Baseline bill and adjusted bill for each step using verified and active tariff calculator.
        - Suppresses monetary bill amounts and savings when provider tariff is unverified or expired.
        - Generates context-grounded recommendations and transparent disclaimers.
        """
        # 1. Run isolated appliance simulation (nameplate hypothetical estimate)
        appliance_res = self.simulate_appliance_scenario(request.appliance_scenario)
        app_baseline_kwh = appliance_res.baseline_kwh
        app_modified_kwh = appliance_res.modified_kwh
        delta_kwh = appliance_res.delta_kwh

        provider_id = (request.provider_id or "").strip().lower()
        phase = request.phase_type or "single_phase"
        sanctioned = request.sanctioned_load_kw

        # 2. Validate Physical Consistency between Appliance and Household Baseline
        # An appliance reduction cannot be subtracted from a household forecast if:
        # a) Assumed appliance baseline exceeds total household baseline (e.g. 348 kWh AC in 2 kWh house)
        # b) Appliance reduction exceeds or equals total household baseline (e.g. saving 43.5 kWh from 2 kWh)
        # c) Subtracting delta produces non-positive household consumption (<= 0 kWh)
        unsupported_steps: List[str] = []
        min_household_base = min(request.baseline_forecast_kwh) if request.baseline_forecast_kwh else 0.0

        for idx, base_kwh in enumerate(request.baseline_forecast_kwh, start=1):
            lbl = request.month_labels[idx - 1] if idx - 1 < len(request.month_labels) else f"Month {idx}"
            if delta_kwh > 0:
                if app_baseline_kwh > base_kwh:
                    unsupported_steps.append(
                        f"In {lbl}, assumed appliance baseline ({app_baseline_kwh:.1f} kWh) exceeds total household forecast ({base_kwh:.1f} kWh)."
                    )
                elif delta_kwh >= base_kwh or (base_kwh - delta_kwh) <= 0.0:
                    unsupported_steps.append(
                        f"In {lbl}, appliance reduction ({delta_kwh:.1f} kWh) exceeds or equals total household forecast ({base_kwh:.1f} kWh)."
                    )
            elif delta_kwh < 0:
                if app_baseline_kwh > base_kwh and request.appliance_scenario.baseline.hours_per_day > 0:
                    unsupported_steps.append(
                        f"In {lbl}, existing appliance baseline ({app_baseline_kwh:.1f} kWh) exceeds total household forecast ({base_kwh:.1f} kWh)."
                    )

        is_adjustment_supported = len(unsupported_steps) == 0
        household_adjustment_reason: Optional[str] = None
        if not is_adjustment_supported:
            household_adjustment_reason = (
                f"Household scenario adjustment withheld: The hypothetical appliance baseline ({app_baseline_kwh:.1f} kWh/month) "
                f"or reduction ({delta_kwh:.1f} kWh/month) exceeds the total household baseline forecast "
                f"({min_household_base:.1f} kWh/month). A household's consumption cannot be reduced "
                "below physical reality, and unmeasured appliance ratings cannot be reconciled with this household's confirmed baseline. "
                f"The hypothetical appliance-only reduction of {delta_kwh:.1f} kWh/month is preserved above, but adjusted household forecast "
                "and bill savings are withheld."
            )

        # 3. Date resolution helper for tariff verification
        def _resolve_step_date(step_idx: int) -> Optional[date]:
            if request.period_starts and step_idx - 1 < len(request.period_starts):
                return request.period_starts[step_idx - 1]
            if step_idx - 1 < len(request.month_labels):
                lbl = request.month_labels[step_idx - 1]
                import re
                months = {
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
                }
                match = re.search(r'([A-Za-z]{3,9})\s*(\d{4}|\d{2})', lbl)
                if match:
                    m_str = match.group(1)[:3].lower()
                    y_str = match.group(2)
                    year = int(y_str) if len(y_str) == 4 else 2000 + int(y_str)
                    m_int = months.get(m_str)
                    if m_int:
                        return date(year, m_int, 1)
            return None

        steps: List[ScenarioComparisonStep] = []
        total_baseline = 0.0
        total_adjusted: Optional[float] = 0.0 if is_adjustment_supported else None
        total_delta: Optional[float] = 0.0 if is_adjustment_supported else None
        total_savings: Optional[float] = 0.0 if is_adjustment_supported else None
        all_tariffs_verified = True
        has_expired_tariff = False

        for idx, base_kwh in enumerate(request.baseline_forecast_kwh, start=1):
            label = request.month_labels[idx - 1] if idx - 1 < len(request.month_labels) else f"Month {idx}"
            total_baseline += base_kwh
            step_date = _resolve_step_date(idx)

            # Check tariff schedule validity for this specific step date
            step_status_info = tariff_registry.get_tariff_status(provider_id=provider_id, as_of_date=step_date)
            step_tariff_verified = step_status_info["is_applicable_schedule_verified"]
            step_tariff_status = step_status_info["status"]
            step_tariff_label = step_status_info["status_label"]
            if not step_tariff_verified:
                all_tariffs_verified = False
                if step_status_info["status"] == "expired_schedule":
                    has_expired_tariff = True

            base_bill: Optional[float] = None
            if step_tariff_verified and provider_id == "msedcl":
                try:
                    res_base = msedcl_calculator.calculate(
                        consumption_kwh=base_kwh,
                        phase_type=phase,
                        sanctioned_load_kw=sanctioned,
                        billing_period_start=step_date,
                    )
                    base_bill = res_base.estimated_bill_amount
                except Exception:
                    base_bill = None

            if not is_adjustment_supported:
                # WITHHELD: Do NOT clamp to 0 or invent fake savings
                steps.append(
                    ScenarioComparisonStep(
                        month_index=idx,
                        month_label=label,
                        baseline_kwh=round(base_kwh, 2),
                        adjusted_kwh=None,
                        delta_kwh=None,
                        baseline_bill_amount=base_bill,
                        adjusted_bill_amount=None,
                        savings_amount=None,
                        is_adjustment_supported=False,
                        unsupported_reason=(
                            f"Assumed appliance baseline ({app_baseline_kwh:.1f} kWh) or reduction ({delta_kwh:.1f} kWh) "
                            f"exceeds household baseline forecast ({base_kwh:.1f} kWh)."
                        ),
                        tariff_status=step_tariff_status,
                        tariff_status_label=step_tariff_label,
                    )
                )
            else:
                # Supported: Physically plausible reduction
                adj_kwh = round(base_kwh - delta_kwh, 2)
                step_delta = round(base_kwh - adj_kwh, 2)
                if total_adjusted is not None:
                    total_adjusted += adj_kwh
                if total_delta is not None:
                    total_delta += step_delta

                adj_bill: Optional[float] = None
                step_saving: Optional[float] = None

                if step_tariff_verified and provider_id == "msedcl":
                    try:
                        res_adj = msedcl_calculator.calculate(
                            consumption_kwh=adj_kwh,
                            phase_type=phase,
                            sanctioned_load_kw=sanctioned,
                            billing_period_start=step_date,
                        )
                        adj_bill = res_adj.estimated_bill_amount
                        if base_bill is not None and adj_bill is not None:
                            step_saving = round(base_bill - adj_bill, 2)
                            if total_savings is not None:
                                total_savings += step_saving
                    except Exception:
                        step_saving = None

                if not step_tariff_verified:
                    step_saving = None

                steps.append(
                    ScenarioComparisonStep(
                        month_index=idx,
                        month_label=label,
                        baseline_kwh=round(base_kwh, 2),
                        adjusted_kwh=adj_kwh,
                        delta_kwh=step_delta,
                        baseline_bill_amount=base_bill,
                        adjusted_bill_amount=adj_bill,
                        savings_amount=step_saving,
                        is_adjustment_supported=True,
                        unsupported_reason=None,
                        tariff_status=step_tariff_status,
                        tariff_status_label=step_tariff_label,
                    )
                )

        if not is_adjustment_supported:
            total_adjusted = None
            total_delta = None
            total_savings = None
        else:
            if not all_tariffs_verified or total_savings is None:
                total_savings = None
            else:
                total_savings = round(total_savings, 2)
            if total_adjusted is not None:
                total_adjusted = round(total_adjusted, 2)
            if total_delta is not None:
                total_delta = round(total_delta, 2)

        # 4. Disclosures & Recommendations
        disclosures: List[str] = [
            f"Appliance simulation ('{request.appliance_scenario.modified.name}') is a hypothetical estimate based on nameplate "
            f"rated power ({request.appliance_scenario.modified.power_watts}W) and user-entered hours, not sub-metered from household lines."
        ]

        if not is_adjustment_supported:
            disclosures.append(f"Physical consistency restriction: {household_adjustment_reason}")

        if all_tariffs_verified and provider_id == "msedcl":
            disclosures.append(
                "Bills evaluated using verified MSEDCL LT-I(B) Residential tariff (MERC Case No. 226 of 2022). "
                "Fixed charges, wheeling charges (₹1.17/kWh), and 16% Electricity Duty are included."
            )
        elif has_expired_tariff and provider_id == "msedcl":
            disclosures.append(
                "Official MSEDCL LT-I(B) tariff schedule expired on 2025-03-31 and is not verified for the forecast period. "
                "Monetary bill amounts and savings are withheld to prevent using outdated rates. Physical energy reductions (kWh) are preserved."
            )
        else:
            if provider_id:
                disclosures.append(
                    f"Official tariff schedules for provider '{provider_id}' are currently unverified. "
                    "Monetary bill amounts and savings are withheld to prevent fabricated estimates. "
                    "Physical energy reductions (kWh) are preserved."
                )
            else:
                disclosures.append(
                    "No electricity provider selected. Monetary savings withheld. Physical energy reductions (kWh) are preserved."
                )

        recommendations: List[CoachRecommendation] = []
        app_name = request.appliance_scenario.modified.name
        cat = request.appliance_scenario.modified.category

        if not is_adjustment_supported:
            recommendations.append(
                CoachRecommendation(
                    id=f"inconsistent_{cat.value}",
                    title=f"Appliance Demand Exceeds Household Forecast ({app_name})",
                    description=(
                        f"The modeled {app_name} baseline usage ({app_baseline_kwh:.1f} kWh/month) exceeds your total household "
                        f"forecast ({min_household_base:.1f} kWh/month). If this appliance is operating at the entered hours, your confirmed "
                        "meter readings may reflect an unoccupied period, or the appliance runs with a lower duty cycle than entered."
                    ),
                    action_type="data_setup",
                    rationale="Physical conservation of energy: appliance demand cannot exceed total household consumption.",
                    is_hypothetical=True,
                )
            )
        else:
            if delta_kwh > 0:
                if all_tariffs_verified and total_savings is not None and total_savings > 0:
                    rec_title = f"Projected {len(steps)}-Month Saving: ₹{total_savings:.0f} via {app_name} Optimization"
                    rec_desc = (
                        f"Adopting the proposed '{app_name}' schedule reduces energy use by {delta_kwh:.1f} kWh each month, "
                        f"amounting to {total_delta:.1f} kWh total across the {len(steps)}-month forecast period. "
                        f"Under verified MSEDCL LT-I(B) rates, this yields ~₹{total_savings:.2f} in cumulative bill savings."
                    )
                else:
                    rec_title = f"Projected {len(steps)}-Month Saving: {total_delta:.1f} kWh via {app_name} Optimization"
                    rec_desc = (
                        f"Adopting the proposed '{app_name}' schedule reduces energy use by {delta_kwh:.1f} kWh each month, "
                        f"amounting to {total_delta:.1f} kWh total across the {len(steps)}-month forecast period."
                    )
                recommendations.append(
                    CoachRecommendation(
                        id=f"scenario_{cat.value}",
                        title=rec_title,
                        description=rec_desc,
                        action_type="efficiency_habit" if "AC" in app_name or "Geyser" in app_name else "appliance_upgrade",
                        estimated_monthly_savings_kwh=delta_kwh,
                        estimated_monthly_savings_inr=steps[0].savings_amount if steps and steps[0].savings_amount is not None else None,
                        rationale=f"Hypothetical simulation based on {request.appliance_scenario.modified.hours_per_day}h/day vs {request.appliance_scenario.baseline.hours_per_day}h/day.",
                        is_hypothetical=True,
                    )
                )
            elif delta_kwh < 0:
                recommendations.append(
                    CoachRecommendation(
                        id=f"scenario_increase_{cat.value}",
                        title=f"Consumption Increase: +{abs(delta_kwh):.1f} kWh/month for {app_name}",
                        description=f"Increasing {app_name} usage adds approximately {abs(total_delta or 0.0):.1f} kWh across the {len(steps)}-month forecast horizon.",
                        action_type="appliance_upgrade",
                        estimated_monthly_savings_kwh=delta_kwh,
                        estimated_monthly_savings_inr=steps[0].savings_amount if steps and steps[0].savings_amount is not None else None,
                        rationale="Hypothetical scenario with increased operating hours or appliance quantity.",
                        is_hypothetical=True,
                    )
                )

        # Duty cycle / safety warnings from appliance results
        for note in appliance_res.appliance_notes:
            if "SAFETY CAUTION" in note:
                recommendations.append(
                    CoachRecommendation(
                        id="safety_refrigeration",
                        title="Essential Safety Rule: Maintain Uninterrupted Refrigeration",
                        description=note,
                        action_type="data_setup",
                        rationale="Food safety requires continuous temperature regulation below 4°C.",
                        is_hypothetical=False,
                    )
                )

        if all_tariffs_verified and is_adjustment_supported:
            overall_t_status = "verified"
            overall_t_label = "Verified"
        elif has_expired_tariff:
            overall_t_status = "expired_schedule"
            overall_t_label = "Expired schedule"
        elif provider_id:
            overall_t_status = "unverified_schedule"
            overall_t_label = "Current tariff unverified"
        else:
            overall_t_status = "no_provider"
            overall_t_label = "No provider selected"

        return HouseholdScenarioComparisonResponse(
            steps=steps,
            total_baseline_kwh=round(total_baseline, 2),
            total_adjusted_kwh=total_adjusted,
            total_delta_kwh=total_delta,
            total_savings_inr=total_savings,
            is_verified_tariff=all_tariffs_verified and is_adjustment_supported,
            tariff_status=overall_t_status,
            tariff_status_label=overall_t_label,
            is_household_adjustment_supported=is_adjustment_supported,
            household_adjustment_reason=household_adjustment_reason,
            appliance_result=appliance_res,
            recommendations=recommendations,
            disclosures=disclosures,
        )


coach_service = CoachService()

