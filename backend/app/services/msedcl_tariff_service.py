"""
MSEDCL (Mahavitaran) LT-I(B) Residential Tariff Calculator.

Verified Primary Authority:
- Maharashtra Electricity Regulatory Commission (MERC) Mid-Term Review (MTR) Order
  in Case No. 226 of 2022 dated 31 March 2023.
- MSEDCL Commercial Circular No. 341 dated 1 April 2023.
- Schedule: LT-I(B) Residential (other than BPL), effective from 1 April 2024 (FY 2024-25 baseline).

Verified Rate Schedule:
1. Energy Charges (Telescopic Slabs per kWh):
   - Slab 1 (0 to 100 kWh): ₹4.41 / kWh
   - Slab 2 (101 to 300 kWh): ₹9.64 / kWh
   - Slab 3 (301 to 500 kWh): ₹13.61 / kWh
   - Slab 4 (Above 500 kWh): ₹15.57 / kWh
2. Wheeling Charge: Uniform flat ₹1.17 / kWh on all consumed units.
3. Fixed Charges (per connection per month):
   - Single-Phase: ₹128.00 / month
   - Three-Phase: ₹385.00 / month
   - Loads exceeding 10 kW: additional ₹200.00 per 10 kW or part thereof.
4. Electricity Duty (ED): 16.0% on taxable base (Energy + Wheeling + Fixed + FAC).
5. Billing Cycle Adjustments: Slabs and fixed charges scale pro-rata for non-30-day cycles (cycle_days / 30.0).
6. FAC: Supported as an explicit monthly parameter. Defaults to 0.0 with transparent disclosure if omitted.
"""

import calendar
import math
from datetime import date
from typing import Any, Dict, List, Optional
from backend.app.schemas.tariffs import BillEstimateResponse


class MSEDCLTariffCalculator:
    """
    Isolated tariff calculation engine strictly adhering to MERC Case No. 226 of 2022.
    """

    OFFICIAL_CITATION = (
        "MERC MTR Order Case No. 226 of 2022 (dated 31 March 2023) & "
        "MSEDCL Commercial Circular No. 341 (FY 2024-25 baseline)"
    )

    # Base rate constants
    SLAB_RATES = [
        (0.0, 100.0, 4.41),    # Slab 1: 0 - 100 kWh @ ₹4.41
        (100.0, 300.0, 9.64),  # Slab 2: 101 - 300 kWh @ ₹9.64
        (300.0, 500.0, 13.61), # Slab 3: 301 - 500 kWh @ ₹13.61
        (500.0, None, 15.57),  # Slab 4: Above 500 kWh @ ₹15.57
    ]
    WHEELING_RATE = 1.17        # Flat ₹1.17 per kWh
    FIXED_SINGLE_PHASE = 128.0  # ₹128.00 / month
    FIXED_THREE_PHASE = 385.0   # ₹385.00 / month
    DUTY_RATE = 0.16            # 16% on taxable base

    def calculate(
        self,
        consumption_kwh: float,
        phase_type: Optional[str] = None,
        cycle_days: Optional[int] = None,
        fac_rate_per_kwh: Optional[float] = None,
        sanctioned_load_kw: Optional[float] = None,
        billing_period_start: Optional[date] = None,
        billing_period_end: Optional[date] = None,
    ) -> BillEstimateResponse:
        """
        Execute verified MSEDCL LT-I(B) residential billing calculation.
        """
        clean_kwh = max(0.0, float(consumption_kwh))
        disclosures: List[str] = [
            f"Calculated using verified rates from {self.OFFICIAL_CITATION} for Category LT-I(B) Residential."
        ]

        # 1. Determine cycle duration and pro-rata factor according to billing-period convention:
        #
        # CONVENTION:
        # A. Full Calendar Month:
        #    If billing_period_start and billing_period_end span a complete calendar month
        #    (start.day == 1 and end.day == last day of that month, within the same month/year),
        #    it represents exactly 1 standard monthly billing cycle under MERC LT-I(B) tariff rules.
        #    Slab thresholds (100, 300, 500 kWh) and fixed charges apply without pro-rata distortion,
        #    regardless of whether the month has 28, 29, 30, or 31 calendar days.
        #    Prorata factor is strictly 1.0 (resolved_days = 30 standard billing units).
        #
        # B. Non-Calendar-Month Cycles (Inclusive Dates):
        #    If start and end dates do NOT form a complete calendar month (e.g. mid-month meter reading
        #    dates or genuinely shorter cycles like a 15-day connection), the duration is computed
        #    inclusively: resolved_days = (billing_period_end - billing_period_start).days + 1.
        #    Prorata factor is resolved_days / 30.0.
        #
        # C. Explicit cycle_days:
        #    If cycle_days is explicitly passed, resolved_days = int(cycle_days), prorata_factor = resolved_days / 30.0.
        #
        # D. Default:
        #    If neither dates nor cycle_days are passed, defaults to standard 30-day monthly cycle (prorata_factor = 1.0).
        resolved_days = 30
        is_full_month = False

        if billing_period_start and billing_period_end:
            s = billing_period_start
            e = billing_period_end
            if s.day == 1 and s.year == e.year and s.month == e.month:
                _, max_day = calendar.monthrange(s.year, s.month)
                if e.day == max_day:
                    is_full_month = True

            if is_full_month:
                resolved_days = 30
                prorata_factor = 1.0
            else:
                # Inclusive day count for non-calendar-month billing intervals
                resolved_days = max(1, (e - s).days + 1)
                prorata_factor = resolved_days / 30.0
        elif cycle_days is not None:
            resolved_days = max(1, int(cycle_days))
            prorata_factor = resolved_days / 30.0
        else:
            resolved_days = 30
            prorata_factor = 1.0

        if not is_full_month and resolved_days != 30:
            disclosures.append(
                f"Billing cycle is {resolved_days} days. Slab thresholds and fixed charges were prorated by {prorata_factor:.3f}."
            )


        # 2. Determine connection phase and fixed charge
        phase_normalized = (phase_type or "single_phase").strip().lower().replace("-", "_").replace(" ", "_")
        if phase_normalized in ["three_phase", "3_phase", "3phase"]:
            is_three_phase = True
            base_fixed = self.FIXED_THREE_PHASE
            disclosures.append("Connection type: Three-Phase (base fixed charge ₹385.00/month).")
        else:
            is_three_phase = False
            base_fixed = self.FIXED_SINGLE_PHASE
            if phase_type is None:
                disclosures.append(
                    "Connection type not specified; defaulted to Single-Phase (₹128.00/month). "
                    "If your home has a Three-Phase connection, actual fixed charges will be ₹385.00/month."
                )
            else:
                disclosures.append("Connection type: Single-Phase (base fixed charge ₹128.00/month).")

        # Additional load fixed charges for loads > 10 kW
        additional_fixed = 0.0
        if sanctioned_load_kw is not None and sanctioned_load_kw > 10.0:
            excess_kw = sanctioned_load_kw - 10.0
            blocks = math.ceil(excess_kw / 10.0)
            additional_fixed = blocks * 200.0
            disclosures.append(
                f"Sanctioned load is {sanctioned_load_kw:.1f} kW (>10 kW). Additional fixed charge of ₹{additional_fixed:.2f} applied."
            )

        total_fixed_charge = round((base_fixed + additional_fixed) * prorata_factor, 2)

        # 3. Calculate telescopic energy slabs
        remaining_units = clean_kwh
        slab_breakdown: Dict[str, float] = {}
        total_energy_charge = 0.0

        for idx, (min_u, max_u, rate) in enumerate(self.SLAB_RATES, start=1):
            scaled_min = min_u * prorata_factor
            scaled_max = (max_u * prorata_factor) if max_u is not None else None

            if clean_kwh <= scaled_min:
                units_in_slab = 0.0
            elif scaled_max is None:
                # Top unbounded slab
                units_in_slab = max(0.0, clean_kwh - scaled_min)
            else:
                units_in_slab = min(clean_kwh, scaled_max) - scaled_min
                units_in_slab = max(0.0, units_in_slab)

            slab_cost = round(units_in_slab * rate, 2)
            slab_breakdown[f"slab_{idx}_units"] = round(units_in_slab, 2)
            slab_breakdown[f"slab_{idx}_rate_inr"] = rate
            slab_breakdown[f"slab_{idx}_cost_inr"] = slab_cost
            total_energy_charge += slab_cost

        total_energy_charge = round(total_energy_charge, 2)

        # 4. Calculate wheeling charge (flat ₹1.17 on all units)
        wheeling_charge = round(clean_kwh * self.WHEELING_RATE, 2)

        # 5. Calculate Fuel Adjustment Charge (FAC)
        if fac_rate_per_kwh is not None:
            clean_fac_rate = max(0.0, float(fac_rate_per_kwh))
            fac_charge = round(clean_kwh * clean_fac_rate, 2)
            disclosures.append(f"Fuel Adjustment Charge (FAC) applied at verified rate of ₹{clean_fac_rate:.4f}/kWh.")
        else:
            fac_charge = 0.0
            disclosures.append(
                "Fuel Adjustment Charge (FAC) is dynamic and published monthly by MSEDCL. "
                "FAC was calculated as ₹0.00. Check your recent electricity bill for the active monthly FAC rate."
            )

        # 6. Taxable Base & Electricity Duty
        taxable_base = round(total_energy_charge + wheeling_charge + total_fixed_charge + fac_charge, 2)
        electricity_duty = round(taxable_base * self.DUTY_RATE, 2)

        # 7. Total Estimated Bill
        estimated_total = round(taxable_base + electricity_duty, 2)

        disclosures.append(
            "Statutory notice: This pre-bill estimate is based on forecasted consumption and standard MERC tariff rules. "
            "It excludes arrears, interest, prompt payment rebates (1%), digital payment discounts (0.25%), meter rent, and local municipal cesses."
        )

        itemized_breakdown: Dict[str, float] = {
            "energy_charges_inr": total_energy_charge,
            "wheeling_charges_inr": wheeling_charge,
            "fixed_charges_inr": total_fixed_charge,
            "fuel_adjustment_charges_inr": fac_charge,
            "taxable_subtotal_inr": taxable_base,
            "electricity_duty_16pct_inr": electricity_duty,
            "estimated_total_bill_inr": estimated_total,
        }
        itemized_breakdown.update(slab_breakdown)

        return BillEstimateResponse(
            provider_id="msedcl",
            provider_name="Maharashtra State Electricity Distribution Company Limited (Mahavitaran)",
            state="Maharashtra",
            is_tariff_supported=True,
            is_tariff_verified=True,
            forecast_kwh=round(clean_kwh, 2),
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            estimated_bill_amount=estimated_total,
            charge_breakdown=itemized_breakdown,
            currency="INR",
            disclosures=disclosures,
        )


# Global isolated MSEDCL calculator instance
msedcl_calculator = MSEDCLTariffCalculator()
