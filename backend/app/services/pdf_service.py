"""
PDF report generation service for persistent household electricity load forecasts.

Generates genuine, publication-grade vector PDF documents in memory:
- 24-hour multi-step point predictions with shaded 90% conformal prediction intervals.
- Key energy KPIs (total kWh, mean load, peak demand, uncertainty margin).
- Full 24-hour tabular breakdown with upper/lower prediction bounds.
- Explicit statutory notes on conformal uncertainty and domain generalization limitations.
"""

import io
import textwrap
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for server environments
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from backend.app.schemas.monthly_forecast import MonthlyForecastRecordDetail
from backend.app.schemas.records import ForecastRecordDetail


def generate_forecast_pdf(record: ForecastRecordDetail) -> bytes:
    """
    Generate a 2-page publication-grade PDF report from a stored successful forecast record.
    Returns the binary PDF bytes.
    """
    if record.status != "succeeded" or not record.predictions_watts or not record.intervals:
        raise ValueError("PDF report can only be generated for successful forecast executions with complete predictions.")

    preds = record.predictions_watts
    intervals = record.intervals
    steps = [step.horizon_hour for step in intervals]
    lower_clipped = [step.lower_bound_clipped_watts for step in intervals]
    upper = [step.upper_bound_watts for step in intervals]

    # Calculations
    sum_watts = sum(preds)
    total_kwh = sum_watts / 1000.0
    avg_watts = sum_watts / len(preds)
    peak_idx = int(np.argmax(preds))
    peak_watts = preds[peak_idx]
    peak_step = steps[peak_idx]
    mean_width = record.mean_interval_width_watts or float(np.mean([u - l for u, l in zip(upper, lower_clipped)]))
    half_width = mean_width / 2.0

    buf = io.BytesIO()

    # Create multi-page PDF document
    with PdfPages(buf) as pdf:
        # =====================================================================
        # PAGE 1: Executive Summary, KPIs, Forecast Chart & Scientific Notices
        # =====================================================================
        fig1 = plt.figure(figsize=(8.5, 11), dpi=150)
        fig1.patch.set_facecolor("white")

        # Top Header Banner
        fig1.text(0.08, 0.94, "HOUSEHOLD ELECTRICITY LOAD FORECAST REPORT", fontsize=15, fontweight="bold", color="#0f172a")
        fig1.text(0.08, 0.92, "Reliable Multimodal Household Energy Forecasting System • Day-Ahead Outlook", fontsize=9, color="#475569")
        fig1.text(0.92, 0.92, f"Run ID: {record.run_id}", fontsize=8, color="#64748b", ha="right", fontfamily="monospace")

        # Divider line
        line_ax = fig1.add_axes([0.08, 0.905, 0.84, 0.002])
        line_ax.axhline(color="#cbd5e1", linewidth=1)
        line_ax.axis("off")

        # Metadata Box (Left & Right columns)
        created_str = record.created_at[:19].replace("T", " ") if record.created_at else "N/A"
        meta_text_left = (
            f"Execution Run ID:   {record.run_id}\n"
            f"Created Time (UTC): {created_str}\n"
            f"Forecast Horizon:   24 Hours (Multi-Step)\n"
            f"Physical Units:     Active Power in Watts (W)"
        )
        meta_text_right = (
            f"Selected Model:     {record.selected_model or 'Early Fusion'}\n"
            f"Ingestion Channel:  {record.input_method.upper()}\n"
            f"User / Identifier:  {record.user_id or 'Anonymous Household'}\n"
            f"Confidence Target:  90.0% Split Conformal PICP"
        )
        fig1.text(0.08, 0.84, meta_text_left, fontsize=8, color="#1e293b", fontfamily="monospace", va="top")
        fig1.text(0.52, 0.84, meta_text_right, fontsize=8, color="#1e293b", fontfamily="monospace", va="top")

        # KPI Summary Cards (4 blocks)
        kpi_y = 0.74
        kpi_box_height = 0.065
        kpi_box_width = 0.195
        kpis = [
            ("24h Projected Energy", f"{total_kwh:.2f} kWh", "Cumulative sum / 1000", "#1d4ed8"),
            ("Average Active Power", f"{avg_watts:.0f} W", "Mean hourly power draw", "#0f172a"),
            ("Projected Peak Load", f"{peak_watts:.0f} W", f"Peak expected at Step +{peak_step}h", "#b45309"),
            ("Uncertainty Margin", f"±{half_width:.0f} W", f"Mean width: {mean_width:.0f} W", "#047857"),
        ]

        for i, (title, val, sub, col) in enumerate(kpis):
            bx = 0.08 + i * (kpi_box_width + 0.02)
            rect = plt.Rectangle((bx, kpi_y), kpi_box_width, kpi_box_height, transform=fig1.transFigure,
                                 facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1, zorder=1)
            fig1.patches.append(rect)
            fig1.text(bx + 0.01, kpi_y + 0.05, title.upper(), fontsize=6.5, fontweight="bold", color="#64748b")
            fig1.text(bx + 0.01, kpi_y + 0.024, val, fontsize=12, fontweight="bold", color=col)
            fig1.text(bx + 0.01, kpi_y + 0.008, sub, fontsize=6, color="#64748b")

        # Chart: 24-Hour Horizon with 90% Conformal Interval
        chart_ax = fig1.add_axes([0.08, 0.35, 0.84, 0.34])
        chart_ax.fill_between(
            steps,
            lower_clipped,
            upper,
            color="#93c5fd",
            alpha=0.35,
            label="90% Conformal Prediction Interval [Lower, Upper]",
            edgecolor="#60a5fa",
            linewidth=0.8,
        )
        chart_ax.plot(
            steps,
            preds,
            color="#1d4ed8",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="Model Point Forecast ŷ (Watts)",
        )
        chart_ax.set_title("24-Hour Electricity Load Forecast & 90% Conformal Prediction Bounds", fontsize=10, fontweight="bold", pad=8, color="#0f172a")
        chart_ax.set_xlabel("Forecast Horizon Step (Hours Ahead: t+1 to t+24)", fontsize=8, labelpad=6, color="#475569")
        chart_ax.set_ylabel("Active Power Demand (Watts)", fontsize=8, labelpad=6, color="#475569")
        chart_ax.set_xticks(range(1, 25))
        chart_ax.set_xticklabels([f"+{h}h" for h in range(1, 25)], fontsize=7, color="#475569")
        chart_ax.tick_params(axis="y", labelsize=7, colors="#475569")
        chart_ax.set_ylim(bottom=0)
        chart_ax.grid(True, linestyle="--", alpha=0.4, color="#cbd5e1")
        chart_ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="white", edgecolor="#cbd5e1")
        for spine in chart_ax.spines.values():
            spine.set_color("#cbd5e1")

        # Statutory Disclaimer & Scientific Integrity Note
        disc_y = 0.12
        disc_rect = plt.Rectangle((0.08, disc_y), 0.84, 0.18, transform=fig1.transFigure,
                                  facecolor="#eff6ff", edgecolor="#bfdbfe", linewidth=1, zorder=1)
        fig1.patches.append(disc_rect)

        fig1.text(0.10, disc_y + 0.155, "STATUTORY SCIENTIFIC & UNCERTAINTY DISCLOSURES", fontsize=7.5, fontweight="bold", color="#1e3a8a")

        uncert_text = (
            "• Conformal Uncertainty Calibration: The shaded prediction interval band represents a distribution-free 90% confidence\n"
            "  envelope (significance level alpha = 0.10) calibrated using Horizon-Wise Split Conformal Prediction across 587 rolling\n"
            "  evaluation windows. These intervals are statistical estimations based on calibrated validation behavior, NOT an operational\n"
            "  guarantee for an individual household. Actual demand may deviate outside this envelope during unpredictable occupant appliance\n"
            "  usage (e.g. EV charging, HVAC spikes, heat pumps) or extreme grid events."
        )
        fig1.text(0.10, disc_y + 0.105, uncert_text, fontsize=6.5, color="#1e293b", va="top", linespacing=1.3)

        generalization_text = (
            "• Domain Generalization Notice: Forecast models were trained on household smart meter and localized weather distributions.\n"
            "  Forecast accuracy and conformal coverage may degrade when applied to homes with substantially different electrical heating,\n"
            "  occupancy schedules, or geographic microclimates. Zero missing data imputation was performed during model inference."
        )
        fig1.text(0.10, disc_y + 0.038, generalization_text, fontsize=6.5, color="#1e293b", va="top", linespacing=1.3)

        # Footer
        fig1.text(0.08, 0.04, "Reliable Multimodal Household Energy Forecasting • Operational Report", fontsize=7, color="#94a3b8")
        fig1.text(0.92, 0.04, "Page 1 of 2", fontsize=7, color="#94a3b8", ha="right")

        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2: 24-Hour Detailed Tabular Schedule & Architectural Provenance
        # =====================================================================
        fig2 = plt.figure(figsize=(8.5, 11), dpi=150)
        fig2.patch.set_facecolor("white")

        # Header
        fig2.text(0.08, 0.94, "24-HOUR HOURLY FORECAST SCHEDULE & METADATA", fontsize=14, fontweight="bold", color="#0f172a")
        fig2.text(0.08, 0.92, f"Detailed Numerical Breakdown for Run ID: {record.run_id}", fontsize=8.5, color="#475569")
        fig2.text(0.92, 0.92, f"Target Horizon: t+1 to t+24", fontsize=8, color="#64748b", ha="right", fontfamily="monospace")

        line_ax2 = fig2.add_axes([0.08, 0.905, 0.84, 0.002])
        line_ax2.axhline(color="#cbd5e1", linewidth=1)
        line_ax2.axis("off")

        # Model Provenance Box
        prov_y = 0.835
        prov_rect = plt.Rectangle((0.08, prov_y), 0.84, 0.065, transform=fig2.transFigure,
                                  facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1, zorder=1)
        fig2.patches.append(prov_rect)
        fig2.text(0.10, prov_y + 0.048, "MODEL ARCHITECTURE & PREPROCESSING PROVENANCE", fontsize=7, fontweight="bold", color="#475569")

        model_meta = record.model_metadata or {}
        model_class = model_meta.get("model_class", "PyTorch Model")
        param_count = model_meta.get("trainable_parameters", 0)
        scaler_type = model_meta.get("energy_scaler_type", "StandardScaler (mean=307.17 W, std=147.99 W)")
        weather_scaler = model_meta.get("weather_scaler_type", "None")

        prov_text = (
            f"Model Architecture: {record.selected_model} ({model_class})  •  Trainable Parameters: {param_count:,}  •  Lookback: 168h\n"
            f"Scaler Configuration: {scaler_type}  •  Weather Scaler: {weather_scaler}  •  Selection Mode: {record.selection_mode or 'auto'}"
        )
        fig2.text(0.10, prov_y + 0.018, prov_text, fontsize=6.5, color="#1e293b", fontfamily="monospace")

        # 24-Hour Detailed Table
        table_ax = fig2.add_axes([0.08, 0.18, 0.84, 0.63])
        table_ax.axis("off")

        headers = ["Step", "Target Timestamp", "Point Forecast (W)", "Lower Bound (≥0W)", "Upper Bound (W)", "Width (W)", "Demand Category"]
        table_data = []

        for step in intervals:
            p = round(step.prediction_watts, 1)
            l = round(step.lower_bound_clipped_watts, 1)
            u = round(step.upper_bound_watts, 1)
            w = round(step.interval_width_watts, 1)
            ts = step.target_timestamp if step.target_timestamp else f"Step +{step.horizon_hour}h"
            cat = "Low (<500W)" if p < 500 else ("High (>1500W)" if p > 1500 else "Moderate")
            table_data.append([
                f"+{step.horizon_hour}h",
                ts[:19].replace("T", " "),
                f"{p:,.1f}",
                f"{l:,.1f}",
                f"{u:,.1f}",
                f"{w:,.1f}",
                cat,
            ])

        tab = table_ax.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(6.8)
        tab.scale(1.0, 1.45)

        # Style Table Header and Rows
        for (row_idx, col_idx), cell in tab.get_celld().items():
            cell.set_edgecolor("#e2e8f0")
            if row_idx == 0:
                cell.set_facecolor("#1e293b")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                if row_idx % 2 == 0:
                    cell.set_facecolor("#f8fafc")
                else:
                    cell.set_facecolor("white")
                if col_idx == 2:  # Point forecast
                    cell.set_text_props(color="#1d4ed8", fontweight="bold")

        # Bottom Privacy & Compliance Note
        fig2.text(
            0.08,
            0.08,
            "Privacy & Security Note: Raw historical 168-hour smart meter readings are discarded upon tensor construction and are never persisted.\n"
            "This report was generated directly from stored model prediction vectors and conformal quantile bounds in the local database.",
            fontsize=6.5,
            color="#64748b",
            linespacing=1.3,
        )

        fig2.text(0.08, 0.04, "Reliable Multimodal Household Energy Forecasting • Operational Report", fontsize=7, color="#94a3b8")
        fig2.text(0.92, 0.04, "Page 2 of 2", fontsize=7, color="#94a3b8", ha="right")

        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()


def format_bullet_entry(category: str, text: str, width: int = 98) -> str:
    """Format a bullet point with hanging indent to avoid text crowding or clipping."""
    full_text = f"• {category}: {text}"
    wrapped = textwrap.wrap(full_text, width=width, subsequent_indent="    ")
    return "\n".join(wrapped)


def collect_disclosures_list(record: MonthlyForecastRecordDetail) -> List[Tuple[str, str]]:
    """Gather all saved warnings, limitations, step disclosures, and tariff notices."""
    items: List[Tuple[str, str]] = []
    for w in (record.warnings or []):
        if w:
            items.append(("Warning", w))
    for lim in (record.limitations or []):
        if lim:
            items.append(("Limitation", lim))

    # Step-level bill estimate disclosures
    step_disc: List[str] = []
    for s in (record.forecast_steps or []):
        if s.bill_estimate and s.bill_estimate.disclosures:
            for d in s.bill_estimate.disclosures:
                if d and d not in step_disc:
                    step_disc.append(d)

    if step_disc:
        for d in step_disc:
            items.append(("Tariff Notice", d))
    elif record.total_estimated_bill_inr is None:
        note = record.bill_estimate_notes or "Estimated bill is Unavailable because tariff schedule is unverified or unsupported."
        items.append(("Tariff Notice", note))

    if not items:
        items.append(("Note", "No methodological warnings or operational limitations were recorded for this forecast run."))

    return items


def generate_monthly_forecast_pdf(record: MonthlyForecastRecordDetail) -> bytes:
    """
    Generate a publication-grade, vector PDF report from a stored monthly forecast snapshot.
    Generated exclusively from the authoritative, persisted database record.
    Returns binary PDF bytes.
    """
    buf = io.BytesIO()
    gen_time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    created_str = str(record.created_at)[:19].replace("T", " ") + " UTC"

    input_recs = record.input_records or []
    n_inputs = len(input_recs)

    # Collect disclosures
    disclosure_items = collect_disclosures_list(record)
    warn_entries = [format_bullet_entry(cat, txt, width=98) for cat, txt in disclosure_items]

    total_warn_lines = sum(e.count("\n") + 1 for e in warn_entries)
    warn_box_h = 0.020 + total_warn_lines * 0.0108 + (len(warn_entries) - 1) * 0.005

    # Method description
    method_name = record.selected_method.value if hasattr(record.selected_method, "value") else str(record.selected_method)
    method_lines = textwrap.wrap(record.method_description or "Standard monthly persistence method.", width=102)
    bm_text = (
        f"Backtest Evaluation: MAE = {record.backtest_metrics.mae_kwh:.2f} kWh  •  RMSE = {record.backtest_metrics.rmse_kwh:.2f} kWh  •  sMAPE = {record.backtest_metrics.smape_percent or 0.0:.1f}% ({record.backtest_metrics.backtest_periods or 1} periods evaluated)"
        if record.backtest_metrics
        else "Backtest Evaluation: Formal holdout backtesting requires ≥12 continuous historical billing cycles."
    )
    bm_lines = textwrap.wrap(bm_text, width=102)
    method_box_h = 0.022 + (1 + len(method_lines) + len(bm_lines)) * 0.011 + 0.010

    # Disclaimers
    d1 = (
        "Statutory Notice: This report contains predictive statistical projections based strictly on the supplied historical monthly "
        "billing records. It does not constitute an official electricity utility bill, a tariff guarantee, or a binding assessment from your distribution licensee."
    )
    d2 = (
        "Operational Caveats: Actual billing amounts and consumption may vary due to weather variations, seasonal factors, changes in household "
        "appliances, utility tariff revisions, fuel adjustment charges (FAC), or meter inaccuracies. Unavailable tariff estimates indicate unverified schedules, not zero-cost billing."
    )
    d3 = (
        "Data Integrity: Raw billing inputs are immutable snapshots locked at forecast creation time. Neither this application nor its operators assume "
        "liability for operational, budgeting, or procurement decisions made on the basis of these projections."
    )

    d1_wrapped = "\n".join(textwrap.wrap(d1, width=102))
    d2_wrapped = "\n".join(textwrap.wrap(d2, width=102))
    d3_wrapped = "\n".join(textwrap.wrap(d3, width=102))
    disc_lines = d1_wrapped.count("\n") + 1 + d2_wrapped.count("\n") + 1 + d3_wrapped.count("\n") + 1
    disclaimer_box_h = 0.020 + disc_lines * 0.0098 + 2 * 0.006

    total_disc_h = 0.065 + warn_box_h + method_box_h + disclaimer_box_h + 0.040
    table_section_h = 0.080 + 0.024 * n_inputs

    chunk_size = 16

    # Dynamic pagination calculation
    if n_inputs <= 6 and (table_section_h + total_disc_h <= 0.80):
        total_pages = 2
        input_chunks = [input_recs]
        has_separate_disclosures = False
    elif n_inputs <= 14 and (total_disc_h <= 0.78):
        total_pages = 3
        input_chunks = [input_recs]
        has_separate_disclosures = True
    else:
        input_chunks = [input_recs[i : i + chunk_size] for i in range(0, n_inputs, chunk_size)]
        total_pages = 1 + len(input_chunks) + 1
        has_separate_disclosures = True

    def _draw_page_header(fig, title: str, subtitle: str, page_num: int):
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.945, "HOUSEHOLD ELECTRICITY LOAD FORECASTING", fontsize=8, fontweight="bold", color="#2563eb", fontfamily="sans-serif")
        fig.text(0.08, 0.920, title, fontsize=14, fontweight="bold", color="#0f172a", fontfamily="sans-serif")
        fig.text(0.08, 0.898, subtitle, fontsize=7.5, color="#64748b", fontfamily="sans-serif")

        fig.text(0.92, 0.945, f"Run ID: {record.run_id}", fontsize=7, fontweight="bold", color="#1e293b", fontfamily="monospace", ha="right")
        fig.text(0.92, 0.930, f"Snapshot: {created_str}", fontsize=6.5, color="#64748b", ha="right")
        fig.text(0.92, 0.915, f"Generated: {gen_time_str}", fontsize=6.5, color="#64748b", ha="right")
        fig.text(0.92, 0.900, f"Owner ID: {record.owner_user_id}", fontsize=6.5, color="#64748b", fontfamily="monospace", ha="right")

        fig.patches.extend([plt.Line2D([0.08, 0.92], [0.888, 0.888], transform=fig.transFigure, color="#cbd5e1", linewidth=1.0)])
        fig.text(0.08, 0.035, f"Reliable Multimodal Household Energy Forecasting • Snapshot [{record.run_id}]", fontsize=6.5, color="#94a3b8")
        fig.text(0.92, 0.035, f"Page {page_num} of {total_pages}", fontsize=6.5, color="#94a3b8", ha="right")

    def _draw_disclosures_section(fig, start_y: float):
        cur_y = start_y

        # 1. Saved Warnings Box
        fig.text(0.08, cur_y, "SAVED WARNINGS, LIMITATIONS & TARIFF NOTICES", fontsize=8.0, fontweight="bold", color="#0f172a")
        cur_y -= 0.015

        fig.patches.extend([
            plt.Rectangle((0.08, cur_y - warn_box_h), 0.84, warn_box_h, transform=fig.transFigure, facecolor="#fffbeb", edgecolor="#fde68a", linewidth=1.0, zorder=1)
        ])

        text_y = cur_y - 0.014
        for entry in warn_entries:
            fig.text(0.10, text_y, entry, fontsize=6.5, color="#92400e", linespacing=1.28, va="top")
            n_lines = entry.count("\n") + 1
            text_y -= (n_lines * 0.0108 + 0.005)

        cur_y = cur_y - warn_box_h - 0.020

        # 2. Methodology Box
        fig.text(0.08, cur_y, "FORECASTING METHODOLOGY & EVALUATION", fontsize=8.0, fontweight="bold", color="#0f172a")
        cur_y -= 0.015

        fig.patches.extend([
            plt.Rectangle((0.08, cur_y - method_box_h), 0.84, method_box_h, transform=fig.transFigure, facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1.0, zorder=1)
        ])

        m_text_y = cur_y - 0.014
        fig.text(0.10, m_text_y, f"Selected Method: {method_name.replace('_', ' ').title()}", fontsize=7.5, fontweight="bold", color="#1e293b", va="top")
        m_text_y -= 0.014

        fig.text(0.10, m_text_y, "\n".join(method_lines), fontsize=6.5, color="#475569", linespacing=1.28, va="top")
        m_text_y -= (len(method_lines) * 0.0108 + 0.006)

        fig.text(0.10, m_text_y, "\n".join(bm_lines), fontsize=6.3, color="#64748b", fontfamily="monospace", linespacing=1.25, va="top")

        cur_y = cur_y - method_box_h - 0.020

        # 3. Disclaimers Box
        fig.text(0.08, cur_y, "STATUTORY & OPERATIONAL DISCLAIMER", fontsize=8.0, fontweight="bold", color="#0f172a")
        cur_y -= 0.015

        fig.patches.extend([
            plt.Rectangle((0.08, cur_y - disclaimer_box_h), 0.84, disclaimer_box_h, transform=fig.transFigure, facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.0, zorder=1)
        ])

        d_text_y = cur_y - 0.014
        for p_wrapped in [d1_wrapped, d2_wrapped, d3_wrapped]:
            fig.text(0.10, d_text_y, p_wrapped, fontsize=6.0, color="#475569", linespacing=1.22, va="top")
            p_lines = p_wrapped.count("\n") + 1
            d_text_y -= (p_lines * 0.0098 + 0.006)

    with PdfPages(buf) as pdf:
        # =====================================================================
        # PAGE 1: Forecast Overview, KPIs, Forecast Chart & Detailed Table
        # =====================================================================
        fig1 = plt.figure(figsize=(8.5, 11), dpi=150)
        fp_str = record.forecast_period_range or "N/A"
        _draw_page_header(
            fig1,
            title="Monthly Household Electricity Forecast",
            subtitle=f"Authoritative Server Snapshot • Forecast Horizon: {record.horizon_months} Months ({fp_str})",
            page_num=1,
        )

        # 4 KPI Summary Cards across y in [0.77, 0.87]
        kpi_y, kpi_h, card_w, card_gap = 0.77, 0.10, 0.195, 0.02
        cards_x = [0.08 + i * (card_w + card_gap) for i in range(4)]
        for cx in cards_x:
            fig1.patches.extend([plt.Rectangle((cx, kpi_y), card_w, kpi_h, transform=fig1.transFigure, facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1, zorder=1)])

        fig1.text(cards_x[0] + 0.01, kpi_y + 0.075, "TOTAL FORECAST", fontsize=6.5, fontweight="bold", color="#64748b")
        fig1.text(cards_x[0] + 0.01, kpi_y + 0.040, f"{record.total_forecast_kwh:,.1f}", fontsize=13, fontweight="bold", color="#1d4ed8")
        fig1.text(cards_x[0] + 0.01, kpi_y + 0.020, f"kWh across {record.horizon_months} billing cycles", fontsize=6.5, color="#475569")

        fig1.text(cards_x[1] + 0.01, kpi_y + 0.075, "AVERAGE MONTHLY", fontsize=6.5, fontweight="bold", color="#64748b")
        fig1.text(cards_x[1] + 0.01, kpi_y + 0.040, f"{record.average_monthly_kwh:,.1f}", fontsize=13, fontweight="bold", color="#0f172a")
        fig1.text(cards_x[1] + 0.01, kpi_y + 0.020, "kWh per billing cycle", fontsize=6.5, color="#475569")

        fig1.text(cards_x[2] + 0.01, kpi_y + 0.075, "ESTIMATED TOTAL BILL", fontsize=6.5, fontweight="bold", color="#64748b")
        if record.total_estimated_bill_inr is not None:
            fig1.text(cards_x[2] + 0.01, kpi_y + 0.040, f"₹{record.total_estimated_bill_inr:,.2f}", fontsize=13, fontweight="bold", color="#15803d")
            fig1.text(cards_x[2] + 0.01, kpi_y + 0.020, f"{record.months_with_bill_estimates} of {record.horizon_months} mos estimated", fontsize=6.5, color="#475569")
        else:
            fig1.text(cards_x[2] + 0.01, kpi_y + 0.040, "Unavailable", fontsize=13, fontweight="bold", color="#b45309")
            fig1.text(cards_x[2] + 0.01, kpi_y + 0.020, "Tariff unverified / unconfigured", fontsize=6.5, color="#b45309")

        conf_val = record.confidence_rating.value if hasattr(record.confidence_rating, "value") else str(record.confidence_rating)
        fig1.text(cards_x[3] + 0.01, kpi_y + 0.075, "METHOD & CONFIDENCE", fontsize=6.5, fontweight="bold", color="#64748b")
        fig1.text(cards_x[3] + 0.01, kpi_y + 0.045, method_name.replace("_", " ").title()[:18], fontsize=9.5, fontweight="bold", color="#0f172a")
        fig1.text(cards_x[3] + 0.01, kpi_y + 0.020, f"Confidence: {conf_val.upper()} ({record.history_length_months} mos)", fontsize=6.5, fontweight="bold", color="#2563eb" if conf_val in ("moderate", "high") else "#d97706")

        # Context pill line across y = 0.730
        ctx_y = 0.730
        fig1.patches.extend([plt.Rectangle((0.08, ctx_y), 0.84, 0.026, transform=fig1.transFigure, facecolor="#f1f5f9", edgecolor="#e2e8f0", linewidth=0.8, zorder=1)])
        mode_str = "Live Projection" if record.is_live else "Historical Backtest"
        prov_str = record.provider_id.upper() if record.provider_id else "Not specified"
        phase_str = record.phase_type.replace("_", " ").title() if record.phase_type else "Single Phase"
        load_str = f"{record.sanctioned_load_kw:.1f} kW" if record.sanctioned_load_kw is not None else "Not specified"
        last_str = record.last_observed_period or "N/A"
        ctx_text = f"Mode: {mode_str}   •   Utility: {prov_str}   •   Phase: {phase_str}   •   Sanctioned Load: {load_str}   •   Last Observed: {last_str}"
        fig1.text(0.10, ctx_y + 0.007, ctx_text, fontsize=6.5, color="#334155")

        # Forecast Chart
        steps = record.forecast_steps
        chart_ax = fig1.add_axes([0.08, 0.445, 0.84, 0.250])

        x_labels = [s.month_label for s in steps]
        y_vals = [s.forecast_kwh for s in steps]
        x_indices = np.arange(len(steps))

        has_intervals = all(s.uncertainty_lower_kwh is not None and s.uncertainty_upper_kwh is not None for s in steps)
        bar_width = 0.4 if len(steps) <= 3 else 0.55
        chart_ax.bar(x_indices, y_vals, width=bar_width, color="#2563eb", alpha=0.9, edgecolor="#1d4ed8", linewidth=1.2, zorder=3, label="Projected Consumption (kWh)")

        if has_intervals:
            yerr_lower = [s.forecast_kwh - (s.uncertainty_lower_kwh or s.forecast_kwh) for s in steps]
            yerr_upper = [(s.uncertainty_upper_kwh or s.forecast_kwh) - s.forecast_kwh for s in steps]
            chart_ax.errorbar(
                x_indices, y_vals, yerr=[yerr_lower, yerr_upper], fmt="none",
                ecolor="#d97706", elinewidth=1.8, capsize=5, capthick=1.5, zorder=4,
                label="90% Uncertainty Interval",
            )

        max_y = max([s.uncertainty_upper_kwh or s.forecast_kwh for s in steps]) if has_intervals else (max(y_vals) if y_vals else 100.0)
        for idx, s in enumerate(steps):
            chart_ax.text(idx, s.forecast_kwh + (max_y * 0.04), f"{s.forecast_kwh:,.1f} kWh", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1e293b")

        chart_ax.set_xticks(x_indices)
        chart_ax.set_xticklabels(x_labels, fontsize=8, fontweight="bold", color="#1e293b")
        chart_ax.set_ylabel("Monthly Consumption (kWh)", fontsize=7.5, color="#475569")
        chart_ax.set_ylim(0, max(1.0, max_y * 1.30))
        chart_ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
        chart_ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            chart_ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            chart_ax.spines[sp].set_color("#cbd5e1")
        chart_ax.tick_params(colors="#64748b", labelsize=7.5)
        chart_ax.set_title("Projected Monthly Consumption with Uncertainty Bounds", fontsize=8.5, fontweight="bold", color="#0f172a", loc="left", pad=8)
        chart_ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=7)

        # Monthly Forecast Breakdown Table
        fig1.text(0.08, 0.405, "MONTHLY FORECAST BREAKDOWN & TARIFF NOTICES", fontsize=8.0, fontweight="bold", color="#0f172a")
        table_ax = fig1.add_axes([0.08, 0.065, 0.84, 0.330])
        table_ax.axis("off")

        headers = ["Cycle", "Forecast Period", "Predicted\n(kWh)", "90% Interval\n(kWh)", "Estimated\nBill", "Tariff Status / Disclosures"]
        col_widths = [0.15, 0.18, 0.13, 0.17, 0.14, 0.23]
        table_data = []

        for s in steps:
            int_str = (
                f"[{s.uncertainty_lower_kwh:,.1f} –\n{s.uncertainty_upper_kwh:,.1f}]"
                if (s.uncertainty_lower_kwh is not None and s.uncertainty_upper_kwh is not None)
                else ("N/A\n(Single obs)" if record.history_length_months <= 1 else "N/A\n(Short history)")
            )
            if s.bill_estimate and s.bill_estimate.estimated_bill_amount is not None:
                bill_str = f"₹{s.bill_estimate.estimated_bill_amount:,.2f}"
                prov = s.bill_estimate.provider_id.upper() if s.bill_estimate.provider_id else "VERIFIED"
                tariff_note = f"Verified\n({prov})"
            else:
                bill_str = "Unavailable"
                tariff_note = "Withheld\n(Tariff unverified)"

            table_data.append([
                f"Month {s.month_index}\n({s.month_label})",
                f"{s.period_start} to\n{s.period_end}",
                f"{s.forecast_kwh:,.1f}",
                int_str,
                bill_str,
                tariff_note,
            ])

        tab = table_ax.table(cellText=table_data, colLabels=headers, colWidths=col_widths, loc="center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7.5 if len(steps) <= 3 else (7.0 if len(steps) <= 6 else 6.2))
        tab.scale(1.0, 1.8 if len(steps) <= 3 else (1.4 if len(steps) <= 6 else 1.15))

        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            if r == 0:
                cell.set_facecolor("#1e293b")
                cell.set_text_props(color="white", fontweight="bold", fontsize=7.2)
            else:
                cell.set_facecolor("#f8fafc" if r % 2 == 0 else "white")
                if c == 2:
                    cell.set_text_props(color="#1d4ed8", fontweight="bold")
                elif c == 4:
                    cell_text = cell.get_text().get_text()
                    if "Unavailable" in cell_text:
                        cell.set_text_props(color="#b45309", fontweight="bold")
                    else:
                        cell.set_text_props(color="#15803d", fontweight="bold")
                elif c == 5:
                    cell.set_text_props(color="#92400e" if "Withheld" in cell.get_text().get_text() else "#15803d", fontsize=7.0)

        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2+: Historical Input Snapshot & Methodological Disclosures
        # =====================================================================
        if not has_separate_disclosures:
            fig2 = plt.figure(figsize=(8.5, 11), dpi=150)
            _draw_page_header(
                fig2,
                title="Historical Billing Inputs & Methodological Disclosures",
                subtitle="Confirmed historical monthly consumption records and complete analytical disclosures.",
                page_num=2,
            )

            # Historical Input Table
            fig2.text(0.08, 0.865, "HISTORICAL INPUT SNAPSHOT", fontsize=8.0, fontweight="bold", color="#0f172a")

            hist_tab_h = min(0.20, 0.040 + 0.024 * n_inputs)
            hist_table_ax = fig2.add_axes([0.08, 0.860 - hist_tab_h, 0.84, hist_tab_h])
            hist_table_ax.axis("off")

            hist_headers = ["#", "Period Start", "Period End", "Observed (kWh)", "Billed Amount", "Input Status"]
            hist_col_widths = [0.08, 0.18, 0.18, 0.20, 0.18, 0.18]
            hist_data = []
            for i, r in enumerate(input_recs):
                b_str = f"₹{r.bill_amount:,.2f}" if r.bill_amount is not None else "Not Provided"
                hist_data.append([str(i + 1), str(r.billing_period_start), str(r.billing_period_end), f"{r.consumption_kwh:,.1f}", b_str, "Confirmed"])

            hist_tab = hist_table_ax.table(cellText=hist_data, colLabels=hist_headers, colWidths=hist_col_widths, loc="center", cellLoc="center")
            hist_tab.auto_set_font_size(False)
            hist_tab.set_fontsize(7.5)
            hist_tab.scale(1.0, 1.45)

            for (r, c), cell in hist_tab.get_celld().items():
                cell.set_edgecolor("#cbd5e1")
                if r == 0:
                    cell.set_facecolor("#1e293b")
                    cell.set_text_props(color="white", fontweight="bold", fontsize=7.2)
                else:
                    cell.set_facecolor("#f8fafc" if r % 2 == 0 else "white")
                    if c == 4:
                        if "Not Provided" in cell.get_text().get_text():
                            cell.set_text_props(color="#64748b")
                        else:
                            cell.set_text_props(color="#0f172a")

            # Draw disclosures on bottom of Page 2
            disc_start_y = 0.860 - hist_tab_h - 0.025
            _draw_disclosures_section(fig2, start_y=disc_start_y)

            pdf.savefig(fig2)
            plt.close(fig2)

        else:
            # Multi-page input table
            current_page_idx = 2
            for chunk_idx, chunk in enumerate(input_chunks):
                fig_chunk = plt.figure(figsize=(8.5, 11), dpi=150)
                _draw_page_header(
                    fig_chunk,
                    title="Historical Billing Inputs Snapshot",
                    subtitle=f"Confirmed historical records (Part {chunk_idx + 1} of {len(input_chunks)})",
                    page_num=current_page_idx,
                )

                chunk_table_ax = fig_chunk.add_axes([0.08, 0.15, 0.84, 0.70])
                chunk_table_ax.axis("off")

                hist_headers = ["#", "Period Start", "Period End", "Observed (kWh)", "Billed Amount", "Input Status"]
                hist_col_widths = [0.08, 0.18, 0.18, 0.20, 0.18, 0.18]
                hist_data = []
                base_idx = chunk_idx * chunk_size
                for i, r in enumerate(chunk):
                    b_str = f"₹{r.bill_amount:,.2f}" if r.bill_amount is not None else "Not Provided"
                    hist_data.append([str(base_idx + i + 1), str(r.billing_period_start), str(r.billing_period_end), f"{r.consumption_kwh:,.1f}", b_str, "Confirmed"])

                chunk_tab = chunk_table_ax.table(cellText=hist_data, colLabels=hist_headers, colWidths=hist_col_widths, loc="center", cellLoc="center")
                chunk_tab.auto_set_font_size(False)
                chunk_tab.set_fontsize(7.5)
                chunk_tab.scale(1.0, 1.4)

                for (r, c), cell in chunk_tab.get_celld().items():
                    cell.set_edgecolor("#cbd5e1")
                    if r == 0:
                        cell.set_facecolor("#1e293b")
                        cell.set_text_props(color="white", fontweight="bold", fontsize=7.2)
                    else:
                        cell.set_facecolor("#f8fafc" if r % 2 == 0 else "white")

                pdf.savefig(fig_chunk)
                plt.close(fig_chunk)
                current_page_idx += 1

            # Final Page: Disclosures & Disclaimers
            fig_disc = plt.figure(figsize=(8.5, 11), dpi=150)
            _draw_page_header(
                fig_disc,
                title="Methodology, Disclosures & Statutory Disclaimers",
                subtitle="Complete methodological transparency and regulatory notices.",
                page_num=current_page_idx,
            )
            _draw_disclosures_section(fig_disc, start_y=0.850)

            pdf.savefig(fig_disc)
            plt.close(fig_disc)

    buf.seek(0)
    return buf.getvalue()

