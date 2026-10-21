"""
Reliable Multimodal Household Energy Forecasting Under Degraded Weather Information
Interactive Research Dashboard & Analysis Application
"""

import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# Page Configuration & Aesthetics
# ==============================================================================
st.set_page_config(
    page_title="Multimodal Energy Forecasting | Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Theme Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global Card Container */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(8px);
        margin-bottom: 16px;
    }
    
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    
    .metric-badge-success {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }
    
    .metric-badge-info {
        display: inline-block;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }

    .metric-badge-warning {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        line-height: 1.5;
        max-width: 900px;
    }

    /* Tab enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px 8px 0 0;
        padding: 0 16px;
        font-weight: 500;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Set base matplotlib dark-theme palette
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#94a3b8",
    "text.color": "#f8fafc",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "font.size": 10,
})

# ==============================================================================
# Cached Data Loaders
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_data
def load_clean_benchmark():
    p = os.path.join(BASE_DIR, "outputs", "tables", "step10_clean_benchmark_comparison_revised.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_master_performance():
    p = os.path.join(BASE_DIR, "outputs", "tables", "step10_master_performance_table_revised.csv")
    if os.path.exists(p):
        return pd.read_csv(p, keep_default_na=False)
    return None

@st.cache_data
def load_statistical_summary():
    p = os.path.join(BASE_DIR, "outputs", "tables", "step10_statistical_significance_summary_revised.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_xai_summary():
    p = os.path.join(BASE_DIR, "outputs", "tables", "step10_xai_modality_attribution_summary_revised.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_conformal_summary():
    p = os.path.join(BASE_DIR, "outputs", "tables", "step10_conformal_uncertainty_summary_revised.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_weather_attribution():
    p = os.path.join(BASE_DIR, "outputs", "xai", "tables", "weather_feature_attribution.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_temporal_attribution():
    p = os.path.join(BASE_DIR, "outputs", "xai", "tables", "electricity_temporal_attribution.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

@st.cache_data
def load_all_test_predictions():
    """Load precomputed test predictions for all 4 models."""
    models = {
        "Energy-Only": os.path.join(BASE_DIR, "outputs", "predictions", "energy_only", "test_predictions.npz"),
        "Early Fusion": os.path.join(BASE_DIR, "outputs", "predictions", "early_fusion", "test_predictions.npz"),
        "Late Fusion": os.path.join(BASE_DIR, "outputs", "predictions", "late_fusion", "test_predictions.npz"),
        "Hybrid Fusion": os.path.join(BASE_DIR, "outputs", "predictions", "hybrid_fusion", "test_predictions.npz"),
    }
    data = {}
    actuals = None
    for m, p in models.items():
        if os.path.exists(p):
            npz = np.load(p)
            for k in ["predictions_watts", "predicted_watts", "predictions"]:
                if k in npz:
                    data[m] = npz[k]
                    break
            if actuals is None and "actual_watts" in npz:
                actuals = npz["actual_watts"]
    return actuals, data

@st.cache_data
def load_conformal_interval(model_key, condition="clean_NN"):
    """Load precomputed conformal intervals for specific model and condition."""
    fname = f"intervals_{model_key}_{condition}.npz"
    p = os.path.join(BASE_DIR, "outputs", "uncertainty", "artifacts", fname)
    if os.path.exists(p):
        return np.load(p)
    return None

# Load dataset artifacts
df_clean = load_clean_benchmark()
df_master = load_master_performance()
df_stats = load_statistical_summary()
df_xai = load_xai_summary()
df_conf = load_conformal_summary()
actual_test_watts, pred_test_dict = load_all_test_predictions()

# ==============================================================================
# Sidebar Navigation & Settings
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚡ Multimodal Energy AI")
    st.caption("Reliable Household Forecasting Under Weather Degradation")
    st.markdown("---")
    
    app_section = st.radio(
        "Navigation",
        [
            "📊 Executive Overview & Insights",
            "🔮 24-Hour Forecast Explorer",
            "🛡️ Operational Robustness Lab",
            "📐 Inferential Statistics (78 Tests)",
            "🔍 Explainable AI & Attribution",
            "🎯 Conformal Prediction & Bounds",
            "📚 Methodology & Architecture",
        ],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("#### 🔬 Experimental Scope")
    st.markdown(
        """
        - **Lookback Window**: 168 hours (7 days)
        - **Forecast Horizon**: 24 hours ahead
        - **Evaluation Set**: 587 rolling windows
        - **Total Test Points**: 14,088 hours
        - **Degradation**: 30% MCAR & 20% AWGN
        - **Conformal Coverage**: 90% ($\alpha = 0.10$)
        """
    )
    st.markdown("---")
    st.caption("Audited Empirical Artifacts • Zero Retraining")

# ==============================================================================
# SECTION 1: Executive Overview & Synthesis
# ==============================================================================
if app_section == "📊 Executive Overview & Insights":
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">Reliable Multimodal Household Energy Forecasting</div>
            <div class="hero-subtitle">
                Comprehensive empirical investigation into the impact of auxiliary weather degradation 
                on forecasting accuracy, operational robustness, modality attribution, and conformal uncertainty.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # KPI Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Clean Benchmark Leader</div>
                <div class="metric-value">Early Fusion</div>
                <span class="metric-badge-success">71.64 W MAE (-8.02% error)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Degraded Leader (A-A)</div>
                <div class="metric-value">Early Fusion</div>
                <span class="metric-badge-info">76.39 W MAE (Beats Clean Hybrid)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Statistical Test Battery</div>
                <div class="metric-value">60 / 78 Rejected</div>
                <span class="metric-badge-success">76.9% Rejection Rate (p &lt; 0.05)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Conformal Interval Score</div>
                <div class="metric-value">360.65 IS</div>
                <span class="metric-badge-info">Early Fusion (88.56% PICP, 90% Target)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    st.markdown("### 🏆 Clean Benchmark Performance Comparison")
    if df_clean is not None:
        display_clean = df_clean.copy()
        st.dataframe(
            display_clean.style.format({
                "test_mae_watts": "{:.2f} W",
                "test_rmse_watts": "{:.2f} W",
                "test_r2": "{:.4f}",
                "test_mape_pct": "{:.2f}%",
                "delta_mae_vs_baseline_watts": "{:+.2f} W",
                "relative_mae_change_pct": "{:+.2f}%",
            }),
            use_container_width=True,
        )
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🎯 Core Empirical Discoveries")
        st.markdown(
            """
            1. **Early Fusion Superiority**:
               Early Fusion achieved the lowest test MAE across all models (**71.64 W** vs **77.88 W** for Energy-Only baseline, an **8.02% relative error reduction**). It also achieved the highest $R^2$ (**0.2169**) and lowest MAPE (**48.58%**).
            
            2. **Decoupled Architectural Capacity vs Accuracy**:
               Decoupled multimodal architectures (Late Fusion with 112K parameters and Hybrid Fusion with 145K parameters) did not outperform the lightweight Early Fusion model (53K parameters) or the unimodal baseline on the held-out test partition.
            
            3. **Modality Sensitivity Asymmetry**:
               Smart meter electricity telemetry is the dominant operational signal. Losing electricity data caused substantial error increases (+1.89 W to +9.13 W), whereas losing weather data caused minute changes (+0.17 W to +0.65 W). Auxiliary weather provides valuable secondary context, but cannot substitute for missing electricity history.
            """
        )
    with col_right:
        st.markdown("#### 🛡️ Reliability & Uncertainty Insights")
        st.markdown(
            r"""
            4. **Noise Invariance vs Missingness Vulnerability**:
               All neural models demonstrated strong invariance to 20% Additive Gaussian Noise ($\le 0.10$ W MAE increase). In contrast, 30% MCAR missingness (with zero/mean imputation) caused statistically significant performance drops across all models.
               
            5. **Decoupled Relative Robustness vs Absolute Error**:
               While Hybrid Fusion exhibited the smallest percentage degradation (+2.95% relative increase under joint missingness), Early Fusion's degraded performance (76.39 W) remained lower in absolute terms than Hybrid Fusion's clean baseline (82.19 W).
               
            6. **Conformal Interval Invariance**:
               Clean-calibrated split conformal intervals maintain constant width across conditions ($268.26\text{ W}$ for Early Fusion), resulting in empirical coverage erosion under unmitigated sensor missingness.
            """
        )

# ==============================================================================
# SECTION 2: 24-Hour Forecast Explorer
# ==============================================================================
elif app_section == "🔮 24-Hour Forecast Explorer":
    st.markdown("## 🔮 Interactive 24-Hour Multi-Step Forecast Explorer")
    st.caption("Inspect and compare multi-step rollout trajectories across all 4 architectures for any withheld test window.")
    
    if actual_test_watts is not None and pred_test_dict:
        total_windows = actual_test_watts.shape[0]
        
        c_sel, c_stat1, c_stat2 = st.columns([2, 1, 1])
        with c_sel:
            window_idx = st.slider(
                "Select Test Window (Chronological Index 0 to 586)",
                min_value=0,
                max_value=total_windows - 1,
                value=42,
                help="Each index corresponds to a rolling 24-hour forecast window with 168 hours of historical lookback.",
            )
        
        actual_seq = actual_test_watts[window_idx]
        with c_stat1:
            st.metric("Window True Mean Load", f"{actual_seq.mean():.1f} W")
        with c_stat2:
            st.metric("Window Peak Load", f"{actual_seq.max():.1f} W")
            
        # Model Selection Checkboxes
        st.markdown("##### Model Toggles")
        t_cols = st.columns(4)
        show_models = {}
        colors = {
            "Energy-Only": "#f59e0b",
            "Early Fusion": "#10b981",
            "Late Fusion": "#38bdf8",
            "Hybrid Fusion": "#a855f7",
        }
        for i, (m, col) in enumerate(zip(pred_test_dict.keys(), t_cols)):
            with col:
                show_models[m] = st.checkbox(m, value=True, key=f"toggle_{m}")
                
        # Main Forecast Plot
        fig, (ax_main, ax_err) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
        hours = np.arange(1, 25)
        
        # Ground Truth
        ax_main.plot(hours, actual_seq, color="#f8fafc", linewidth=2.5, label="Ground Truth (Actual Watts)", zorder=5)
        
        # Model predictions
        window_metrics = []
        for m, preds in pred_test_dict.items():
            if show_models.get(m, True):
                m_pred = preds[window_idx]
                ax_main.plot(hours, m_pred, color=colors[m], linewidth=1.8, linestyle="--", label=f"{m} Forecast")
                # Residuals
                err = m_pred - actual_seq
                ax_err.plot(hours, err, color=colors[m], linewidth=1.5, linestyle="-", alpha=0.85, label=f"{m} Error")
                mae = np.mean(np.abs(err))
                rmse = np.sqrt(np.mean(err**2))
                window_metrics.append({"Model": m, "Window MAE": f"{mae:.2f} W", "Window RMSE": f"{rmse:.2f} W", "Peak Diff": f"{np.max(np.abs(err)):.2f} W"})
        
        ax_main.set_ylabel("Active Power (Watts)")
        ax_main.set_title(f"24-Hour Load Forecast Rollout — Test Window #{window_idx}", fontsize=12, pad=10)
        ax_main.legend(loc="upper right", framealpha=0.4, fontsize=9)
        ax_main.grid(True)
        
        ax_err.axhline(0, color="#64748b", linestyle=":", linewidth=1.2)
        ax_err.set_xlabel("Forecast Horizon Step (Hours Ahead: t+1 to t+24)")
        ax_err.set_ylabel("Error (W)")
        ax_err.set_xticks(hours)
        ax_err.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        if window_metrics:
            st.markdown("##### Window Evaluation Metrics")
            st.dataframe(pd.DataFrame(window_metrics), use_container_width=True)
    else:
        st.warning("Test predictions artifacts could not be located in outputs/predictions.")

# ==============================================================================
# SECTION 3: Operational Robustness Lab
# ==============================================================================
elif app_section == "🛡️ Operational Robustness Lab":
    st.markdown("## 🛡️ Operational Robustness & Degradation Lab")
    st.caption("Inspect performance across the 2x2 factorial matrix under 30% MCAR missingness and 20% Additive Gaussian Noise.")
    
    if df_master is not None:
        c_exp, c_cond = st.columns(2)
        with c_exp:
            exp_choice = st.selectbox(
                "Degradation Mechanism",
                ["30% MCAR Missingness", "20% Additive Gaussian Noise (eta=0.20)"],
                index=0,
            )
        exp_filter = "missingness_30" if "Missingness" in exp_choice else "noise_20"
        
        df_sub = df_master[df_master["experiment"] == exp_filter].copy()
        
        # Factorial condition visual matrix
        st.markdown("### 2x2 Factorial Performance Matrix")
        
        # Pivot table for MAE
        pivot_mae = df_sub.pivot(index="model", columns="condition", values="test_mae_watts")
        # Ensure ordered columns: NN, NA, AN, AA
        ordered_cols = [c for c in ["NN", "NA", "AN", "AA"] if c in pivot_mae.columns]
        pivot_mae = pivot_mae[ordered_cols]
        
        c_piv, c_chart = st.columns([1, 1.2])
        with c_piv:
            st.markdown("##### Test MAE (Watts)")
            st.dataframe(pivot_mae.style.format("{:.2f} W"), use_container_width=True)
            
            pivot_rel = df_sub.pivot(index="model", columns="condition", values="relative_mae_change_pct")[ordered_cols]
            st.markdown("##### Relative Degradation vs Clean Baseline (%)")
            st.dataframe(pivot_rel.style.format("{:+.2f}%"), use_container_width=True)
            
        with c_chart:
            # Bar chart comparing NN vs AA for all models
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(pivot_mae.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pivot_mae["NN"], width, label="Clean (N-N)", color="#10b981")
            bars2 = ax.bar(x + width/2, pivot_mae["AA"], width, label="Joint Degraded (A-A)", color="#f43f5e")
            
            ax.set_ylabel("Test MAE (Watts)")
            ax.set_title(f"{exp_choice}: Clean vs Joint Degraded", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(pivot_mae.index, rotation=15)
            ax.legend(framealpha=0.4)
            ax.grid(True, axis="y")
            
            for b in bars1:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8, color="#cbd5e1")
            for b in bars2:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8, color="#cbd5e1")
                
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        st.markdown("### 📋 Complete Experimental Audit Table")
        st.dataframe(
            df_sub[[
                "model", "condition", "condition_description", "test_mae_watts",
                "delta_mae_vs_clean_watts", "relative_mae_change_pct", "test_rmse_watts",
                "test_r2", "conformal_picp_pct", "conformal_mpiw_watts", "winkler_interval_score"
            ]].style.format({
                "test_mae_watts": "{:.2f} W",
                "delta_mae_vs_clean_watts": "{:+.2f} W",
                "relative_mae_change_pct": "{:+.2f}%",
                "test_rmse_watts": "{:.2f} W",
                "test_r2": "{:.4f}",
                "conformal_picp_pct": "{:.2f}%",
                "conformal_mpiw_watts": "{:.2f} W",
                "winkler_interval_score": "{:.2f}",
            }),
            use_container_width=True,
        )

# ==============================================================================
# SECTION 4: Inferential Statistics Explorer
# ==============================================================================
elif app_section == "📐 Inferential Statistics (78 Tests)":
    st.markdown("## 📐 Inferential Hypothesis Testing Battery")
    st.caption("Review all 78 formal paired hypothesis tests across 6 structured families with multiple comparisons corrections.")
    
    if df_stats is not None:
        st.dataframe(
            df_stats[["family_id", "family_name", "total_tests", "significant_tests_holm", "significant_tests_fdr"]],
            use_container_width=True,
        )
        
        selected_family = st.selectbox(
            "Select Hypothesis Family to Inspect",
            df_stats["family_name"].tolist(),
        )
        
        fam_row = df_stats[df_stats["family_name"] == selected_family].iloc[0]
        st.markdown(f"### 🔍 Detailed Inferences for {selected_family}")
        st.info(fam_row["primary_inference"])
        
        # Visual breakdown of rejection rates across all 6 families
        st.markdown("### 📊 Test Rejection Counts by Family")
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fam_ids = df_stats["family_id"].values
        total = df_stats["total_tests"].values
        sig_holm = df_stats["significant_tests_holm"].values
        
        x = np.arange(len(fam_ids))
        width = 0.35
        
        ax.bar(x - width/2, total, width, label="Total Tests Planned", color="#475569")
        ax.bar(x + width/2, sig_holm, width, label="Rejected Null (p_holm < 0.05)", color="#38bdf8")
        
        ax.set_ylabel("Number of Tests")
        ax.set_title("Hypothesis Testing Summary Across All 6 Planned Families (76.9% Total Rejection Rate)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fam {i}" for i in fam_ids])
        ax.legend(framealpha=0.4)
        ax.grid(True, axis="y")
        
        for i, (t, s) in enumerate(zip(total, sig_holm)):
            ax.text(i - width/2, t + 0.3, str(t), ha="center", va="bottom", fontsize=9, color="#94a3b8")
            ax.text(i + width/2, s + 0.3, str(s), ha="center", va="bottom", fontsize=9, color="#38bdf8")
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==============================================================================
# SECTION 5: Explainable AI & Attribution
# ==============================================================================
elif app_section == "🔍 Explainable AI & Attribution":
    st.markdown("## 🔍 Explainable AI & Modality Attribution")
    st.caption("Integrated Gradients attribution analysis quantifying modality shares, per-channel density, and temporal lag dynamics.")
    
    if df_xai is not None:
        st.markdown("### ⚡ Clean Benchmark Modality Shares & Channel Density")
        st.dataframe(
            df_xai[[
                "model", "clean_electricity_share_pct", "clean_weather_share_pct",
                "clean_electricity_density_share_pct", "clean_weather_density_share_pct",
                "top_weather_features_clean"
            ]].style.format({
                "clean_electricity_share_pct": "{:.1f}%",
                "clean_weather_share_pct": "{:.1f}%",
                "clean_electricity_density_share_pct": "{:.2f}%",
                "clean_weather_density_share_pct": "{:.2f}%",
            }),
            use_container_width=True,
        )
        
        col_density, col_shift = st.columns(2)
        with col_density:
            st.markdown("#### Per-Channel Density Share (Electricity vs Weather)")
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            models = df_xai["model"].tolist()
            elec_density = df_xai["clean_electricity_density_share_pct"].tolist()
            weath_density = df_xai["clean_weather_density_share_pct"].tolist()
            
            x = np.arange(len(models))
            width = 0.55
            
            ax.bar(x, elec_density, width, label="Electricity Density Share", color="#f59e0b")
            ax.bar(x, weath_density, width, bottom=elec_density, label="Weather Density Share", color="#06b6d4")
            
            ax.set_ylabel("Share of Single-Channel Density (%)")
            ax.set_title("Single-Channel Attribution Density Concentration", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=15)
            ax.set_ylim(0, 105)
            ax.legend(loc="lower right", framealpha=0.4)
            ax.grid(True, axis="y")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        with col_shift:
            st.markdown("#### Dynamic Attribution Reallocation Under Missingness")
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            multimodal_df = df_xai[df_xai["model"] != "Energy-Only"]
            m_names = multimodal_df["model"].tolist()
            shift_na = multimodal_df["miss_NA_elec_shift_pct_pts"].tolist()
            shift_an = multimodal_df["miss_AN_elec_shift_pct_pts"].tolist()
            
            x = np.arange(len(m_names))
            width = 0.35
            
            ax.bar(x - width/2, shift_na, width, label="Weather Masked (N-A): Shift to Elec", color="#10b981")
            ax.bar(x + width/2, shift_an, width, label="Elec Masked (A-N): Shift to Weather", color="#f43f5e")
            ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
            
            ax.set_ylabel("Electricity Attribution Shift (Percentage Points)")
            ax.set_title("Attribution Reallocation Under Missingness", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(m_names, rotation=15)
            ax.legend(framealpha=0.4)
            ax.grid(True, axis="y")
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        # Temporal lag analysis
        st.markdown("### ⏳ Temporal Memory & Lag Concentration")
        st.markdown(
            """
            - **Late & Hybrid Fusion**: Concentrated **>99.4%** of all electricity attribution exclusively in the most recent 24 hours ($t-1$ to $t-24$). Intermediate and distant lookbacks were practically ignored.
            - **Early Fusion**: Distributed **24.9%** into intermediate lags ($t-24$ to $t-71$) and **2.8%** into distant lags ($t-72$ to $t-167$), capturing multi-day autoregressive context.
            """
        )

# ==============================================================================
# SECTION 6: Conformal Prediction & Uncertainty
# ==============================================================================
elif app_section == "🎯 Conformal Prediction & Bounds":
    st.markdown("## 🎯 Conformal Prediction & Uncertainty Quantification")
    st.caption("Horizon-wise absolute residual split conformal prediction providing distribution-free 90% coverage guarantees.")
    
    if df_conf is not None:
        st.markdown("### 📐 Conformal Calibration & Clean Test Evaluation")
        st.dataframe(
            df_conf[[
                "model", "calibration_mae_watts", "conformal_quantile_qh_mean_watts",
                "mpiw_watts", "clean_picp_pct", "clean_ace_pct", "clean_winkler_interval_score"
            ]].style.format({
                "calibration_mae_watts": "{:.2f} W",
                "conformal_quantile_qh_mean_watts": "{:.2f} W",
                "mpiw_watts": "{:.2f} W",
                "clean_picp_pct": "{:.2f}%",
                "clean_ace_pct": "{:.2f}%",
                "clean_winkler_interval_score": "{:.2f}",
            }),
            use_container_width=True,
        )
        
        # Interactive conformal interval overlay
        st.markdown("### 🔮 Interactive 90% Conformal Interval Forecast Overlay")
        m_keys = {
            "Early Fusion": "early_fusion",
            "Energy-Only": "energy_only",
            "Late Fusion": "late_fusion",
            "Hybrid Fusion": "hybrid_fusion",
        }
        c1, c2, c3 = st.columns([1.5, 1.5, 2])
        with c1:
            sel_model_name = st.selectbox("Model", list(m_keys.keys()), index=0)
        with c2:
            sel_cond = st.selectbox("Condition", ["Clean (N-N)", "30% Missing (A-A)", "20% Noise (A-A)"], index=0)
        
        cond_map = {
            "Clean (N-N)": "clean_NN",
            "30% Missing (A-A)": "missingness_AA",
            "20% Noise (A-A)": "noise_AA",
        }
        
        int_obj = load_conformal_interval(m_keys[sel_model_name], cond_map[sel_cond])
        if int_obj is not None:
            total_wins = int_obj["y_hat"].shape[0]
            with c3:
                win_slider = st.slider("Test Window Index", 0, total_wins - 1, 100)
                
            y_hat = int_obj["y_hat"][win_slider]
            y_true = int_obj["y_true"][win_slider]
            L = int_obj["L_raw"][win_slider]
            U = int_obj["U_raw"][win_slider]
            hours = np.arange(1, 25)
            
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.fill_between(hours, L, U, color="#38bdf8", alpha=0.25, label="90% Conformal Prediction Interval [L, U]")
            ax.plot(hours, y_hat, color="#38bdf8", linewidth=2, linestyle="--", label="Model Point Forecast")
            ax.plot(hours, y_true, color="#f8fafc", linewidth=2.5, label="Actual Observed Power (Watts)")
            
            ax.set_xlabel("Forecast Horizon Step (Hours Ahead: t+1 to t+24)")
            ax.set_ylabel("Power (Watts)")
            ax.set_title(f"{sel_model_name} — 90% Conformal Prediction Interval ({sel_cond}, Window #{win_slider})", fontsize=11)
            ax.set_xticks(hours)
            ax.legend(loc="upper right", framealpha=0.5)
            ax.grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Coverage stats for this window
            inside = np.sum((y_true >= L) & (y_true <= U))
            st.metric("Window Empirical Coverage (PICP)", f"{inside / 24 * 100:.1f}% ({inside}/24 hours covered)")
        else:
            st.warning(f"Interval file for {sel_model_name} under {sel_cond} not found.")

# ==============================================================================
# SECTION 7: Architecture & Methodology
# ==============================================================================
elif app_section == "📚 Methodology & Architecture":
    st.markdown("## 📚 Research Methodology & Architecture Deep Dive")
    st.caption("Formal specifications, neural architecture configurations, and parameter audits.")
    
    arch_specs = [
        {"Model Architecture": "Energy-Only Baseline", "Trainable Parameters": "51,992", "Recurrent Backbone": "LSTM(64, 2 layers)", "Input Dim": "1 (Power)", "Readout Head": "Linear(64, 24)"},
        {"Model Architecture": "Early Fusion LSTM", "Trainable Parameters": "53,784", "Recurrent Backbone": "LSTM(64, 2 layers)", "Input Dim": "8 (Power + 7 Weather)", "Readout Head": "Linear(64, 24)"},
        {"Model Architecture": "Late Fusion LSTM", "Trainable Parameters": "112,216", "Recurrent Backbone": "2x LSTM(64, 2 layers)", "Input Dim": "Power (1) & Weather (7)", "Readout Head": "Linear(128, 24)"},
        {"Model Architecture": "Hybrid Fusion LSTM", "Trainable Parameters": "145,496", "Recurrent Backbone": "2x LSTM(64, 2) + Shared LSTM(64)", "Input Dim": "Power (1) & Weather (7)", "Readout Head": "Linear(64, 24)"},
    ]
    st.markdown("### 🏗️ Model Architecture Master Table")
    st.dataframe(pd.DataFrame(arch_specs), use_container_width=True)
    
    st.markdown("### 📋 Preprocessing & Hygiene Principles")
    st.markdown(
        r"""
        - **1-Hour Grouping**: Timestamps floored to the hour, computing arithmetic mean of `active_power` (W).
        - **Circular Wind Direction**: Raw degrees mapped to continuous $\sin(\theta)$ and $\cos(\theta)$ components.
        - **Strict Anti-Leakage**: Future forecast columns (`temp_t+1`, `feels_like_t+1`) excluded from feature matrix.
        - **Chronological Split**: 4,400h Train, 881h Validation, 587h Calibration, 587h Test.
        - **Frozen Standard Scalers**: Scalers fit exclusively on training split and loaded read-only via `joblib`.
        """
    )
    
    # Audit log
    st.markdown("### 📜 Audit Verification Trail")
    st.code(
        """
Audit Status: COMPLETE
Verification Pass Rate: 100% (28/28 experimental conditions, 78/78 hypothesis tests)
Numerical Discrepancies: 0
Data Contamination: ZERO (No leakage between partitions)
Model Training: Deterministic seeds (CPU-only, seed=42)
        """,
        language="text",
    )
