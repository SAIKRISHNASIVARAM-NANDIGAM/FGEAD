"""
app/streamlit_app.py

FGEAD — Premium Enterprise Observability Dashboard
Consumes the FGEAD FastAPI backend serving the Server Machine Dataset (SMD) Machine 1-1.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FGEAD — Telemetry Intelligence Operations Center",
    page_icon="🧠",
    layout="wide",
)

# ============================================================================
# API ENDPOINT DEFINITION
# ============================================================================

API_URL = "http://127.0.0.1:8000"

# ============================================================================
# PREMIUM OBSERVED STYLING (Datadog / Grafana Aesthetic)
# ============================================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700;800&display=swap');

  /* Global Body override */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0a0b0d !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Outfit" !important;
  }

  /* Sidebar Styling */
  [data-testid="stSidebar"] {
    background-color: #0d0e12 !important;
    border-right: 1px solid #21262d !important;
  }

  /* Enterprise Card Panel */
  .enterprise-panel {
    background-color: #0f1115;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .enterprise-panel-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 12px;
  }

  /* Compact Metric Cards */
  .metric-card-enterprise {
    background-color: #0f1115;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 12px 16px;
    text-align: left;
    height: 100%;
  }

  .metric-value-enterprise {
    font-size: 1.6rem;
    font-weight: 700;
    color: #58a6ff; /* Cyan/blue for general info */
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
  }

  .metric-value-anomaly {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ff7b72; /* Red/orange for anomaly */
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
  }

  .metric-label-enterprise {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Badges */
  .badge-anomaly {
    background-color: rgba(248, 81, 73, 0.1);
    color: #ff7b72;
    border: 1px solid rgba(248, 81, 73, 0.2);
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
  }

  .badge-normal {
    background-color: rgba(56, 139, 253, 0.1);
    color: #58a6ff;
    border: 1px solid rgba(56, 139, 253, 0.2);
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
  }

  /* Alert Banners */
  .alert-banner {
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 16px;
    border-left: 4px solid;
    font-weight: 500;
    color: #c9d1d9;
    background-color: #0f1115;
    border-top: 1px solid #21262d;
    border-right: 1px solid #21262d;
    border-bottom: 1px solid #21262d;
  }
  .alert-critical { border-left-color: #ff7b72; }
  .alert-high { border-left-color: #ffa657; }
  .alert-medium { border-left-color: #f2cc60; }
  .alert-low { border-left-color: #7ee787; }

  /* Table styling */
  .enterprise-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 0.88rem;
  }

  .enterprise-table th {
    background-color: #161b22;
    border: 1px solid #21262d;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    color: #c9d1d9;
  }

  .enterprise-table td {
    border: 1px solid #21262d;
    padding: 8px 12px;
    color: #8b949e;
  }

  .enterprise-table tr:hover {
    background-color: #1f242c;
  }

  code {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #161b22;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.85rem !important;
  }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API CLIENT SERVICES
# ============================================================================

def fetch_api_health() -> Tuple[Dict[str, Any] | None, bool]:
    """Retrieve FastAPI health status."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2.0)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
        pass
    return None, False


def fetch_api_dataset() -> Dict[str, Any] | None:
    """Retrieve telemetry dataset stats."""
    try:
        r = requests.get(f"{API_URL}/dataset", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_machine() -> Dict[str, Any] | None:
    """Retrieve loaded machine characteristics."""
    try:
        r = requests.get(f"{API_URL}/machine", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data
def fetch_api_prediction(window_idx: int, window_data: List[List[float]]) -> Dict[str, Any]:
    """Submit raw window data to backend API for prediction and explanation."""
    payload = {
        "data": window_data,
        "window_start_abs": int(window_idx * 5)
    }
    # Increased timeout to 30.0 seconds to eliminate read time-outs
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30.0)
    if r.status_code == 200:
        return r.json()
    else:
        raise RuntimeError(
            f"Prediction failed with status {r.status_code}: {r.text}"
        )


# ============================================================================
# DATA HANDLING & LOCAL CACHING
# ============================================================================

@st.cache_data
def load_raw_smd_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    """Load real SMD Machine 1-1 test dataset and precompute window anomaly labels."""
    project_root = Path(__file__).resolve().parent.parent
    test_path = project_root / "data" / "SMD" / "test" / "machine-1-1.txt"
    label_path = project_root / "data" / "SMD" / "test_label" / "machine-1-1.txt"

    if not test_path.exists() or not label_path.exists():
        return None, None, None

    # Load data in Float32 to preserve memory space
    test_data = np.loadtxt(test_path, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(label_path, delimiter=",", dtype=np.int32)

    # Precompute window labels (sliding window size 60, stride 5)
    n_windows = (len(test_data) - 60) // 5 + 1
    window_labels = []
    for i in range(n_windows):
        start = i * 5
        end = start + 60
        window_labels.append(1 if np.any(test_labels[start:end] > 0) else 0)
    window_labels = np.array(window_labels)

    return test_data, test_labels, window_labels


# ============================================================================
# COMPACT LAYOUT TIMELINE HELPERS
# ============================================================================

def get_anomaly_intervals(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Determine continuous anomaly window segments."""
    intervals = []
    in_anom = False
    start = 0
    for i, l in enumerate(labels):
        if l > 0 and not in_anom:
            start = i
            in_anom = True
        elif l == 0 and in_anom:
            intervals.append((start, i - 1))
            in_anom = False
    if in_anom:
        intervals.append((start, len(labels) - 1))
    return intervals


def plot_telemetry_timeline(test_data: np.ndarray, test_labels: np.ndarray, selected_window_idx: int, feature_names: List[str]) -> go.Figure:
    """Plot downsampled raw telemetry with shaded anomaly regions and selected window outline."""
    # Downsample by 10 to make rendering extremely fast
    step = 10
    indices = np.arange(0, len(test_data), step)
    ds_data = test_data[::step]
    ds_labels = test_labels[::step]

    fig = go.Figure()

    # Shade point-level anomaly regions (Ground Truth)
    intervals = get_anomaly_intervals(ds_labels)
    for idx, (start, end) in enumerate(intervals):
        orig_start = start * step
        orig_end = end * step
        fig.add_vrect(
            x0=orig_start, x1=orig_end,
            fillcolor="#ff7b72", opacity=0.12, line_width=0,
            name="Anomaly Interval" if idx == 0 else ""
        )

    # Plot first 3 streams as representative telemetry
    for f_i in range(min(3, test_data.shape[1])):
        fig.add_trace(go.Scatter(
            x=indices.tolist(),
            y=ds_data[:, f_i].tolist(),
            mode="lines",
            name=feature_names[f_i],
            line=dict(width=1),
            hovertemplate="Time: %{x}<br>Val: %{y:.4f}<extra></extra>"
        ))

    # Draw highlighted vertical span for selected window
    w_start = selected_window_idx * 5
    w_end = w_start + 59
    fig.add_vrect(
        x0=w_start, x1=w_end,
        fillcolor="#58a6ff", opacity=0.25,
        line=dict(color="#58a6ff", width=1),
        name="Selected Window Explorer"
    )

    fig.update_layout(
        title="System Telemetry Timeline & Window Tracker",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c0e12",
        height=360,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(gridcolor="#1f242d", title="Absolute Timestep", zeroline=False),
        yaxis=dict(gridcolor="#1f242d", title="Sensor Value Scale", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ============================================================================
# MAIN INITIALIZATION
# ============================================================================

def main():
    # 1. Health check FastAPI
    health_info, connected = fetch_api_health()

    if not connected:
        st.markdown("""
            <div style="text-align:center; padding: 10px 0 30px 0;">
                <div style="font-size:2.4rem; font-weight:800; font-family:Outfit;
                    background: linear-gradient(90deg, #ff7b72, #ffa657);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; display:inline-block;">
                    FGEAD Operations Center
                </div>
            </div>
        """, unsafe_allow_html=True)
        show_offline_screen()
        return

    # 2. Load dataset
    test_data, test_labels, test_window_labels = load_raw_smd_dataset()

    if test_data is None:
        st.error(
            "CRITICAL: SMD Machine 1-1 telemetry data not found in "
            "`data/SMD/`. Please load the SMD datasets."
        )
        return

    n_windows = len(test_window_labels)
    threshold = 2.073376

    # ========================================================================
    # LEFT SIDEBAR NAVIGATION
    # ========================================================================
    st.sidebar.markdown(
        "<div style='font-size: 1.6rem; font-weight: 800; color: #58a6ff; margin-bottom: 2px; font-family: Outfit;'>FGEAD</div>"
        "<div style='font-size: 0.70rem; color: #8b949e; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 20px; font-family: Outfit;'>Enterprise observability</div>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown("### TARGET TELEMETRY")
    st.sidebar.selectbox("Active Device", ["Machine 1-1"], disabled=True)

    st.sidebar.markdown("### OPERATIONS CONSOLE")
    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "Overview",
            "Anomaly Detection",
            "Explainability",
            "Feature Graph",
            "Timeline",
            "Model Performance",
            "System Health"
        ],
        label_visibility="collapsed"
    )

    # ========================================================================
    # TOP STATUS BAR (Observability Summary)
    # ========================================================================
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"""
        <div style="background-color: #0f1115; padding: 12px 18px; border: 1px solid #21262d; border-radius: 4px; margin-bottom: 20px; font-size: 0.88rem; font-family: 'JetBrains Mono', monospace;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div><strong>MACHINE:</strong> <span style="color: #58a6ff;">1-1</span> &nbsp;|&nbsp;
                     <strong>MODEL:</strong> <span style="color: #58a6ff;">Active</span> &nbsp;|&nbsp;
                     <strong>API STATUS:</strong> <span style="color: #7ee787;">● CONNECTED</span></div>
                <div><strong>REFRESHED:</strong> <span style="color: #8b949e;">{now_str}</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ========================================================================
    # COMPACT SETTINGS & CONTROL AREA
    # ========================================================================
    col_c1, col_c2, col_c3 = st.columns([4, 4, 4])
    with col_c1:
        # Visual threshold parameter slider
        threshold_k = st.slider(
            "Visual Threshold Offset (k·σ)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            key="threshold_k"
        )
    with col_c2:
        # Default index set to verified anomaly window 3497 (absolute start 17485)
        window_idx = st.number_input(
            "Selected Window Index",
            min_value=0,
            max_value=n_windows - 1,
            value=17485 // 5,
            step=1,
            key="window_idx"
        )
    with col_c3:
        w_start_abs = int(window_idx * 5)
        w_end_abs = w_start_abs + 59
        st.markdown(f"""
            <div style="padding-top: 24px; font-size: 0.9rem; color: #8b949e; font-family: 'JetBrains Mono', monospace;">
                <strong>Absolute Window Timeline:</strong><br/>
                <code>t={w_start_abs}</code> &rarr; <code>t={w_end_abs}</code> (60m)
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 15px 0 25px 0; border-color: #21262d;' />", unsafe_allow_html=True)

    # 3. Pull predict response for active window (with spinner)
    active_window_data = test_data[w_start_abs : w_start_abs + 60].tolist()

    with st.spinner("Analyzing telemetry window via FGEAD backend..."):
        try:
            predict_res = fetch_api_prediction(window_idx, active_window_data)
        except Exception as e:
            st.error(f"Failed to query backend prediction API: {e}")
            return

    # Extract metadata features from `/machine` API response (cache-friendly)
    machine_info = fetch_api_machine()
    feature_names = (
        machine_info["feature_names"]
        if machine_info
        else [f"feature_{i:02d}" for i in range(38)]
    )

    # ========================================================================
    # NAVIGATION LOGIC
    # ========================================================================

    if page == "Overview":
        render_overview_page(predict_res, test_data, test_labels, window_idx, feature_names)

    elif page == "Anomaly Detection":
        render_anomaly_detection_page(predict_res, test_window_labels, threshold, window_idx)

    elif page == "Explainability":
        render_explainability_page(predict_res, test_data[w_start_abs : w_start_abs + 60], feature_names)

    elif page == "Feature Graph":
        render_feature_graph_page(predict_res)

    elif page == "Timeline":
        render_timeline_page(test_labels)

    elif page == "Model Performance":
        render_model_performance_page()

    elif page == "System Health":
        render_system_health_page(health_info)


# ============================================================================
# PAGE RENDERERS
# ============================================================================

def render_overview_page(predict_res: Dict[str, Any], test_data: np.ndarray, test_labels: np.ndarray, selected_idx: int, feature_names: List[str]):
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #c9d1d9; margin-top: 0;'>System Observability Overview</h2>", unsafe_allow_html=True)

    # Metric Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Detector Status</div>
                <div class="metric-value-enterprise" style="color: #7ee787;">ACTIVE</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        is_anom = predict_res["is_anomaly"]
        status_html = '<span class="badge-anomaly">🔴 ANOMALY</span>' if is_anom else '<span class="badge-normal">🟢 NORMAL</span>'
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Active Anomaly Status</div>
                <div style="margin-top: 8px;">{status_html}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Feature Streams</div>
                <div class="metric-value-enterprise">38/38</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        latency = predict_res["latency_ms"]
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Inference Latency</div>
                <div class="metric-value-enterprise" style="color: #7ee787;">{latency:.2f} ms</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # Metric Row 2 (Dataset specifications)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Train Timesteps</div>
                <div class="metric-value-enterprise" style="color: #8b949e;">28,479</div>
            </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Test Timesteps</div>
                <div class="metric-value-enterprise" style="color: #8b949e;">28,479</div>
            </div>
        """, unsafe_allow_html=True)
    with c7:
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Test Anomalous Steps</div>
                <div class="metric-value-anomaly">2,694</div>
            </div>
        """, unsafe_allow_html=True)
    with c8:
        st.markdown(f"""
            <div class="metric-card-enterprise">
                <div class="metric-label-enterprise">Anomaly rate (Ground Truth)</div>
                <div class="metric-value-anomaly" style="font-size: 1.4rem;">9.46%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

    # Timeline Chart
    st.markdown("<div class='enterprise-panel-header'>Telemetry Timeline Explorer</div>", unsafe_allow_html=True)

    # Render telemetry-based timeline with anomaly bands
    fig = plot_telemetry_timeline(test_data, test_labels, selected_idx, feature_names)
    st.plotly_chart(fig, use_container_width=True)


def render_anomaly_detection_page(predict_res: Dict[str, Any], window_labels: np.ndarray, threshold: float, selected_idx: int):
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #c9d1d9; margin-top: 0;'>Anomaly Detection Console</h2>", unsafe_allow_html=True)

    is_anom = predict_res["is_anomaly"]
    alert_level = predict_res["alert_level"]
    confidence_pct = predict_res["confidence_pct"]

    # Alerts matching Datadog incident style
    if is_anom:
        sev_class = (
            "alert-critical"
            if "CRIT" in alert_level.upper()
            else "alert-high"
            if "HIGH" in alert_level.upper()
            else "alert-medium"
        )
        st.markdown(f"""
            <div class="alert-banner {sev_class}">
                <div style="font-size: 1.1rem; font-weight: 700; color: #ff7b72; margin-bottom: 4px;">ACTIVE INCIDENT TRIGGERED</div>
                <div>The FGEAD detector has flagged the selected telemetry window as highly anomalous.</div>
                <div style="margin-top: 10px; font-family: monospace; font-size: 0.9rem;">
                    <strong>Alert Level:</strong> {alert_level} &nbsp;|&nbsp;
                    <strong>Score:</strong> {predict_res['anomaly_score']:.4f} &nbsp;|&nbsp;
                    <strong>Confidence:</strong> {confidence_pct}
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="alert-banner alert-low">
                <div style="font-size: 1.1rem; font-weight: 700; color: #7ee787; margin-bottom: 4px;">NOMINAL SYSTEM STATE</div>
                <div>Telemetry flows represent standard operating conditions. Anomaly scores are within safe bounds.</div>
            </div>
        """, unsafe_allow_html=True)

    # Columns of indicators
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='enterprise-panel-header'>Active Incident Details</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <table class="enterprise-table">
                <tr><td>Detector Decision</td><td>{'🔴 ANOMALY' if is_anom else '🟢 NOMINAL'}</td></tr>
                <tr><td>Anomaly Score</td><td><code>{predict_res['anomaly_score']:.6f}</code></td></tr>
                <tr><td>Calibrated Threshold</td><td><code>{threshold:.6f}</code></td></tr>
                <tr><td>Classification Confidence</td><td>{confidence_pct}</td></tr>
                <tr><td>Peak Timestep in Window</td><td>t = {predict_res['peak_timestep']}</td></tr>
                <tr><td>Anomaly Duration</td><td>{predict_res['anomaly_duration_steps']} steps</td></tr>
                <tr><td>Root Cause Summary</td><td>{predict_res['root_cause']}</td></tr>
            </table>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='enterprise-panel-header'>Incident Trigger Audit Log</div>", unsafe_allow_html=True)
        # Create a clean DataFrame of ground truth anomalous windows
        anom_indices = np.where(window_labels == 1)[0]

        audit_df = pd.DataFrame({
            "Window Index": anom_indices,
            "Absolute Range": [f"t={idx*5} → t={idx*5+59}" for idx in anom_indices],
            "Classification": ["Ground Truth Incident" for _ in anom_indices]
        })

        st.dataframe(audit_df, use_container_width=True, height=220, hide_index=True)
        st.caption(
            f"Audit log lists all {len(anom_indices)} window segments known to contain anomalous timesteps."
        )


def render_explainability_page(predict_res: Dict[str, Any], window_data: np.ndarray, feature_names: List[str]):
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #c9d1d9; margin-top: 0;'>Explainable AI (XAI) Diagnostics</h2>", unsafe_allow_html=True)

    # 1. WHY IS THIS ANOMALOUS?
    st.markdown("<div class='enterprise-panel-header'>Q1–Q5: Explainability Report Summary</div>", unsafe_allow_html=True)

    is_anom = predict_res["is_anomaly"]
    alert_level = str(predict_res.get("alert_level", "UNKNOWN"))
    alert_upper = alert_level.upper()

    # Respect the severity returned by the FastAPI backend instead of
    # labelling every anomalous window as CRITICAL.
    if not is_anom:
        status_text = "🟢 NOMINAL WINDOW"
        color_code = "#58a6ff"
    elif "CRITICAL" in alert_upper:
        status_text = "🔴 CRITICAL ANOMALY"
        color_code = "#ff7b72"
    elif "HIGH" in alert_upper:
        status_text = "🟠 HIGH ANOMALY"
        color_code = "#f0883e"
    elif "MEDIUM" in alert_upper:
        status_text = "🟡 MEDIUM ANOMALY"
        color_code = "#d29922"
    else:
        status_text = "🔵 ANOMALY DETECTED"
        color_code = "#58a6ff"

    # Prefer the actual temporal ranges when the API's aggregate duration
    # is zero or unavailable.
    ranges = predict_res.get("anomaly_ranges", [])
    if isinstance(ranges, list) and ranges:
        range_duration = sum(
            int(r.get("duration_steps", 0) or 0)
            for r in ranges
            if isinstance(r, dict)
        )
    else:
        range_duration = 0

    api_duration = int(predict_res.get("anomaly_duration_steps", 0) or 0)
    display_duration = max(api_duration, range_duration)

    st.markdown(f"""
        <div style="background-color: #0f1115; border: 1px solid #21262d; padding: 16px; border-radius: 4px; margin-bottom: 25px;">
            <div style="font-size: 1.2rem; font-weight: 800; color: {color_code}; margin-bottom: 8px;">{status_text}</div>
            <div style="display: flex; gap: 40px; flex-wrap: wrap; font-size: 0.9rem; color: #8b949e; font-family: monospace;">
                <div><strong>Score:</strong> <code>{predict_res['anomaly_score']:.4f}</code></div>
                <div><strong>Threshold:</strong> <code>{predict_res['threshold']:.4f}</code></div>
                <div><strong>Confidence:</strong> {predict_res['confidence_pct']}</div>
                <div><strong>Alert Level:</strong> {alert_level}</div>
                <div><strong>Peak Step:</strong> t = {predict_res['peak_timestep']}</div>
                <div><strong>Duration:</strong> {display_duration} steps</div>
            </div>
            <div style="margin-top: 12px; border-top: 1px solid #21262d; padding-top: 8px; font-size: 0.92rem;">
                <strong>Likely Root Cause Analysis:</strong><br/>
                <span style="color: #c9d1d9;">{predict_res['root_cause']}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. TOP CONTRIBUTING FEATURES
    st.markdown("<div class='enterprise-panel-header'>Top Contributing Telemetry Features</div>", unsafe_allow_html=True)

    top_features = predict_res.get("top_features", [])

    if not top_features:
        st.info("No contributing features returned for this window.")
    else:
        # HTML styled progress bar contribution table
        html_table = "<table class='enterprise-table'>"
        html_table += "<tr><th>Feature</th><th>Actual</th><th>Predicted</th><th>Deviation</th><th>Z-Score</th><th>Direction</th><th>Relative Impact</th></tr>"

        # Determine maximum z_score to scale impact bars
        max_z = max([float(x["z_score"]) for x in top_features]) if top_features else 1.0
        max_z = max(max_z, 1e-6)

        for f in top_features:
            z = float(f["z_score"])
            bar_width = int((z / max_z) * 100)
            bar_width = min(max(bar_width, 2), 100)

            bar_color = "#ff7b72" if z > 2.2 else "#ffa657" if z > 1.2 else "#58a6ff"
            bar_html = f"<div style='background-color: #21262d; width: 100px; height: 10px; border-radius: 2px;'><div style='background-color: {bar_color}; width: {bar_width}px; height: 10px; border-radius: 2px;'></div></div>"

            html_table += f"<tr>"
            html_table += f"<td><code>{f['feature']}</code></td>"
            html_table += f"<td>{f['actual']:.4f}</td>"
            html_table += f"<td>{f['predicted']:.4f}</td>"
            html_table += f"<td>{f['deviation']:.4f}</td>"
            html_table += f"<td>{z:.1f}&sigma;</td>"
            html_table += f"<td>{f['direction']}</td>"
            html_table += f"<td>{bar_html}</td>"
            html_table += f"</tr>"
        html_table += "</table>"

        st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # 3. ACTUAL VS PREDICTED (Plotly Chart)
    st.markdown("<div class='enterprise-panel-header'>Actual vs Forecast Telemetry Profile</div>", unsafe_allow_html=True)

    top_feature_names = [f["feature"] for f in top_features] if top_features else feature_names[:5]
    selected_feat = st.selectbox(
        "Select Telemetry Stream to Visualize Actual vs Forecast",
        top_feature_names
    )

    peak_t = predict_res["peak_timestep"]

    fig = plot_actual_vs_predicted(window_data, feature_names, selected_feat, peak_t, top_features)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    # 4. RELATIONSHIP BREAKDOWN & TEMPORAL RANGES
    col_break, col_range = st.columns([7, 5])

    with col_break:
        st.markdown("<div class='enterprise-panel-header'>🔗 Changed Feature Relationships</div>", unsafe_allow_html=True)
        broken_pairs = predict_res.get("broken_pairs", [])
        if not broken_pairs:
            st.info("All feature relationships remain within nominal bounds.")
        else:
            broken_rows = []
            for bp in broken_pairs:
                broken_rows.append({
                    "Feature A": bp["feature_a"],
                    "Feature B": bp["feature_b"],
                    "Expected": bp["expected_corr"],
                    "Observed": bp["observed_corr"],
                    "Gap": bp["gap"],
                    "Type": bp["break_type"],
                    "Severity": bp["severity"],
                    "Interpretation": bp["likely_meaning"]
                })
            st.dataframe(pd.DataFrame(broken_rows), use_container_width=True, hide_index=True)

    with col_range:
        st.markdown("<div class='enterprise-panel-header'>⏱ Temporal Anomaly Ranges</div>", unsafe_allow_html=True)
        ranges = predict_res.get("anomaly_ranges", [])
        if not ranges:
            st.info("No temporal anomaly ranges flagged inside this window.")
        else:
            range_rows = []
            for r in ranges:
                range_rows.append({
                    "Start Index": r["window_start"],
                    "End Index": r["window_end"],
                    "Duration": r["duration_human"],
                    "Peak Score": r["peak_score"],
                    "Mean Score": r["mean_score"]
                })
            st.dataframe(pd.DataFrame(range_rows), use_container_width=True, hide_index=True)


def plot_actual_vs_predicted(window_data: np.ndarray, feature_names: List[str], selected_feat: str, peak_t: int, top_features: List[Dict[str, Any]]) -> go.Figure:
    """Plot actual values line chart and compare with the predicted value at the anomaly peak."""
    feat_idx = feature_names.index(selected_feat)
    actual_series = window_data[:, feat_idx]

    fig = go.Figure()

    # Plot actual sequence
    fig.add_trace(go.Scatter(
        x=list(range(60)),
        y=actual_series.tolist(),
        mode="lines",
        name="Actual Value",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="Step: %{x}<br>Actual: %{y:.4f}<extra></extra>"
    ))

    # Find prediction values at peak
    pred_val = None
    act_val = None
    for f in top_features:
        if f["feature"] == selected_feat:
            pred_val = float(f["predicted"])
            act_val = float(f["actual"])
            break

    if pred_val is not None:
        # Plot predicted value at peak
        fig.add_trace(go.Scatter(
            x=[peak_t],
            y=[pred_val],
            mode="markers",
            name="Forecasted Value (at peak)",
            marker=dict(symbol="x", size=10, color="#7ee787", line=dict(width=2)),
            hovertemplate="Forecasted: %{y:.4f}<extra></extra>"
        ))

        # Plot actual value marker at peak
        fig.add_trace(go.Scatter(
            x=[peak_t],
            y=[act_val],
            mode="markers",
            name="Observed Value (at peak)",
            marker=dict(symbol="circle", size=10, color="#ff7b72"),
            hovertemplate="Observed: %{y:.4f}<extra></extra>"
        ))

    # Add vertical peak indicator line
    fig.add_vline(
        x=peak_t,
        line_dash="dash",
        line_color="#ff7b72",
        annotation_text="Anomaly Peak",
        annotation_position="top right"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c0e12",
        xaxis=dict(gridcolor="#1f242d", title="Relative Step inside Window", zeroline=False),
        yaxis=dict(gridcolor="#1f242d", title="Sensor Metric Value", zeroline=False),
        height=320,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def render_feature_graph_page(predict_res: Dict[str, Any]):
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #c9d1d9; margin-top: 0;'>Active Breakdown Graph</h2>", unsafe_allow_html=True)
    st.caption("Visually inspects feature dependency changes based strictly on API reports.")

    broken_pairs = predict_res.get("broken_pairs", [])

    if not broken_pairs:
        st.info("Nominal graph state: No broken relationships detected in this window.")
        return

    # Form node circular coordinates
    nodes = set()
    for p in broken_pairs:
        nodes.add(p["feature_a"])
        nodes.add(p["feature_b"])
    nodes = sorted(list(nodes))

    import math
    N = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / N
        pos[node] = (math.cos(angle), math.sin(angle))

    fig = go.Figure()

    # Draw edges
    for p in broken_pairs:
        x0, y0 = pos[p["feature_a"]]
        x1, y1 = pos[p["feature_b"]]

        sev = p["severity"].upper()
        edge_color = "#ff7b72" if "CRIT" in sev else "#ffa657" if "HIGH" in sev else "#f2cc60"

        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=float(p["gap"]) * 4.5, color=edge_color),
            hoverinfo="none",
            showlegend=False
        ))

    # Draw nodes
    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=14, color="#58a6ff", line=dict(width=2, color="#0f1115")),
        text=nodes,
        textposition="top center",
        hoverinfo="text",
        hovertext=nodes,
        textfont=dict(family="Outfit, monospace", size=10, color="#c9d1d9"),
        showlegend=False
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=520,
        margin=dict(l=40, r=40, t=20, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_timeline_page(test_labels: np.ndarray):
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #c9d1d9; margin-top: 0;'>Chronological Anomaly Events</h2>", unsafe_allow_html=True)
    st.caption("Chronological list of all ground-truth anomaly events in the test dataset.")

    # Calculate intervals using test_labels (point-level ground truth)
    intervals = get_anomaly_intervals(test_labels)
    events = []
    for idx, (s, e) in enumerate(intervals):
        duration = e - s + 1
        events.append({
            "Incident ID": f"GT-{s:05d}",
            "Start Step (Abs)": s,
            "End Step (Abs)": e,
            "Timeline Range": f"t={s} → t={e}",
            "Duration": f"{duration} timesteps ({duration}m)",
            "Severity Alert": "🔴 GROUND TRUTH ANOMALY"
        })

    if not events:
        st.info("No anomaly events recorded.")
    else:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


def render_model_performance_page():
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #c9d1d9; margin-top: 0;'>Model Performance & Validation</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='enterprise-panel-header'>FGEAD Baseline Benchmark Comparison</div>", unsafe_allow_html=True)
        # Read from official comparison reports
        benchmark = pd.DataFrame({
            "Model Name": ["FGEAD (Proposed)", "LSTM Autoencoder", "Isolation Forest"],
            "Window F1-Score": ["0.958", "0.918", "0.105"],
            "Window ROC-AUC": ["0.932", "0.927", "0.866"],
            "Point-level F1-Score": ["0.651", "0.612", "0.084"]
        })
        st.table(benchmark)
        st.caption(
            "Note: Isolation Forest suffers low recall due to time series flattening."
        )

    with col2:
        st.markdown("<div class='enterprise-panel-header'>Hyperparameters & Thresholds</div>", unsafe_allow_html=True)
        st.markdown("""
            <table class="enterprise-table">
                <tr><td>Attention Window size</td><td>60 steps</td></tr>
                <tr><td>Window Stride</td><td>5 steps</td></tr>
                <tr><td>Sparsity Regularization (L1)</td><td>&lambda; = 0.01</td></tr>
                <tr><td>Sparsity Threshold</td><td>0.30</td></tr>
                <tr><td>Encoder GCN Output Dimension</td><td>64</td></tr>
                <tr><td>Decoder LSTM Hidden State Dimension</td><td>128</td></tr>
                <tr><td>Calibrated Evaluation Threshold</td><td><code>2.073376</code></td></tr>
            </table>
        """, unsafe_allow_html=True)


def render_system_health_page(health_info: Dict[str, Any]):
    st.markdown("<h2 style='font-family: Outfit; font-weight: 700; color: #c9d1d9; margin-top: 0;'>FastAPI Observability & Health</h2>", unsafe_allow_html=True)

    st.markdown("<div class='enterprise-panel-header'>Uvicorn REST Health Check</div>", unsafe_allow_html=True)
    st.json(health_info)


def show_offline_screen():
    """Display instructions when FastAPI backend is disconnected."""
    st.markdown("""
        <div style="text-align:center; padding: 40px 0;">
            <h1 style="color: #ff7b72; font-family: Outfit; font-weight: 700;">FASTAPI API DISCONNECTED</h1>
            <p style="color: #8b949e; font-size: 1.1rem; max-width: 600px; margin: 10px auto 30px auto;">
                The operations dashboard is offline because the FGEAD FastAPI backend could not be reached at
                <code>http://127.0.0.1:8000</code>.
            </p>

            <div style="background-color: #0f1115; border: 1px solid #21262d; border-radius: 4px; padding: 24px; max-width: 520px; margin: 0 auto; text-align: left;">
                <p style="margin-top: 0; font-weight: 700; color: #c9d1d9; font-size: 0.95rem;">START BACKEND COMMAND:</p>
                <pre style="background-color: #161b22; padding: 12px; border-radius: 3px; color: #58a6ff; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; overflow-x: auto; border: 1px solid #21262d;">
uvicorn api.main:app --reload --port 8000
                </pre>
                <p style="font-size: 0.82rem; color: #8b949e; margin-bottom: 0; margin-top: 12px; line-height: 1.4;">
                    Ensure uvicorn is launched within the root directory of your python virtual environment with required libraries installed.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns([5, 2, 5])
    with col_r2:
        if st.button("Retry Connection", use_container_width=True):
            st.rerun()


if __name__ == "__main__":
    main()
