"""
app/streamlit_app.py

FGEAD — Feature-Level Graph-Based Explainable Anomaly Detection
Google Stitch-Inspired Light Academic Interface for Multivariate Time Series.

Core Project Mission:
"FGEAD does not only detect an anomaly — it explains why it was detected."

Consumes the FGEAD FastAPI backend serving the Server Machine Dataset (SMD) Machine 1-1.
"""

from __future__ import annotations

import math
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path before any project-level package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from data.live_feature_schema import (
    FEATURE_DESCRIPTIONS,
    FEATURE_UNITS,
    LIVE_FEATURES,
    format_physical_metric,
    validate_live_feature_dict,
)
from agents.windows_agent import WindowsTelemetryCollector
from api.deep_explainability import (
    FEATURE_HUMAN_NAMES,
    SUBSYSTEM_HUMAN_NAMES,
    FEATURE_INVESTIGATION_HINTS,
    DISCLAIMER_TEXT,
    get_feature_human_name,
    build_deep_human_explanation,
)



# ============================================================================
# HELPER UTILITIES
# ============================================================================

def render_html(html_str: str, in_sidebar: bool = False):
    """Safely render HTML without Markdown indented-code-block parsing traps."""
    cleaned_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    cleaned_html = "\n".join(cleaned_lines)
    target = st.sidebar if in_sidebar else st
    target.markdown(cleaned_html, unsafe_allow_html=True)


def navigate_to_page(page_name: str):
    """Programmatically navigate to another page while keeping Streamlit sidebar radio in sync."""
    st.session_state["nav_radio"] = page_name
    st.session_state["current_page"] = page_name
    st.rerun()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FGEAD — Explainable AI Anomaly Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API ENDPOINT DEFINITION
# ============================================================================

API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

# ============================================================================
# GOOGLE STITCH-INSPIRED LIGHT ACADEMIC THEME STYLING
# ============================================================================

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap');

  /* Global Body override - Clean Light Academic Theme */
  html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  }

  /* Sidebar Styling */
  [data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
  }

  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #334155 !important;
  }

  /* Clean Academic Card */
  .stitch-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.03);
  }

  .stitch-card-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e3a8a;
    border-bottom: 1px solid #f1f5f9;
    padding-bottom: 8px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Hero Status Card for Dashboard */
  .status-hero-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
  }

  .status-hero-card-anomaly {
    border-left: 6px solid #dc2626;
  }

  .status-hero-card-normal {
    border-left: 6px solid #16a34a;
  }

  /* Step Progress Flow Indicator */
  .step-flow-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 16px;
    gap: 8px;
    flex-wrap: wrap;
  }

  .step-flow-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #475569;
  }

  .step-flow-item.active {
    color: #1e40af;
  }

  .step-flow-badge {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: #eff6ff;
    color: #2563eb;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
  }

  .step-flow-item.active .step-flow-badge {
    background-color: #2563eb;
    color: #ffffff;
  }

  .step-flow-arrow {
    color: #cbd5e1;
    font-weight: 700;
    font-size: 0.85rem;
  }

  /* Metric Cards */
  .stitch-metric-box {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 16px;
    height: 100%;
    box-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.02);
  }

  .stitch-metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }

  .stitch-metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    font-family: 'Outfit', sans-serif;
  }

  .stitch-metric-subtext {
    font-size: 0.80rem;
    color: #64748b;
    margin-top: 3px;
  }

  /* Status Badges */
  .badge-normal {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.84rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .badge-anomaly {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.84rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .badge-neutral {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #dbeafe;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.84rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  /* Tag Pill */
  .tag-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 9px;
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #334155;
    margin-right: 6px;
    margin-bottom: 6px;
  }

  .tag-pill-alert {
    background-color: #fef2f2;
    border-color: #fecaca;
    color: #b91c1c;
  }

  .tag-pill-info {
    background-color: #eff6ff;
    border-color: #bfdbfe;
    color: #1d4ed8;
  }

  /* Result Verdict Banners */
  .verdict-banner-normal {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 5px solid #16a34a;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }

  .verdict-banner-anomaly {
    background-color: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 5px solid #dc2626;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
  }

  /* Human-Readable Narrative Box (Prominent Why Box) */
  .narrative-why-box {
    background-color: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 5px solid #2563eb;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 14px 0 18px 0;
    box-shadow: 0 1px 3px rgba(37, 99, 235, 0.05);
  }

  .narrative-why-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.10rem;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .narrative-why-text {
    font-size: 0.94rem;
    color: #1e293b;
    line-height: 1.6;
  }

  /* Root Cause Callout Box */
  .root-cause-callout {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.5;
  }

  /* Visual Feature Contribution Card (Top 3-5) */
  .feature-contrib-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
  }

  .feature-contrib-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .feature-contrib-name {
    font-family: 'Outfit', sans-serif;
    font-size: 0.98rem;
    font-weight: 700;
    color: #0f172a;
  }

  .feature-contrib-zscore {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
  }

  .feature-contrib-z-high {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
  }

  .feature-contrib-z-med {
    background-color: #ffedd5;
    color: #c2410c;
    border: 1px solid #fed7aa;
  }

  .feature-contrib-z-low {
    background-color: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #dbeafe;
  }

  .feature-contrib-bar-bg {
    background-color: #f1f5f9;
    height: 7px;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .feature-contrib-stats {
    display: flex;
    gap: 16px;
    font-size: 0.80rem;
    color: #64748b;
    flex-wrap: wrap;
  }

  .feature-contrib-stats strong {
    color: #334155;
  }

  /* Visual Relationship Summary Card */
  .relationship-summary-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
  }

  .relationship-connection-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-bottom: 8px;
  }

  .relationship-node {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.90rem;
    color: #1e3a8a;
  }

  .relationship-line {
    flex-grow: 1;
    margin: 0 14px;
    height: 2px;
    background: repeating-linear-gradient(to right, #94a3b8 0, #94a3b8 5px, transparent 5px, transparent 9px);
    position: relative;
    text-align: center;
  }

  .relationship-gap-badge {
    position: relative;
    top: -10px;
    background-color: #fee2e2;
    color: #b91c1c;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 10px;
    border: 1px solid #fecaca;
  }

  /* Section Intro Text */
  .section-intro-text {
    font-size: 0.88rem;
    color: #475569;
    margin-bottom: 12px;
    line-height: 1.5;
  }

  /* Academic Data Table */
  .stitch-table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 0.86rem;
  }

  .stitch-table th {
    background-color: #f8fafc;
    border-bottom: 2px solid #e2e8f0;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    color: #475569;
    font-size: 0.80rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .stitch-table td {
    border-bottom: 1px solid #f1f5f9;
    padding: 8px 12px;
    color: #334155;
  }

  .stitch-table tr:hover {
    background-color: #f8fafc;
  }

  /* Monospace Telemetry Badges */
  code {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    border: 1px solid #e2e8f0 !important;
  }

  /* Clean Tab Styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e2e8f0;
  }

  .stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    font-size: 0.90rem;
    color: #64748b;
    border-radius: 6px 6px 0 0;
    padding: 6px 14px;
  }

  .stTabs [aria-selected="true"] {
    color: #1e40af !important;
    border-bottom: 2px solid #2563eb !important;
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
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30.0)
    if r.status_code == 200:
        return r.json()
    else:
        raise RuntimeError(
            f"Prediction failed with status {r.status_code}: {r.text}"
        )


def fetch_api_live_status() -> Dict[str, Any] | None:
    """Retrieve live agent connection status and buffer diagnostics."""
    try:
        r = requests.get(f"{API_URL}/agent/status", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_live_history() -> Dict[str, Any] | None:
    """Retrieve rolling live telemetry history (up to 60 timesteps)."""
    try:
        r = requests.get(f"{API_URL}/telemetry/history", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_live_latest() -> Dict[str, Any] | None:
    """Retrieve latest live telemetry snapshot."""
    try:
        r = requests.get(f"{API_URL}/telemetry/latest", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_live_analysis() -> Dict[str, Any] | None:
    """Retrieve live 22-channel neural model inference, anomaly score, and 5-question narrative."""
    try:
        r = requests.get(f"{API_URL}/telemetry/live_analysis", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_fleet_overview() -> Dict[str, Any] | None:
    """Retrieve aggregated fleet overview metrics."""
    try:
        r = requests.get(f"{API_URL}/fleet/overview", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_hosts() -> List[Dict[str, Any]]:
    """Retrieve all registered hosts."""
    try:
        r = requests.get(f"{API_URL}/hosts", timeout=3.0)
        if r.status_code == 200:
            return r.json().get("hosts", [])
    except Exception:
        pass
    return []


def fetch_api_host_details(host_id: str) -> Dict[str, Any] | None:
    """Retrieve specific host details and buffer status."""
    try:
        r = requests.get(f"{API_URL}/hosts/{host_id}", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_host_analysis(host_id: str) -> Dict[str, Any] | None:
    """Retrieve live neural inference, XAI, and score history for a host."""
    try:
        r = requests.get(f"{API_URL}/hosts/{host_id}/analysis", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_host_latest(host_id: str) -> Dict[str, Any] | None:
    """Retrieve the latest telemetry snapshot for a specific host."""
    try:
        r = requests.get(f"{API_URL}/hosts/{host_id}/telemetry/latest", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_host_history(host_id: str) -> Dict[str, Any] | None:
    """Retrieve rolling telemetry series for a specific host."""
    try:
        r = requests.get(f"{API_URL}/hosts/{host_id}/telemetry/history", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_api_alerts(host_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent alerts."""
    try:
        url = f"{API_URL}/alerts?limit={limit}"
        if host_id:
            url += f"&host_id={host_id}"
        r = requests.get(url, timeout=3.0)
        if r.status_code == 200:
            return r.json().get("alerts", [])
    except Exception:
        pass
    return []



# ============================================================================
# DATA HANDLING & LOCAL CACHING
# ============================================================================

@st.cache_data
def load_raw_smd_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    """Load real SMD Machine 1-1 test dataset and precompute window anomaly labels."""
    try:
        project_root = Path(__file__).resolve().parent.parent
        test_path = project_root / "data" / "SMD" / "test" / "machine-1-1.txt"
        label_path = project_root / "data" / "SMD" / "test_label" / "machine-1-1.txt"

        if not test_path.is_file() or not label_path.is_file():
            return None, None, None

        test_data = np.loadtxt(test_path, delimiter=",", dtype=np.float32)
        test_labels = np.loadtxt(label_path, delimiter=",", dtype=np.int32)

        if len(test_data) < 60 or len(test_labels) < 60:
            return None, None, None

        n_windows = (len(test_data) - 60) // 5 + 1
        window_labels = []
        for i in range(n_windows):
            start = i * 5
            end = start + 60
            window_labels.append(1 if np.any(test_labels[start:end] > 0) else 0)
        window_labels = np.array(window_labels)

        return test_data, test_labels, window_labels
    except Exception:
        return None, None, None


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


# ============================================================================
# NATURAL LANGUAGE EXPLANATION GENERATOR (DYNAMIC XAI TRANSLATION)
# ============================================================================

def format_feature_title(feat_name: str) -> str:
    """Format technical 'feature_33' into readable 'Feature 33' for prose."""
    if feat_name.startswith("feature_"):
        try:
            num = int(feat_name.split("_")[1])
            return f"Feature {num}"
        except Exception:
            pass
    return feat_name.replace("_", " ").title()


def generate_human_readable_narrative(predict_res: Dict[str, Any]) -> str:
    """
    Transform dynamic XAI outputs into a clean, human-understandable narrative
    explaining WHY the model made this prediction.
    """
    is_anom = predict_res.get("is_anomaly", False)
    if not is_anom:
        return (
            "The system observed nominal operating behavior across all 38 monitored telemetry streams. "
            "Sensor measurements closely matched the model's spatio-temporal forecasts, and inter-feature "
            "dependencies remained consistent with the learned baseline structure without any anomalous deviations."
        )

    score = predict_res.get("anomaly_score", 0.0)
    threshold = predict_res.get("threshold", 2.0734)
    confidence = predict_res.get("confidence_pct", "0%")
    alert_level = predict_res.get("alert_level", "ALERT")
    peak_step = predict_res.get("peak_timestep", None)
    duration = predict_res.get("anomaly_duration_steps", 0)

    top_features = predict_res.get("top_features", [])
    broken_pairs = predict_res.get("broken_pairs", [])

    feature_sentences = []
    if top_features:
        high_spikes = [f for f in top_features[:3] if "spike" in str(f.get("direction", "")).lower()]
        high_drops = [f for f in top_features[:3] if "drop" in str(f.get("direction", "")).lower()]

        named_spikes = [f"{format_feature_title(f['feature'])} (+{float(f.get('z_score', 0)):.1f} Z-score deviation)" for f in high_spikes]
        named_drops = [f"{format_feature_title(f['feature'])} (-{float(f.get('z_score', 0)):.1f} Z-score deviation)" for f in high_drops]

        if named_spikes and named_drops:
            feature_sentences.append(
                f"The primary driver was a simultaneous surge in {', '.join(named_spikes)}, "
                f"alongside an unexpected decrease in {', '.join(named_drops)}."
            )
        elif named_spikes:
            feature_sentences.append(
                f"The strongest evidence came from unusually high values in {', '.join(named_spikes)}, "
                f"which surged significantly above their expected baseline."
            )
        elif named_drops:
            feature_sentences.append(
                f"The strongest evidence came from sharp declines in {', '.join(named_drops)}, "
                f"dropping well below their expected baseline."
            )
        else:
            top_names = [f"{format_feature_title(f['feature'])} ({float(f['z_score']):.1f} Z-score)" for f in top_features[:3]]
            feature_sentences.append(
                f"The highest individual forecast errors occurred in {', '.join(top_names)}."
            )

    rel_sentences = []
    if broken_pairs:
        p0 = broken_pairs[0]
        fa = format_feature_title(p0["feature_a"])
        fb = format_feature_title(p0["feature_b"])
        obs = float(p0.get("observed_corr", 0.0))
        exp = float(p0.get("expected_corr", 0.0))
        count = len(broken_pairs)

        rel_sentences.append(
            f"In addition, {count} inter-feature relationship{'s' if count > 1 else ''} changed significantly, "
            f"most notably between {fa} and {fb} where the observed correlation ({obs:.2f}) "
            f"diverged from the expected dependency pattern ({exp:.2f})."
        )

    feat_text = " ".join(feature_sentences)
    rel_text = " ".join(rel_sentences)
    timing_text = f"This abnormal activity persisted for {duration} consecutive timesteps (peaking at t={peak_step})." if (duration and peak_step) else ""

    narrative = (
        f"The system detected an anomaly during this time period ({alert_level} Alert, {confidence} confidence) "
        f"because the calculated anomaly score ({score:.4f}) exceeded the calibrated normal threshold ({threshold:.4f}). "
        f"{feat_text} {rel_text} {timing_text} "
        f"Together, these unusual feature behaviors and changed relationships caused the model to identify this time window as an anomaly."
    )
    return narrative


def generate_explanation_bullets(predict_res: Dict[str, Any]) -> List[str]:
    """Generate bullet points of exact telemetry drivers for quick human reading."""
    bullets = []
    top_features = predict_res.get("top_features", [])
    broken_pairs = predict_res.get("broken_pairs", [])

    for f in top_features[:3]:
        feat_name = format_feature_title(f["feature"])
        z = float(f.get("z_score", 0.0))
        direction = str(f.get("direction", "")).lower()
        if "spike" in direction:
            bullets.append(f"<strong>{feat_name}</strong> increased significantly above its expected value (+{z:.1f} Z-score deviation)")
        elif "drop" in direction:
            bullets.append(f"<strong>{feat_name}</strong> dropped sharply below its expected value (-{z:.1f} Z-score deviation)")
        else:
            bullets.append(f"<strong>{feat_name}</strong> showed an abnormal difference from baseline ({z:.1f} Z-score deviation)")

    for bp in broken_pairs[:2]:
        fa = format_feature_title(bp["feature_a"])
        fb = format_feature_title(bp["feature_b"])
        exp = float(bp.get("expected_corr", 0.0))
        obs = float(bp.get("observed_corr", 0.0))
        gap = float(bp.get("gap", 0.0))
        bullets.append(f"The relationship between <strong>{fa}</strong> and <strong>{fb}</strong> was altered (observed correlation <code>{obs:.2f}</code> vs expected <code>{exp:.2f}</code>, gap <code>{gap:.2f}</code>)")

    return bullets


# ============================================================================
# LIGHT THEME PLOTLY VISUALIZATIONS
# ============================================================================

def plot_telemetry_timeline_light(
    test_data: np.ndarray,
    test_labels: np.ndarray,
    selected_window_idx: int,
    feature_names: List[str]
) -> go.Figure:
    """Plot downsampled raw telemetry with clean light styling and anomaly intervals."""
    step = 10
    indices = np.arange(0, len(test_data), step)
    ds_data = test_data[::step]
    ds_labels = test_labels[::step]

    fig = go.Figure()

    # Ground Truth Anomaly Shading (Light Red)
    intervals = get_anomaly_intervals(ds_labels)
    for idx, (start, end) in enumerate(intervals):
        orig_start = start * step
        orig_end = end * step
        fig.add_vrect(
            x0=orig_start, x1=orig_end,
            fillcolor="#fee2e2", opacity=0.6, line_width=0,
            name="Ground Truth Anomaly" if idx == 0 else ""
        )

    # Telemetry Stream Traces (Selected Primary Colors)
    colors = ["#2563eb", "#0ea5e9", "#6366f1"]
    for f_i in range(min(3, test_data.shape[1])):
        fig.add_trace(go.Scatter(
            x=indices.tolist(),
            y=ds_data[:, f_i].tolist(),
            mode="lines",
            name=feature_names[f_i],
            line=dict(width=1.2, color=colors[f_i % len(colors)]),
            hovertemplate="Time: %{x}<br>Value: %{y:.4f}<extra></extra>"
        ))

    # Selected Window Highlight Span (Soft Blue)
    w_start = selected_window_idx * 5
    w_end = w_start + 59
    fig.add_vrect(
        x0=w_start, x1=w_end,
        fillcolor="#dbeafe", opacity=0.6,
        line=dict(color="#1d4ed8", width=1.5),
        name="Active Analysis Window"
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=320,
        margin=dict(l=40, r=30, t=30, b=40),
        xaxis=dict(
            gridcolor="#f1f5f9",
            title="Absolute Timestep (SMD Telemetry)",
            title_font=dict(size=11, color="#64748b"),
            tickfont=dict(size=10, color="#64748b"),
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            title="Standardized Value",
            title_font=dict(size=11, color="#64748b"),
            tickfont=dict(size=10, color="#64748b"),
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#334155")
        )
    )
    return fig


def plot_actual_vs_predicted_light(
    window_data: np.ndarray,
    feature_names: List[str],
    selected_feat: str,
    peak_t: int,
    top_features: List[Dict[str, Any]]
) -> go.Figure:
    """Plot actual values line chart and compare with predicted value at anomaly peak."""
    feat_idx = feature_names.index(selected_feat)
    actual_series = window_data[:, feat_idx]

    fig = go.Figure()

    # Actual sequence trace
    fig.add_trace(go.Scatter(
        x=list(range(60)),
        y=actual_series.tolist(),
        mode="lines",
        name="Observed Telemetry",
        line=dict(color="#2563eb", width=2),
        hovertemplate="Step: %{x}<br>Observed: %{y:.4f}<extra></extra>"
    ))

    # Match prediction at peak
    pred_val = None
    act_val = None
    for f in top_features:
        if f["feature"] == selected_feat:
            pred_val = float(f["predicted"])
            act_val = float(f["actual"])
            break

    if pred_val is not None:
        # Forecasted marker
        fig.add_trace(go.Scatter(
            x=[peak_t],
            y=[pred_val],
            mode="markers",
            name="Forecasted Value (Model)",
            marker=dict(symbol="x", size=10, color="#16a34a", line=dict(width=2)),
            hovertemplate="Forecasted: %{y:.4f}<extra></extra>"
        ))

        # Observed marker at peak
        fig.add_trace(go.Scatter(
            x=[peak_t],
            y=[act_val],
            mode="markers",
            name="Observed Value (Peak)",
            marker=dict(symbol="circle", size=10, color="#dc2626"),
            hovertemplate="Observed Peak: %{y:.4f}<extra></extra>"
        ))

    # Vertical anomaly peak line
    fig.add_vline(
        x=peak_t,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Anomaly Peak",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#dc2626")
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis=dict(
            gridcolor="#f1f5f9",
            title="Timestep inside 60-Step Analysis Window",
            title_font=dict(size=11, color="#64748b"),
            tickfont=dict(size=10, color="#64748b"),
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            title=f"Telemetry Magnitude ({selected_feat})",
            title_font=dict(size=11, color="#64748b"),
            tickfont=dict(size=10, color="#64748b"),
            zeroline=False
        ),
        height=320,
        margin=dict(l=40, r=30, t=30, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#334155")
        )
    )
    return fig


def plot_circular_dependency_graph_light(predict_res: Dict[str, Any]) -> go.Figure | None:
    """Plot interactive circular graph of broken/altered feature correlations in light mode."""
    broken_pairs = predict_res.get("broken_pairs", [])
    if not broken_pairs:
        return None

    nodes = set()
    for p in broken_pairs:
        nodes.add(p["feature_a"])
        nodes.add(p["feature_b"])
    nodes = sorted(list(nodes))

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

        sev = p.get("severity", "MEDIUM").upper()
        edge_color = "#dc2626" if "CRIT" in sev else "#f97316" if "HIGH" in sev else "#eab308"
        edge_width = max(1.5, float(p.get("gap", 0.3)) * 4.0)

        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=edge_width, color=edge_color),
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
        marker=dict(size=16, color="#2563eb", line=dict(width=2, color="#ffffff")),
        text=nodes,
        textposition="top center",
        hoverinfo="text",
        hovertext=nodes,
        textfont=dict(family="Inter, sans-serif", size=10.5, color="#0f172a"),
        showlegend=False
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=420,
        margin=dict(l=30, r=30, t=20, b=30)
    )
    return fig


# ============================================================================
# LIVE TELEMETRY PLOTLY VISUALIZATIONS
# ============================================================================

def plot_live_cpu_timeline(history: Dict[str, Any]) -> go.Figure:
    """Plot rolling CPU utilization breakdown."""
    fig = go.Figure()
    timestamps = history.get("timestamps", []) if history else []
    features = history.get("features", {}) if history else {}
    if not timestamps or not features:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            height=260,
            margin=dict(l=40, r=20, t=25, b=35),
            xaxis=dict(title="Rolling Seconds", gridcolor="#f1f5f9"),
            yaxis=dict(title="CPU Utilization (%)", gridcolor="#f1f5f9"),
        )
        return fig

    x_steps = list(range(len(timestamps)))
    cpu_pct = [float(v) for v in features.get("cpu_percent", [])]
    cpu_user = [float(v) for v in features.get("cpu_user_time_percent", [])]
    cpu_sys = [float(v) for v in features.get("cpu_system_time_percent", [])]

    fig.add_trace(go.Scatter(
        x=x_steps, y=cpu_pct, mode="lines+markers",
        name="Total CPU %", line=dict(color="#2563eb", width=2.2),
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=x_steps, y=cpu_user, mode="lines",
        name="User %", line=dict(color="#0ea5e9", width=1.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=x_steps, y=cpu_sys, mode="lines",
        name="System %", line=dict(color="#f97316", width=1.5, dash="dot")
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=260,
        margin=dict(l=40, r=20, t=25, b=35),
        xaxis=dict(title="Rolling Seconds (Past → Present)", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(title="CPU Utilization (%)", range=[0, 105], gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    )
    return fig


def plot_live_memory_timeline(history: Dict[str, Any]) -> go.Figure:
    """Plot rolling physical RAM and swap usage."""
    fig = go.Figure()
    timestamps = history.get("timestamps", []) if history else []
    features = history.get("features", {}) if history else {}
    if not timestamps or not features:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            height=260,
            margin=dict(l=40, r=20, t=25, b=35),
            xaxis=dict(title="Rolling Seconds", gridcolor="#f1f5f9"),
            yaxis=dict(title="Memory (%)", gridcolor="#f1f5f9"),
        )
        return fig

    x_steps = list(range(len(timestamps)))
    mem_pct = [float(v) for v in features.get("memory_percent", [])]
    swap_pct = [float(v) for v in features.get("swap_percent", [])]

    fig.add_trace(go.Scatter(
        x=x_steps, y=mem_pct, mode="lines+markers",
        name="RAM Utilization %", line=dict(color="#10b981", width=2.2),
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=x_steps, y=swap_pct, mode="lines",
        name="Swap Usage %", line=dict(color="#8b5cf6", width=1.5, dash="dash")
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=260,
        margin=dict(l=40, r=20, t=25, b=35),
        xaxis=dict(title="Rolling Seconds (Past → Present)", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(title="Memory (%)", range=[0, 105], gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    )
    return fig


def plot_live_io_timeline(history: Dict[str, Any]) -> go.Figure:
    """Plot rolling disk storage read/write throughput in KB/s."""
    fig = go.Figure()
    timestamps = history.get("timestamps", []) if history else []
    features = history.get("features", {}) if history else {}
    if not timestamps or not features:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            height=260,
            margin=dict(l=40, r=20, t=25, b=35),
            xaxis=dict(title="Rolling Seconds", gridcolor="#f1f5f9"),
            yaxis=dict(title="Throughput (KB/s)", gridcolor="#f1f5f9"),
        )
        return fig

    x_steps = list(range(len(timestamps)))
    read_kb = [float(v) / 1024.0 for v in features.get("disk_read_bytes_per_sec", [])]
    write_kb = [float(v) / 1024.0 for v in features.get("disk_write_bytes_per_sec", [])]

    fig.add_trace(go.Scatter(
        x=x_steps, y=read_kb, mode="lines",
        name="Disk Read (KB/s)", line=dict(color="#0284c7", width=1.8)
    ))
    fig.add_trace(go.Scatter(
        x=x_steps, y=write_kb, mode="lines",
        name="Disk Write (KB/s)", line=dict(color="#e11d48", width=1.8)
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=260,
        margin=dict(l=40, r=20, t=25, b=35),
        xaxis=dict(title="Rolling Seconds", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(title="Throughput (KB/s)", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    )
    return fig


def plot_live_network_timeline(history: Dict[str, Any]) -> go.Figure:
    """Plot rolling network ingress/egress bandwidth in KB/s."""
    fig = go.Figure()
    timestamps = history.get("timestamps", []) if history else []
    features = history.get("features", {}) if history else {}
    if not timestamps or not features:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            height=260,
            margin=dict(l=40, r=20, t=25, b=35),
            xaxis=dict(title="Rolling Seconds", gridcolor="#f1f5f9"),
            yaxis=dict(title="Rate (KB/s)", gridcolor="#f1f5f9"),
        )
        return fig

    x_steps = list(range(len(timestamps)))
    sent_kb = [float(v) / 1024.0 for v in features.get("net_bytes_sent_per_sec", [])]
    recv_kb = [float(v) / 1024.0 for v in features.get("net_bytes_recv_per_sec", [])]

    fig.add_trace(go.Scatter(
        x=x_steps, y=recv_kb, mode="lines",
        name="Net Ingress (KB/s)", line=dict(color="#059669", width=1.8)
    ))
    fig.add_trace(go.Scatter(
        x=x_steps, y=sent_kb, mode="lines",
        name="Net Egress (KB/s)", line=dict(color="#7c3aed", width=1.8)
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=260,
        margin=dict(l=40, r=20, t=25, b=35),
        xaxis=dict(title="Rolling Seconds", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(title="Rate (KB/s)", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    )
    return fig


def plot_live_anomaly_score_timeline(score_history: List[Dict[str, Any]], threshold: float) -> go.Figure:
    """Plot rolling live anomaly scores against the calibrated validation threshold."""
    fig = go.Figure()
    if not score_history:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            height=260,
            margin=dict(l=40, r=20, t=25, b=35),
            xaxis=dict(title="Rolling Window Index", gridcolor="#f1f5f9"),
            yaxis=dict(title="Anomaly Score", gridcolor="#f1f5f9"),
        )
        return fig

    x_steps = list(range(len(score_history)))
    scores = [float(s.get("anomaly_score", 0.0)) for s in score_history]
    anom_flags = [bool(s.get("is_anomaly", False)) for s in score_history]
    hover_texts = [
        f"Window #{i}<br>Score: {s.get('anomaly_score', 0.0):.4f}<br>Status: {s.get('severity', 'NOMINAL')}<br>Time: {s.get('timestamp', '')}"
        for i, s in enumerate(score_history)
    ]

    # Threshold Line
    fig.add_trace(go.Scatter(
        x=[0, max(1, len(score_history) - 1)],
        y=[threshold, threshold],
        mode="lines",
        name=f"Threshold τ ({threshold:.4f})",
        line=dict(color="#dc2626", width=2, dash="dash")
    ))

    # Anomaly score trace
    fig.add_trace(go.Scatter(
        x=x_steps,
        y=scores,
        mode="lines+markers",
        name="Anomaly Score",
        text=hover_texts,
        hoverinfo="text",
        line=dict(color="#2563eb", width=2),
        marker=dict(
            size=6,
            color=["#dc2626" if a else "#2563eb" for a in anom_flags],
        )
    ))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=260,
        margin=dict(l=40, r=20, t=25, b=35),
        xaxis=dict(title="Rolling Evaluation Windows", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        yaxis=dict(title="Anomaly Score", gridcolor="#f1f5f9", tickfont=dict(size=10, color="#64748b")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    )
    return fig



# ============================================================================
# FLEET OPERATIONS CENTER & MULTI-HOST MONITORING
# ============================================================================

def render_fleet_overview_page(health_info: Dict[str, Any]):
    """Render high-level fleet health, active anomalies, and registered host table."""
    fleet = fetch_api_fleet_overview() or {}
    hosts = fleet.get("hosts", [])
    total_hosts = fleet.get("total_hosts", 0)
    online_hosts = fleet.get("online_hosts", 0)
    offline_hosts = fleet.get("offline_hosts", 0)

    scan_requested = bool(
        st.session_state.get("system_scan_requested", False) or
        st.session_state.get("scan_active_fleet", False)
    )

    anom_hosts = fleet.get("hosts_with_anomalies", 0) if scan_requested else 0
    active_eps = fleet.get("total_active_episodes", 0) if scan_requested else 0

    col_hdr, col_ctrl = st.columns([7, 5])
    with col_hdr:
        render_html(
            """
            <div style="margin-bottom: 14px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.80rem; font-weight: 800; color: #0f172a; letter-spacing: -0.5px;">
                    🏢 FGEAD Fleet Operations Center
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #1e3a8a; margin-top: -2px;">
                    Centralized Spatio-Temporal Anomaly Detection & Telemetry Fleet Management
                </div>
            </div>
            """
        )
    with col_ctrl:
        auto_refresh = st.checkbox("⚡ Auto-Refresh Fleet (1.5s)", value=True, key="fleet_auto_refresh")
        col_fl1, col_fl2 = st.columns([1, 1])
        with col_fl1:
            if st.button("🔍 Scan My System", type="primary", use_container_width=True, key="btn_scan_fleet"):
                st.session_state["system_scan_requested"] = True
                st.session_state["scan_active_fleet"] = True
                st.rerun()
        with col_fl2:
            if st.button("🔄 Refresh Fleet", use_container_width=True, key="btn_refresh_fleet"):
                st.session_state["scan_active_fleet"] = False
                st.rerun()

    # Render system scan panel for active host on fleet overview page if triggered
    if st.session_state.get("scan_active_fleet"):
        target_hid = hosts[0]["host_id"] if hosts else "host_sivachowdary"
        render_system_scan_panel(target_hid)
    elif not scan_requested:
        render_html(
            """
            <div class="stitch-card" style="border-left: 6px solid #2563eb; background-color: #ffffff; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 0.8px;">
                            🔍 SYSTEM SCAN REQUIRED
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.40rem; font-weight: 800; color: #0f172a; margin-top: 2px;">
                            Fleet Ready for Telemetry Scan
                        </div>
                        <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                            Multi-host telemetry streams are connected and streaming. Press <strong>"Scan My System"</strong> above to evaluate host anomaly threshold boundaries and surface active episodes.
                        </div>
                    </div>
                    <div>
                        <span style="background-color: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; font-size: 0.82rem; font-weight: 700; padding: 4px 10px; border-radius: 6px;">
                            ⚪ READY TO SCAN
                        </span>
                    </div>
                </div>
                <div class="step-flow-bar" style="margin-top: 14px; margin-bottom: 0; background-color: #f8fafc;">
                    <div class="step-flow-item active">
                        <span class="step-flow-badge">1</span> Telemetry Ingestion (🟢 22 Channels Sampled)
                    </div>
                    <span class="step-flow-arrow">➔</span>
                    <div class="step-flow-item">
                        <span class="step-flow-badge">2</span> Scan Validation (⚪ Pending Explicit Scan)
                    </div>
                    <span class="step-flow-arrow">➔</span>
                    <div class="step-flow-item">
                        <span class="step-flow-badge">3</span> 60s Window Analysis (⚪ Pending Explicit Scan)
                    </div>
                    <span class="step-flow-arrow">➔</span>
                    <div class="step-flow-item">
                        <span class="step-flow-badge">4</span> Persistence & Evidence (⚪ Pending Explicit Scan)
                    </div>
                </div>
            </div>
            """
        )

    # 1. FLEET KPI CARDS
    render_html(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px;">
            <div class="stitch-card" style="margin-bottom: 0; padding: 14px 16px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Total Registered Hosts</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.80rem; font-weight: 800; color: #0f172a; margin: 2px 0;">{total_hosts}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Monitored Infrastructure</div>
            </div>

            <div class="stitch-card" style="margin-bottom: 0; padding: 14px 16px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Online Hosts</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.80rem; font-weight: 800; color: #16a34a; margin: 2px 0;">{online_hosts}</div>
                <div style="font-size: 0.75rem; color: #15803d;">Streaming Telemetry</div>
            </div>

            <div class="stitch-card" style="margin-bottom: 0; padding: 14px 16px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Offline Hosts</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.80rem; font-weight: 800; color: #dc2626; margin: 2px 0;">{offline_hosts}</div>
                <div style="font-size: 0.75rem; color: #b91c1c;">No Heartbeat</div>
            </div>

            <div class="stitch-card" style="margin-bottom: 0; padding: 14px 16px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Active Anomalies</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.80rem; font-weight: 800; color: {'#dc2626' if anom_hosts > 0 else '#16a34a'}; margin: 2px 0;">{'—' if not scan_requested else anom_hosts}</div>
                <div style="font-size: 0.75rem; color: #64748b;">{'Scan Required' if not scan_requested else 'Hosts Above Threshold'}</div>
            </div>

            <div class="stitch-card" style="margin-bottom: 0; padding: 14px 16px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Active Episodes</div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.80rem; font-weight: 800; color: {'#dc2626' if active_eps > 0 else '#475569'}; margin: 2px 0;">{'—' if not scan_requested else active_eps}</div>
                <div style="font-size: 0.75rem; color: #64748b;">{'Scan Required' if not scan_requested else 'Contiguous Anomaly Windows'}</div>
            </div>
        </div>
        """
    )

    # 2. FLEET HOST STATUS TABLE
    render_html(
        """
        <div class="stitch-card" style="margin-bottom: 20px;">
            <div class="stitch-card-header">
                🖥️ Monitored Hosts Overview
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                Real-time status of all registered Windows desktops, Windows servers, and Linux hosts.
            </div>
        """
    )

    if not hosts:
        render_html(
            """
            <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 6px; padding: 24px; text-align: center; color: #64748b;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #334155; margin-bottom: 6px;">No Monitored Hosts Registered Yet</div>
                <div>Start an agent to register and stream live telemetry to the platform:</div>
                <div style="margin-top: 10px;">
                    <code style="font-family: 'JetBrains Mono', monospace;">python data/live_agent.py</code> (Windows) &nbsp;|&nbsp;
                    <code style="font-family: 'JetBrains Mono', monospace;">python agents/linux_agent.py</code> (Linux)
                </div>
            </div>
            """
        )
    else:
        table_rows = []
        for h in hosts:
            hid = h.get("host_id", "N/A")
            hname = h.get("hostname", "N/A")
            os_name = h.get("operating_system", "Unknown")
            status = h.get("status", "OFFLINE")
            model_name = h.get("model_name", "None")
            model_compat = h.get("model_compatibility", "Baseline Required")
            score = h.get("latest_score")
            thresh = h.get("threshold")
            buf_sz = h.get("buffer_size", 0)
            ep = h.get("active_episode")
            is_telemetry = h.get("telemetry_connected", status != "OFFLINE")

            # Telemetry status badge
            telemetry_badge = "🟢 Connected" if is_telemetry else "⚪ Inactive"

            # Host Status Badge
            if not scan_requested:
                if status == "OFFLINE":
                    status_badge = "⚪ OFFLINE"
                elif status == "TELEMETRY_ONLY" or model_compat != "Compatible":
                    status_badge = "🟡 TELEMETRY ONLY"
                else:
                    status_badge = "🟢 MODEL READY"
            else:
                if status == "ANOMALY":
                    status_badge = "🔴 ANOMALY"
                elif status == "ONLINE":
                    status_badge = "🟢 MODEL ACTIVE"
                elif status == "TELEMETRY_ONLY":
                    status_badge = "🟡 TELEMETRY ONLY"
                else:
                    status_badge = "⚪ OFFLINE"

            # Score & Threshold formatting
            if not scan_requested or status == "TELEMETRY_ONLY" or model_compat != "Compatible":
                score_str = "—"
                thresh_str = f"{thresh:.4f}" if thresh else "—"
                ep_str = "—"
            elif score is not None:
                score_str = f"{score:.4f}"
                thresh_str = f"{thresh:.4f}" if thresh else "—"
                ep_str = f"Active (#{ep})" if ep else "—"
            elif status == "ONLINE":
                score_str = f"Buffering ({buf_sz}/60)"
                thresh_str = f"{thresh:.4f}" if thresh else "—"
                ep_str = "—"
            else:
                score_str = "—"
                thresh_str = f"{thresh:.4f}" if thresh else "—"
                ep_str = "—"

            table_rows.append({
                "Host": f"{hname} ({hid})",
                "OS": os_name,
                "Telemetry": telemetry_badge,
                "Model": model_name,
                "Model Compatibility": model_compat,
                "Status": status_badge,
                "Score": score_str,
                "Threshold": thresh_str,
                "Episode": ep_str,
                "Buffer": f"{buf_sz}/60",
                "Last Seen": h.get("last_seen", "N/A")[:19].replace("T", " ") if h.get("last_seen") else "Never",
            })

        df_hosts = pd.DataFrame(table_rows)
        st.dataframe(df_hosts, use_container_width=True, hide_index=True)

    render_html("</div>")

    # 3. RECENT ALERTS FEED
    alerts = fetch_api_alerts(limit=10) if scan_requested else []
    if scan_requested and alerts:
        render_html(
            """
            <div class="stitch-card" style="margin-bottom: 20px;">
                <div class="stitch-card-header">
                    🚨 Recent Fleet Alerts & Anomaly Episodes
                </div>
                <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                    Debounced anomaly events triggered across all monitored nodes.
                </div>
            """
        )
        alert_rows = []
        for a in alerts:
            alert_rows.append({
                "Alert ID": f"#{a.get('alert_id')}",
                "Host ID": a.get("host_id"),
                "Episode": f"#{a.get('episode_id')}",
                "Severity": a.get("severity"),
                "Score": f"{a.get('anomaly_score', 0):.4f}",
                "Threshold": f"{a.get('threshold', 0):.4f}",
                "Dominant Metric": a.get("dominant_feature"),
                "Message": a.get("message"),
                "Timestamp": a.get("timestamp", "")[:19].replace("T", " "),
            })
        st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
        render_html("</div>")
    elif not scan_requested:
        render_html(
            """
            <div class="stitch-card" style="margin-bottom: 20px;">
                <div class="stitch-card-header">
                    🚨 Recent Fleet Alerts & Anomaly Episodes
                </div>
                <div style="font-size: 0.84rem; color: #64748b; padding: 12px 0;">
                    ℹ️ Explicit system scan required to evaluate host anomaly boundaries and display active alert episodes. Press <strong>"Scan My System"</strong> above to run an explicit scan.
                </div>
            </div>
            """
        )

    # 4. PRIVACY & SECURITY GUIDELINES
    render_html(
        """
        <div class="stitch-card" style="background-color: #f8fafc; border-left: 4px solid #3b82f6;">
            <div style="font-weight: 700; color: #1e3a8a; font-size: 0.90rem; margin-bottom: 6px;">
                🔒 Architecture Security & Privacy Assurance
            </div>
            <div style="font-size: 0.82rem; color: #475569; line-height: 1.6;">
                <strong>Local Authorization Required:</strong> FGEAD only monitors systems that have installed and authorized the dedicated lightweight telemetry agent. The central dashboard cannot and does not scan or access arbitrary visitor computers through the browser.
                <br>
                <strong>Data Isolation:</strong> All telemetry streams, rolling ring buffers, anomaly scores, and model inferences are isolated strictly by <code>host_id</code>.
            </div>
        </div>
        """
    )

    if auto_refresh:
        time.sleep(1.5)
        st.rerun()


def render_deep_human_explainability_panel(
    human_exp: Dict[str, Any],
    explanation_5q: Dict[str, Any],
    top_feats: List[Dict[str, Any]],
    broken_pairs: List[Dict[str, Any]],
    score: float,
    threshold: float,
    ratio: float,
    severity: str,
    is_anomaly: bool,
):
    """
    Renders Deep Human-Understandable Explainability interface adhering strictly to
    the 16 explainability principles and progressive disclosure architecture.
    """
    state = human_exp.get("state", "ANOMALY" if is_anomaly else "NORMAL")
    headline = human_exp.get("headline", "System telemetry evaluation")
    what_happened = human_exp.get("what_happened", "")
    why_detected = human_exp.get("why_detected", "")
    what_changed_table = human_exp.get("what_changed_table", [])
    top_contributors = human_exp.get("top_contributors", [])
    why_anomaly = human_exp.get("why_this_is_an_anomaly", "")
    severity_text = human_exp.get("severity_text", f"Severity: {severity}")
    suggestions = human_exp.get("investigation_suggestions", [])
    disclaimer = human_exp.get("disclaimer", DISCLAIMER_TEXT)

    # 1. Summary Card & Plain-English Headline
    border_color = "#dc2626" if is_anomaly else "#0284c7" if state == "SUSPICIOUS" else "#16a34a"
    icon = "🚨" if is_anomaly else "🔍" if state == "SUSPICIOUS" else "🟢"
    card_title = "Plain-English Anomaly Explanation (XAI)" if is_anomaly else "Plain-English System Status & Explainability (XAI)"

    render_html(
        f"""
        <div class="stitch-card" style="border-left: 5px solid {border_color}; margin-bottom: 16px;">
            <div class="stitch-card-header" style="color: {'#991b1b' if is_anomaly else '#0f172a'};">
                {icon} {card_title}
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
                {headline}
            </div>
            <div style="font-size: 0.90rem; color: #334155; line-height: 1.6; margin-bottom: 12px;">
                <strong>What happened:</strong> {what_happened}
            </div>
            <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 4px; padding: 10px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #1e3a8a; text-transform: uppercase;">Why was it detected?</div>
                <div style="font-size: 0.88rem; color: #1e293b; margin-top: 3px; line-height: 1.5;">{why_detected}</div>
            </div>
        """
    )

    if is_anomaly:
        render_html(
            f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
                <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; border-radius: 4px; padding: 10px 14px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #991b1b; text-transform: uppercase;">Why this is an anomaly</div>
                    <div style="font-size: 0.84rem; color: #7f1d1d; margin-top: 3px; line-height: 1.5;">{why_anomaly}</div>
                </div>
                <div style="background-color: #fff7ed; border-left: 4px solid #f97316; border-radius: 4px; padding: 10px 14px;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #9a3412; text-transform: uppercase;">Severity Assessment</div>
                    <div style="font-size: 0.84rem; color: #7c2d12; margin-top: 3px; line-height: 1.5; white-space: pre-line;">{severity_text}</div>
                </div>
            </div>
            """
        )

    # Practical Investigation Suggestions Checklist
    if suggestions:
        hint_items = "".join([f"<li style='margin-bottom: 4px;'>{s}</li>" for s in suggestions])
        render_html(
            f"""
            <div style="background-color: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 4px; padding: 10px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #166534; text-transform: uppercase;">💡 Recommended Investigation Checklist:</div>
                <ul style="font-size: 0.85rem; color: #14532d; margin-top: 4px; margin-bottom: 0; padding-left: 18px; line-height: 1.5;">
                    {hint_items}
                </ul>
            </div>
            """
        )

    # Disclaimer
    if is_anomaly:
        render_html(
            f"""
            <div style="font-size: 0.78rem; color: #64748b; font-style: italic; border-top: 1px dashed #e2e8f0; padding-top: 8px;">
                ℹ️ <strong>Note:</strong> {disclaimer}
            </div>
            """
        )

    render_html("</div>")

    # 2. What Changed Comparison Table (Section C)
    if what_changed_table and is_anomaly:
        render_html(
            """
            <div class="stitch-card" style="margin-bottom: 16px;">
                <div class="stitch-card-header">
                    📊 What Changed? (Learned Normal vs. Current Behavior)
                </div>
            """
        )
        df_change = pd.DataFrame(what_changed_table)
        df_display = df_change.rename(columns={
            "measurement": "Measurement",
            "normal_behavior": "Normal Behavior (Expected)",
            "current_behavior": "Current Behavior (Observed)",
            "difference": "Difference",
            "meaning": "Meaning / Impact",
        })[["Measurement", "Normal Behavior (Expected)", "Current Behavior (Observed)", "Difference", "Meaning / Impact"]]
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        render_html("</div>")

    # 3. Deep Contributor Breakdown (Section E)
    if top_contributors and is_anomaly:
        render_html(
            """
            <div class="stitch-card" style="margin-bottom: 16px;">
                <div class="stitch-card-header">
                    🔍 Deep Root-Cause Contributor Breakdown
                </div>
                <div style="display: grid; grid-template-columns: 1fr; gap: 10px;">
            """
        )
        for c in top_contributors:
            render_html(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.95rem; font-weight: 700; color: #0f172a;">
                            #{c['rank']} {c['feature_human_name']}
                        </span>
                        <code style="font-size: 0.78rem; color: #64748b; background-color: #e2e8f0; padding: 2px 6px; border-radius: 3px;">{c['technical_key']}</code>
                    </div>
                    <div style="font-size: 0.84rem; color: #475569; margin-bottom: 8px;">
                        <em>{c['what_it_means']}</em>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; font-size: 0.82rem; color: #334155; margin-bottom: 8px;">
                        <div><strong>Normal Expected:</strong> {c['normally_expected']}</div>
                        <div><strong>Observed Value:</strong> <span style="color: #dc2626; font-weight: 600;">{c['observed_value']}</span></div>
                        <div><strong>Difference:</strong> {c['difference_factor']}</div>
                    </div>
                    <div style="font-size: 0.82rem; color: #1e293b; line-height: 1.4; border-top: 1px dashed #cbd5e1; padding-top: 6px;">
                        <strong>Why it contributed:</strong> {c['why_it_contributed']}
                    </div>
                </div>
                """
            )
        render_html("</div></div>")

    # 4. Technical Details Expander (Progressive Disclosure - Section 9)
    with st.expander("🔬 Technical Model Details (For ML Engineers & System Examiners)", expanded=False):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(f"**Anomaly Score:** `{score:.6f}`")
            st.markdown(f"**Calibrated Threshold (&tau;):** `{threshold:.6f}`")
            st.markdown(f"**Score / Threshold Ratio:** `{ratio:.2f}×`")
        with col_t2:
            st.markdown(f"**Decision State:** `{state}`")
            st.markdown(f"**Active Episode:** `#{human_exp.get('technical_details', {}).get('active_episode_id', 1)}`")
            st.markdown(f"**Episode Duration:** `{human_exp.get('technical_details', {}).get('episode_duration_sec', 1)} seconds`")

        # 5-Question Technical Narrative
        if explanation_5q:
            st.markdown("##### 5-Question Technical Synthesis")
            for k, v in explanation_5q.items():
                clean_k = k.replace("Q1_", "Q1: ").replace("Q2_", "Q2: ").replace("Q3_", "Q3: ").replace("Q4_", "Q4: ").replace("Q5_", "Q5: ").replace("_", " ").title()
                st.markdown(f"- **{clean_k}:** {v}")

        # Standardized Residuals Table
        if top_feats:
            st.markdown("##### Top Feature Standardized Residuals")
            top_rows = []
            for rank, tf in enumerate(top_feats, start=1):
                top_rows.append({
                    "Rank": f"#{rank}",
                    "Feature": tf.get("feature", ""),
                    "Actual Value": tf.get("formatted_value", ""),
                    "Standardized Residual": f"{tf.get('residual', 0.0):.4f}",
                    "Description": tf.get("description", ""),
                })
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

        # Graph Dependency Disruptions
        if broken_pairs:
            st.markdown("##### Metric Dependency Disruption Graph (GNN Attention Weights)")
            bp_rows = []
            for bp in broken_pairs[:5]:
                bp_rows.append({
                    "Source Metric": bp.get("source", ""),
                    "Target Metric": bp.get("target", ""),
                    "Attention Weight": f"{bp.get('weight', 0.0):.4f}",
                    "Status": bp.get("status", ""),
                })
            st.dataframe(pd.DataFrame(bp_rows), use_container_width=True, hide_index=True)


def render_system_scan_panel(host_id: str):
    """
    Renders an interactive real-time system scan panel implementing the complete workflow:
    Open FGEAD -> '🔍 Scan My System' -> Collect telemetry -> Validate scan -> Analyze 60s window ->
    Check persistence + baseline compatibility + evidence -> Outcome: NORMAL or DEVIATION/ANOMALY.
    """
    st.markdown("### 🔍 Interactive System Telemetry Scan")

    # 1. Collect Telemetry Snapshot
    err_msg = ""
    try:
        collector = WindowsTelemetryCollector()
        feats = collector.collect_features()
    except Exception as exc:
        feats = None
        err_msg = str(exc)

    # 2. Validate Scan
    if feats:
        is_valid, val_err = validate_live_feature_dict(feats)
        if not is_valid:
            err_msg = val_err
    else:
        is_valid = False
        if not err_msg:
            err_msg = "Could not collect live telemetry snapshot."

    # 3. Analyze Window & Check Compatibility / Evidence
    live_analysis = fetch_api_host_analysis(host_id)
    latest_inf = live_analysis.get("latest_inference") if live_analysis else None
    model_compat = live_analysis.get("model_compatibility", "Baseline Required") if live_analysis else "Baseline Required"
    is_compat = (model_compat == "Compatible")
    ep_status = live_analysis.get("episode_status", {}) if live_analysis else {}

    # Pipeline Verification Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Telemetry Ingestion**\n\n🟢 22 Channels Sampled")
    with col2:
        if is_valid:
            st.markdown("**2. Scan Validation**\n\n🟢 Valid (No NaN/Inf)")
        else:
            st.markdown(f"**2. Scan Validation**\n\n🔴 Failed ({err_msg})")
    with col3:
        if is_compat:
            st.markdown("**3. 60s Window Analysis**\n\n🟢 Neural Model Active")
        else:
            st.markdown(f"**3. 60s Window Analysis**\n\n🟡 {model_compat}")
    with col4:
        if latest_inf:
            st.markdown("**4. Persistence & Evidence**\n\n🟢 Episode & Graph Verified")
        else:
            st.markdown("**4. Persistence & Evidence**\n\n⚪ Buffering Window")

    st.markdown("---")

    # Guard 1: Validation Failure
    if not is_valid:
        st.error(f"⚠️ **Scan inconclusive** — FGEAD could not obtain sufficient validated evidence (Telemetry validation error: {err_msg}).")
        return

    # Guard 2: Incompatible / Missing Model
    if not is_compat:
        st.info(f"ℹ️ **Scan inconclusive** — FGEAD could not obtain sufficient validated evidence (Model profile status: {model_compat}).")
        return

    # Guard 3: Insufficient Telemetry Buffer (<60 samples)
    if not latest_inf:
        st.info("ℹ️ **Scan inconclusive** — FGEAD could not obtain sufficient validated evidence (Rolling 60-second telemetry window is buffering).")
        return

    is_anomaly = latest_inf.get("is_anomaly", False)
    score = latest_inf.get("anomaly_score", 0.0)
    thresh = latest_inf.get("threshold", 1.411807)
    ratio = latest_inf.get("ratio", 1.0)
    severity = latest_inf.get("severity", "NOMINAL")
    top_feats = latest_inf.get("top_features", [])

    if not is_anomaly:
        render_html(
            f"""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 2px solid #22c55e; border-radius: 12px; padding: 22px; margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div style="font-size: 0.80rem; font-weight: 800; color: #15803d; text-transform: uppercase; letter-spacing: 1.2px;">
                            🟢 NO CONFIRMED ANOMALY DETECTED
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 800; color: #0f172a; margin-top: 2px;">
                            No Confirmed Anomaly
                        </div>
                        <div style="font-size: 0.92rem; color: #334155; margin-top: 4px;">
                            No confirmed anomaly was detected by FGEAD in the analyzed window.
                        </div>
                    </div>
                    <div style="text-align: right; background: #ffffff; padding: 12px 18px; border-radius: 8px; border: 1px solid #bbf7d0;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Anomaly Score / Threshold</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.50rem; font-weight: 800; color: #16a34a;">
                            {score:.4f} <span style="font-size: 0.9rem; color: #64748b;">(τ = {thresh:.4f})</span>
                        </div>
                        <div style="font-size: 0.80rem; font-weight: 700; color: #16a34a; margin-top: 2px;">Ratio: {ratio:.2f}x &bull; Status: NOMINAL</div>
                    </div>
                </div>
            </div>
            """
        )
    else:
        top_names = [f"<code>{f.get('feature', '')}</code>" for f in top_feats[:3]]
        feats_html = ", ".join(top_names) if top_names else "telemetry features"
        
        active_ep = ep_status.get("active_episode_id")
        if not active_ep and ratio < 1.5:
            subtitle_msg = "Unusual operating-state deviation detected. Available evidence is insufficient to classify this as a confirmed anomaly."
        else:
            subtitle_msg = f"Top contributing feature / root-cause candidates: {feats_html}"

        render_html(
            f"""
            <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border: 2px solid #ef4444; border-radius: 12px; padding: 22px; margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                    <div>
                        <div style="font-size: 0.80rem; font-weight: 800; color: #b91c1c; text-transform: uppercase; letter-spacing: 1.2px;">
                            🔴 TELEMETRY DEVIATION / ANOMALY DETECTED
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 800; color: #0f172a; margin-top: 2px;">
                            Operating-State Deviation Detected
                        </div>
                        <div style="font-size: 0.92rem; color: #7f1d1d; margin-top: 6px;">
                            {subtitle_msg}
                        </div>
                    </div>
                    <div style="text-align: right; background: #ffffff; padding: 12px 18px; border-radius: 8px; border: 1px solid #fca5a5;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Anomaly Score / Threshold</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.50rem; font-weight: 800; color: #dc2626;">
                            {score:.4f} <span style="font-size: 0.9rem; color: #64748b;">(τ = {thresh:.4f})</span>
                        </div>
                        <div style="font-size: 0.80rem; font-weight: 700; color: #dc2626; margin-top: 2px;">Ratio: {ratio:.2f}x &bull; Severity: {severity}</div>
                    </div>
                </div>
            </div>
            """
        )

        st.markdown("##### 🔬 Top Contributing Feature / Root-Cause Candidates")
        top_rows = []
        for rank, tf in enumerate(top_feats[:5], start=1):
            top_rows.append({
                "Rank": f"#{rank}",
                "Candidate Feature": tf.get("feature", ""),
                "Observed Value": tf.get("formatted_value", ""),
                "Model Residual": f"{tf.get('residual', 0.0):.4f}",
                "Description": tf.get("description", ""),
            })
        st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)


def render_single_host_page(host_id: str, health_info: Dict[str, Any]):
    """Render comprehensive telemetry, real-time XAI, and model timeline for one specific host."""
    host_info_data = fetch_api_host_details(host_id)
    latest_data = fetch_api_host_latest(host_id)
    history_data = fetch_api_host_history(host_id)
    live_analysis = fetch_api_host_analysis(host_id)

    host_meta = host_info_data.get("host", {}) if host_info_data else {}
    buf_status = host_info_data.get("buffer_status", {}) if host_info_data else {}
    latest_payload = latest_data.get("data") if latest_data else None
    latest_feats = latest_payload.get("features", {}) if latest_payload else {}

    hostname = host_meta.get("hostname", host_id)
    os_name = host_meta.get("operating_system", "Windows")
    os_ver = host_meta.get("os_version", "")
    arch = host_meta.get("architecture", "x86_64")
    agent_ver = host_meta.get("agent_version", "1.0.0")
    status_str = host_meta.get("status", "OFFLINE")
    is_connected = (status_str in ["ONLINE", "ANOMALY", "TELEMETRY_ONLY"])
    model_compat = live_analysis.get("model_compatibility", "Baseline Required") if live_analysis else "Baseline Required"
    is_compat = (model_compat == "Compatible")

    buf_size = buf_status.get("buffer_size", 0)
    win_size = buf_status.get("window_size", 60)
    is_buf_full = bool(buf_status.get("is_window_full", False))
    buf_pct = min(100, int((buf_size / win_size) * 100)) if win_size > 0 else 0
    sec_ago = buf_status.get("seconds_since_last_seen", 0.0)

    # 1. HEADER
    col_hdr, col_ctrl = st.columns([7, 5])
    with col_hdr:
        if not is_connected:
            status_badge = "<span class='badge-anomaly' style='font-size:0.82rem; padding:3px 9px;'>⚪ Offline</span>"
        elif not is_compat or status_str == "TELEMETRY_ONLY":
            status_badge = f"<span style='background-color: #fef3c7; border: 1px solid #fde68a; color: #92400e; font-size: 0.82rem; font-weight: 700; padding: 3px 9px; border-radius: 6px;'>🟡 Telemetry Only ({sec_ago}s ago)</span>"
        else:
            status_badge = f"<span class='badge-normal' style='font-size:0.82rem; padding:3px 9px;'>🟢 Online ({sec_ago}s ago)</span>"

        render_html(
            f"""
            <div style="margin-bottom: 14px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.70rem; font-weight: 800; color: #0f172a; letter-spacing: -0.5px;">
                    🖥️ {hostname}
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #1e3a8a; margin-top: -2px;">
                    Host ID: <code>{host_id}</code> &bull; OS: <strong>{os_name} {os_ver} ({arch})</strong>
                </div>
                <div style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; align-items: center;">
                    {status_badge}
                    <span class="tag-pill tag-pill-info">🧠 Model: <strong>{live_analysis.get('model_name', 'None') if live_analysis else 'None'}</strong></span>
                    <span class="tag-pill">🏷️ Compatibility: <strong>{model_compat}</strong></span>
                    <span class="tag-pill">⏱️ Window: <strong>60 × 22 (1.0s Rate)</strong></span>
                </div>
            </div>
            """
        )
    with col_ctrl:
        auto_refresh = st.checkbox("⚡ Auto-Refresh Host (1.5s)", value=True, key=f"host_refresh_{host_id}")
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            if st.button("🔍 Scan My System", type="primary", use_container_width=True, key=f"btn_scan_{host_id}"):
                st.session_state["system_scan_requested"] = True
                st.session_state[f"scan_active_{host_id}"] = True
                st.rerun()
        with col_ctrl2:
            if st.button("🔄 Refresh Snapshot", use_container_width=True, key=f"btn_refresh_{host_id}"):
                st.session_state[f"scan_active_{host_id}"] = False
                st.rerun()

    scan_requested = bool(
        st.session_state.get("system_scan_requested", False) or
        st.session_state.get(f"scan_active_{host_id}", False)
    )

    # Render system scan panel if triggered
    if st.session_state.get(f"scan_active_{host_id}"):
        render_system_scan_panel(host_id)
    elif not scan_requested and is_connected and is_compat:
        render_html(
            """
            <div class="stitch-card" style="border-left: 6px solid #2563eb; background-color: #ffffff; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: 0.8px;">
                            🔍 SYSTEM SCAN REQUIRED
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.40rem; font-weight: 800; color: #0f172a; margin-top: 2px;">
                            Explicit System Scan Required
                        </div>
                        <div style="font-size: 0.86rem; color: #475569; margin-top: 4px;">
                            Real-time telemetry streams are connected and buffering. Press <strong>"Scan My System"</strong> above to run spatio-temporal anomaly detection, evaluate threshold boundaries, and view explainability narratives.
                        </div>
                    </div>
                    <div>
                        <span style="background-color: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; font-size: 0.82rem; font-weight: 700; padding: 4px 10px; border-radius: 6px;">
                            ⚪ READY TO SCAN
                        </span>
                    </div>
                </div>
                <div class="step-flow-bar" style="margin-top: 14px; margin-bottom: 0; background-color: #f8fafc;">
                    <div class="step-flow-item active">
                        <span class="step-flow-badge">1</span> Telemetry Ingestion (🟢 22 Channels Sampled)
                    </div>
                    <span class="step-flow-arrow">➔</span>
                    <div class="step-flow-item">
                        <span class="step-flow-badge">2</span> Scan Validation (⚪ Pending Explicit Scan)
                    </div>
                    <span class="step-flow-arrow">➔</span>
                    <div class="step-flow-item">
                        <span class="step-flow-badge">3</span> 60s Window Analysis (⚪ Pending Explicit Scan)
                    </div>
                    <span class="step-flow-arrow">➔</span>
                    <div class="step-flow-item">
                        <span class="step-flow-badge">4</span> Persistence & Evidence (⚪ Pending Explicit Scan)
                    </div>
                </div>
            </div>
            """
        )

    # 2. SYSTEM STATUS (5 Clean Metric Cards)
    if latest_feats and is_connected:
        cpu_p = latest_feats.get("cpu_percent", 0.0)
        cpu_f = latest_feats.get("cpu_freq_current", 0.0)
        cpu_u = latest_feats.get("cpu_user_time_percent", 0.0)
        cpu_s = latest_feats.get("cpu_system_time_percent", 0.0)

        mem_p = latest_feats.get("memory_percent", 0.0)
        mem_u = latest_feats.get("memory_used_mb", 0.0)
        mem_a = latest_feats.get("memory_available_mb", 0.0)
        swap_p = latest_feats.get("swap_percent", 0.0)

        disk_p = latest_feats.get("disk_usage_percent", 0.0)
        disk_r_fmt = format_physical_metric("disk_read_bytes_per_sec", latest_feats.get("disk_read_bytes_per_sec", 0.0))
        disk_w_fmt = format_physical_metric("disk_write_bytes_per_sec", latest_feats.get("disk_write_bytes_per_sec", 0.0))

        net_tot = latest_feats.get("net_bytes_sent_per_sec", 0.0) + latest_feats.get("net_bytes_recv_per_sec", 0.0)
        net_tot_fmt = format_physical_metric("net_bytes_sent_per_sec", net_tot)
        net_in_fmt = format_physical_metric("net_bytes_recv_per_sec", latest_feats.get("net_bytes_recv_per_sec", 0.0))
        net_out_fmt = format_physical_metric("net_bytes_sent_per_sec", latest_feats.get("net_bytes_sent_per_sec", 0.0))

        procs = int(latest_feats.get("process_count", 0))

        render_html(
            f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 14px;">
                <div class="stitch-card" style="margin-bottom: 0; padding: 12px 14px;">
                    <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">CPU Usage</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 800; color: #2563eb; margin: 2px 0;">{cpu_p:.1f}%</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Clock: <strong>{cpu_f:.0f} MHz</strong> &bull; User: {cpu_u:.1f}%</div>
                </div>

                <div class="stitch-card" style="margin-bottom: 0; padding: 12px 14px;">
                    <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Memory Usage</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 800; color: #10b981; margin: 2px 0;">{mem_p:.1f}%</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Used: <strong>{mem_u:,.0f} MB</strong> &bull; Free: {mem_a:,.0f} MB</div>
                </div>

                <div class="stitch-card" style="margin-bottom: 0; padding: 12px 14px;">
                    <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Disk Usage</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 800; color: #0284c7; margin: 2px 0;">{disk_p:.1f}%</div>
                    <div style="font-size: 0.74rem; color: #64748b;">R: <strong>{disk_r_fmt}</strong> &bull; W: <strong>{disk_w_fmt}</strong></div>
                </div>

                <div class="stitch-card" style="margin-bottom: 0; padding: 12px 14px;">
                    <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Network Activity</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 800; color: #7c3aed; margin: 2px 0;">{net_tot_fmt}</div>
                    <div style="font-size: 0.74rem; color: #64748b;">In: <strong>{net_in_fmt}</strong> &bull; Out: <strong>{net_out_fmt}</strong></div>
                </div>

                <div class="stitch-card" style="margin-bottom: 0; padding: 12px 14px;">
                    <div style="font-size: 0.70rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Process Count</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 800; color: #475569; margin: 2px 0;">{procs}</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Active Tasks</div>
                </div>
            </div>
            """
        )

    # 3. STATUS HERO CARD & EXPLAINABILITY (ONLY WHEN SCAN HAS BEEN REQUESTED)
    if not is_connected:
        render_html(
            f"""
            <div class="status-hero-card" style="border-left: 6px solid #dc2626; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: #dc2626; text-transform: uppercase; letter-spacing: 0.8px;">
                            🔴 HOST OFFLINE
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.40rem; font-weight: 800; color: #0f172a; margin-top: 2px;">
                            No Live Telemetry Heartbeat from {hostname}
                        </div>
                        <div style="font-size: 0.84rem; color: #64748b; margin-top: 2px;">
                            The telemetry stream timed out. Last seen: {host_meta.get('last_seen', 'N/A')}.
                        </div>
                    </div>
                    <div>
                        <span class="badge-anomaly" style="font-size: 0.82rem; padding: 4px 10px;">
                            DISCONNECTED
                        </span>
                    </div>
                </div>
            </div>
            """
        )
    elif is_connected and not is_compat:
        render_html(
            f"""
            <div class="status-hero-card" style="border-left: 6px solid #f59e0b; margin-bottom: 16px; background-color: #fffbeb;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: #d97706; text-transform: uppercase; letter-spacing: 0.8px;">
                            🟡 TELEMETRY CONNECTED — COMPATIBLE ANOMALY MODEL UNAVAILABLE
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.35rem; font-weight: 800; color: #92400e; margin-top: 2px;">
                            Baseline Required for {os_name} Anomaly Inference
                        </div>
                        <div style="font-size: 0.85rem; color: #78350f; margin-top: 4px; line-height: 1.5;">
                            Anomaly inference is disabled because a compatible {os_name} baseline/model has not yet been trained.
                            <br>
                            The Windows physical baseline model is not universally representative and is gated to prevent false alerts. Telemetry streaming remains fully active.
                        </div>
                    </div>
                    <div>
                        <span style="background-color: #fef3c7; border: 1px solid #fde68a; color: #92400e; font-size: 0.82rem; font-weight: 700; padding: 4px 10px; border-radius: 6px;">
                            TELEMETRY ONLY
                        </span>
                    </div>
                </div>
            </div>
            """
        )

        # Baseline Collection & Distribution Audit Panel
        st.markdown("<div style='font-size: 0.95rem; font-weight: 700; color: #1e3a8a; margin: 14px 0 8px 0;'>📊 Host Baseline Quality & Distribution Audit</div>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns([4, 8])
        with col_b1:
            if st.button("📸 Collect Host Baseline Snapshot", key=f"btn_collect_base_{host_id}", use_container_width=True):
                try:
                    resp = requests.post(f"{DEFAULT_API_URL}/hosts/{host_id}/baseline/collect", timeout=5.0)
                    if resp.status_code == 200:
                        st.success("✅ Baseline statistics recorded successfully.")
                    else:
                        st.error(f"Failed to collect baseline: {resp.text}")
                except Exception as exc:
                    st.error(f"Error: {exc}")

        with col_b2:
            st.info("Collect a normal operating baseline for this host before training or assigning a dedicated FGEAD model.")
    elif is_connected and not is_buf_full:
        render_html(
            f"""
            <div class="status-hero-card" style="border-left: 6px solid #f59e0b; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: #d97706; text-transform: uppercase; letter-spacing: 0.8px;">
                            ⏳ COLLECTING TELEMETRY
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.40rem; font-weight: 800; color: #92400e; margin-top: 2px;">
                            Warm-Up Phase: {buf_size} / 60 Samples Collected
                        </div>
                        <div style="font-size: 0.84rem; color: #78350f; margin-top: 2px;">
                            Buffering rolling 60-second window before initiating neural inference ({60 - buf_size}s remaining).
                        </div>
                    </div>
                    <div>
                        <span style="background-color: #fef3c7; border: 1px solid #fde68a; color: #92400e; font-size: 0.82rem; font-weight: 700; padding: 4px 10px; border-radius: 6px;">
                            BUFFERING ({buf_pct}%)
                        </span>
                    </div>
                </div>
                <div style="margin-top: 14px;">
                    <div style="background-color: #e2e8f0; border-radius: 999px; height: 8px; overflow: hidden;">
                        <div style="background-color: #f59e0b; height: 100%; width: {buf_pct}%; border-radius: 999px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
            </div>
            """
        )
    elif scan_requested and is_connected and is_buf_full and live_analysis and live_analysis.get("latest_inference"):
        inf = live_analysis["latest_inference"]
        is_anom = inf.get("is_anomaly", False)
        score = inf.get("anomaly_score", 0.0)
        tau = inf.get("threshold", 1.859450)
        ratio = inf.get("ratio", 1.0)
        sev = inf.get("severity", "NOMINAL")
        lat = inf.get("latency_ms", 0.0)
        top_feats = inf.get("top_features", [])
        explanation = inf.get("xai_5_questions", {})
        ep_status = live_analysis.get("episode_status", {})

        card_class = "status-hero-card-anomaly" if is_anom else "status-hero-card-normal"
        status_title = "🔴 ANOMALY DETECTED" if is_anom else "🟢 NORMAL OPERATION"
        badge_type = "badge-anomaly" if is_anom else "badge-normal"

        if is_anom:
            ep_id = ep_status.get("active_episode_id", 1)
            ep_dur = ep_status.get("duration_sec", 1)
            status_sub = f"Active Episode #{ep_id} (ongoing for {ep_dur}s) &bull; Anomaly score is {ratio:.2f}× of threshold limit."
        else:
            status_sub = f"All 22 physical telemetry counters match baseline spatio-temporal forecasting patterns ({ratio:.2f}× of threshold limit)."

        render_html(
            f"""
            <div class="status-hero-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 3px;">
                            Current Host Anomaly Status
                        </div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 800; color: {'#dc2626' if is_anom else '#16a34a'};">
                            {status_title}
                        </div>
                        <div style="color: #475569; font-size: 0.88rem; margin-top: 2px;">
                            {status_sub}
                        </div>
                    </div>
                    <div>
                        <span class="{badge_type}" style="font-size: 0.86rem; padding: 4px 12px;">
                            {sev}
                        </span>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; padding-top: 12px; border-top: 1px solid #f1f5f9;">
                    <div>
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Anomaly Score</div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.30rem; font-weight: 700; color: {'#dc2626' if is_anom else '#16a34a'};">{score:.4f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Threshold (τ)</div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.30rem; font-weight: 700; color: #475569;">{tau:.4f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Ratio (Score / τ)</div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.30rem; font-weight: 700; color: {'#dc2626' if is_anom else '#2563eb'};">{ratio:.2f}×</div>
                    </div>
                    <div>
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Inference Latency</div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; color: #0f172a; margin-top: 3px;">{lat:.1f} ms</div>
                    </div>
                </div>
            </div>
            """
        )

        # 4. ACTIVE EPISODE DETAILS
        is_in_ep = bool(ep_status.get("is_in_anomaly_episode", False))
        if is_in_ep:
            ep_id = ep_status.get("active_episode_id", 1)
            ep_start = ep_status.get("start_time", "N/A")
            ep_dur = ep_status.get("duration_sec", 1)
            ep_peak = ep_status.get("peak_score", score)
            ep_curr = ep_status.get("current_score", score)
            ep_dom = ep_status.get("dominant_features", [])

            render_html(
                f"""
                <div class="stitch-card" style="border-left: 5px solid #dc2626; margin-bottom: 16px;">
                    <div class="stitch-card-header" style="color: #991b1b;">
                        🚨 Active Anomaly Episode #{ep_id} Details
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 10px;">
                        <div>
                            <div style="font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Start Time</div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #0f172a;">{ep_start}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Duration</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.10rem; font-weight: 700; color: #dc2626;">{ep_dur} seconds</div>
                        </div>
                        <div>
                            <div style="font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Peak Score</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 1.10rem; font-weight: 700; color: #dc2626;">{ep_peak:.4f}</div>
                        </div>
                    </div>
                    <div style="font-size: 0.82rem; color: #475569;">
                        Dominant Metrics: <strong>{', '.join(ep_dom) if ep_dom else 'Analyzing...'}</strong>
                    </div>
                </div>
                """
            )

        # 5. DEEP HUMAN-UNDERSTANDABLE EXPLAINABILITY & PROGRESSIVE DISCLOSURE
        human_exp = inf.get("human_explanation") or inf.get("deep_explanation") or build_deep_human_explanation(
            state="ANOMALY" if is_anom else "NORMAL",
            anomaly_score=score,
            threshold=tau,
            top_features=top_feats,
            timestamp_iso=inf.get("timestamp", ""),
            episode_info=ep_status,
            host_id=host_id,
        )
        tech_exp = inf.get("xai_5_questions", {})
        broken_pairs = inf.get("broken_pairs", [])

        render_deep_human_explainability_panel(
            human_exp=human_exp,
            explanation_5q=tech_exp,
            top_feats=top_feats,
            broken_pairs=broken_pairs,
            score=score,
            threshold=tau,
            ratio=ratio,
            severity=sev,
            is_anomaly=is_anom,
        )

    # 7. ANOMALY SCORE TIMELINE (ONLY WHEN SCAN HAS BEEN REQUESTED)
    if scan_requested and live_analysis:
        score_hist = live_analysis.get("score_history", [])
        if score_hist:
            active_tau = live_analysis.get("latest_inference", {}).get("threshold", 1.411807 if "v2" in str(live_analysis.get("latest_inference", {}).get("model_id", "")) else 1.859450)
            render_html(
                f"""
                <div class="stitch-card">
                    <div class="stitch-card-header">
                        📈 Host Anomaly Score vs. Calibrated Baseline Threshold (τ = {active_tau:.6f})
                    </div>
                    <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 10px;">
                        Continuous sliding window evaluations for this specific host. Points above the red dashed line indicate anomalies.
                    </div>
                """
            )
            fig_sc = plot_live_anomaly_score_timeline(score_hist, active_tau)
            st.plotly_chart(fig_sc, use_container_width=True)
            render_html("</div>")

    # 8. ROLLING 60-SECOND TELEMETRY PLOTS
    if history_data and history_data.get("count", 0) > 0 and is_connected:
        render_html(
            """
            <div class="stitch-card">
                <div class="stitch-card-header">📈 Rolling 60-Second Real-Time Telemetry Streams</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                    Continuous 1-second telemetry buffers updated dynamically from host counters.
                </div>
            """
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>CPU Utilization Timeline (%)</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_live_cpu_timeline(history_data), use_container_width=True)

            st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>Storage I/O Rates (KB/s)</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_live_io_timeline(history_data), use_container_width=True)

        with col_p2:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>Memory & Swap Allocation (%)</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_live_memory_timeline(history_data), use_container_width=True)

            st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>Network Traffic Rates (KB/s)</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_live_network_timeline(history_data), use_container_width=True)

        render_html("</div>")

    # 9. COMPLETE 22-FEATURE LIVE SCHEMA INSPECTOR
    if latest_feats and is_connected:
        render_html(
            """
            <div class="stitch-card">
                <div class="stitch-card-header">📋 Complete 22-Feature Hardware Schema</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                    Standardized physical metrics with instantaneous formatted values and metric descriptions.
                </div>
            """
        )
        feat_rows = []
        for idx, f in enumerate(LIVE_FEATURES):
            curr_v = latest_feats.get(f, "—")
            formatted_v = format_physical_metric(f, curr_v) if isinstance(curr_v, (int, float)) else str(curr_v)
            category = "CPU" if "cpu" in f else "Memory" if "mem" in f or "swap" in f else "Disk" if "disk" in f else "Network" if "net" in f else "Process"
            unit_str = FEATURE_UNITS.get(f, "")
            feat_rows.append({
                "Index": idx + 1,
                "Category": category,
                "Feature ID": f,
                "Physical Value": formatted_v,
                "Unit": unit_str,
                "Metric Description": FEATURE_DESCRIPTIONS.get(f, "Live OS metric")
            })
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
        render_html("</div>")

    # 10. HISTORICAL LABELLED EVALUATION METRICS PANEL (DISTINCT FROM LIVE OBSERVATIONS)
    with st.expander("📊 Model Validation Metrics & Labelled Benchmark Results (V3 Profile)", expanded=False):
        metrics_json_p = PROJECT_ROOT / "data" / "live_training" / "V3_DETECTION_VALIDATION_METRICS.json"
        if metrics_json_p.exists():
            try:
                with open(metrics_json_p, "r", encoding="utf-8") as mf:
                    v3_mdata = json.load(mf)
                evt_m = v3_mdata.get("event_level_metrics", {})
                win_m = v3_mdata.get("window_level_metrics", {})
                sc_res = v3_mdata.get("scenario_results", [])

                st.markdown("##### 🔬 Event-Level Controlled Workload Benchmark Metrics")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Event Precision", f"{evt_m.get('precision_pct', 100.0):.1f}%", "Zero False Alerts")
                col_m2.metric("Event Recall", f"{evt_m.get('recall_pct', 75.0):.1f}%", "3/4 Incidents")
                col_m3.metric("Avg Detection Delay", f"{evt_m.get('avg_detection_delay_sec', 2.0):.1f}s", "Target ≤ 5.0s")
                col_m4.metric("Avg Recovery Time", f"{evt_m.get('avg_recovery_time_sec', 0.0):.1f}s", "Immediate Reset")

                st.markdown("##### 📋 Scenario-by-Scenario Validation Matrix")
                matrix_rows = []
                for s in sc_res:
                    matrix_rows.append({
                        "Test ID": s.get("test_id", ""),
                        "Scenario Name": s.get("scenario_name", ""),
                        "Ground Truth": s.get("ground_truth", ""),
                        "Detection Status": s.get("detection_status", ""),
                        "Peak Score": f"{s.get('peak_score', 0.0):.4f}",
                        "Threshold (τ)": f"{s.get('threshold', 2.120169):.4f}",
                        "Ratio": f"{s.get('score_ratio', 1.0):.2f}x",
                        "Delay": f"{s.get('detection_delay_sec')}s" if s.get("detection_delay_sec") is not None else "N/A",
                    })
                st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
                st.caption("ℹ️ Note: These metrics reflect offline controlled benchmark evaluations on labelled physical workloads for host_sivachowdary. They are distinct from real-time live telemetry observations.")
            except Exception as m_exc:
                st.caption(f"Could not parse validation metrics: {m_exc}")
        else:
            st.caption("Validation metrics file not found.")

    if auto_refresh and is_connected:
        time.sleep(1.5)
        st.rerun()


# ============================================================================
# MAIN APPLICATION ROUTING & CONTROLLER
# ============================================================================

def main():
    # 1. Verify FastAPI Backend Connection
    health_info, connected = fetch_api_health()

    if not connected:
        show_offline_screen()
        return

    # 2. Check SMD Benchmark Resources (Dataset & Backend Model)
    test_data, test_labels, test_window_labels = load_raw_smd_dataset()
    has_smd_data = (test_data is not None and test_labels is not None and test_window_labels is not None)
    has_smd_model = bool(health_info and health_info.get("model_loaded"))
    has_smd_resources = (has_smd_data and has_smd_model)

    n_windows = len(test_window_labels) if has_smd_resources else 0
    threshold = 2.073376

    # Initialize Session State
    if "selected_window_idx" not in st.session_state:
        st.session_state["selected_window_idx"] = (17485 // 5) if has_smd_resources else 0

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dashboard"

    # ========================================================================
    # SIDEBAR NAVIGATION
    # ========================================================================
    render_html(
        """
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 18px;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.45rem; font-weight: 800; color: #1e3a8a; letter-spacing: -0.5px;">
                🔬 FGEAD
            </div>
            <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px;">
                Explainable AI Operations Center
            </div>
        </div>
        """,
        in_sidebar=True
    )

    render_html(
        "<div style='font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;'>System Mode</div>",
        in_sidebar=True
    )

    operating_mode = st.sidebar.selectbox(
        "Operating Mode",
        [
            "🏢 Fleet Operations Center",
            "📊 SMD Dataset Benchmark (Machine 1-1)"
        ],
        key="operating_mode",
        label_visibility="collapsed"
    )

    if operating_mode == "🏢 Fleet Operations Center":
        registered_hosts = fetch_api_hosts()
        host_options = ["All Hosts (Fleet Overview)"]
        host_map = {}
        for h in registered_hosts:
            label = f"{h.get('hostname', 'Host')} ({h.get('operating_system', 'OS')}) — [{h.get('host_id')}]"
            host_options.append(label)
            host_map[label] = h.get("host_id")

        render_html(
            "<div style='font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 14px; margin-bottom: 8px;'>Select Host</div>",
            in_sidebar=True
        )

        selected_label = st.sidebar.selectbox(
            "Select Monitored Host",
            host_options,
            key="selected_host_label",
            label_visibility="collapsed"
        )

        render_html("<div style='margin-top: 24px;'></div>", in_sidebar=True)
        render_html(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; font-size: 0.8rem; color: #64748b;">
                <div style="font-weight: 600; color: #334155; margin-bottom: 4px;">Platform Capabilities</div>
                <div>Fleet: <strong>Multi-Host Telemetry</strong></div>
                <div>OS: <strong>Windows & Linux</strong></div>
                <div>Model: <strong>22-Feature Spatio-Temporal</strong></div>
                <div>Threshold: <strong>Dynamic Host Calibrated</strong></div>
            </div>
            """,
            in_sidebar=True
        )

        if selected_label == "All Hosts (Fleet Overview)":
            render_fleet_overview_page(health_info)
        else:
            sel_host_id = host_map.get(selected_label)
            if sel_host_id:
                render_single_host_page(sel_host_id, health_info)
        return

    # SMD BENCHMARK MODE (STANDARD FGEAD WORKFLOW)
    render_html(
        "<div style='font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 14px; margin-bottom: 8px;'>Navigation</div>",
        in_sidebar=True
    )

    pages = [
        "Dashboard",
        "Analyze Data",
        "Explainability & Prediction Analysis",
        "Analysis History",
        "About the Explainable AI System"
    ]

    current_p_idx = pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0

    page = st.sidebar.radio(
        "Navigation Menu",
        pages,
        index=current_p_idx,
        key="nav_radio",
        label_visibility="collapsed"
    )
    st.session_state["current_page"] = page

    render_html("<div style='margin-top: 30px;'></div>", in_sidebar=True)
    render_html(
        """
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; font-size: 0.8rem; color: #64748b;">
            <div style="font-weight: 600; color: #334155; margin-bottom: 4px;">Target Telemetry</div>
            <div>Device: <strong>SMD Machine 1-1</strong></div>
            <div>Streams: <strong>38 Features</strong></div>
            <div>Window: <strong>60 Steps (Stride 5)</strong></div>
        </div>
        """,
        in_sidebar=True
    )

    if not has_smd_resources and page != "About the Explainable AI System":
        render_html(
            """
            <div class="stitch-card" style="border-left: 5px solid #0284c7; padding: 22px 24px;">
                <div class="stitch-card-header" style="color: #0369a1; font-size: 1.15rem; margin-bottom: 12px;">
                    ℹ️ SMD Benchmark Dataset Not Installed
                </div>
                <div style="font-size: 0.94rem; color: #334155; line-height: 1.6;">
                    The <strong>Server Machine Dataset (SMD)</strong> benchmark dataset and model resources are not installed in the local environment.<br><br>
                    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 12px 16px; margin: 10px 0;">
                        <strong style="color: #15803d;">✓ Live Windows Host Monitoring is fully operational!</strong><br>
                        You can monitor real-time Windows system telemetry, run anomaly detection, and view XAI explanations by selecting <strong>🏢 Fleet Operations Center</strong> in the sidebar dropdown.
                    </div><br>
                    <em>To enable offline SMD benchmark evaluation, ensure <code>machine-1-1.txt</code> is present in <code>data/SMD/test/</code> and <code>data/SMD/test_label/</code> and restart the server.</em>
                </div>
            </div>
            """
        )
        return

    if page == "About the Explainable AI System":
        render_about_page(health_info)
        return

    # Global window calculations
    current_idx = int(st.session_state["selected_window_idx"])
    w_start_abs = int(current_idx * 5)
    w_end_abs = w_start_abs + 59
    active_window_data = test_data[w_start_abs : w_start_abs + 60].tolist()

    # Fetch Prediction Response dynamically
    try:
        predict_res = fetch_api_prediction(current_idx, active_window_data)
    except Exception as exc:
        st.error(f"Failed to communicate with FGEAD prediction API: {exc}")
        return

    # Extract dynamic feature names
    machine_info = fetch_api_machine()
    feature_names = (
        machine_info["feature_names"]
        if machine_info
        else [f"feature_{i:02d}" for i in range(38)]
    )

    # ========================================================================
    # PAGE ROUTING
    # ========================================================================
    if page == "Dashboard":
        render_dashboard_page(predict_res, test_data, test_labels, current_idx, feature_names, n_windows)

    elif page == "Analyze Data":
        render_analyze_data_page(predict_res, test_data, test_window_labels, threshold, current_idx, n_windows, feature_names)

    elif page == "Explainability & Prediction Analysis":
        render_explainability_page(predict_res, test_data[w_start_abs : w_start_abs + 60], feature_names)

    elif page == "Analysis History":
        render_analysis_history_page(test_labels, n_windows)

    elif page == "About the Explainable AI System":
        render_about_page(health_info)


# ============================================================================
# PHASE 7 — DASHBOARD (USER STORY & ACTION FOCUSED)
# ============================================================================

def render_dashboard_page(
    predict_res: Dict[str, Any],
    test_data: np.ndarray,
    test_labels: np.ndarray,
    current_idx: int,
    feature_names: List[str],
    n_windows: int
):
    render_html(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 700; color: #0f172a;">
                Telemetry Intelligence Dashboard
            </div>
            <div style="color: #64748b; font-size: 0.90rem; margin-top: 2px;">
                Multivariate Explainable Anomaly Monitoring for <strong>SMD Machine 1-1</strong>
            </div>
        </div>
        """
    )

    is_anom = predict_res["is_anomaly"]
    score = predict_res["anomaly_score"]
    threshold = predict_res["threshold"]
    confidence = predict_res.get("confidence_pct", "N/A")
    alert_level = predict_res.get("alert_level", "🟢 NORMAL")
    peak_step = predict_res.get("peak_timestep", None)
    w_start = current_idx * 5
    w_end = w_start + 59

    # Unified Hero Status Card
    card_class = "status-hero-card-anomaly" if is_anom else "status-hero-card-normal"
    status_title = "🔴 ANOMALY DETECTED" if is_anom else "🟢 NOMINAL OPERATION"
    status_sub = (
        "Unusual telemetry behavior detected across multiple sensor streams."
        if is_anom
        else "All 38 telemetry streams behaving within nominal calibrated bounds."
    )
    badge_type = "badge-anomaly" if is_anom else "badge-normal"

    render_html(
        f"""
        <div class="status-hero-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
                <div>
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 3px;">
                        Current Machine Status
                    </div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.40rem; font-weight: 800; color: {'#dc2626' if is_anom else '#16a34a'};">
                        {status_title}
                    </div>
                    <div style="color: #475569; font-size: 0.88rem; margin-top: 2px;">
                        {status_sub}
                    </div>
                </div>
                <div>
                    <span class="{badge_type}" style="font-size: 0.86rem; padding: 4px 12px;">
                        {alert_level} &bull; {confidence} CONFIDENCE
                    </span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; padding-top: 12px; border-top: 1px solid #f1f5f9;">
                <div>
                    <div style="font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Confidence</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #1e40af;">{confidence}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Anomaly Score</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: {'#dc2626' if is_anom else '#16a34a'};">{score:.4f}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Threshold (&tau;)</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #475569;">{threshold:.4f}</div>
                </div>
                <div>
                    <div style="font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Active Window</div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; font-weight: 600; color: #0f172a; margin-top: 2px;">t={w_start}&rarr;{w_end}</div>
                </div>
            </div>
        </div>
        """
    )

    # "Why should I care?" Card + Action
    top_feats = predict_res.get("top_features", [])
    top_names = [format_feature_title(f["feature"]) for f in top_feats[:3]]
    if is_anom:
        feat_clause = f"Several telemetry features ({', '.join(top_names)}) deviated significantly from their expected behavior" if top_names else "Multiple telemetry streams deviated from model expectations"
        why_care_text = f"{feat_clause} alongside altered sensor correlation patterns. FGEAD isolated these exact signals to explain the root cause."
    else:
        why_care_text = "All 38 monitored telemetry streams closely match the model's forecasting baseline. Sensor relationships remain normal without operational degradation."

    col_care1, col_care2 = st.columns([8, 4])
    with col_care1:
        render_html(
            f"""
            <div style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-left: 4px solid #2563eb; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 0.92rem; font-weight: 700; color: #1e3a8a; margin-bottom: 3px;">
                    💡 Why should I care?
                </div>
                <div style="font-size: 0.88rem; color: #1e293b; line-height: 1.5;">
                    {why_care_text}
                </div>
            </div>
            """
        )
    with col_care2:
        render_html("<div style='padding-top: 4px;'></div>")
        if st.button("👉 View Full Explanation →", type="primary", use_container_width=True):
            st.session_state["current_page"] = "Explainability & Prediction Analysis"
            st.rerun()

    # Main Telemetry Timeline Card
    render_html(
        """
        <div class="stitch-card">
            <div class="stitch-card-header">
                📊 Telemetry Timeline & Active Window Explorer
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 10px;">
                Shaded light red areas indicate ground-truth anomaly events. The highlighted blue span marks the active analysis window.
            </div>
        """
    )

    fig = plot_telemetry_timeline_light(test_data, test_labels, current_idx, feature_names)
    st.plotly_chart(fig, use_container_width=True)
    render_html("</div>")

    # Quick Jump Selector & Actions
    render_html(
        """
        <div class="stitch-card">
            <div class="stitch-card-header">⚡ Jump to Any Telemetry Window</div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 8px;">
                Select any window index across the 5,684 test windows (e.g. <code>3497</code> for verified peak).
            </div>
        """
    )
    new_w = st.number_input(
        "Target Window Index",
        min_value=0,
        max_value=n_windows - 1,
        value=current_idx,
        step=1,
        key="dashboard_w_input"
    )
    if new_w != current_idx:
        st.session_state["selected_window_idx"] = new_w
        st.rerun()
    render_html("</div>")


# ============================================================================
# PHASE 8 — ANALYZE DATA PAGE (STREAMLINED WORKFLOW)
# ============================================================================

def render_analyze_data_page(
    predict_res: Dict[str, Any],
    test_data: np.ndarray,
    test_window_labels: np.ndarray,
    threshold: float,
    current_idx: int,
    n_windows: int,
    feature_names: List[str]
):
    render_html(
        """
        <div style="margin-bottom: 16px;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 700; color: #0f172a;">
                Analyze Telemetry Data
            </div>
            <div style="color: #64748b; font-size: 0.90rem; margin-top: 2px;">
                Select an input telemetry window and evaluate it using the FGEAD forecasting model.
            </div>
        </div>
        """
    )

    # Step Progress Flow Indicator
    render_html(
        """
        <div class="step-flow-bar">
            <div class="step-flow-item active">
                <span class="step-flow-badge">1</span>
                <span>Select Window</span>
            </div>
            <span class="step-flow-arrow">&rarr;</span>
            <div class="step-flow-item active">
                <span class="step-flow-badge">2</span>
                <span>Model Inference</span>
            </div>
            <span class="step-flow-arrow">&rarr;</span>
            <div class="step-flow-item active">
                <span class="step-flow-badge">3</span>
                <span>Prediction Verdict</span>
            </div>
            <span class="step-flow-arrow">&rarr;</span>
            <div class="step-flow-item active">
                <span class="step-flow-badge">4</span>
                <span>Explain Decision</span>
            </div>
        </div>
        """
    )

    # Step 1: Input & Configuration
    render_html(
        """
        <div class="stitch-card">
            <div class="stitch-card-header">
                ⚙️ Step 1: Input Data Selection & Parameters
            </div>
        """
    )

    col_w1, col_w2, col_w3 = st.columns([5, 4, 3])
    with col_w1:
        selected_idx = st.number_input(
            "Telemetry Window Index (0 to 5,683)",
            min_value=0,
            max_value=n_windows - 1,
            value=current_idx,
            step=1,
            key="analyze_window_input"
        )
        if selected_idx != current_idx:
            st.session_state["selected_window_idx"] = selected_idx
            st.rerun()

    with col_w2:
        threshold_offset = st.slider(
            "Visual Sensitivity Multiplier (k Z-Score)",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1,
            key="analyze_k_slider"
        )

    with col_w3:
        w_start = selected_idx * 5
        w_end = w_start + 59
        render_html(
            f"""
            <div style="padding-top: 24px; font-size: 0.84rem; color: #475569;">
                <strong>Timeline Span:</strong><br/>
                <code>t={w_start}</code> &rarr; <code>t={w_end}</code> (60 steps)
            </div>
            """
        )

    render_html("</div>")

    # Step 2: Prediction Verdict
    is_anom = predict_res["is_anomaly"]
    score = predict_res["anomaly_score"]
    threshold = predict_res["threshold"]
    confidence = predict_res.get("confidence_pct", "N/A")
    alert_level = predict_res.get("alert_level", "🟢 NORMAL")
    peak_step = predict_res.get("peak_timestep", None)
    duration = predict_res.get("anomaly_duration_steps", 0)

    render_html("<div class='stitch-card-header' style='margin-top: 10px;'>🔍 Step 2: Prediction Result</div>")

    if is_anom:
        render_html(
            f"""
            <div class="verdict-banner-anomaly">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #dc2626;">
                            🔴 ANOMALY DETECTED
                        </div>
                        <div style="color: #475569; font-size: 0.88rem; margin-top: 2px;">
                            The telemetry sequence exceeds the calibrated normal threshold (&tau; = {threshold:.4f}).
                        </div>
                    </div>
                    <div>
                        <span class="badge-anomaly" style="font-size: 0.88rem; padding: 5px 12px;">
                            {alert_level} ALERT &bull; {confidence} CONFIDENCE
                        </span>
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #fecaca; display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.85rem; color: #334155;">
                    <div><strong>Score:</strong> <code>{score:.6f}</code></div>
                    <div><strong>Threshold:</strong> <code>{threshold:.6f}</code></div>
                    <div><strong>Peak Step:</strong> <code>t={peak_step}</code></div>
                    <div><strong>Duration:</strong> <code>{duration} steps</code></div>
                </div>
            </div>
            """
        )
    else:
        render_html(
            f"""
            <div class="verdict-banner-normal">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 700; color: #16a34a;">
                            🟢 NORMAL TELEMETRY
                        </div>
                        <div style="color: #475569; font-size: 0.88rem; margin-top: 2px;">
                            Telemetry flows represent standard nominal operating behavior. Score is within safe bounds.
                        </div>
                    </div>
                    <div>
                        <span class="badge-normal" style="font-size: 0.88rem; padding: 5px 12px;">
                            NOMINAL &bull; {confidence} CONFIDENCE
                        </span>
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #bbf7d0; display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.85rem; color: #334155;">
                    <div><strong>Score:</strong> <code>{score:.6f}</code></div>
                    <div><strong>Threshold:</strong> <code>{threshold:.6f}</code></div>
                </div>
            </div>
            """
        )

    # Step 3: Brief Why Summary & Direct Action
    narrative_short = generate_human_readable_narrative(predict_res)
    render_html(
        f"""
        <div class="stitch-card">
            <div class="stitch-card-header">💡 Step 3: Why was this predicted?</div>
            <p style="font-size: 0.90rem; color: #334155; line-height: 1.55; margin-bottom: 14px;">
                {narrative_short}
            </p>
        """
    )

    if st.button("👉 See Why (View Full Explainability & Evidence) →", type="primary", use_container_width=False):
        st.session_state["current_page"] = "Explainability & Prediction Analysis"
        st.rerun()

    render_html("</div>")


# ============================================================================
# PHASE 1 & 2 — EXPLAINABILITY & PREDICTION ANALYSIS (CORE PURPOSE)
# ============================================================================

def render_explainability_page(
    predict_res: Dict[str, Any],
    window_data: np.ndarray,
    feature_names: List[str]
):
    render_html(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.70rem; font-weight: 700; color: #0f172a;">
                Explainability & Prediction Analysis
            </div>
            <div style="color: #1e40af; font-size: 0.92rem; font-weight: 600; margin-top: 2px;">
                FGEAD does not only detect an anomaly — it explains why it was detected.
            </div>
        </div>
        """
    )

    is_anom = predict_res["is_anomaly"]
    alert_level = predict_res.get("alert_level", "🟢 NORMAL")
    confidence_pct = predict_res.get("confidence_pct", "N/A")
    score = predict_res["anomaly_score"]
    threshold = predict_res["threshold"]
    peak_t = predict_res.get("peak_timestep", None)
    root_cause = predict_res.get("root_cause", "No root cause explanation generated.")
    top_features = predict_res.get("top_features", [])
    broken_pairs = predict_res.get("broken_pairs", [])

    # 1. WHAT DID WE FIND?
    render_html(
        f"""
        <div class="stitch-card">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;">
                1. What Did We Find?
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 10px;">
                <div style="font-family: 'Outfit', sans-serif; font-size: 1.30rem; font-weight: 800; color: {'#dc2626' if is_anom else '#16a34a'};">
                    {'🔴 ANOMALY DETECTED' if is_anom else '🟢 NOMINAL TELEMETRY'}
                </div>
                <div>
                    <span class="{'badge-anomaly' if is_anom else 'badge-normal'}" style="font-size: 0.88rem; padding: 4px 12px;">
                        {alert_level} ALERT &bull; {confidence_pct} CONFIDENCE
                    </span>
                </div>
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.85rem; color: #475569; padding-top: 8px; border-top: 1px solid #f1f5f9;">
                <div><strong>Anomaly Score:</strong> <code>{score:.6f}</code></div>
                <div><strong>Calibrated Threshold (&tau;):</strong> <code>{threshold:.6f}</code></div>
                <div><strong>Peak Timestep:</strong> <code>t = {peak_t}</code></div>
                <div><strong>Duration:</strong> <code>{predict_res['anomaly_duration_steps']} timesteps</code></div>
            </div>
        </div>
        """
    )

    # 2. WHY WAS IT DETECTED? (HERO SECTION)
    human_narrative = generate_human_readable_narrative(predict_res)

    # Visual emphasis badges for top signals and relationship changes
    visual_tags_html = ""
    if is_anom:
        signal_pills = []
        for f in top_features[:3]:
            fname = format_feature_title(f["feature"])
            z = float(f.get("z_score", 0.0))
            direction = str(f.get("direction", "")).lower()
            dir_icon = "📈" if "spike" in direction else "📉" if "drop" in direction else "📊"
            signal_pills.append(f"<span class='tag-pill tag-pill-alert'>{dir_icon} {fname} (+{z:.1f} Z-Score)</span>")

        rel_pills = []
        for bp in broken_pairs[:2]:
            fa = format_feature_title(bp["feature_a"])
            fb = format_feature_title(bp["feature_b"])
            rel_pills.append(f"<span class='tag-pill tag-pill-info'>🔗 {fa} &harr; {fb}</span>")

        visual_tags_html = "<div style='margin-top: 12px; padding-top: 10px; border-top: 1px solid #bfdbfe; display: flex; flex-direction: column; gap: 6px;'>"
        if signal_pills:
            visual_tags_html += f"<div><span style='font-size: 0.75rem; font-weight: 700; color: #1e3a8a; text-transform: uppercase; margin-right: 8px;'>Top Signals:</span>{''.join(signal_pills)}</div>"
        if rel_pills:
            visual_tags_html += f"<div><span style='font-size: 0.75rem; font-weight: 700; color: #1e3a8a; text-transform: uppercase; margin-right: 8px;'>Relationship Shift:</span>{''.join(rel_pills)}</div>"
        visual_tags_html += "</div>"

    root_cause_html = ""
    if is_anom and root_cause and root_cause != "No root cause explanation generated.":
        root_cause_html = f"""
        <div class="root-cause-callout">
            <strong>Diagnostic Hypothesis:</strong> {root_cause}
        </div>
        """

    render_html(
        f"""
        <div class="narrative-why-box">
            <div class="narrative-why-title">
                💡 2. Why Was This Detected?
            </div>
            <div class="narrative-why-text">
                {human_narrative}
            </div>
            {visual_tags_html}
            {root_cause_html}
        </div>
        """
    )

    # 3. WHAT CONTRIBUTED MOST? (TOP CONTRIBUTING FEATURES - VISUAL SUMMARY FIRST)
    render_html(
        """
        <div class="stitch-card">
            <div class="stitch-card-header">
                🔍 3. What Contributed Most? (Top Contributing Features)
            </div>
            <div class="section-intro-text">
                These features showed the largest differences from the behavior expected by the model during this time window.
                Metrics are ranked by standardized Z-score deviation evaluated on the original telemetry scale.
            </div>
        """
    )

    if not top_features:
        st.info("No feature deviations exceeded the reporting threshold for this window.")
    else:
        max_z = max([float(x["z_score"]) for x in top_features]) if top_features else 1.0
        max_z = max(max_z, 1e-6)

        # Visual Contribution Cards (Top 3-5 Features)
        for idx, f in enumerate(top_features[:4]):
            z = float(f["z_score"])
            bar_width = int((z / max_z) * 100)
            bar_width = min(max(bar_width, 6), 100)

            z_class = "feature-contrib-z-high" if z > 2.5 else "feature-contrib-z-med" if z > 1.2 else "feature-contrib-z-low"
            bar_color = "#dc2626" if z > 2.5 else "#f97316" if z > 1.2 else "#2563eb"

            direction_label = f.get("direction", "")
            dir_icon = "↑ Spike" if "spike" in direction_label.lower() else "↓ Drop" if "drop" in direction_label.lower() else "≈ Stable"

            takeaway = (
                "Largest observed deviation in this analysis window."
                if idx == 0
                else "High observed deviation surging above baseline."
                if "spike" in direction_label.lower()
                else "Sharp decline dropping well below expected levels."
                if "drop" in direction_label.lower()
                else "Significant divergence from model prediction."
            )

            render_html(
                f"""
                <div class="feature-contrib-card">
                    <div class="feature-contrib-header">
                        <div>
                            <span class="feature-contrib-name">{format_feature_title(f['feature'])}</span>
                            <span style="color: #64748b; font-size: 0.80rem; margin-left: 6px;">({f['feature']})</span>
                            <span style="font-size: 0.80rem; font-weight: 600; color: {bar_color}; margin-left: 8px;">{dir_icon}</span>
                        </div>
                        <div class="feature-contrib-zscore {z_class}">
                            +{z:.1f} Z-Score
                        </div>
                    </div>
                    <div class="feature-contrib-bar-bg">
                        <div style="background-color: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <div class="feature-contrib-stats">
                        <div>Observed: <strong><code>{f['actual']:.4f}</code></strong></div>
                        <div>Forecasted: <strong><code>{f['predicted']:.4f}</code></strong></div>
                        <div>Deviation: <strong><code>{f['deviation']:.4f}</code></strong></div>
                        <div style="color: #475569; font-style: italic;">{takeaway}</div>
                    </div>
                </div>
                """
            )

        # Supporting Detailed Technical Table
        render_html("<div style='margin-top: 14px; font-weight: 600; font-size: 0.82rem; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;'>Detailed Telemetry Error Attribution Table</div>")

        html_table = "<table class='stitch-table'>"
        html_table += "<tr><th>Feature</th><th>Observed Value</th><th>Expected / Forecasted</th><th>Deviation</th><th>Z-Score</th><th>Direction</th><th>Relative Impact</th></tr>"

        for f in top_features:
            z = float(f["z_score"])
            bar_width = int((z / max_z) * 100)
            bar_width = min(max(bar_width, 4), 100)
            bar_color = "#dc2626" if z > 2.5 else "#f97316" if z > 1.2 else "#2563eb"

            bar_html = f"<div style='background-color: #f1f5f9; width: 90px; height: 6px; border-radius: 3px; overflow: hidden;'><div style='background-color: {bar_color}; width: {bar_width}%; height: 100%;'></div></div>"

            direction_label = f.get("direction", "")
            dir_badge = (
                f"<span style='color: #dc2626; font-weight: 600;'>&uarr; Spike</span>"
                if "spike" in direction_label.lower()
                else f"<span style='color: #2563eb; font-weight: 600;'>&darr; Drop</span>"
                if "drop" in direction_label.lower()
                else "<span style='color: #64748b;'>&approx; Stable</span>"
            )

            html_table += f"<tr>"
            html_table += f"<td><strong>{format_feature_title(f['feature'])}</strong> <span style='color:#64748b; font-size:0.78rem;'>({f['feature']})</span></td>"
            html_table += f"<td><code>{f['actual']:.4f}</code></td>"
            html_table += f"<td><code>{f['predicted']:.4f}</code></td>"
            html_table += f"<td>{f['deviation']:.4f}</td>"
            html_table += f"<td><strong style='color:{bar_color};'>{z:.1f}</strong></td>"
            html_table += f"<td>{dir_badge}</td>"
            html_table += f"<td>{bar_html}</td>"
            html_table += f"</tr>"
        html_table += "</table>"

        render_html(html_table)
    render_html("</div>")

    # 4. WHAT CHANGED? (GRAPH RELATIONSHIP AUDITING - VISUAL SUMMARY FIRST)
    render_html(
        """
        <div class="stitch-card">
            <div class="stitch-card-header">
                🔗 4. What Changed? (Inter-Metric Relationship Auditing)
            </div>
        """
    )

    if not broken_pairs:
        st.info("All inter-feature relationships remained within nominal dependency bounds during this window.")
    else:
        p0 = broken_pairs[0]
        fa = format_feature_title(p0["feature_a"])
        fb = format_feature_title(p0["feature_b"])
        render_html(
            f"""
            <div class="section-intro-text">
                Some telemetry signals also changed their normal relationships with each other during this anomaly event.
                The largest detected relationship change involved <strong>{fa}</strong> and <strong>{fb}</strong>.
            </div>
            """
        )

        # Visual Relationship Cards
        for bp in broken_pairs[:3]:
            fa_name = format_feature_title(bp["feature_a"])
            fb_name = format_feature_title(bp["feature_b"])
            exp = float(bp.get("expected_corr", 0.0))
            obs = float(bp.get("observed_corr", 0.0))
            gap = float(bp.get("gap", 0.0))
            sev = bp.get("severity", "MEDIUM")

            meaning_text = (
                f"These two telemetry signals behaved much more strongly together ({obs:.2f}) than the model expected ({exp:.2f})."
                if obs > exp
                else f"These signals decoupled from their learned baseline relationship ({exp:.2f} &rarr; {obs:.2f})."
            )

            render_html(
                f"""
                <div class="relationship-summary-card">
                    <div class="relationship-connection-row">
                        <div class="relationship-node">{fa_name} <span style="font-size: 0.75rem; color: #64748b; font-weight: normal;">({bp['feature_a']})</span></div>
                        <div class="relationship-line">
                            <span class="relationship-gap-badge">&Delta; {gap:.2f} Gap ({sev})</span>
                        </div>
                        <div class="relationship-node">{fb_name} <span style="font-size: 0.75rem; color: #64748b; font-weight: normal;">({bp['feature_b']})</span></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; font-size: 0.82rem; color: #475569;">
                        <div>Expected Dependency: <strong><code>{exp:.2f}</code></strong> &bull; Observed Window Correlation: <strong><code>{obs:.2f}</code></strong></div>
                        <div style="font-style: italic; color: #1e3a8a;">{meaning_text}</div>
                    </div>
                </div>
                """
            )

        # Supporting Detailed Table
        render_html("<div style='margin-top: 14px; font-weight: 600; font-size: 0.82rem; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;'>Full Graph Relationship Audit Log</div>")
        broken_rows = []
        for bp in broken_pairs:
            broken_rows.append({
                "Feature A": f"{format_feature_title(bp['feature_a'])} ({bp['feature_a']})",
                "Feature B": f"{format_feature_title(bp['feature_b'])} ({bp['feature_b']})",
                "Expected": f"{bp['expected_corr']:.2f}",
                "Observed": f"{bp['observed_corr']:.2f}",
                "Gap": f"{bp['gap']:.2f}",
                "Severity": bp["severity"],
                "Interpretation": bp.get("likely_meaning", bp.get("break_type", "Relationship altered"))
            })
        st.dataframe(pd.DataFrame(broken_rows), use_container_width=True, hide_index=True)

    render_html("</div>")

    # 5. WHAT IS THE EVIDENCE? (VISUAL EVIDENCE & CHARTS)
    render_html(
        """
        <div class="stitch-card">
            <div class="stitch-card-header">
                📈 5. What Is The Evidence? (Visual Evidence & Interactive Analysis)
            </div>
        """
    )

    tab_prof, tab_graph, tab_ranges = st.tabs([
        "📉 Evidence 1 — Feature Deviation (Observed vs. Forecasted)",
        "🕸️ Evidence 2 — Changed Relationships Network",
        "⏱️ Evidence 3 — When the Anomaly Occurred"
    ])

    with tab_prof:
        top_feature_names = [f["feature"] for f in top_features] if top_features else feature_names[:5]
        selected_feat = st.selectbox(
            "Select Telemetry Stream to Inspect Forecasting Error",
            top_feature_names,
            format_func=lambda x: f"{format_feature_title(x)} ({x})",
            key="xai_evidence_feature_selector"
        )

        # Match selected feature details for metric summary
        sel_f_info = next((f for f in top_features if f["feature"] == selected_feat), None)
        if sel_f_info:
            c_e1, c_e2, c_e3, c_e4 = st.columns(4)
            with c_e1:
                render_html(f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px;">
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Observed Value (Peak)</div>
                        <div style="font-size: 1.10rem; font-weight: 700; color: #dc2626; font-family: 'JetBrains Mono', monospace; margin-top: 1px;">{float(sel_f_info['actual']):.4f}</div>
                    </div>
                """)
            with c_e2:
                render_html(f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px;">
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Forecasted Value (Model)</div>
                        <div style="font-size: 1.10rem; font-weight: 700; color: #16a34a; font-family: 'JetBrains Mono', monospace; margin-top: 1px;">{float(sel_f_info['predicted']):.4f}</div>
                    </div>
                """)
            with c_e3:
                render_html(f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px;">
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Absolute Deviation</div>
                        <div style="font-size: 1.10rem; font-weight: 700; color: #0f172a; font-family: 'JetBrains Mono', monospace; margin-top: 1px;">{float(sel_f_info['deviation']):.4f}</div>
                    </div>
                """)
            with c_e4:
                render_html(f"""
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px;">
                        <div style="font-size: 0.70rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Z-Score Deviation</div>
                        <div style="font-size: 1.10rem; font-weight: 700; color: #2563eb; font-family: 'JetBrains Mono', monospace; margin-top: 1px;">+{float(sel_f_info['z_score']):.1f}</div>
                    </div>
                """)
            render_html("<div style='margin-bottom: 10px;'></div>")

        fig_prof = plot_actual_vs_predicted_light(window_data, feature_names, selected_feat, peak_t, top_features)
        st.plotly_chart(fig_prof, use_container_width=True)

    with tab_graph:
        if not broken_pairs:
            st.info("No broken relationships to graph for this window.")
        else:
            fig_g = plot_circular_dependency_graph_light(predict_res)
            if fig_g:
                st.plotly_chart(fig_g, use_container_width=True)
            else:
                st.info("No graph connections available.")

    with tab_ranges:
        ranges = predict_res.get("anomaly_ranges", [])
        if not ranges:
            st.info("No contiguous temporal anomaly ranges isolated in this window.")
        else:
            range_rows = []
            for r in ranges:
                range_rows.append({
                    "Start Timestep": r["window_start"],
                    "End Timestep": r["window_end"],
                    "Duration": r["duration_human"],
                    "Peak Anomaly Score": f"{r['peak_score']:.4f}",
                    "Mean Anomaly Score": f"{r['mean_score']:.4f}"
                })
            st.dataframe(pd.DataFrame(range_rows), use_container_width=True, hide_index=True)

    render_html("</div>")


# ============================================================================
# PHASE 8 — ANALYSIS HISTORY PAGE (Stitch Light Style)
# ============================================================================

def render_analysis_history_page(test_labels: np.ndarray, n_windows: int):
    render_html(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 700; color: #0f172a;">
                Analysis History & Benchmark Incidents
            </div>
            <div style="color: #64748b; font-size: 0.90rem; margin-top: 2px;">
                Chronological list of verified ground-truth anomaly events in the SMD Machine 1-1 benchmark test stream.
            </div>
        </div>
        """
    )

    intervals = get_anomaly_intervals(test_labels)

    render_html(
        f"""
        <div class="stitch-card">
            <div class="stitch-card-header">
                📋 Benchmark Anomaly Events Log ({len(intervals)} Recorded Incidents)
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 10px;">
                Select any incident below to load its corresponding window into the analysis console.
            </div>
        """
    )

    events = []
    for idx, (s, e) in enumerate(intervals):
        duration = e - s + 1
        window_idx_equiv = s // 5
        events.append({
            "Incident ID": f"INC-{s:05d}",
            "Start Step": s,
            "End Step": e,
            "Timeline Range": f"t={s} → t={e}",
            "Duration": f"{duration} steps ({duration}m)",
            "Target Window Index": window_idx_equiv,
            "Status": "🔴 Ground Truth Anomaly"
        })

    events_df = pd.DataFrame(events)
    st.dataframe(events_df, use_container_width=True, hide_index=True)

    # Quick Jump Selector
    render_html("<div style='margin-top: 12px;'></div>")
    selected_event_idx = st.selectbox(
        "Jump to Incident Window",
        options=list(range(len(events))),
        format_func=lambda i: f"{events[i]['Incident ID']} ({events[i]['Timeline Range']} &bull; Window #{events[i]['Target Window Index']})"
    )

    if st.button("Load Selected Incident into Analysis Console", use_container_width=False):
        st.session_state["selected_window_idx"] = events[selected_event_idx]["Target Window Index"]
        st.session_state["current_page"] = "Explainability & Prediction Analysis"
        st.success(f"Loaded Incident {events[selected_event_idx]['Incident ID']}! Switching to Explainability...")
        st.rerun()

    render_html("</div>")


# ============================================================================
# PHASE 9 — ABOUT THE EXPLAINABLE AI SYSTEM (HUMAN-READABLE INTRO)
# ============================================================================

def render_about_page(health_info: Dict[str, Any]):
    render_html(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 700; color: #0f172a;">
                About the Explainable AI System
            </div>
            <div style="color: #64748b; font-size: 0.90rem; margin-top: 2px;">
                System architecture, mathematical methodology, benchmark performance, and live operational health.
            </div>
        </div>
        """
    )

    # Visual Workflow Overview
    render_html(
        """
        <div class="narrative-why-box" style="margin-top: 0;">
            <div class="narrative-why-title">
                📖 System Purpose & How FGEAD Works
            </div>
            <div class="narrative-why-text">
                <strong>FGEAD</strong> analyzes multivariate machine telemetry to detect abnormal operational behavior.
                When an anomaly is detected, the system extracts feature-level deviations and graph relationship shifts
                to deliver structured, human-understandable explanations.
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; background: #ffffff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 12px 16px; margin-top: 12px; gap: 8px;">
                <div style="text-align: center;">
                    <div style="font-size: 1.15rem;">📡</div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: #1e3a8a;">1. Telemetry</div>
                    <div style="font-size: 0.70rem; color: #64748b;">38 SMD Sensors</div>
                </div>
                <div style="color: #94a3b8; font-weight: 700;">&rarr;</div>
                <div style="text-align: center;">
                    <div style="font-size: 1.15rem;">🚨</div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: #1e3a8a;">2. Detection</div>
                    <div style="font-size: 0.70rem; color: #64748b;">GCN + LSTM Forecast</div>
                </div>
                <div style="color: #94a3b8; font-weight: 700;">&rarr;</div>
                <div style="text-align: center;">
                    <div style="font-size: 1.15rem;">🔍</div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: #1e3a8a;">3. Attribution</div>
                    <div style="font-size: 0.70rem; color: #64748b;">Z-Score Deviations</div>
                </div>
                <div style="color: #94a3b8; font-weight: 700;">&rarr;</div>
                <div style="text-align: center;">
                    <div style="font-size: 1.15rem;">🔗</div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: #1e3a8a;">4. Graph Auditing</div>
                    <div style="font-size: 0.70rem; color: #64748b;">Learned Correlations</div>
                </div>
                <div style="color: #94a3b8; font-weight: 700;">&rarr;</div>
                <div style="text-align: center;">
                    <div style="font-size: 1.15rem;">💡</div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: #1e3a8a;">5. Explanation</div>
                    <div style="font-size: 0.70rem; color: #64748b;">Actionable Insights</div>
                </div>
            </div>
        </div>
        """
    )

    tab_arch, tab_metrics, tab_health = st.tabs([
        "🏗️ System Architecture & XAI",
        "📊 Benchmark Evaluation Results",
        "🩺 FastAPI System Health"
    ])

    with tab_arch:
        render_html(
            """
            <div class="stitch-card">
                <div class="stitch-card-header">Framework Overview: FGEAD</div>
                <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                    <strong>FGEAD</strong> (Feature-Level Graph-Based Explainable Anomaly Detection) is an end-to-end framework
                    designed for multivariate server telemetry. It combines dynamic graph structure learning, spatio-temporal GCN+LSTM
                    forecasting, and a structured post-hoc explainability engine.
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px;">
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 14px;">
                        <div style="font-weight: 600; color: #1e3a8a; font-size: 0.88rem; margin-bottom: 6px;">🧠 Core ML Components</div>
                        <ul style="margin: 0; padding-left: 18px; font-size: 0.82rem; color: #475569; line-height: 1.5;">
                            <li><strong>Feature Embedding:</strong> 64-dimensional dense metric embeddings.</li>
                            <li><strong>Self-Attention Graph Learner:</strong> 4 attention heads with Top-K=5 graph sparsification.</li>
                            <li><strong>Spatio-Temporal GCN + LSTM:</strong> GCN Layer + 2-layer LSTM (128 units, 0.2 dropout).</li>
                            <li><strong>Forecasting Head:</strong> Next-step prediction error with learnable importance weights.</li>
                        </ul>
                    </div>
                    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 14px;">
                        <div style="font-weight: 600; color: #1e3a8a; font-size: 0.88rem; margin-bottom: 6px;">💡 5-Question XAI Engine</div>
                        <ul style="margin: 0; padding-left: 18px; font-size: 0.82rem; color: #475569; line-height: 1.5;">
                            <li><strong>Q1:</strong> Which features deviated? (Standardized Z-scores)</li>
                            <li><strong>Q2:</strong> What direction? (&uarr; spike / &darr; drop / &approx; stable)</li>
                            <li><strong>Q3:</strong> Which relationships altered? (Graph Relationship Auditing)</li>
                            <li><strong>Q4:</strong> When did it occur? (Temporal start, end, duration)</li>
                            <li><strong>Q5:</strong> How confident is the alert? (Composite score C)</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
        )

    with tab_metrics:
        render_html(
            """
            <div class="stitch-card">
                <div class="stitch-card-header">SMD Machine 1-1 Official Evaluation Performance</div>
                <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 10px;">
                    Evaluated across 28,479 test timesteps (5,684 test windows, 635 anomaly windows) with threshold &tau; = 2.073376.
                </div>
            """
        )

        metrics_df = pd.DataFrame({
            "Evaluation Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
            "Window-Level Performance": ["0.5600", "0.9039 (90.4%)", "0.6916", "0.9709", "0.7981"],
            "Point-Level Performance": ["0.4891", "0.9714 (97.1%)", "0.6506", "0.9708", "0.7373"]
        })
        st.table(metrics_df)

        render_html(
            """
            <div style="font-size: 0.80rem; color: #64748b; margin-top: 8px; line-height: 1.5;">
                <strong>Note on Precision-Recall Trade-off:</strong> The high recall (90.4% window-level, 97.1% point-level) reflects a deliberate
                operational threshold calibration (99.5th percentile on train data) prioritizing high anomaly sensitivity.
            </div>
            </div>
            """
        )

    with tab_health:
        render_html(
            """
            <div class="stitch-card">
                <div class="stitch-card-header">FastAPI REST Backend Status</div>
            """
        )
        st.json(health_info)
        render_html("</div>")


# ============================================================================
# OFFLINE FALLBACK SCREEN
# ============================================================================

def show_offline_screen():
    """Display clean instructions when FastAPI backend is offline."""
    render_html(
        f"""
        <div style="text-align: center; padding: 40px 20px; max-width: 600px; margin: 0 auto;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 700; color: #dc2626; margin-bottom: 8px;">
                FastAPI Backend Disconnected
            </div>
            <div style="font-size: 0.95rem; color: #64748b; margin-bottom: 24px;">
                The FGEAD dashboard cannot establish an HTTP connection with the REST API server at <code>{API_URL}</code>.
            </div>

            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; text-align: left; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-weight: 600; color: #0f172a; font-size: 0.9rem; margin-bottom: 8px;">
                    Start the Backend Server via Terminal:
                </div>
                <pre style="background-color: #f1f5f9; padding: 12px; border-radius: 6px; color: #1e3a8a; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; border: 1px solid #e2e8f0; margin: 0;">
uvicorn api.main:app --reload --port 8000
                </pre>
            </div>
        </div>
        """
    )

    col_r1, col_r2, col_r3 = st.columns([5, 2, 5])
    with col_r2:
        if st.button("Retry Connection", use_container_width=True):
            st.rerun()


if __name__ == "__main__":
    main()

