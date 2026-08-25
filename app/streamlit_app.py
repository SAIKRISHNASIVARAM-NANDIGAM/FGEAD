"""
app/streamlit_app.py
FGEAD — AI-Powered Telemetry Intelligence Platform
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path fix so imports work from the app/ subfolder ─────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocessor import TimeSeriesPreprocessor
from models.fgead import FGEAD
from models.explainer import FGEADExplainer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FGEAD — AI-Powered Telemetry Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ── Premium Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #0a0c10 !important;
    font-family: 'Outfit', sans-serif !important;
  }

  .metric-card {
    background: linear-gradient(135deg, rgba(20,26,40,0.6) 0%, rgba(10,15,26,0.9) 100%);
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.4);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    margin-bottom: 10px;
  }
  .metric-card:hover {
    border-color: rgba(0,242,254,0.35);
    box-shadow: 0 8px 24px 0 rgba(0,242,254,0.08);
    transform: translateY(-2px);
  }
  .metric-value {
    font-size: 2.1rem; font-weight: 800;
    background: linear-gradient(45deg, #00f2fe 10%, #4facfe 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .metric-label {
    font-size: 0.75rem; font-weight: 600; color: #8a99ad;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 4px;
  }
  .metric-icon { font-size: 1.4rem; margin-bottom: 4px; }

  .alert-banner {
    border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;
    border-left: 5px solid; font-weight: 600; color: #f1f5f9;
    box-shadow: 0 4px 20px 0 rgba(0,0,0,0.3);
  }
  .alert-critical { background: rgba(239,68,68,0.08); border-color: #ef4444;
    border-top: 1px solid rgba(239,68,68,0.15); border-right: 1px solid rgba(239,68,68,0.15); border-bottom: 1px solid rgba(239,68,68,0.15); }
  .alert-high { background: rgba(249,115,22,0.08); border-color: #f97316;
    border-top: 1px solid rgba(249,115,22,0.15); border-right: 1px solid rgba(249,115,22,0.15); border-bottom: 1px solid rgba(249,115,22,0.15); }
  .alert-medium { background: rgba(234,179,8,0.08); border-color: #eab308;
    border-top: 1px solid rgba(234,179,8,0.15); border-right: 1px solid rgba(234,179,8,0.15); border-bottom: 1px solid rgba(234,179,8,0.15); }
  .alert-low { background: rgba(16,185,129,0.08); border-color: #10b981;
    border-top: 1px solid rgba(16,185,129,0.15); border-right: 1px solid rgba(16,185,129,0.15); border-bottom: 1px solid rgba(16,185,129,0.15); }

  .section-header {
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px; margin-top: 32px; margin-bottom: 16px;
    text-transform: uppercase; letter-spacing: 0.8px;
  }

  .glass-panel {
    background: linear-gradient(135deg, rgba(15,23,42,0.7) 0%, rgba(10,15,26,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    backdrop-filter: blur(12px);
  }

  .kpi-row {
    display: flex; justify-content: center; gap: 32px; padding: 14px 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin: 16px 0;
  }
  .kpi-item { font-size: 0.92rem; font-weight: 600; }

  .timeline-item {
    display: flex; gap: 14px; padding: 8px 0;
    border-left: 2px solid rgba(255,255,255,0.08); margin-left: 8px; padding-left: 16px;
  }
  .timeline-time { color: #64748b; font-size: 0.82rem; min-width: 70px; font-family: 'JetBrains Mono', monospace; }
  .timeline-event { color: #e2e8f0; font-size: 0.88rem; }

  code { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_data():
    proc = TimeSeriesPreprocessor(window_size=60, stride=5)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_data.csv")

    if not os.path.exists(data_path):
        from data.synthetic_generator import generate_synthetic_data
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        generate_synthetic_data()

    data, labels = proc.load_csv(data_path)
    data = proc.handle_missing(data)
    data, labels = proc.remove_duplicates(data, labels)

    n = len(data)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)

    train_norm, _, test_norm = proc.normalize(
        data[:t_end], data[t_end:v_end], data[v_end:]
    )

    test_win, test_lbl = proc.create_windows(test_norm, labels[v_end:])
    n_features = data.shape[1]

    feature_names = [
        "cpu_usage", "cpu_temp", "memory_usage", "memory_free",
        "disk_io_read", "disk_io_write", "net_in", "net_out",
        "process_count", "context_switches", "cache_hits",
        "cache_misses", "load_avg_1m", "load_avg_5m", "load_avg_15m",
        "swap_usage", "iowait", "kernel_threads", "open_files", "network_errors",
    ][:n_features]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = FGEAD(n_features=n_features).to(device)

    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
    model_loaded = False
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        model_loaded = True

    explainer = FGEADExplainer(model, feature_names)

    return {
        "model": model, "explainer": explainer,
        "test_win": test_win, "test_lbl": test_lbl,
        "feature_names": feature_names, "n_features": n_features,
        "device": device, "model_loaded": model_loaded, "proc": proc,
    }


@st.cache_data
def compute_all_scores(_model_ref, _test_win, _device):
    scores = []
    _model_ref.eval()
    with torch.no_grad():
        for i in range(len(_test_win)):
            x = torch.FloatTensor(_test_win[i:i+1]).to(_device)
            _, _, s = _model_ref(x)
            scores.append(float(s.max().cpu()))
    return np.array(scores)


# ── Plotly Helpers ────────────────────────────────────────────────────────────

def plot_anomaly_scores(scores, labels, threshold):
    fig = go.Figure()
    in_anomaly, a_start = False, 0
    for i, lbl in enumerate(labels):
        if lbl == 1 and not in_anomaly:
            a_start, in_anomaly = i, True
        elif lbl == 0 and in_anomaly:
            fig.add_vrect(x0=a_start, x1=i, fillcolor="#ef4444", opacity=0.14, line_width=0)
            in_anomaly = False
    if in_anomaly:
        fig.add_vrect(x0=a_start, x1=len(labels), fillcolor="#ef4444", opacity=0.14, line_width=0)

    fig.add_trace(go.Scatter(x=list(range(len(scores))), y=scores, mode="lines",
        name="Anomaly Score", line=dict(color="#00f2fe", width=2.0)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
        annotation_text=f"Threshold = {threshold:.3f}",
        annotation_position="top left",
        annotation_font=dict(color="#ef4444", size=11, family="Outfit"))
    fig.update_layout(
        title="Real-Time Anomaly Score Timeline",
        xaxis_title="Window Index", yaxis_title="Anomaly Score",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)", font=dict(family="Outfit, sans-serif"),
        height=340, margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    )
    return fig


def plot_feature_deviations(top_features):
    names = [f["feature"] for f in top_features]
    devs  = [f["deviation"] for f in top_features]
    z_sc  = [f["z_score"] for f in top_features]
    colors = ["#ef4444" if z > 2.2 else "#f97316" if z > 1.2 else "#3b82f6" for z in z_sc]

    fig = go.Figure(go.Bar(x=devs, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
        text=[f"z={z:.1f}σ" for z in z_sc], textposition="outside",
        textfont=dict(family="Outfit", size=11)))
    fig.update_layout(
        title="Feature Deviation Ranking", xaxis_title="Deviation |actual − predicted|",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)", font=dict(family="Outfit, sans-serif"),
        height=300, margin=dict(l=20, r=40, t=50, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)"),
    )
    return fig


def plot_heatmap(attn_matrix, feature_names):
    mat = np.array(attn_matrix)
    n   = min(len(feature_names), mat.shape[0])
    fig = px.imshow(mat[:n, :n], x=feature_names[:n], y=feature_names[:n],
        color_continuous_scale="Turbo", title="Dynamic Dependency Intelligence Map",
        template="plotly_dark", aspect="auto")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(family="Outfit, sans-serif"), height=400,
        margin=dict(l=20, r=20, t=50, b=40), coloraxis_showscale=False,
    )
    return fig


def plot_gauge(confidence_pct, severity_word):
    val = float(confidence_pct.replace("%", ""))
    # Dynamic coloring: 0-40 -> green, 40-70 -> yellow, 70-100 -> red
    color = "#10b981" if val <= 40 else "#eab308" if val <= 70 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val, number=dict(suffix="%", font=dict(size=42, family="Outfit", color="#f1f5f9")),
        title=dict(text="Anomaly Risk Score", font=dict(size=14, family="Outfit", color="#94a3b8")),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#64748b", size=10)),
            bar=dict(color=color),
            bgcolor="rgba(15,23,42,0.4)",
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="rgba(16,185,129,0.12)"),
                dict(range=[40, 70], color="rgba(234,179,8,0.12)"),
                dict(range=[70, 100], color="rgba(239,68,68,0.12)"),
            ],
            threshold=dict(line=dict(color="#f1f5f9", width=2), thickness=0.8, value=val),
        )
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Outfit"),
        height=250, margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def plot_window_features(window, feature_names, peak_t, n_show=6):
    T, M = window.shape
    n_show = min(n_show, M)
    fig = make_subplots(rows=n_show, cols=1, shared_xaxes=True)
    for i in range(n_show):
        fig.add_trace(go.Scatter(x=list(range(T)), y=window[:, i].tolist(), mode="lines",
            name=feature_names[i], line=dict(width=1.5)), row=i+1, col=1)
        if 0 <= peak_t < T:
            fig.add_vline(x=peak_t, line_dash="dot", line_color="#ef4444", row=i+1, col=1)
    fig.update_layout(
        title="Signal Activity Profiles (red dashed = anomaly peak)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)", font=dict(family="Outfit, sans-serif"),
        height=550, margin=dict(l=40, r=40, t=50, b=30), showlegend=True,
    )
    return fig


# ── Main Application ──────────────────────────────────────────────────────────

def main():
    # ── 11. Project Branding ──────────────────────────────────────────────────
    st.markdown("""
        <div style="text-align:center; padding: 10px 0 0 0;">
            <div style="font-size:2.6rem; font-weight:800; font-family:Outfit;
                background: linear-gradient(90deg, #00f2fe, #4facfe, #a78bfa);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                FGEAD
            </div>
            <div style="font-size:0.85rem; color:#64748b; letter-spacing:2px; text-transform:uppercase; margin-top:-2px;">
                Feature Graph Explainable Anomaly Detection
            </div>
            <div style="font-size:1.05rem; color:#94a3b8; margin-top:8px; font-weight:300;">
                AI-Powered Telemetry Intelligence Platform
            </div>
            <div style="margin-top:12px; display:inline-block; padding: 5px 14px; 
                border-radius: 20px; font-size: 0.74rem; font-weight: 600; font-family:Outfit;
                background: rgba(0, 242, 254, 0.06); color: #00f2fe; 
                border: 1px solid rgba(0, 242, 254, 0.25);
                letter-spacing: 0.8px; text-transform: uppercase;">
                Graph Autoencoder + LSTM + Correlation Intelligence
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("---")

    # Load
    ctx = load_model_and_data()
    if not ctx["model_loaded"]:
        st.warning("⚠️ No trained model found. Run `python train.py` first, then refresh.")
        return

    # Sidebar
    st.sidebar.markdown("<h2 style='color:#00f2fe; font-family:Outfit; font-weight:700; font-size:1.3rem;'>⚙️ Control Panel</h2>", unsafe_allow_html=True)
    st.sidebar.write("---")
    threshold_k = st.sidebar.slider("Threshold (k·σ)", 1.0, 4.0, 2.0, 0.1)
    top_k       = st.sidebar.slider("Top K Features", 3, 10, 5)
    window_idx  = st.sidebar.number_input("Window Index", min_value=0,
                    max_value=len(ctx["test_win"]) - 1, value=20)

    # Compute scores
    scores = compute_all_scores(ctx["model"], ctx["test_win"], ctx["device"])
    labels = ctx["test_lbl"]
    thr    = scores.mean() + threshold_k * scores.std()
    preds  = (scores >= thr).astype(int)

    # ── Run explanation for selected window ───────────────────────────────────
    window = torch.FloatTensor(ctx["test_win"][window_idx:window_idx+1])
    report = ctx["explainer"].explain(window, window_start_abs=window_idx, device=ctx["device"])
    conf       = report["Q5_confidence"]
    alert      = conf["alert_level"]
    feat_data  = report["Q1_Q2_feature_analysis"]
    graph_data = report["Q3_graph_analysis"]
    rng        = report["Q4_temporal_range"]

    severity_icon = "🔴" if "CRITICAL" in alert else "🟠" if "HIGH" in alert else "🟡" if "MEDIUM" in alert else "🟢"
    severity_word = "CRITICAL" if "CRITICAL" in alert else "HIGH" if "HIGH" in alert else "MEDIUM" if "MEDIUM" in alert else "LOW"
    css_cls = "alert-critical" if "CRITICAL" in alert else "alert-high" if "HIGH" in alert else "alert-medium" if "MEDIUM" in alert else "alert-low"

    # ── 1. Section: AI-Powered Anomaly Intelligence Center ────────────────────
    st.markdown("<div class='section-header'>🧠 AI-Powered Anomaly Intelligence Center</div>", unsafe_allow_html=True)

    # ── 2. Status Cards with Colors ───────────────────────────────────────────
    health_pct = (1.0 - preds.mean()) * 100
    health_icon = "🟢" if health_pct > 90 else "🟡" if health_pct > 70 else "🔴"
    n_incidents = int(preds.sum())

    # Compute detection accuracy (F1 on test set as proxy)
    from sklearn.metrics import f1_score as sk_f1
    det_acc = sk_f1(labels, preds, zero_division=0) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">{health_icon}</div>
            <div class="metric-value">{health_pct:.1f}%</div>
            <div class="metric-label">System Health</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">🔴</div>
            <div class="metric-value">{n_incidents}</div>
            <div class="metric-label">Active Incidents</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{ctx['n_features']}</div>
            <div class="metric-label">Signals Monitored</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-value">{det_acc:.1f}%</div>
            <div class="metric-label">Detection Accuracy</div></div>""", unsafe_allow_html=True)

    # ── 9. Mini KPI Row ───────────────────────────────────────────────────────
    top_feats_all = feat_data["top_features"]
    n_critical = sum(1 for f in top_feats_all if f["z_score"] > 2.2)
    n_warning  = sum(1 for f in top_feats_all if 1.2 < f["z_score"] <= 2.2)
    n_healthy  = ctx["n_features"] - n_critical - n_warning

    st.markdown(f"""
        <div class="kpi-row">
            <span class="kpi-item" style="color:#10b981;">🟢 Healthy Signals: {n_healthy}</span>
            <span class="kpi-item" style="color:#eab308;">🟡 Warning Signals: {n_warning}</span>
            <span class="kpi-item" style="color:#ef4444;">🔴 Critical Signals: {n_critical}</span>
        </div>
    """, unsafe_allow_html=True)

    # ── Anomaly score timeline ────────────────────────────────────────────────
    st.plotly_chart(plot_anomaly_scores(scores, labels, thr), use_container_width=True)

    # ── 2. Section: Incident Investigation Console ────────────────────────────
    st.markdown("<div class='section-header'>🔬 Incident Investigation Console</div>", unsafe_allow_html=True)
    st.caption(f"Window **{window_idx}**  ·  Status: {'🔴 ANOMALY' if labels[window_idx] else '🟢 NORMAL'}")

    # ── 3. Glassmorphism Incident Card + 12. Gauge ────────────────────────────
    card_col, gauge_col = st.columns([3, 2])

    with card_col:
        # Affected components
        affected = [f["feature"] for f in top_feats_all[:5] if f["z_score"] >= 1.0]
        affected_html = "".join(f"<div style='padding:2px 0;'>• {name}</div>" for name in affected) if affected else "<div style='padding:2px 0;'>• No significant deviations</div>"

        # Recommended action
        root = report["root_cause"]
        if "memory" in root.lower() or "cache" in root.lower():
            action = "Investigate memory leaks or abnormal allocation patterns."
        elif "cpu" in root.lower():
            action = "Check for runaway processes or unexpected compute spikes."
        elif "disk" in root.lower() or "i/o" in root.lower():
            action = "Investigate disk throughput bottlenecks and I/O queue depth."
        elif "network" in root.lower() or "net" in root.lower():
            action = "Inspect network traffic patterns and connection pools."
        else:
            action = "Review top deviating features and correlate with recent deployments."

        # Primary cause (headline only, before the colon if present)
        primary = root.split(":")[0] if ":" in root else root

        st.markdown(f"""
            <div class="glass-panel">
                <div style="font-size:1.2rem; font-weight:700; margin-bottom:14px;">{severity_icon} 🚨 INCIDENT DETECTED</div>
                <div style="display:flex; gap:24px; flex-wrap:wrap;">
                    <div style="flex:1; min-width:160px;">
                        <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Severity</div>
                        <div style="font-size:1.05rem; font-weight:700; margin-top:2px;">{severity_icon} {severity_word}</div>
                    </div>
                    <div style="flex:1; min-width:160px;">
                        <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Confidence</div>
                        <div style="font-size:1.05rem; font-weight:700; margin-top:2px; color:#7c9df0;">{conf['confidence_pct']}</div>
                    </div>
                </div>
                <div style="margin-top:16px;">
                    <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Primary Cause</div>
                    <div style="font-size:0.95rem; color:#e2e8f0; margin-top:4px;">{primary}</div>
                </div>
                <div style="margin-top:12px;">
                    <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Affected Components</div>
                    <div style="font-size:0.88rem; color:#cbd5e1; margin-top:4px;">{affected_html}</div>
                </div>
                <div style="margin-top:12px;">
                    <div style="color:#64748b; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;">Recommended Action</div>
                    <div style="font-size:0.88rem; color:#10b981; margin-top:4px;">{action}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with gauge_col:
        st.plotly_chart(plot_gauge(conf["confidence_pct"], severity_word), use_container_width=True)

    # ── 7. AI Insight Panel ───────────────────────────────────────────────────
    # Build narrative from explainer data
    top3 = top_feats_all[:3]
    narrative_parts = []
    if top3:
        narrative_parts.append(f"The anomaly appears to originate from <b>{top3[0]['feature']}</b> ({top3[0]['direction']})")
        if len(top3) > 1:
            narrative_parts.append(f"with correlated impact on <b>{top3[1]['feature']}</b> and <b>{top3[2]['feature'] if len(top3) > 2 else 'related signals'}</b>.")
        else:
            narrative_parts.append(".")

    effects = [f"• {f['feature']} {f['direction']}" for f in top_feats_all[1:4] if f["z_score"] >= 1.0]
    broken_pairs = graph_data["broken_pairs"]
    for bp in broken_pairs[:1]:
        effects.append(f"• {bp['feature_a']} / {bp['feature_b']} {bp['break_type'].lower()}")

    effects_html = "<br/>".join(effects) if effects else "• No secondary effects detected"

    st.markdown(f"""
        <div class="glass-panel">
            <div style="font-size:1.05rem; font-weight:700; margin-bottom:10px;">🧠 AI Incident Summary</div>
            <div style="font-size:0.9rem; color:#cbd5e1; line-height:1.6;">
                {" ".join(narrative_parts)}
            </div>
            <div style="margin-top:10px; font-size:0.82rem; color:#64748b; text-transform:uppercase; letter-spacing:1px;">Observed Effects</div>
            <div style="font-size:0.88rem; color:#94a3b8; margin-top:4px; line-height:1.7;">{effects_html}</div>
            <div style="margin-top:10px; font-size:0.88rem; color:#64748b;">Confidence: <span style="color:#7c9df0; font-weight:700;">{conf['confidence_pct']}</span></div>
        </div>
    """, unsafe_allow_html=True)

    # ── Charts: Deviation + Heatmap ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_feature_deviations(feat_data["top_features"][:top_k]), use_container_width=True)

        # ── 6. Feature Table with Risk column ─────────────────────────────────
        feat_rows = []
        for f in feat_data["top_features"][:top_k]:
            risk = "🔴 Critical" if f["z_score"] > 2.2 else "🟠 High" if f["z_score"] > 1.2 else "🟡 Moderate" if f["z_score"] > 0.5 else "🟢 Low"
            feat_rows.append({
                "Feature":   f["feature"],
                "Actual":    f"{f['actual']:.4f}",
                "Predicted": f"{f['predicted']:.4f}",
                "Deviation": f"{f['deviation']:.4f}",
                "Risk":      risk,
            })
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

    with col2:
        # ── 4 + 8. Heatmap with Turbo colorscale ─────────────────────────────
        st.plotly_chart(plot_heatmap(graph_data["attn_matrix"], ctx["feature_names"]), use_container_width=True)

    # ── Signal profiles ───────────────────────────────────────────────────────
    st.plotly_chart(plot_window_features(ctx["test_win"][window_idx], ctx["feature_names"],
        feat_data["peak_timestep"]), use_container_width=True)

    # ── 5. Broken relationships with severity chips ───────────────────────────
    broken = graph_data["broken_pairs"]
    if broken:
        st.markdown("<div class='section-header'>🔗 Broken Feature Relationships</div>", unsafe_allow_html=True)
        sev_map = {"CRITICAL": "🔴 CRITICAL", "HIGH": "🟠 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}
        rows = []
        for p in broken:
            rows.append({
                "Feature A":     p["feature_a"],
                "Feature B":     p["feature_b"],
                "Expected":      p["expected_corr"],
                "Observed":      p["observed_corr"],
                "Gap":           p["gap"],
                "Type":          p["break_type"],
                "Severity":      sev_map.get(p["severity"], p["severity"]),
                "Interpretation": p["likely_meaning"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("All feature relationships remain within expected bounds.")

    # ── 10. Incident Timeline ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📅 Incident Timeline</div>", unsafe_allow_html=True)

    base_minute = window_idx  # use window index as proxy for minute offset
    timeline_events = []

    # Build timeline from actual data
    if rng["ranges"]:
        r0 = rng["ranges"][0]
        t_start = r0["abs_start"]
        timeline_events.append((base_minute, "Window ingested for analysis"))
        timeline_events.append((base_minute + 1, f"Anomaly onset detected at t={t_start}"))

    # Add top feature spikes
    for i, f in enumerate(top_feats_all[:2]):
        if f["z_score"] >= 1.0:
            timeline_events.append((base_minute + 2 + i, f"{f['feature']} {f['direction']} (z={f['z_score']:.1f}σ)"))

    if broken_pairs:
        bp = broken_pairs[0]
        timeline_events.append((base_minute + 4, f"Relationship breakdown: {bp['feature_a']} ↔ {bp['feature_b']}"))

    timeline_events.append((base_minute + 5, f"Alert generated — {severity_icon} {severity_word} ({conf['confidence_pct']})"))

    timeline_html = ""
    for minute_offset, event in timeline_events:
        h = 10 + (minute_offset // 60)
        m = minute_offset % 60
        timeline_html += f"""<div class="timeline-item">
            <span class="timeline-time">{h:02d}:{m:02d}</span>
            <span class="timeline-event">{event}</span>
        </div>"""

    st.markdown(f'<div class="glass-panel">{timeline_html}</div>', unsafe_allow_html=True)

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    st.write("---")
    with st.expander("📄 Raw Explanation JSON"):
        safe = {k: v for k, v in report.items() if k != "Q3_graph_analysis"}
        safe["Q3_graph_analysis"] = {
            "broken_pairs": report["Q3_graph_analysis"]["broken_pairs"],
            "n_broken": report["Q3_graph_analysis"]["n_broken"],
        }
        st.json(safe)


if __name__ == "__main__":
    main()
