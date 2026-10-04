"""
models/explainer.py

FGEAD Explainability Layer

Answers:
Q1 - Which features deviated most?
Q2 - What did they deviate from?
Q3 - Which feature relationships changed?
Q4 - When did the anomaly start and end?
Q5 - How confident is the alert?

IMPORTANT:
The FGEAD model operates on normalized data.

For human-readable explanations:
    normalized model input
        ->
    model prediction
        ->
    inverse transformation
        ->
    original data scale

Therefore actual and predicted values are always compared
in the SAME scale.
"""

from __future__ import annotations

from typing import Any, Dict, List
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch


# ============================================================================
# Q1 + Q2
# FEATURE DEVIATION ANALYZER
# ============================================================================

class FeatureDeviationAnalyzer:
    """
    Finds the most important feature deviations at the peak anomaly.

    Actual and predicted values MUST be in the same scale.
    """

    def __init__(
        self,
        feature_names: List[str],
        scaler=None,
    ):
        self.feature_names = feature_names
        self.scaler = scaler

    def analyze(
        self,
        x_window: np.ndarray,
        pred_window: np.ndarray,
        anomaly_scores: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        x_window = np.asarray(x_window, dtype=np.float64)
        pred_window = np.asarray(pred_window, dtype=np.float64)
        anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64)

        T, M = x_window.shape

        if len(anomaly_scores) == 0:
            return {
                "peak_timestep": 0,
                "peak_score": 0.0,
                "top_features": [],
            }

        # Scores correspond to prediction of t=1 ... T-1.
        score_index = int(np.argmax(anomaly_scores))

        # Actual timestep being predicted.
        peak_t = score_index + 1

        actual_at_peak = x_window[peak_t]
        pred_at_peak = pred_window[score_index]

        deviations = np.abs(
            actual_at_peak - pred_at_peak
        )

        # Robust scale estimation.
        #
        # Standard deviation can become extremely small for SMD
        # features that are almost constant inside a short window.
        window_std = np.std(
            x_window,
            axis=0,
            ddof=0,
        )

        window_std = np.maximum(
            window_std,
            1e-8,
        )

        z_scores = deviations / window_std

        top_idx = np.argsort(
            deviations
        )[::-1][:top_k]

        top_features = []

        for i in top_idx:

            i = int(i)

            name = (
                self.feature_names[i]
                if i < len(self.feature_names)
                else f"feature_{i + 1:02d}"
            )

            actual = float(actual_at_peak[i])
            predicted = float(pred_at_peak[i])
            deviation = float(deviations[i])
            z_score = float(z_scores[i])

            if actual > predicted:
                direction = "↑ spike"
            elif actual < predicted:
                direction = "↓ drop"
            else:
                direction = "≈ stable"

            top_features.append(
                {
                    "feature": name,
                    "actual": round(actual, 4),
                    "predicted": round(predicted, 4),
                    "deviation": round(deviation, 4),
                    "z_score": round(z_score, 2),
                    "direction": direction,
                }
            )

        return {
            "peak_timestep": peak_t,
            "peak_score": round(
                float(anomaly_scores[score_index]),
                4,
            ),
            "top_features": top_features,
        }


# ============================================================================
# Q3
# GRAPH RELATIONSHIP AUDITOR
# ============================================================================

class GraphRelationshipAuditor:
    """
    Compares learned graph relationships with observed
    relationships around the anomaly.

    IMPORTANT:
    The attention matrix is NOT a Pearson correlation matrix.

    It represents learned feature dependency strength.

    Therefore this module reports a relationship change rather
    than claiming that attention values are literal correlations.
    """

    def __init__(
        self,
        feature_names: List[str],
        break_threshold: float = 0.20,
        min_expected_strength: float = 0.10,
    ):
        self.feature_names = feature_names
        self.break_threshold = break_threshold
        self.min_expected_strength = min_expected_strength

    def audit(
        self,
        x_window: np.ndarray,
        attn_weights: np.ndarray,
        anomaly_scores: np.ndarray,
    ) -> Dict[str, Any]:

        x_window = np.asarray(
            x_window,
            dtype=np.float64,
        )

        attn_weights = np.asarray(
            attn_weights,
            dtype=np.float64,
        )

        anomaly_scores = np.asarray(
            anomaly_scores,
            dtype=np.float64,
        )

        M = x_window.shape[1]

        broken_pairs = []

        # ------------------------------------------------------------
        # Locate anomaly region
        # ------------------------------------------------------------

        if len(anomaly_scores) > 0:

            peak_t = int(
                np.argmax(anomaly_scores)
            ) + 1

        else:

            peak_t = min(
                len(x_window) // 2,
                len(x_window) - 1,
            )

        start = max(
            0,
            peak_t - 10,
        )

        end = min(
            len(x_window),
            peak_t + 11,
        )

        local_window = x_window[start:end]

        # ------------------------------------------------------------
        # Observed correlation
        # ------------------------------------------------------------

        if len(local_window) >= 3:

            with np.errstate(
                divide="ignore",
                invalid="ignore",
            ):

                obs_corr = np.corrcoef(
                    local_window.T
                )

            obs_corr = np.nan_to_num(
                obs_corr,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        else:

            obs_corr = np.eye(M)

        # ------------------------------------------------------------
        # Compare relationships
        # ------------------------------------------------------------

        for i in range(M):

            for j in range(i + 1, M):

                expected = float(
                    attn_weights[i, j]
                )

                observed = float(
                    obs_corr[i, j]
                )

                if expected < self.min_expected_strength:
                    continue

                # Attention is non-negative.
                # Normalize it to a relationship magnitude.
                #
                # This makes the comparison more interpretable.
                expected_relationship = min(
                    1.0,
                    max(
                        0.0,
                        expected,
                    ),
                )

                observed_strength = abs(
                    observed
                )

                gap = abs(
                    observed_strength
                    - expected_relationship
                )

                if gap < self.break_threshold:
                    continue

                if (
                    expected_relationship >= 0.70
                    and observed_strength < 0.10
                ):

                    break_type = (
                        "COMPLETE DECOUPLING"
                    )

                elif (
                    observed < 0
                    and expected_relationship >= 0.20
                ):

                    break_type = (
                        "SIGN REVERSAL"
                    )

                else:

                    break_type = (
                        "RELATIONSHIP CHANGE"
                    )

                name_a = (
                    self.feature_names[i]
                    if i < len(self.feature_names)
                    else f"feature_{i + 1:02d}"
                )

                name_b = (
                    self.feature_names[j]
                    if j < len(self.feature_names)
                    else f"feature_{j + 1:02d}"
                )

                if gap >= 0.70:
                    severity = "CRITICAL"

                elif gap >= 0.50:
                    severity = "HIGH"

                else:
                    severity = "MEDIUM"

                broken_pairs.append(
                    {
                        "feature_a": name_a,
                        "feature_b": name_b,
                        "expected_corr": round(
                            expected_relationship,
                            3,
                        ),
                        "observed_corr": round(
                            observed,
                            3,
                        ),
                        "gap": round(
                            gap,
                            3,
                        ),
                        "break_type": break_type,
                        "severity": severity,
                        "likely_meaning":
                            self._infer_meaning(
                                name_a,
                                name_b,
                                break_type,
                            ),
                    }
                )

        broken_pairs.sort(
            key=lambda x: x["gap"],
            reverse=True,
        )

        return {
            "broken_pairs": broken_pairs,
            "n_broken": len(broken_pairs),
            "obs_corr_matrix": obs_corr.tolist(),
            "attn_matrix": attn_weights.tolist(),
        }

    def _infer_meaning(
        self,
        feat_a: str,
        feat_b: str,
        break_type: str,
    ) -> str:

        a = feat_a.lower()
        b = feat_b.lower()

        if (
            "cpu" in a
            or "cpu" in b
        ):

            return (
                "CPU-related feature relationship changed"
            )

        if (
            "mem" in a
            or "mem" in b
            or "swap" in a
            or "swap" in b
        ):

            return (
                "Memory-related feature relationship changed"
            )

        if (
            "disk" in a
            or "disk" in b
            or "io" in a
            or "io" in b
        ):

            return (
                "I/O-related feature relationship changed"
            )

        if (
            "net" in a
            or "net" in b
        ):

            return (
                "Network-related feature relationship changed"
            )

        return (
            f"Relationship between {feat_a} and "
            f"{feat_b} changed ({break_type})"
        )


# ============================================================================
# Q4
# ANOMALY RANGE DETECTOR
# ============================================================================

class AnomalyRangeDetector:

    def __init__(
        self,
        k_sigma: float = 2.0,
        merge_gap: int = 3,
        min_duration: int = 1,
    ):
        self.k_sigma = k_sigma
        self.merge_gap = merge_gap
        self.min_duration = min_duration

    def detect(
        self,
        anomaly_scores: np.ndarray,
        window_start_abs: int = 0,
        timestep_seconds: int = 60,
    ) -> Dict[str, Any]:

        scores = np.asarray(
            anomaly_scores,
            dtype=np.float64,
        )

        if len(scores) == 0:

            return {
                "threshold": 0.0,
                "n_anomaly_ranges": 0,
                "ranges": [],
                "total_anomalous_steps": 0,
                "anomaly_fraction": 0.0,
            }

        mean = float(
            np.mean(scores)
        )

        std = float(
            np.std(scores)
        )

        threshold = mean + self.k_sigma * std

        is_anomalous = (
            scores > threshold
        ).astype(int)

        padded = np.concatenate(
            [
                [0],
                is_anomalous,
                [0],
            ]
        )

        diff = np.diff(
            padded
        )

        starts = np.where(
            diff == 1
        )[0]

        ends = (
            np.where(
                diff == -1
            )[0]
            - 1
        )

        merged = []

        for s, e in zip(
            starts,
            ends,
        ):

            s = int(s)
            e = int(e)

            if (
                merged
                and (
                    s
                    - merged[-1][1]
                    <= self.merge_gap
                )
            ):

                merged[-1] = (
                    merged[-1][0],
                    e,
                )

            else:

                merged.append(
                    (s, e)
                )

        ranges = []

        for s, e in merged:

            duration_steps = (
                e - s + 1
            )

            if (
                duration_steps
                < self.min_duration
            ):
                continue

            peak = (
                int(
                    np.argmax(
                        scores[
                            s:e + 1
                        ]
                    )
                )
                + s
            )

            duration_seconds = (
                duration_steps
                * timestep_seconds
            )

            ranges.append(
                {
                    "window_start": s,
                    "window_end": e,
                    "abs_start":
                        window_start_abs + s,
                    "abs_end":
                        window_start_abs + e,
                    "peak_timestep": peak,
                    "duration_steps":
                        duration_steps,
                    "duration_secs":
                        duration_seconds,
                    "duration_human":
                        self._human_duration(
                            duration_seconds
                        ),
                    "peak_score":
                        round(
                            float(
                                scores[peak]
                            ),
                            4,
                        ),
                    "mean_score":
                        round(
                            float(
                                scores[
                                    s:e + 1
                                ].mean()
                            ),
                            4,
                        ),
                }
            )

        return {
            "threshold":
                round(
                    threshold,
                    4,
                ),
            "n_anomaly_ranges":
                len(ranges),
            "ranges": ranges,
            "total_anomalous_steps":
                int(
                    is_anomalous.sum()
                ),
            "anomaly_fraction":
                round(
                    float(
                        is_anomalous.mean()
                    ),
                    3,
                ),
        }

    @staticmethod
    def _human_duration(
        seconds: int,
    ) -> str:

        if seconds < 60:
            return f"{seconds}s"

        if seconds < 3600:
            return (
                f"{seconds // 60}m "
                f"{seconds % 60}s"
            )

        return (
            f"{seconds // 3600}h "
            f"{(seconds % 3600) // 60}m"
        )


# ============================================================================
# Q5
# CONFIDENCE SCORER
# ============================================================================

class ConfidenceScorer:

    def __init__(
        self,
        w_score: float = 0.40,
        w_duration: float = 0.25,
        w_features: float = 0.20,
        w_graph: float = 0.15,
    ):

        total = (
            w_score
            + w_duration
            + w_features
            + w_graph
        )

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Confidence weights must sum to 1.0"
            )

        self.w_score = w_score
        self.w_duration = w_duration
        self.w_features = w_features
        self.w_graph = w_graph

    def score(
        self,
        peak_score: float,
        threshold: float,
        anomaly_duration: int,
        window_size: int,
        n_deviating_feats: int,
        n_total_feats: int,
        n_broken_pairs: int,
        n_total_edges: int,
    ) -> Dict[str, Any]:

        threshold = max(
            float(threshold),
            1e-8,
        )

        score_signal = min(
            1.0,
            max(
                0.0,
                (
                    peak_score
                    - threshold
                )
                / (
                    0.25
                    * threshold
                ),
            ),
        )

        duration_signal = min(
            1.0,
            anomaly_duration
            / max(
                window_size * 0.30,
                1,
            ),
        )

        feature_signal = min(
            1.0,
            n_deviating_feats
            / max(
                n_total_feats * 0.30,
                1,
            ),
        )

        graph_signal = min(
            1.0,
            n_broken_pairs
            / max(
                n_total_edges * 0.20,
                1,
            ),
        )

        confidence = (
            self.w_score
            * score_signal
            + self.w_duration
            * duration_signal
            + self.w_features
            * feature_signal
            + self.w_graph
            * graph_signal
        )

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

        if confidence >= 0.80:

            alert_level = (
                "🔴 CRITICAL — Act immediately"
            )

        elif confidence >= 0.60:

            alert_level = (
                "🟠 HIGH — Investigate now"
            )

        elif confidence >= 0.40:

            alert_level = (
                "🟡 MEDIUM — Monitor closely"
            )

        else:

            alert_level = (
                "🟢 LOW — Log and watch"
            )

        return {
            "confidence":
                round(confidence, 3),

            "confidence_pct":
                f"{confidence * 100:.1f}%",

            "alert_level":
                alert_level,

            "signals": {
                "score_signal":
                    round(
                        score_signal,
                        3,
                    ),
                "duration_signal":
                    round(
                        duration_signal,
                        3,
                    ),
                "feature_signal":
                    round(
                        feature_signal,
                        3,
                    ),
                "graph_signal":
                    round(
                        graph_signal,
                        3,
                    ),
            },
        }


# ============================================================================
# MASTER EXPLAINER
# ============================================================================

class FGEADExplainer:

    def __init__(
        self,
        model,
        feature_names: List[str],
        scaler=None,
        top_k: int = 5,
        timestep_seconds: int = 60,
    ):

        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        self.top_k = top_k
        self.timestep_seconds = timestep_seconds

        self.feat_analyzer = (
            FeatureDeviationAnalyzer(
                feature_names,
                scaler,
            )
        )

        self.graph_auditor = (
            GraphRelationshipAuditor(
                feature_names,
            )
        )

        self.range_detector = (
            AnomalyRangeDetector()
        )

        self.conf_scorer = (
            ConfidenceScorer()
        )

    def explain(
        self,
        window: torch.Tensor,
        window_start_abs: int = 0,
        device: str = "cpu",
    ) -> Dict[str, Any]:

        self.model.eval()

        # ------------------------------------------------------------
        # MODEL INFERENCE
        # ------------------------------------------------------------

        with torch.no_grad():

            predictions, attn_weights, anomaly_scores = (
                self.model(
                    window.to(device)
                )
            )

        # ------------------------------------------------------------
        # NORMALIZED MODEL VALUES
        # ------------------------------------------------------------

        x_np = (
            window
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        pred_np = (
            predictions
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        score_np = (
            anomaly_scores
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        attn_np = (
            attn_weights
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        T, M = x_np.shape

        # ------------------------------------------------------------
        # IMPORTANT SCALE FIX
        # ------------------------------------------------------------
        #
        # FGEAD works on normalized data.
        #
        # For the explanation we inverse-transform BOTH:
        #
        #   actual values
        #   predicted values
        #
        # This ensures actual and predicted are comparable.
        # ------------------------------------------------------------

        if self.scaler is not None:

            try:

                x_display = (
                    self.scaler
                    .inverse_transform(
                        x_np.reshape(
                            -1,
                            M,
                        )
                    )
                    .reshape(
                        T,
                        M,
                    )
                )

                pred_display = (
                    self.scaler
                    .inverse_transform(
                        pred_np.reshape(
                            -1,
                            M,
                        )
                    )
                    .reshape(
                        T - 1,
                        M,
                    )
                )

                scale_status = (
                    "Original SMD scale"
                )

            except Exception as exc:

                print(
                    "[Warning] Explanation "
                    "inverse transformation failed:"
                )

                print(
                    f"          {exc}"
                )

                x_display = x_np
                pred_display = pred_np

                scale_status = (
                    "Normalized scale"
                )

        else:

            x_display = x_np
            pred_display = pred_np

            scale_status = (
                "Normalized scale "
                "(no scaler supplied)"
            )

        # ------------------------------------------------------------
        # Q1 + Q2
        # ------------------------------------------------------------

        feat_result = (
            self.feat_analyzer.analyze(
                x_display,
                pred_display,
                score_np,
                self.top_k,
            )
        )

        # ------------------------------------------------------------
        # Q3
        # ------------------------------------------------------------

        graph_result = (
            self.graph_auditor.audit(
                x_display,
                attn_np,
                score_np,
            )
        )

        # ------------------------------------------------------------
        # Q4
        # ------------------------------------------------------------

        range_result = (
            self.range_detector.detect(
                score_np,
                window_start_abs,
                self.timestep_seconds,
            )
        )

        # ------------------------------------------------------------
        # Q5
        # ------------------------------------------------------------

        n_deviating = sum(
            1
            for feature in feat_result[
                "top_features"
            ]
            if feature["z_score"] > 2.0
        )

        n_edges = max(
            int(
                (
                    attn_np > 0.5
                ).sum()
                // 2
            ),
            1,
        )

        duration = (
            range_result[
                "total_anomalous_steps"
            ]
        )

        conf_result = (
            self.conf_scorer.score(
                peak_score=
                    feat_result[
                        "peak_score"
                    ],

                threshold=
                    range_result[
                        "threshold"
                    ],

                anomaly_duration=
                    duration,

                window_size=T,

                n_deviating_feats=
                    n_deviating,

                n_total_feats=M,

                n_broken_pairs=
                    graph_result[
                        "n_broken"
                    ],

                n_total_edges=n_edges,
            )
        )

        # ------------------------------------------------------------
        # ROOT CAUSE
        # ------------------------------------------------------------

        root_cause = (
            self._infer_root_cause(
                feat_result[
                    "top_features"
                ],
                graph_result[
                    "broken_pairs"
                ],
            )
        )

        return {
            "Q1_Q2_feature_analysis":
                feat_result,

            "Q3_graph_analysis":
                graph_result,

            "Q4_temporal_range":
                range_result,

            "Q5_confidence":
                conf_result,

            "root_cause":
                root_cause,

            "metadata": {
                "window_start_abs":
                    window_start_abs,

                "window_size":
                    T,

                "n_features":
                    M,

                "timestep_seconds":
                    self.timestep_seconds,

                "display_scale":
                    scale_status,
            },
        }

    # ========================================================================
    # ROOT CAUSE
    # ========================================================================

    def _infer_root_cause(
        self,
        top_feats,
        broken_pairs,
    ) -> str:

        names = [
            f["feature"].lower()
            for f in top_feats
        ]

        has_cpu = any(
            "cpu" in n
            for n in names
        )

        has_mem = any(
            (
                "mem" in n
                or "swap" in n
                or "cache" in n
            )
            for n in names
        )

        has_disk = any(
            (
                "disk" in n
                or "io" in n
            )
            for n in names
        )

        has_net = any(
            "net" in n
            for n in names
        )

        if has_cpu and has_mem:

            headline = (
                "Likely CPU and memory pressure event"
            )

        elif has_mem:

            headline = (
                "Likely memory-pressure event"
            )

        elif has_cpu:

            headline = (
                "Likely CPU-related anomaly"
            )

        elif has_disk and has_net:

            headline = (
                "Likely I/O and network disturbance"
            )

        elif has_disk:

            headline = (
                "Likely disk/I/O anomaly"
            )

        elif has_net:

            headline = (
                "Likely network-related anomaly"
            )

        else:

            headline = (
                "Unusual multivariate pattern detected"
            )

        parts = []

        for feature in top_feats[:3]:

            if feature["z_score"] >= 1.0:

                parts.append(
                    f"{feature['feature']} "
                    f"{feature['direction']}"
                )

        if broken_pairs:

            pair = broken_pairs[0]

            parts.append(
                f"{pair['feature_a']}/"
                f"{pair['feature_b']} "
                f"{pair['break_type'].lower()}"
            )

        if parts:

            return (
                f"{headline}: "
                + ", ".join(parts)
            )

        return headline

    # ========================================================================
    # FORMAT REPORT
    # ========================================================================

    def format_report(
        self,
        report: Dict[str, Any],
    ) -> str:

        lines = [
            "=" * 64,
            "   FGEAD — ANOMALY EXPLANATION REPORT",
            "=" * 64,
        ]

        metadata = report[
            "metadata"
        ]

        lines.append(
            f"\n📐 Display scale: "
            f"{metadata['display_scale']}"
        )

        feat = report[
            "Q1_Q2_feature_analysis"
        ]

        peak_window_t = (
            feat["peak_timestep"]
        )

        peak_absolute_t = (
            metadata[
                "window_start_abs"
            ]
            + peak_window_t
        )

        lines.append(
            "\n📍 Peak anomaly:"
        )

        lines.append(
            f"   Window timestep : "
            f"t={peak_window_t}"
        )

        lines.append(
            f"   Absolute timestep: "
            f"t={peak_absolute_t}"
        )

        lines.append(
            f"   Score           : "
            f"{feat['peak_score']:.4f}"
        )

        lines.append(
            "\n🪟 Analysis Window:"
        )

        lines.append(
            f"   Absolute start  : "
            f"t={metadata['window_start_abs']}"
        )

        lines.append(
            f"   Absolute end    : "
            f"t={metadata['window_start_abs'] + metadata['window_size'] - 1}"
        )

        lines.append(
            "\n🔍 Top Contributing Features:"
        )

        for feature in feat[
            "top_features"
        ]:

            lines.append(
                f"   {feature['feature']:18s} "
                f"actual={feature['actual']:12.4f} "
                f"predicted={feature['predicted']:12.4f} "
                f"dev={feature['deviation']:12.4f} "
                f"z={feature['z_score']:.1f}σ "
                f"{feature['direction']}"
            )

        graph = report[
            "Q3_graph_analysis"
        ]

        if graph[
            "broken_pairs"
        ]:

            lines.append(
                f"\n🔗 Changed Feature Relationships "
                f"({graph['n_broken']} found):"
            )

            for pair in graph[
                "broken_pairs"
            ][:5]:

                lines.append(
                    f"   {pair['feature_a']} ↔ "
                    f"{pair['feature_b']}  "
                    f"expected={pair['expected_corr']:.2f}  "
                    f"observed={pair['observed_corr']:.2f}  "
                    f"[{pair['break_type']}]"
                )

        else:

            lines.append(
                "\n🔗 No major feature relationship "
                "changes detected."
            )

        rng = report[
            "Q4_temporal_range"
        ]

        lines.append(
            f"\n⏱ Anomaly Duration: "
            f"{rng['total_anomalous_steps']} steps "
            f"({rng['anomaly_fraction']:.1%} of window)"
        )

        if rng["ranges"]:

            r0 = rng["ranges"][0]

            lines.append(
                f"   Start: t={r0['abs_start']}  "
                f"End: t={r0['abs_end']}  "
                f"Duration: {r0['duration_human']}"
            )

            lines.append(
                f"   Range peak: "
                f"t={metadata['window_start_abs'] + r0['peak_timestep']}"
            )

        conf = report[
            "Q5_confidence"
        ]

        lines.append(
            f"\n📊 Alert Confidence: "
            f"{conf['confidence_pct']}  →  "
            f"{conf['alert_level']}"
        )

        lines.append(
            "\n💡 Likely Root Cause:"
        )

        lines.append(
            f"   {report['root_cause']}"
        )

        lines.append(
            "\nℹ️ Confidence is an "
            "explainability-based alert score, "
            "not a calibrated probability."
        )

        lines.append(
            "=" * 64
        )

        return "\n".join(lines)