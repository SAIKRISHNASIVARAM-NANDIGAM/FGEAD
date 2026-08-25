"""
api/main.py

FGEAD — FastAPI Backend
Server Machine Dataset (SMD) inference API.

Run:
    uvicorn api.main:app --reload --port 8000

Docs:
    http://127.0.0.1:8000/docs

Health:
    http://127.0.0.1:8000/health
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# FGEAD IMPORTS
# ============================================================================

from data.smd_loader import SMDLoader
from models.fgead import FGEAD
from models.explainer import FGEADExplainer


# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW_SIZE = 60
STRIDE = 5

MACHINE_ID = "1-1"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "fgead_smd_machine_1_1.pt"
)

SMD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "SMD"
)

# Official threshold from evaluate_smd_final.py
OFFICIAL_WINDOW_THRESHOLD = 2.073376

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="FGEAD — Explainable Anomaly Detection API",
    description=(
        "Feature Graph-based Explainable Anomaly Detector "
        "for the Server Machine Dataset."
    ),
    version="2.1.0",
)


# ============================================================================
# GLOBAL STATE
# ============================================================================

_model: Optional[FGEAD] = None
_explainer: Optional[FGEADExplainer] = None
_loader: Optional[SMDLoader] = None

_train_data: Optional[np.ndarray] = None
_test_data: Optional[np.ndarray] = None
_test_labels: Optional[np.ndarray] = None

_n_features: int = 38
_model_loaded: bool = False


# ============================================================================
# FEATURE NAMES
# ============================================================================

FEATURE_NAMES = [
    f"feature_{i:02d}"
    for i in range(38)
]


# ============================================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================================

class WindowInput(BaseModel):
    """
    One SMD telemetry window.

    Expected shape:
        [60, 38]

    Rows:
        timesteps

    Columns:
        features
    """

    data: List[List[float]] = Field(
        ...,
        description="60 timesteps × 38 SMD features.",
    )

    window_start_abs: int = Field(
        0,
        description="Absolute start timestep of this window.",
    )


class PredictResponse(BaseModel):
    machine: str

    is_anomaly: bool

    anomaly_score: float

    threshold: float

    confidence_pct: str

    alert_level: str

    peak_timestep: Optional[int]

    window_start: int

    window_end: int

    anomaly_duration_steps: int

    top_features: List[Dict[str, Any]]

    broken_pairs: List[Dict[str, Any]]

    anomaly_ranges: List[Dict[str, Any]]

    root_cause: str

    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    machine: str
    n_features: int
    window_size: int
    stride: int


class DatasetResponse(BaseModel):
    machine: str
    train_timesteps: int
    test_timesteps: int
    features: int
    test_anomalies: int
    anomaly_rate: float
    train_windows: int
    test_windows: int


class MachineResponse(BaseModel):
    machine: str
    features: int
    feature_names: List[str]


# ============================================================================
# CHECKPOINT
# ============================================================================

def load_checkpoint() -> Dict[str, Any]:
    """
    Load the SMD FGEAD checkpoint.

    Expected checkpoint structure:

        model_state_dict
        machine
        n_features
        config
        best_loss
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"FGEAD checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format. "
            "Expected a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    return checkpoint


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
def startup_event() -> None:

    global _model
    global _explainer
    global _loader
    global _train_data
    global _test_data
    global _test_labels
    global _n_features
    global _model_loaded

    print()
    print("=" * 70)
    print("FGEAD — FASTAPI BACKEND STARTUP")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Machine    : {MACHINE_ID}")
    print(f"SMD root   : {SMD_ROOT}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")

    try:

        # ------------------------------------------------------------------
        # 1. LOAD DATASET
        # ------------------------------------------------------------------

        print()
        print("[1/4] Loading SMD dataset...")

        _loader = SMDLoader(
            root=str(SMD_ROOT),
            window_size=WINDOW_SIZE,
            stride=STRIDE,
        )

        (
            _train_data,
            _test_data,
            _test_labels,
        ) = _loader.load_machine(
            MACHINE_ID
        )

        _n_features = int(
            _train_data.shape[1]
        )

        print(
            f"[OK] SMD Machine {MACHINE_ID} loaded."
        )

        # ------------------------------------------------------------------
        # 2. NORMALIZATION
        # ------------------------------------------------------------------

        print()
        print("[2/4] Fitting TRAIN-only normalization...")

        _loader.normalize(
            _train_data,
            _test_data,
        )

        print(
            "[OK] Scaler fitted using TRAIN data only."
        )

        # ------------------------------------------------------------------
        # 3. LOAD MODEL
        # ------------------------------------------------------------------

        print()
        print("[3/4] Loading FGEAD checkpoint...")

        checkpoint = load_checkpoint()

        checkpoint_features = int(
            checkpoint.get(
                "n_features",
                _n_features,
            )
        )

        if checkpoint_features != _n_features:
            raise RuntimeError(
                "Checkpoint/data feature mismatch:\n"
                f"checkpoint = {checkpoint_features}\n"
                f"dataset    = {_n_features}"
            )

        _model = FGEAD(
            n_features=_n_features
        ).to(DEVICE)

        _model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        _model.eval()

        print("[OK] FGEAD model loaded.")

        if "machine" in checkpoint:
            print(
                f"     Checkpoint machine : "
                f"{checkpoint['machine']}"
            )

        if "best_loss" in checkpoint:
            print(
                f"     Checkpoint loss    : "
                f"{checkpoint['best_loss']}"
            )

        # ------------------------------------------------------------------
        # 4. EXPLAINER
        # ------------------------------------------------------------------

        print()
        print("[4/4] Initializing explainability engine...")

        _explainer = FGEADExplainer(
            _model,
            FEATURE_NAMES,
        )

        print(
            "[OK] FGEAD explainability engine ready."
        )

        _model_loaded = True

        print()
        print("=" * 70)
        print("FGEAD API READY")
        print("=" * 70)

        print(
            f"Machine       : {MACHINE_ID}"
        )

        print(
            f"Features      : {_n_features}"
        )

        print(
            f"Window        : {WINDOW_SIZE}"
        )

        print(
            f"Stride        : {STRIDE}"
        )

        print(
            f"Threshold     : "
            f"{OFFICIAL_WINDOW_THRESHOLD:.6f}"
        )

        print(
            f"Device        : {DEVICE}"
        )

        print("=" * 70)
        print()

    except Exception as exc:

        _model_loaded = False

        print()
        print(
            "[ERROR] FGEAD API startup failed."
        )

        print(
            f"        {exc}"
        )

        print()

        _model = None
        _explainer = None


# ============================================================================
# HEALTH
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:

    return HealthResponse(
        status=(
            "ok"
            if _model_loaded
            else "model_not_loaded"
        ),
        model_loaded=_model_loaded,
        device=DEVICE,
        machine=MACHINE_ID,
        n_features=_n_features,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )


# ============================================================================
# DATASET INFORMATION
# ============================================================================

@app.get(
    "/dataset",
    response_model=DatasetResponse,
)
def dataset_info() -> DatasetResponse:

    if _loader is None:
        raise HTTPException(
            status_code=503,
            detail="SMD loader is not initialized.",
        )

    if _train_data is None:
        raise HTTPException(
            status_code=503,
            detail="Training data is not loaded.",
        )

    if _test_data is None:
        raise HTTPException(
            status_code=503,
            detail="Test data is not loaded.",
        )

    if _test_labels is None:
        raise HTTPException(
            status_code=503,
            detail="Test labels are not loaded.",
        )

    train_windows = max(
        0,
        (
            len(_train_data)
            - WINDOW_SIZE
        )
        // STRIDE
        + 1,
    )

    test_windows = max(
        0,
        (
            len(_test_data)
            - WINDOW_SIZE
        )
        // STRIDE
        + 1,
    )

    anomaly_count = int(
        np.sum(_test_labels)
    )

    anomaly_rate = (
        anomaly_count / len(_test_labels)
        if len(_test_labels) > 0
        else 0.0
    )

    return DatasetResponse(
        machine=MACHINE_ID,
        train_timesteps=int(
            len(_train_data)
        ),
        test_timesteps=int(
            len(_test_data)
        ),
        features=int(_n_features),
        test_anomalies=anomaly_count,
        anomaly_rate=float(
            anomaly_rate
        ),
        train_windows=int(
            train_windows
        ),
        test_windows=int(
            test_windows
        ),
    )


# ============================================================================
# MACHINE INFORMATION
# ============================================================================

@app.get(
    "/machine",
    response_model=MachineResponse,
)
def machine_info() -> MachineResponse:

    return MachineResponse(
        machine=MACHINE_ID,
        features=_n_features,
        feature_names=FEATURE_NAMES,
    )


# ============================================================================
# PREDICTION
# ============================================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(
    payload: WindowInput,
) -> PredictResponse:

    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "FGEAD model is not loaded. "
                "Check /health."
            ),
        )

    if _loader is None:
        raise HTTPException(
            status_code=503,
            detail="SMD loader is unavailable.",
        )

    if _explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine is unavailable.",
        )

    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="FGEAD model is unavailable.",
        )

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------

    try:

        arr = np.asarray(
            payload.data,
            dtype=np.float32,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid numeric input: {exc}"
            ),
        )

    expected_shape = (
        WINDOW_SIZE,
        _n_features,
    )

    if arr.shape != expected_shape:

        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid window shape. "
                f"Expected {expected_shape}, "
                f"received {arr.shape}."
            ),
        )

    if not np.isfinite(arr).all():

        raise HTTPException(
            status_code=422,
            detail=(
                "Input contains NaN or infinite values."
            ),
        )

    # ------------------------------------------------------------------
    # TRAIN-ONLY NORMALIZATION
    # ------------------------------------------------------------------

    try:

        arr_norm = _loader.scaler.transform(
            arr
        ).astype(np.float32)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Normalization failed: {exc}"
            ),
        )

    # ------------------------------------------------------------------
    # TORCH WINDOW
    # ------------------------------------------------------------------

    window = torch.from_numpy(
        arr_norm
    ).unsqueeze(0).to(DEVICE)

    # ------------------------------------------------------------------
    # IMPORTANT:
    # OFFICIAL EVALUATOR SCORE
    #
    # evaluate_smd_final.py does:
    #
    #     _, _, anomaly_scores = model(x)
    #     score = float(
    #         anomaly_scores.max()
    #         .detach()
    #         .cpu()
    #     )
    #
    # We MUST use exactly the same calculation here.
    # ------------------------------------------------------------------

    try:

        with torch.no_grad():

            (
                predictions,
                attention,
                anomaly_scores,
            ) = _model(
                window
            )

        anomaly_scores_cpu = (
            anomaly_scores
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        if anomaly_scores_cpu.size == 0:

            raise RuntimeError(
                "Model returned an empty anomaly-score array."
            )

        # EXACT official evaluation score.
        anomaly_score = float(
            np.max(
                anomaly_scores_cpu
            )
        )

        # Same peak timestep convention:
        # model scores correspond to prediction timesteps
        # 1 ... T-1, therefore +1 relative to window index 0.
        score_peak_index = int(
            np.argmax(
                anomaly_scores_cpu
            )
        )

        peak_timestep = (
            score_peak_index + 1
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"FGEAD model scoring failed: {exc}"
            ),
        )

    # ------------------------------------------------------------------
    # EXPLAINABILITY
    # ------------------------------------------------------------------

    try:

        report = _explainer.explain(
            window,
            window_start_abs=(
                payload.window_start_abs
            ),
            device=DEVICE,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"FGEAD explanation failed: {exc}"
            ),
        )

    # ------------------------------------------------------------------
    # REPORT EXTRACTION
    # ------------------------------------------------------------------

    try:

        confidence = report[
            "Q5_confidence"
        ]

        feature_analysis = report[
            "Q1_Q2_feature_analysis"
        ]

        temporal = report[
            "Q4_temporal_range"
        ]

        graph = report[
            "Q3_graph_analysis"
        ]

        root_cause = report.get(
            "root_cause",
            "No root cause explanation available.",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Invalid explanation report: {exc}"
            ),
        )

    # ------------------------------------------------------------------
    # OFFICIAL ANOMALY DECISION
    # ------------------------------------------------------------------

    is_anomaly = (
        anomaly_score
        >= OFFICIAL_WINDOW_THRESHOLD
    )

    # ------------------------------------------------------------------
    # Prefer model-derived peak timestep.
    # ------------------------------------------------------------------

    explanation_peak = (
        feature_analysis.get(
            "peak_timestep"
        )
    )

    if explanation_peak is not None:

        try:
            peak_timestep = int(
                explanation_peak
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # DURATION
    # ------------------------------------------------------------------

    anomaly_duration_steps = 0

    if (
        "anomaly_duration_steps"
        in temporal
    ):

        try:

            anomaly_duration_steps = int(
                temporal[
                    "anomaly_duration_steps"
                ]
            )

        except Exception:
            anomaly_duration_steps = 0

    elif "duration_steps" in temporal:

        try:

            anomaly_duration_steps = int(
                temporal[
                    "duration_steps"
                ]
            )

        except Exception:
            anomaly_duration_steps = 0

    # ------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------

    top_features = feature_analysis.get(
        "top_features",
        [],
    )

    if not isinstance(
        top_features,
        list,
    ):
        top_features = []

    # ------------------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------------------

    broken_pairs = graph.get(
        "broken_pairs",
        [],
    )

    if not isinstance(
        broken_pairs,
        list,
    ):
        broken_pairs = []

    # ------------------------------------------------------------------
    # TEMPORAL RANGES
    # ------------------------------------------------------------------

    anomaly_ranges = temporal.get(
        "ranges",
        [],
    )

    if not isinstance(
        anomaly_ranges,
        list,
    ):
        anomaly_ranges = []

    # ------------------------------------------------------------------
    # WINDOW BOUNDARIES
    # ------------------------------------------------------------------

    window_start = int(
        payload.window_start_abs
    )

    window_end = (
        window_start
        + WINDOW_SIZE
        - 1
    )

    # ------------------------------------------------------------------
    # LATENCY
    # ------------------------------------------------------------------

    latency_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    # ------------------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------------------

    return PredictResponse(

        machine=MACHINE_ID,

        is_anomaly=bool(
            is_anomaly
        ),

        # THIS IS NOW THE SAME SCORE
        # USED BY evaluate_smd_final.py.
        anomaly_score=round(
            anomaly_score,
            6,
        ),

        threshold=float(
            OFFICIAL_WINDOW_THRESHOLD
        ),

        confidence_pct=str(
            confidence.get(
                "confidence_pct",
                "0%",
            )
        ),

        alert_level=str(
            confidence.get(
                "alert_level",
                "UNKNOWN",
            )
        ),

        peak_timestep=peak_timestep,

        window_start=window_start,

        window_end=window_end,

        anomaly_duration_steps=(
            anomaly_duration_steps
        ),

        top_features=top_features,

        broken_pairs=broken_pairs[:10],

        anomaly_ranges=(
            anomaly_ranges[:10]
        ),

        root_cause=str(
            root_cause
        ),

        latency_ms=round(
            latency_ms,
            2,
        ),
    )


# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
def root() -> Dict[str, Any]:

    return {
        "name": (
            "FGEAD — Explainable "
            "Anomaly Detection API"
        ),
        "version": "2.1.0",
        "status": (
            "ready"
            if _model_loaded
            else "model_not_loaded"
        ),
        "machine": MACHINE_ID,
        "features": _n_features,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "threshold": OFFICIAL_WINDOW_THRESHOLD,
        "device": DEVICE,
        "endpoints": {
            "health": "/health",
            "dataset": "/dataset",
            "machine": "/machine",
            "predict": "/predict",
            "docs": "/docs",
        },
    }