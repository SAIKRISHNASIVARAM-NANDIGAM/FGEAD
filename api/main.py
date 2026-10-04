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

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
from data.live_buffer import LiveRingBuffer
from data.live_feature_schema import LIVE_FEATURES, validate_live_feature_dict
from models.fgead import FGEAD
from models.explainer import FGEADExplainer
from api.live_inference import get_live_inference_service, LiveInferenceService
from api.host_registry import get_host_registry, HostRegistry
from data.multihost_buffer import get_multihost_buffer_manager, MultiHostBufferManager
from api.multihost_inference import get_multihost_inference_manager, MultiHostInferenceManager



# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "60"))
STRIDE = int(os.getenv("STRIDE", "5"))

MACHINE_ID = os.getenv("MACHINE_ID", os.getenv("SMD_MACHINE", "1-1"))

CHECKPOINT_PATH = Path(
    os.getenv(
        "CHECKPOINT_PATH",
        str(PROJECT_ROOT / "checkpoints" / f"fgead_smd_machine_{MACHINE_ID.replace('-', '_')}.pt")
    )
)

SMD_ROOT = Path(
    os.getenv("SMD_ROOT", str(PROJECT_ROOT / "data" / "SMD"))
)

# Official threshold from evaluate_smd_final.py
OFFICIAL_WINDOW_THRESHOLD = float(os.getenv("WINDOW_THRESHOLD", "2.073376"))

DEVICE = os.getenv(
    "DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Thread-safe Live Telemetry Ring Buffer (Window = 60s)
_live_buffer: LiveRingBuffer = LiveRingBuffer(window_size=WINDOW_SIZE)


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
    model_config = {"protected_namespaces": ()}
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


class LiveTelemetryPayload(BaseModel):
    machine_id: str = Field(..., description="Unique machine identifier or hostname")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    machine_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hardware specs and host metadata")
    features: Dict[str, float] = Field(..., description="Live Windows telemetry metrics matching LIVE_FEATURES schema")


class LiveTelemetryResponse(BaseModel):
    status: str
    buffer_size: int
    window_size: int
    is_window_full: bool
    samples_received: int


class LivePredictRequest(BaseModel):
    data: List[List[float]] = Field(
        ...,
        description="60 timesteps × 22 live physical telemetry features.",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp string of the window end.",
    )


class HostRegisterRequest(BaseModel):
    hostname: str = Field(..., description="Hostname or node identifier")
    operating_system: str = Field(..., description="Operating System (Windows, Linux, etc.)")
    os_version: str = Field(..., description="OS version or release kernel")
    architecture: str = Field("x86_64", description="CPU Architecture")
    agent_version: str = Field("1.0.0", description="Agent software version")
    schema_version: str = Field("1.0", description="Telemetry schema version")
    model_id: str = Field("windows_default", description="Model profile identifier")
    custom_host_id: Optional[str] = Field(None, description="Optional custom host ID")
    machine_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hardware specs")


class HostRegisterResponse(BaseModel):
    host_id: str
    hostname: str
    operating_system: str
    agent_id: str
    agent_token: str
    status: str
    model_id: str
    created_at: str
    message: str = "Host registered successfully. Store the agent_token securely."


class HostTelemetryPayload(BaseModel):
    host_id: Optional[str] = Field(None, description="Host identifier")
    machine_id: Optional[str] = Field(None, description="Machine ID fallback")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    machine_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hardware specs")
    features: Dict[str, float] = Field(..., description="22-channel physical telemetry metrics")


class HostBaselineRequest(BaseModel):
    model_id: str = Field("windows_default", description="Model ID")
    threshold: float = Field(1.859450, description="Baseline threshold")
    sample_count: int = Field(3600, description="Number of baseline samples")
    notes: Optional[str] = Field("", description="Baseline documentation")



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
            scaler=_loader.scaler,
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
# LIVE TELEMETRY ENDPOINTS (Physical Host)
# ============================================================================

@app.post("/telemetry", response_model=LiveTelemetryResponse)
def ingest_live_telemetry(payload: LiveTelemetryPayload) -> LiveTelemetryResponse:
    """
    Ingest a single live telemetry snapshot from the physical Windows monitoring agent.
    Validates features against the versioned live feature schema.
    Triggers automated live FGEAD neural inference whenever the rolling 60s window is full.
    """
    is_valid, err_msg = validate_live_feature_dict(payload.features)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail=f"Telemetry validation error: {err_msg}",
        )

    _live_buffer.add_sample(
        timestamp=payload.timestamp,
        features=payload.features,
        machine_info=payload.machine_info,
    )

    # Automated real-time inference on full rolling 60s window
    window_arr, is_full = _live_buffer.get_window_tensor()
    if is_full and window_arr is not None:
        try:
            live_service = get_live_inference_service()
            if live_service.is_ready:
                live_service.infer_window(window_arr, timestamp_iso=payload.timestamp)
        except Exception as inf_err:
            print(f"[LIVE TELEMETRY] Automated inference notice: {inf_err}")

    status_dict = _live_buffer.get_status()
    return LiveTelemetryResponse(
        status="success",
        buffer_size=status_dict["buffer_size"],
        window_size=status_dict["window_size"],
        is_window_full=status_dict["is_window_full"],
        samples_received=status_dict["total_samples_received"],
    )


@app.post("/predict/live")
def predict_live(payload: LivePredictRequest) -> Dict[str, Any]:
    """
    Execute inference on an explicit 60x22 live physical telemetry window.
    """
    live_service = get_live_inference_service()
    if not live_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Live 22-channel FGEAD model service is not ready: {live_service.init_error}",
        )

    try:
        arr = np.asarray(payload.data, dtype=np.float32)
        result = live_service.infer_window(arr, timestamp_iso=payload.timestamp)
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Live inference execution error: {exc}")


@app.get("/telemetry/latest")
def get_latest_telemetry() -> Dict[str, Any]:
    """
    Fetch the most recent live telemetry sample collected from the Windows host.
    """
    latest = _live_buffer.get_latest()
    if latest is None:
        return {
            "status": "empty",
            "message": "No live telemetry has been received yet. Start data/live_agent.py to begin streaming.",
            "data": None,
        }
    return {
        "status": "ok",
        "data": latest,
    }


@app.get("/telemetry/live_analysis")
def get_live_analysis() -> Dict[str, Any]:
    """
    Fetch the latest live model inference, anomaly score, 5-question narrative, and score history.
    """
    live_service = get_live_inference_service()
    status_buf = _live_buffer.get_status()

    if not live_service.is_ready:
        return {
            "status": "error",
            "is_ready": False,
            "error": live_service.init_error,
            "buffer_status": status_buf,
            "latest_inference": None,
        }

    latest = live_service.get_latest_inference()
    score_hist = live_service.get_score_history()
    ep_status = live_service.episode_tracker.get_status()

    return {
        "status": "ok",
        "is_ready": True,
        "model_name": "fgead_live_windows_22ch",
        "threshold": live_service.threshold,
        "buffer_status": status_buf,
        "is_window_full": status_buf["is_window_full"],
        "episode_status": ep_status,
        "latest_inference": latest,
        "score_history": score_hist,
    }


@app.get("/agent/status")
def get_agent_status() -> Dict[str, Any]:
    """
    Diagnostic status of the live telemetry agent, ring buffer, and live model.
    """
    status_dict = _live_buffer.get_status()
    live_service = get_live_inference_service()
    latest_inf = live_service.get_latest_inference() if live_service.is_ready else None

    status_dict["live_model_ready"] = live_service.is_ready
    status_dict["live_model_threshold"] = live_service.threshold if live_service.is_ready else None
    status_dict["live_model_error"] = live_service.init_error
    status_dict["episode_status"] = live_service.episode_tracker.get_status() if live_service.is_ready else None
    status_dict["latest_score"] = latest_inf["anomaly_score"] if latest_inf else None
    status_dict["is_anomaly"] = latest_inf["is_anomaly"] if latest_inf else None
    status_dict["severity"] = latest_inf["severity"] if latest_inf else None

    return status_dict


@app.get("/telemetry/history")
def get_telemetry_history() -> Dict[str, Any]:
    """
    Return recent rolling telemetry history for real-time visualization.
    """
    return _live_buffer.get_history()


# ============================================================================
# MULTI-HOST PLATFORM ENDPOINTS
# ============================================================================

@app.post("/hosts/register", response_model=HostRegisterResponse)
def register_host_endpoint(payload: HostRegisterRequest) -> HostRegisterResponse:
    """
    Register a new host machine or update existing agent registration.
    Generates a secure agent authentication token.
    """
    reg = get_host_registry()
    host_dict, raw_token = reg.register_host(
        hostname=payload.hostname,
        operating_system=payload.operating_system,
        os_version=payload.os_version,
        architecture=payload.architecture,
        agent_version=payload.agent_version,
        schema_version=payload.schema_version,
        model_id=payload.model_id,
        machine_info=payload.machine_info,
        custom_host_id=payload.custom_host_id,
    )
    return HostRegisterResponse(
        host_id=host_dict["host_id"],
        hostname=host_dict["hostname"],
        operating_system=host_dict["operating_system"],
        agent_id=host_dict["agent_id"],
        agent_token=raw_token,
        status=host_dict["status"],
        model_id=host_dict["model_id"],
        created_at=host_dict["created_at"],
    )


@app.get("/hosts")
def list_hosts_endpoint() -> Dict[str, Any]:
    """
    List all registered hosts with their current online/offline status, model profile, and latest metrics.
    """
    reg = get_host_registry()
    hosts = reg.get_all_hosts()
    inf_mgr = get_multihost_inference_manager()
    buf_mgr = get_multihost_buffer_manager()

    enriched_hosts = []
    for h in hosts:
        hid = h["host_id"]
        latest_inf = inf_mgr.get_latest_inference(hid)
        buf_status = buf_mgr.get_status(hid)
        ep_tracker = inf_mgr.get_episode_tracker(hid)
        ep_status = ep_tracker.get_status()

        h_copy = dict(h)
        h_copy["buffer_size"] = buf_status["buffer_size"]
        h_copy["is_window_full"] = buf_status["is_window_full"]
        h_copy["latest_score"] = latest_inf["anomaly_score"] if latest_inf else None
        h_copy["threshold"] = latest_inf["threshold"] if latest_inf else 1.859450
        h_copy["is_anomaly"] = latest_inf["is_anomaly"] if latest_inf else False
        h_copy["severity"] = latest_inf["severity"] if latest_inf else "NOMINAL"
        h_copy["active_episode"] = ep_status.get("active_episode_id")
        enriched_hosts.append(h_copy)

    return {
        "status": "ok",
        "count": len(enriched_hosts),
        "hosts": enriched_hosts,
    }


@app.get("/hosts/{host_id}")
def get_host_details_endpoint(host_id: str) -> Dict[str, Any]:
    """
    Retrieve metadata, buffer status, and model profile for a specific host.
    """
    reg = get_host_registry()
    host_dict = reg.get_host(host_id)
    if not host_dict:
        raise HTTPException(status_code=404, detail=f"Host '{host_id}' not found.")

    buf_mgr = get_multihost_buffer_manager()
    inf_mgr = get_multihost_inference_manager()
    
    return {
        "status": "ok",
        "host": host_dict,
        "buffer_status": buf_mgr.get_status(host_id),
        "latest_inference": inf_mgr.get_latest_inference(host_id),
        "episode_status": inf_mgr.get_episode_tracker(host_id).get_status(),
    }


@app.post("/hosts/{host_id}/telemetry")
def ingest_host_telemetry(
    host_id: str,
    payload: HostTelemetryPayload,
    x_agent_token: Optional[str] = Header(None, alias="X-Agent-Token"),
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Secure telemetry ingestion endpoint for authenticated host agents.
    Validates token, schema (22 features), NaN/Inf, physical bounds.
    Maintains isolated rolling ring buffer for the host and triggers real-time inference.
    """
    # 1. Token extraction
    token = x_agent_token
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Authentication failed: Missing X-Agent-Token header.")

    # 2. Host Authentication
    reg = get_host_registry()
    is_auth, err_msg, host_data = reg.authenticate_agent(host_id, token)
    if not is_auth:
        status_code = 404 if "Unknown host" in str(err_msg) else 403
        raise HTTPException(status_code=status_code, detail=f"Authentication failed: {err_msg}")

    # 3. Telemetry Feature Validation (22 features, NaN, Inf, bounds)
    is_valid, feat_err = validate_live_feature_dict(payload.features)
    if not is_valid:
        raise HTTPException(status_code=422, detail=f"Telemetry validation error: {feat_err}")

    # 4. Isolated Buffer Ingestion
    buf_mgr = get_multihost_buffer_manager()
    buf = buf_mgr.add_sample(
        host_id=host_id,
        timestamp=payload.timestamp,
        features=payload.features,
        machine_info=payload.machine_info,
    )

    # 5. Heartbeat Update
    reg.update_heartbeat(host_id)

    # 6. Automated Real-time Inference on full 60s window (Gated by Model Compatibility)
    inf_mgr = get_multihost_inference_manager()
    window_arr, is_full = buf.get_window_tensor()
    latest_inf = None
    model_id = host_data.get("model_id", "windows_default")
    is_compat, compat_msg, compat_details = inf_mgr.check_compatibility(host_data, model_id)

    if not is_compat:
        # Gated: Linux host or unsupported OS without dedicated trained model
        reg.set_host_status(host_id, "TELEMETRY_ONLY")
        reg.set_host_model_status(host_id, "none", "BASELINE_REQUIRED")
    elif is_full and window_arr is not None:
        try:
            latest_inf = inf_mgr.infer_host_window(
                host_id=host_id,
                window_raw=window_arr,
                timestamp_iso=payload.timestamp,
                model_id=model_id,
                host_data=host_data,
            )
            # Update host status
            new_status = "ANOMALY" if latest_inf["is_anomaly"] else "ONLINE"
            reg.set_host_status(host_id, new_status)
        except Exception as inf_err:
            print(f"[MULTI-HOST INFERENCE NOTICE] Host {host_id} error: {inf_err}")

    buf_status = buf.get_status()
    return {
        "status": "success",
        "host_id": host_id,
        "buffer_size": buf_status["buffer_size"],
        "window_size": buf_status["window_size"],
        "is_window_full": buf_status["is_window_full"],
        "samples_received": buf_status["total_samples_received"],
        "model_status": compat_details.get("model_status", "BASELINE_REQUIRED") if not is_compat else "COMPATIBLE",
        "is_anomaly": latest_inf["is_anomaly"] if latest_inf else False,
        "anomaly_score": latest_inf["anomaly_score"] if latest_inf else None,
        "compatibility_message": compat_msg if not is_compat else "Compatible",
    }


@app.post("/hosts/{host_id}/predict/live")
def predict_host_live(host_id: str, payload: Optional[LivePredictRequest] = None) -> Dict[str, Any]:
    """
    Run neural anomaly inference and 5-question XAI on host's dedicated buffer or supplied window.
    Strictly gates execution based on OS and model compatibility.
    """
    reg = get_host_registry()
    host_dict = reg.get_host(host_id)
    if not host_dict:
        raise HTTPException(status_code=404, detail=f"Host '{host_id}' not found.")

    inf_mgr = get_multihost_inference_manager()
    buf_mgr = get_multihost_buffer_manager()
    model_id = host_dict.get("model_id", "windows_default")

    # Compatibility check
    is_compat, compat_msg, compat_details = inf_mgr.check_compatibility(host_dict, model_id)
    if not is_compat:
        raise HTTPException(
            status_code=400,
            detail=f"Inference error for host {host_id}: {compat_msg}",
        )

    if payload and payload.data:
        arr = np.asarray(payload.data, dtype=np.float32)
        ts = payload.timestamp
    else:
        arr, is_full = buf_mgr.get_window_tensor(host_id)
        if not is_full or arr is None:
            raise HTTPException(
                status_code=400,
                detail=f"Host '{host_id}' rolling buffer is not full yet ({buf_mgr.get_status(host_id)['buffer_size']}/60 samples).",
            )
        ts = datetime.now(timezone.utc).isoformat()

    try:
        res = inf_mgr.infer_host_window(
            host_id=host_id,
            window_raw=arr,
            timestamp_iso=ts,
            model_id=model_id,
            host_data=host_dict,
        )
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error for host {host_id}: {exc}")


@app.get("/hosts/{host_id}/telemetry/latest")
def get_host_latest_telemetry(host_id: str) -> Dict[str, Any]:
    """
    Fetch the most recent telemetry sample for a specific host.
    """
    buf_mgr = get_multihost_buffer_manager()
    latest = buf_mgr.get_latest(host_id)
    if latest is None:
        return {
            "status": "empty",
            "host_id": host_id,
            "message": f"No telemetry received yet for host '{host_id}'.",
            "data": None,
        }
    return {
        "status": "ok",
        "host_id": host_id,
        "data": latest,
    }


@app.get("/hosts/{host_id}/telemetry/history")
def get_host_telemetry_history(host_id: str) -> Dict[str, Any]:
    """
    Fetch rolling telemetry series for a specific host.
    """
    buf_mgr = get_multihost_buffer_manager()
    history = buf_mgr.get_history(host_id)
    history["host_id"] = host_id
    return history


@app.get("/hosts/{host_id}/analysis")
def get_host_analysis(host_id: str) -> Dict[str, Any]:
    """
    Fetch the latest model inference, score history, 5-question XAI, and episode state for a host.
    """
    reg = get_host_registry()
    host_dict = reg.get_host(host_id)
    if not host_dict:
        raise HTTPException(status_code=404, detail=f"Host '{host_id}' not found.")

    buf_mgr = get_multihost_buffer_manager()
    inf_mgr = get_multihost_inference_manager()
    
    buf_status = buf_mgr.get_status(host_id)
    latest_inf = inf_mgr.get_latest_inference(host_id)
    score_hist = inf_mgr.get_score_history(host_id)
    ep_status = inf_mgr.get_episode_tracker(host_id).get_status()
    
    model_id = host_dict.get("model_id", "none")
    profile = inf_mgr.get_profile(model_id) if model_id != "none" else None
    is_compat, compat_msg, compat_details = inf_mgr.check_compatibility(host_dict, model_id)

    return {
        "status": "ok",
        "host_id": host_id,
        "hostname": host_dict.get("hostname"),
        "operating_system": host_dict.get("operating_system"),
        "model_id": profile.profile_id if profile else "none",
        "model_name": profile.name if profile else "None (Baseline Required)",
        "model_profile_type": profile.profile_type if profile else "none",
        "model_status": host_dict.get("model_status", "BASELINE_REQUIRED" if not is_compat else "COMPATIBLE"),
        "model_compatibility": "Compatible" if is_compat else compat_details.get("status", "Baseline Required"),
        "compatibility_message": compat_msg,
        "threshold": profile.threshold if profile else 1.859450,
        "buffer_status": buf_status,
        "is_window_full": buf_status["is_window_full"],
        "episode_status": ep_status,
        "latest_inference": latest_inf if is_compat else None,
        "score_history": score_hist if is_compat else [],
    }


@app.get("/hosts/{host_id}/episodes")
def get_host_episodes(host_id: str) -> Dict[str, Any]:
    """
    Fetch active and completed anomaly episodes for a specific host.
    """
    inf_mgr = get_multihost_inference_manager()
    tracker = inf_mgr.get_episode_tracker(host_id)
    return {
        "status": "ok",
        "host_id": host_id,
        "current_episode": tracker.current_episode,
        "completed_episodes": tracker.completed_episodes,
    }


@app.post("/hosts/{host_id}/baseline")
def create_host_baseline_endpoint(host_id: str, payload: HostBaselineRequest) -> Dict[str, Any]:
    """
    Associate a baseline dataset/model with a host.
    """
    reg = get_host_registry()
    host_dict = reg.get_host(host_id)
    if not host_dict:
        raise HTTPException(status_code=404, detail=f"Host '{host_id}' not found.")

    res = reg.register_baseline(
        host_id=host_id,
        model_id=payload.model_id,
        threshold=payload.threshold,
        sample_count=payload.sample_count,
        notes=payload.notes or "",
    )
    return {"status": "success", "baseline": res}


@app.post("/hosts/{host_id}/baseline/collect")
def collect_host_baseline_endpoint(
    host_id: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Collect baseline statistics from the host's current rolling buffer or submitted telemetry series.
    Computes mean, std, min, max, P95, P99 for all 22 features, verifies NaN/Inf, and saves stats.
    Does NOT modify the existing model checkpoint.
    """
    from data.baseline_compatibility import compute_feature_distribution, compare_baseline_distributions

    reg = get_host_registry()
    host_dict = reg.get_host(host_id)
    if not host_dict:
        raise HTTPException(status_code=404, detail=f"Host '{host_id}' not found.")

    buf_mgr = get_multihost_buffer_manager()
    history = buf_mgr.get_history(host_id)
    samples = history.get("samples", [])

    if payload and "samples" in payload:
        samples = payload["samples"]

    if len(samples) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient telemetry samples to compute baseline ({len(samples)}/10 minimum).",
        )

    # Compute distribution
    stats = compute_feature_distribution(samples)
    reg.save_host_baseline_stats(host_id, stats)

    # Compare with reference training baseline
    comp_report = compare_baseline_distributions(stats)

    return {
        "status": "success",
        "host_id": host_id,
        "sample_count": len(samples),
        "baseline_stats": stats,
        "reference_comparison": comp_report,
    }


@app.get("/hosts/{host_id}/baseline/comparison")
def get_host_baseline_comparison(host_id: str) -> Dict[str, Any]:
    """
    Retrieve baseline distribution comparison between host and model training reference.
    """
    from data.baseline_compatibility import compare_baseline_distributions

    reg = get_host_registry()
    host_dict = reg.get_host(host_id)
    if not host_dict:
        raise HTTPException(status_code=404, detail=f"Host '{host_id}' not found.")

    stats = reg.get_host_baseline_stats(host_id)
    if not stats:
        return {
            "status": "not_collected",
            "host_id": host_id,
            "message": "Host baseline has not been collected yet.",
            "comparison": None,
        }

    comp = compare_baseline_distributions(stats)
    return {
        "status": "ok",
        "host_id": host_id,
        "comparison": comp,
    }


@app.get("/models/metadata")
def get_models_metadata() -> Dict[str, Any]:
    """
    Retrieve explicit compatibility metadata for all registered models.
    """
    inf_mgr = get_multihost_inference_manager()
    metadata = {}
    for pid, prof in inf_mgr.profiles.items():
        metadata[pid] = prof.to_metadata_dict()
    return {
        "status": "ok",
        "models": metadata,
    }


@app.post("/hosts/reconcile-duplicates")
def reconcile_duplicate_hosts_endpoint(target_hostname: Optional[str] = None) -> Dict[str, Any]:
    """
    Clean up duplicate registrations for the same hostname.
    """
    reg = get_host_registry()
    cleaned = reg.reconcile_duplicate_hosts(target_hostname)
    return {
        "status": "ok",
        "cleaned_count": len(cleaned),
        "cleaned_host_ids": cleaned,
    }


@app.get("/fleet/overview")
def get_fleet_overview() -> Dict[str, Any]:
    """
    Aggregated fleet overview metrics across all registered hosts.
    Displays explicit model compatibility and proper status.
    """
    reg = get_host_registry()
    hosts = reg.get_all_hosts()
    inf_mgr = get_multihost_inference_manager()
    buf_mgr = get_multihost_buffer_manager()

    total_hosts = len(hosts)
    online_count = 0
    offline_count = 0
    anomalous_hosts = 0
    total_active_episodes = 0

    enriched_hosts = []
    for h in hosts:
        hid = h["host_id"]
        status = h.get("status", "OFFLINE")
        if status in ["ONLINE", "ANOMALY", "TELEMETRY_ONLY"]:
            online_count += 1
        else:
            offline_count += 1

        model_id = h.get("model_id", "none")
        is_compat, compat_msg, compat_details = inf_mgr.check_compatibility(h, model_id)
        latest_inf = inf_mgr.get_latest_inference(hid) if is_compat else None
        ep_status = inf_mgr.get_episode_tracker(hid).get_status()
        buf_status = buf_mgr.get_status(hid)

        if latest_inf and latest_inf.get("is_anomaly"):
            anomalous_hosts += 1
        if ep_status.get("is_in_anomaly_episode") and is_compat:
            total_active_episodes += 1

        # Determine display status
        if status == "OFFLINE":
            display_status = "OFFLINE"
        elif not is_compat or status == "TELEMETRY_ONLY":
            display_status = "TELEMETRY_ONLY"
        elif latest_inf and latest_inf.get("is_anomaly"):
            display_status = "ANOMALY"
        else:
            display_status = "ONLINE"

        profile = inf_mgr.get_profile(model_id) if model_id != "none" else None

        h_summary = {
            "host_id": hid,
            "hostname": h.get("hostname"),
            "operating_system": h.get("operating_system"),
            "status": display_status,
            "telemetry_connected": status != "OFFLINE",
            "last_seen": h.get("last_seen"),
            "model_id": profile.profile_id if profile else "none",
            "model_name": profile.name if profile else "None",
            "model_compatibility": "Compatible" if is_compat else compat_details.get("status", "Baseline Required"),
            "compatibility_message": compat_msg,
            "buffer_size": buf_status["buffer_size"],
            "latest_score": latest_inf["anomaly_score"] if (latest_inf and is_compat) else None,
            "threshold": profile.threshold if (profile and is_compat) else None,
            "is_anomaly": latest_inf["is_anomaly"] if (latest_inf and is_compat) else False,
            "severity": latest_inf["severity"] if (latest_inf and is_compat) else ("TELEMETRY_ONLY" if not is_compat else "NOMINAL"),
            "active_episode": ep_status.get("active_episode_id") if is_compat else None,
        }
        enriched_hosts.append(h_summary)

    return {
        "status": "ok",
        "total_hosts": total_hosts,
        "online_hosts": online_count,
        "offline_hosts": offline_count,
        "hosts_with_anomalies": anomalous_hosts,
        "total_active_episodes": total_active_episodes,
        "hosts": enriched_hosts,
    }


@app.get("/alerts")
def get_fleet_alerts(host_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """
    Retrieve recent alert events across the fleet or for a specific host.
    """
    reg = get_host_registry()
    alerts = reg.get_alerts(host_id=host_id, limit=limit)
    return {
        "status": "ok",
        "count": len(alerts),
        "alerts": alerts,
    }


# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
def root() -> Dict[str, Any]:

    return {
        "name": (
            "FGEAD — Explainable "
            "Anomaly Detection API & Multi-Host Platform"
        ),
        "version": "2.2.0",
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
            "predict_live": "POST /predict/live",
            "telemetry_post": "POST /telemetry",
            "telemetry_latest": "GET /telemetry/latest",
            "agent_status": "GET /agent/status",
            "telemetry_live_analysis": "GET /telemetry/live_analysis",
            "telemetry_history": "GET /telemetry/history",
            "hosts_register": "POST /hosts/register",
            "hosts_list": "GET /hosts",
            "hosts_telemetry": "POST /hosts/{host_id}/telemetry",
            "hosts_predict": "POST /hosts/{host_id}/predict/live",
            "hosts_analysis": "GET /hosts/{host_id}/analysis",
            "fleet_overview": "GET /fleet/overview",
            "alerts": "GET /alerts",
            "docs": "/docs",
        },
    }