"""
config/settings.py

FGEAD Centralized Environment-Based Configuration System.
Supports DEVELOPMENT, TEST, and PRODUCTION environments with zero hardcoded secrets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    """
    Application settings loaded from environment variables with safe defaults.
    """

    def __init__(self):
        # Environment: DEVELOPMENT, TEST, PRODUCTION
        self.ENVIRONMENT: str = os.getenv("FGEAD_ENV", os.getenv("ENVIRONMENT", "DEVELOPMENT")).upper()

        # Server network configuration
        default_host = "0.0.0.0" if self.ENVIRONMENT == "PRODUCTION" else "127.0.0.1"
        self.SERVER_HOST: str = os.getenv("FGEAD_SERVER_HOST", default_host)
        self.SERVER_PORT: int = int(os.getenv("FGEAD_SERVER_PORT", "8000"))
        self.STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

        # CORS Allowed Origins
        cors_raw = os.getenv("FGEAD_CORS_ORIGINS", "*")
        self.CORS_ORIGINS: List[str] = [orig.strip() for orig in cors_raw.split(",") if orig.strip()]

        # Database Configuration
        default_db_path = str(PROJECT_ROOT / "data" / "fgead_multihost.db")
        self.DATABASE_PATH: str = os.getenv("FGEAD_DB_PATH", default_db_path)
        self.OFFLINE_TIMEOUT_SEC: float = float(os.getenv("FGEAD_OFFLINE_TIMEOUT_SEC", "15.0"))

        # Security & Authentication Secrets
        self.API_SECRET_KEY: str = os.getenv("FGEAD_API_SECRET_KEY", "fgead-dev-secret-change-in-production")
        self.AGENT_TOKEN_SALT: str = os.getenv("FGEAD_AGENT_TOKEN_SALT", "fgead-token-salt-change-in-production")

        if self.ENVIRONMENT == "PRODUCTION":
            if self.API_SECRET_KEY == "fgead-dev-secret-change-in-production":
                print("[WARNING] FGEAD_API_SECRET_KEY is using default development secret in PRODUCTION!")

        # SMD 38-Channel Benchmark Model Paths
        default_smd_root = PROJECT_ROOT / "data" / "SMD"
        default_smd_ckpt = PROJECT_ROOT / "checkpoints" / "fgead_smd_machine_1_1.pt"
        self.SMD_DATA_DIR: Path = Path(os.getenv("FGEAD_SMD_DATA_DIR", str(default_smd_root)))
        self.SMD_CHECKPOINT_PATH: Path = Path(os.getenv("FGEAD_SMD_CHECKPOINT_PATH", str(default_smd_ckpt)))
        self.SMD_MACHINE_ID: str = os.getenv("FGEAD_SMD_MACHINE_ID", "1-1")
        self.SMD_THRESHOLD: float = float(os.getenv("FGEAD_SMD_THRESHOLD", "2.073376"))

        # Live 22-Channel Physical Windows Model Paths
        default_live_ckpt = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch.pt"
        default_live_scaler = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler.joblib"
        default_live_config = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_config.json"
        default_live_threshold = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold.json"

        self.LIVE_CHECKPOINT_PATH: Path = Path(os.getenv("FGEAD_LIVE_CHECKPOINT_PATH", str(default_live_ckpt)))
        self.LIVE_SCALER_PATH: Path = Path(os.getenv("FGEAD_LIVE_SCALER_PATH", str(default_live_scaler)))
        self.LIVE_CONFIG_PATH: Path = Path(os.getenv("FGEAD_LIVE_CONFIG_PATH", str(default_live_config)))
        self.LIVE_THRESHOLD_PATH: Path = Path(os.getenv("FGEAD_LIVE_THRESHOLD_PATH", str(default_live_threshold)))
        self.LIVE_THRESHOLD_DEFAULT: float = float(os.getenv("FGEAD_LIVE_THRESHOLD_DEFAULT", "1.859450"))

        # Hardware & Compute Device
        self.DEVICE: Optional[str] = os.getenv("FGEAD_DEVICE", None)

        # Logging
        self.LOG_LEVEL: str = os.getenv("FGEAD_LOG_LEVEL", "INFO").upper()
        self.STRUCTURED_LOGS: bool = os.getenv("FGEAD_STRUCTURED_LOGS", "true").lower() in ("true", "1", "yes")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "PRODUCTION"

    @property
    def is_test(self) -> bool:
        return self.ENVIRONMENT == "TEST"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "DEVELOPMENT"


# Global singleton settings
settings = Settings()
