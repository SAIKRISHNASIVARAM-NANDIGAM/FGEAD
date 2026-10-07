"""
scripts/preflight_check.py

FGEAD Automated Pre-Flight Environment & System Integrity Check.
Verifies system readiness without modifying any model weights, data, or configuration files.
"""

import sys
import os
import importlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Set ANSI colors for Windows console if supported
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def print_status(component: str, status: str, detail: str = ""):
    if status == "PASS":
        badge = f"{GREEN}[PASS]{RESET}"
    elif status == "WARN":
        badge = f"{YELLOW}[WARN]{RESET}"
    else:
        badge = f"{RED}[FAIL]{RESET}"
    
    msg = f" {badge} {component:<35} {detail}"
    print(msg)


def check_python_version() -> bool:
    v = sys.version_info
    v_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and 10 <= v.minor <= 12:
        print_status("Python Version", "PASS", f"Detected Python {v_str}")
        return True
    elif v.major == 3 and v.minor > 12:
        print_status("Python Version", "WARN", f"Detected Python {v_str} (Tested on 3.10-3.12)")
        return True
    else:
        print_status("Python Version", "FAIL", f"Detected Python {v_str} (Required: 3.10-3.12)")
        return False


def check_dependencies() -> bool:
    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("plotly", "Plotly"),
        ("streamlit", "Streamlit"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("pydantic", "Pydantic"),
        ("httpx", "HTTPX"),
        ("psutil", "PSUtil"),
    ]
    
    all_ok = True
    for module_name, display_name in required_packages:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed")
            print_status(f"Package: {display_name}", "PASS", f"v{ver}")
        except ImportError:
            print_status(f"Package: {display_name}", "FAIL", "Not installed")
            all_ok = False
    return all_ok


def check_directories() -> bool:
    required_dirs = [
        "api",
        "app",
        "data",
        "checkpoints",
        "config",
        "core",
        "agents",
        "data/multihost_validation",
    ]
    all_ok = True
    for rel_dir in required_dirs:
        dir_path = PROJECT_ROOT / rel_dir
        if dir_path.is_dir():
            print_status(f"Directory: {rel_dir}", "PASS", "Exists")
        else:
            print_status(f"Directory: {rel_dir}", "FAIL", "Missing")
            all_ok = False
    return all_ok


def check_configurations() -> bool:
    all_ok = True
    settings_file = PROJECT_ROOT / "config" / "settings.py"
    if settings_file.is_file():
        print_status("Config: settings.py", "PASS", "Exists")
    else:
        print_status("Config: settings.py", "FAIL", "Missing config/settings.py")
        all_ok = False

    env_example = PROJECT_ROOT / ".env.example"
    if env_example.is_file():
        print_status("Config: .env.example", "PASS", "Exists")
    else:
        print_status("Config: .env.example", "FAIL", "Missing .env.example")
        all_ok = False

    return all_ok


def check_model_artifacts() -> bool:
    artifacts = [
        ("Live Model Checkpoint", "checkpoints/fgead_live_windows_22ch.pt"),
        ("Live Feature Scaler", "checkpoints/fgead_live_scaler.joblib"),
        ("Live Model Config", "checkpoints/fgead_live_windows_22ch_config.json"),
        ("Live Anomaly Threshold", "checkpoints/fgead_live_threshold.json"),
        ("SMD Benchmark Checkpoint", "checkpoints/fgead_smd_machine_1_1.pt"),
    ]
    all_ok = True
    for label, rel_path in artifacts:
        path = PROJECT_ROOT / rel_path
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print_status(label, "PASS", f"{rel_path} ({size_mb:.2f} MB)")
        else:
            print_status(label, "FAIL", f"Missing {rel_path}")
            all_ok = False
    return all_ok


def check_model_config_integrity() -> bool:
    config_path = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_config.json"
    if not config_path.is_file():
        print_status("Model Config Integrity", "FAIL", "Config file missing")
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        
        n_feat = cfg.get("n_features", cfg.get("input_dim"))
        feats = cfg.get("feature_names", cfg.get("features"))
        win_size = cfg.get("window_size")

        if n_feat is None or feats is None or win_size is None:
            print_status("Model Config Integrity", "FAIL", "Missing required feature/window keys in config")
            return False

        if n_feat != 22 or len(feats) != 22:
            print_status("Model Config Integrity", "FAIL", f"Invalid feature count n_features={n_feat} (expected 22)")
            return False

        print_status("Model Config Integrity", "PASS", f"Validated n_features=22, window_size={win_size}")
        return True
    except Exception as e:
        print_status("Model Config Integrity", "FAIL", f"Parse error: {e}")
        return False


def check_schema_availability() -> bool:
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES
        if N_LIVE_FEATURES == 22 and len(LIVE_FEATURES) == 22:
            print_status("Schema Availability", "PASS", "Validated 22-feature Windows live schema")
            return True
        else:
            print_status("Schema Availability", "FAIL", f"Unexpected feature count: {N_LIVE_FEATURES}")
            return False
    except Exception as e:
        print_status("Schema Availability", "FAIL", f"Failed to import live_feature_schema: {e}")
        return False


def main():
    print("==================================================================")
    print(" FGEAD Pre-Flight System Verification")
    print("==================================================================")
    
    results = [
        ("Python Environment", check_python_version()),
        ("Required Packages", check_dependencies()),
        ("Directory Structure", check_directories()),
        ("Configuration Files", check_configurations()),
        ("Model Checkpoints & Scalers", check_model_artifacts()),
        ("Model Config Integrity", check_model_config_integrity()),
        ("Telemetry Schema Verification", check_schema_availability()),
    ]

    print("==================================================================")
    print(" PRE-FLIGHT SUMMARY")
    print("==================================================================")
    failed = [name for name, passed in results if not passed]
    if not failed:
        print(f"{GREEN}[SUCCESS] All pre-flight checks passed! FGEAD environment is fully ready.{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}[FAILURE] {len(failed)} verification group(s) failed: {', '.join(failed)}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
