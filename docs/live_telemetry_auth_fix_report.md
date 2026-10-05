# FGEAD Live Telemetry Authentication Fix Report

**Date:** 2026-10-05
**Component:** Live Agent Authentication & Telemetry Ingestion
**Status:** RESOLVED & VERIFIED

---

## 1. Authentication Mechanism
FGEAD utilizes a two-tier token-based authentication architecture:
1. **Tier 1 — Host Registration:**
   - The monitoring agent issues a `POST /hosts/register` request.
   - If an API Secret Key is configured, the agent proves authorization using the registration header.
   - Upon successful registration or identity matching, the backend dynamically generates a cryptographically secure random token (`fgead_<urlsafe_token>`), stores its SHA-256 hash in SQLite (`hosts.token_hash`), and returns the raw token in the response payload (`agent_token`).
2. **Tier 2 — Live Telemetry Ingestion:**
   - The monitoring agent sends periodic telemetry payloads to `POST /hosts/{host_id}/telemetry`.
   - The agent supplies its allocated host token via the required agent authentication header.
   - The backend hashes the provided token with SHA-256 and validates it against `hosts.token_hash` using constant-time equality verification (`secrets.compare_digest`).

---

## 2. Configuration Sources
- **Environment Variables:**
  - `FGEAD_API_SECRET_KEY`: System-wide API secret key for administrative actions and host registration gating.
  - `FGEAD_API_URL`: Backend base URL (defaults to `http://127.0.0.1:8000`).
- **CLI Arguments:**
  - `--api-key`: Provided directly to the telemetry agent to authenticate during registration.
  - `--force-register`: Explicitly requests a fresh registration token from the backend.
- **Client Configuration Cache:**
  - `data/live_agent_config.json` / `data/windows_agent_config.json`: Local JSON file storing the currently active `host_id`, `agent_token`, and `api_url`.

---

## 3. Headers and Auth Methods
| Endpoint | Expected Header | Auth Purpose |
| :--- | :--- | :--- |
| `POST /hosts/register` | `X-API-Key: <secret_key>` or `Authorization: Bearer <secret_key>` | Validates permission to register/update host metadata. |
| `POST /hosts/{host_id}/telemetry` | `X-Agent-Token: <agent_token>` or `Authorization: Bearer <agent_token>` | Authenticates telemetry submissions for the host ring buffer. |

---

## 4. Root Cause
1. **Stale Cached Token Reuse:**
   - The client configuration file (`data/live_agent_config.json`) contained a historical `agent_token` from a prior database session where `host_sivachowdary` had a different hash in SQLite.
   - `data/live_agent.py` checked file existence before verifying token validity. It returned the stale token and bypassed re-registration even when `--api-key` was passed.
2. **Missing Re-Authentication on HTTP 401/403:**
   - When the backend returned `403 Authentication failed: Invalid agent authentication token`, the live agent did not attempt to invalidate the local cache and re-register. It remained in a loop submitting the invalid token.

---

## 5. Files Changed
1. [`data/live_agent.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/live_agent.py):
   - Added support for `DEFAULT_API_KEY` from `FGEAD_API_SECRET_KEY` / `FGEAD_API_KEY`.
   - Added `force_register` argument to `load_or_register_agent()` and CLI parser (`--force-register`).
   - Implemented automatic token invalidation and re-registration handling upon receiving HTTP 401 / 403 telemetry responses.
2. [`agents/windows_agent.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/agents/windows_agent.py):
   - Added `--api-key`, `FGEAD_API_SECRET_KEY` fallback, `force_register`, and HTTP 401/403 auto-recovery.
3. [`agents/linux_agent.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/agents/linux_agent.py):
   - Added `--api-key`, `FGEAD_API_SECRET_KEY` fallback, `force_register`, and HTTP 401/403 auto-recovery.

---

## 6. Verification Results
1. **Automated Recovery Test:**
   - Executed: `python -m data.live_agent --api-url http://127.0.0.1:8000 --interval 1.0 --custom-host-id host_sivachowdary --count 3`
   - Initial request detected the stale token, auto-authenticated with `POST /hosts/register`, acquired a fresh `agent_token`, updated `data/live_agent_config.json`, and succeeded with `HTTP 200 (POST OK)`.
2. **Subsequent Ingestion Run:**
   - Samples streamed cleanly with immediate `HTTP 200 (POST OK)`.
3. **Database & Invariant Integrity:**
   - Host `host_sivachowdary` received live heartbeat and telemetry samples.
   - Multi-host isolation invariant test passed with zero host duplication or synthetic pollution (exactly 2 legitimate hosts preserved).
