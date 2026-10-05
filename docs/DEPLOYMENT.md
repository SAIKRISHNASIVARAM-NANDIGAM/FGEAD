# FGEAD Production Deployment & Infrastructure Guide

Comprehensive guide for deploying the **Feature Graph Explainable Anomaly Detection (FGEAD)** platform to cloud environments (AWS EC2, Google Cloud Compute Engine, Azure Virtual Machines, or on-premise Linux servers).

---

## 1. System Architecture Overview

```
                          Internet / Fleet Network
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Nginx Reverse Proxy      │
                      │  SSL/TLS (Port 80 / 443)    │
                      └──────────────┬──────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   │                                   │
                   ▼ (Port 8000)                       ▼ (Port 8501)
     ┌───────────────────────────┐       ┌───────────────────────────┐
     │      FastAPI Backend      │       │    Streamlit Dashboard    │
     │  (fgead-api.service)      │       │ (fgead-dashboard.service) │
     │  - Live Neural Inference  │       │  - Fleet Ops Center       │
     │  - SMD Benchmark Engine   │       │  - Single Host Monitoring │
     │  - 5-Question XAI Root    │       │  - SMD Benchmark Explorer │
     │  - Host Registry (SQLite) │       │  - Model Gating Status    │
     └─────────────┬─────────────┘       └─────────────┬─────────────┘
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
     ┌───────────────────────────────────────────────────────────────┐
     │                      Persistent Storage                       │
     │  - SQLite Database (`data/fgead_multihost.db`)                │
     │  - Neural Checkpoints (`checkpoints/*.pt`)                    │
     │  - StandardScalers & Thresholds (`checkpoints/*.joblib`)      │
     └───────────────────────────────────────────────────────────────┘
```

---

## 2. Server Requirements & Prerequisites

### Hardware Recommendations
- **Operating System:** Ubuntu 22.04 LTS / Debian 12 / RHEL 9
- **CPU:** 4 vCPUs or higher (x86_64 / ARM64)
- **RAM:** 8 GB minimum (16 GB recommended for high fleet counts)
- **Storage:** 20 GB SSD
- **Compute:** CPU or NVIDIA GPU (CUDA 11.8+ / 12.x supported)

### Software Dependencies
- Python 3.10, 3.11, or 3.12
- `python3-venv`, `python3-pip`, `git`, `nginx`, `certbot`, `python3-certbot-nginx`

---

## 3. Server Provisioning & Installation

### Step 3.1: System Updates & Base Packages
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git nginx certbot python3-certbot-nginx ufw htop
```

### Step 3.2: Create Dedicated Service User
```bash
sudo useradd -m -s /bin/bash fgead
sudo usermod -aG sudo fgead
sudo su - fgead
```

### Step 3.3: Clone Repository and Create Virtualenv
```bash
git clone https://github.com/SAIKRISHNASIVARAM-NANDIGAM/FGEAD.git /home/fgead/FGEAD
cd /home/fgead/FGEAD

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 3.4: Configure Environment Variables
Copy the production environment template:
```bash
cp .env.example .env
nano .env
```

Ensure production values are set:
```ini
FGEAD_ENV=PRODUCTION
FGEAD_SERVER_HOST=0.0.0.0
FGEAD_SERVER_PORT=8000
STREAMLIT_SERVER_PORT=8501
FGEAD_API_SECRET_KEY=YOUR_SECURE_RANDOM_SECRET_KEY_HERE
FGEAD_AGENT_TOKEN_SALT=YOUR_SECURE_TOKEN_SALT_HERE
FGEAD_DB_PATH=/home/fgead/FGEAD/data/fgead_multihost.db
FGEAD_LOG_LEVEL=INFO
FGEAD_STRUCTURED_LOGS=true
```

---

## 4. Systemd Service Configuration

Create systemd unit files to manage the FastAPI API Gateway and Streamlit Dashboard as background daemons with automatic restart on failure.

### 4.1: FastAPI Gateway Service (`/etc/systemd/system/fgead-api.service`)
```ini
[Unit]
Description=FGEAD FastAPI Backend Service
After=network.target

[Service]
Type=simple
User=fgead
WorkingDirectory=/home/fgead/FGEAD
EnvironmentFile=/home/fgead/FGEAD/.env
ExecStart=/home/fgead/FGEAD/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=fgead-api

[Install]
WantedBy=multi-user.target
```

### 4.2: Streamlit Dashboard Service (`/etc/systemd/system/fgead-dashboard.service`)
```ini
[Unit]
Description=FGEAD Streamlit Dashboard Service
After=network.target fgead-api.service

[Service]
Type=simple
User=fgead
WorkingDirectory=/home/fgead/FGEAD
EnvironmentFile=/home/fgead/FGEAD/.env
ExecStart=/home/fgead/FGEAD/venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=fgead-dashboard

[Install]
WantedBy=multi-user.target
```

### 4.3: Enable and Start Services
```bash
sudo systemctl daemon-reload
sudo systemctl enable fgead-api fgead-dashboard
sudo systemctl start fgead-api fgead-dashboard

# Verify service status
sudo systemctl status fgead-api
sudo systemctl status fgead-dashboard
```

---

## 5. Nginx Reverse Proxy & SSL/TLS Configuration

Configure Nginx to serve the API and Dashboard behind HTTPS with WebSocket support for Streamlit.

### 5.1: Nginx Site Configuration (`/etc/nginx/sites-available/fgead.conf`)
```nginx
server {
    listen 80;
    server_name your-fgead-domain.com; # Replace with your domain or server IP

    # Streamlit Web Dashboard
    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # Streamlit WebSocket endpoint
    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    # FastAPI REST & Telemetry Endpoints
    location ~ ^/(api|health|docs|openapi.json|predict|telemetry|agent|hosts|fleet|alerts|dataset|machine) {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### 5.2: Enable Site and Obtain SSL Certificate
```bash
sudo ln -s /etc/nginx/sites-available/fgead.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Obtain free Let's Encrypt SSL certificate
sudo certbot --nginx -d your-fgead-domain.com
```

---

## 6. Security Hardening & Firewall Setup

Configure UFW (Uncomplicated Firewall) to block unauthorized ports:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow http
sudo ufw allow https
sudo ufw enable
```

---

## 7. Health Probes & Monitoring

The FGEAD API provides dedicated health endpoints for Kubernetes, AWS ALB, and uptime monitors:

| Endpoint | Method | Expected Status | Purpose |
| :--- | :---: | :---: | :--- |
| `/health` | `GET` | `200 OK` | Comprehensive runtime metadata & device status |
| `/health/live` | `GET` | `200 OK` | Liveness check for container / process heartbeat |
| `/health/ready` | `GET` | `200 OK` / `503` | Readiness check verifying DB and Model loaded |

### Health Check Verification Commands:
```bash
# Liveness Check
curl -s http://127.0.0.1:8000/health/live

# Readiness Check
curl -s http://127.0.0.1:8000/health/ready
```

---

## 8. Multi-Host Agent Deployment

To monitor remote physical/virtual machines:

### On Remote Windows Host:
```powershell
python -m data.live_agent --api-url https://your-fgead-domain.com --rate 1.0 --auto-register
```

### On Remote Linux Host:
```bash
python3 -m data.live_agent --api-url https://your-fgead-domain.com --rate 1.0 --auto-register
```
*Note: Linux hosts will automatically be placed in `TELEMETRY_ONLY` mode until a dedicated Linux baseline model is registered.*

---

## 9. Log Management and Maintenance

View real-time structured logs via systemd journal:
```bash
# View API logs
journalctl -u fgead-api -f

# View Dashboard logs
journalctl -u fgead-dashboard -f
```

To restart the full FGEAD stack:
```bash
sudo systemctl restart fgead-api fgead-dashboard
```

---

## 10. Troubleshooting & Diagnostic Playbook

### 10.1: Service Fails to Start
- **Symptoms:** `systemctl status fgead-api` returns `failed`.
- **Diagnostic:** Check journal logs: `journalctl -u fgead-api -n 50 --no-pager`.
- **Common Fixes:**
  - Verify Python virtualenv path in `/etc/systemd/system/fgead-api.service`.
  - Check file permissions: `chown -R fgead:fgead /home/fgead/FGEAD`.
  - Ensure checkpoints exist: `ls -l /home/fgead/FGEAD/checkpoints/`.

### 10.2: Remote Agent Cannot Connect
- **Symptoms:** Agent outputs `Waiting for API... (ConnectionError)`.
- **Common Fixes:**
  - Ensure port 80/443 is open in cloud security groups (AWS EC2 Security Group / Azure NSG).
  - Verify Nginx is active: `sudo systemctl status nginx`.
  - Check API key in `X-API-Key` matches `FGEAD_API_SECRET_KEY` in `.env`.

### 10.3: Telemetry Gated as "TELEMETRY ONLY"
- **Cause:** Telemetry received from a Linux host without a trained Linux GNN baseline model.
- **Resolution:** This is intentional behavior under Phase 6.1 model gating. The Windows model is strictly isolated to Windows baselines to eliminate false alarms.

---

## 11. Rollback & Disaster Recovery Procedure

In the event of an unexpected release regression or failure:

1. **Stop Services:**
   ```bash
   sudo systemctl stop fgead-api fgead-dashboard
   ```
2. **Revert Git Repository to Last Known Good Tag/Commit:**
   ```bash
   cd /home/fgead/FGEAD
   git checkout tags/v2.1.0-checkpoint # or previous stable SHA
   ```
3. **Restore SQLite Database (if migration issue occurred):**
   ```bash
   cp /home/fgead/FGEAD/data/fgead_multihost.db.bak /home/fgead/FGEAD/data/fgead_multihost.db
   ```
4. **Restart Stack and Verify Health:**
   ```bash
   sudo systemctl start fgead-api fgead-dashboard
   curl -s http://127.0.0.1:8000/health/ready
   ```
