# Odyssai Services

> Personal cockpit for non-distributed local services. Part of **odyssai.eu**
> alongside **Odysseus** (distributed inference engine) and **Companion** (client).

Manages services that don't fit Odysseus's "MLX distributed" scope:

| Service | Host | Port | Role |
|---|---|---|---|
| **mlx-vlm** | ultra-96A (`192.168.86.49`) | 8080 | Vision-language server (Qwen3.6-35B-VL) |
| **mlx-coder** | max-64 (`192.168.86.50`) | 8086 | Autocomplete (Qwen2.5-Coder 1.5B) |

## What it does

- Show service status (running / stopped) via SSH process+port probe
- Start / Stop / Restart each service
- Switch model (for services with a `state_file` — mlx-vlm)
- Tail logs over SSH
- Reboot the host (sudo NOPASSWD or osascript fallback)
- Liveness probe + RAM info per host

Pure orchestration over SSH. No compute. Runs in a container on the same
Docker host as Odysseus (mini-i3, `192.168.86.141`) but on port **8001**.

## Quick start

```bash
# Generate a strong token
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Bring up (replace with your generated token)
ODYSSAI_SERVICES_TOKEN=xxx docker compose up -d --build

# UI: http://localhost:8001 (or http://192.168.86.141:8001 in prod)
```

The dashboard prompts for the token on first load and stores it in
localStorage. EventSource streams accept `?token=...` query fallback.

## Endpoints

| Method | Path | Auth |
|---|---|---|
| GET  | `/` | public (dashboard HTML) |
| GET  | `/health` | public |
| GET  | `/.well-known/inference-engine.json` | public (declares `kind: service-admin`) |
| GET  | `/admin/services` | bearer |
| GET  | `/admin/services/{id}` | bearer |
| POST | `/admin/services/{id}/start` | bearer |
| POST | `/admin/services/{id}/stop` | bearer |
| POST | `/admin/services/{id}/restart` | bearer |
| POST | `/admin/services/{id}/load` | bearer (body: `{ "model": "<path>" }`) |
| GET  | `/admin/services/{id}/models` | bearer |
| POST | `/admin/services/{id}/host/reboot` | bearer |
| GET  | `/admin/services/{id}/logs?lines=100&follow=false` | bearer |

## Service config

Hard-coded in `scripts/api.py` under `SERVICES = {...}`. To add a new managed
service (e.g. a future ComfyUI sidecar):

1. Add an entry with `host_id`, `ssh`, `port`, `binary`, `watchdog`, `log_files`
2. Optional `state_file` if the service supports live model switching
3. Rebuild + redeploy

This is intentionally minimal — odyssai-services is an **internal cockpit**,
not a product. Configuration via code is fine here.

## Deploy

```bash
# From the repo
scp scripts/api.py scripts/dashboard.html admin@192.168.86.141:/tmp/
ssh admin@192.168.86.141 'docker cp /tmp/api.py odyssai-services:/app/api.py'
ssh admin@192.168.86.141 'docker cp /tmp/dashboard.html odyssai-services:/app/dashboard.html'
ssh admin@192.168.86.141 'docker restart odyssai-services'  # for api.py changes
```

For full rebuild :

```bash
scp -r . admin@192.168.86.141:~/odyssai-services-build/
ssh admin@192.168.86.141 'cd ~/odyssai-services-build && docker compose up -d --build'
```

## Why a separate dashboard?

- **Odysseus** = distributed MLX inference engine (Argo, Hades). Compute-heavy.
- **Odyssai Services** = sidecars and standalone services. SSH-only orchestration.

Keeping them split:
- Reflects the real architecture (different hardware constraints, different audience)
- Lets Odysseus stay focused — `ABOUT.md` doesn't have to caveat "also does VLM kinda"
- Companion talks to Odysseus for inference; never needs to know about this service
- Easy to extend (mlx-coder today, ComfyUI tomorrow, monitoring custom another day)

## Hardware assumptions

- SSH keys (`~/.ssh/id_*`) already configured between mini-i3 (host) and the
  managed Macs. The container mounts `${HOME}/.ssh:ro`.
- Each managed Mac has the watchdog script in `~/admin/bin/` + cron entry
  that respawns the service every minute if not running.
- For host reboot: `sudo -n /sbin/shutdown -r now` requires NOPASSWD sudo for
  `admin`. Falls back to `osascript` if not configured.
