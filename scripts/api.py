"""Odyssai Services — admin cockpit for non-distributed local services.

Manages services that don't fit Odysseus's "MLX distributed" scope:
  - mlx-vlm  (ultra-96A) — vision-language model server
  - mlx-coder (max-64)   — autocomplete server (Qwen2.5-Coder 1.5B)
  - comfyui  (ultra-96b) — ComfyUI image generation server (Flux schnell,
                            Flux dev, Wan2.1, LTX-2.3 — anything in the
                            registry of OdyssAI-Imager on .141)

Pure orchestration over SSH. No compute here. Lives on the same Docker host
as Odysseus (mini-i3:.141) but on a different port (8001 by convention).
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                                PlainTextResponse, StreamingResponse)
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────────
# Static service registry
# ──────────────────────────────────────────────────────────────────────────────
SERVICES: dict[str, dict] = {
    "vlm": {
        "name": "mlx-vlm",
        "description": "Vision-language server (Qwen3.6-35B-VL family)",
        "host_id": "ultra-96A",
        "ssh": "admin@192.168.86.49",
        "port": 8080,
        "binary": "mlx_vlm.server",
        "models_dirs": ["/Volumes/models/mlx-vlm"],
        "state_file": "/Users/admin/.mlx-vlm-current-model",
        "prev_state_file": "/Users/admin/.mlx-vlm-prev-model",
        "watchdog": "/Users/admin/bin/mlx-vlm-watchdog.sh",
        "log_files": [
            "/Users/admin/Library/Logs/mlx-vlm/mlx-vlm.out.log",
            "/Users/admin/Library/Logs/mlx-vlm/mlx-vlm.err.log",
        ],
        "kind": "mlx-vlm",
    },
    "coder": {
        "name": "mlx-coder",
        "description": "Autocomplete server (Qwen2.5-Coder 1.5B-Instruct)",
        "host_id": "max-64",
        "ssh": "admin@192.168.86.50",
        "port": 8086,
        "binary": "mlx_lm.server",
        "models_dirs": ["/Volumes/models/mlx-coder"],
        "state_file": None,  # hardcoded in watchdog script
        "prev_state_file": None,
        "watchdog": "/Users/admin/bin/mlx-coder-watchdog.sh",
        "log_files": [
            "/Users/admin/Library/Logs/mlx-lm/mlx-coder.out.log",
            "/Users/admin/Library/Logs/mlx-lm/mlx-coder.err.log",
        ],
        "kind": "mlx-lm",
    },
    "comfyui": {
        "name": "comfyui",
        "description": "ComfyUI image/video generation server. Backed by the OdyssAI-Imager bridge on .141.",
        "host_id": "ultra-96b",
        "ssh": "admin@192.168.86.42",
        "port": 8188,
        # pgrep -f matches the full command line. The script launches with
        # `python main.py --port 8188`; we anchor on the script name so the
        # watchdog can also locate it for respawn.
        "binary": "main.py --port 8188",
        "models_dirs": ["/Volumes/models/comfyui"],
        "state_file": None,  # model selection lives in the workflow templates on the bridge, not on disk
        "prev_state_file": None,
        "watchdog": "/Users/admin/bin/comfyui-watchdog.sh",
        "log_files": [
            "/tmp/comfy.log",
        ],
        "kind": "comfyui",
    },
}

_HERE = Path(__file__).resolve().parent
DASHBOARD_FILE = Path(os.environ.get("DASHBOARD_FILE", str(_HERE / "dashboard.html")))
ODYSSAI_SERVICES_TOKEN = (os.environ.get("ODYSSAI_SERVICES_TOKEN") or "").strip()


# ──────────────────────────────────────────────────────────────────────────────
# SSH helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ssh_run(ssh: str, cmd: str, timeout: int = 10) -> tuple[int, str, str]:
    """Sync SSH exec. Used from asyncio via to_thread."""
    p = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return p.returncode, p.stdout, p.stderr


async def _ssh(ssh: str, cmd: str, timeout: int = 10) -> tuple[int, str, str]:
    return await asyncio.to_thread(_ssh_run, ssh, cmd, timeout)


# ──────────────────────────────────────────────────────────────────────────────
# Service status / lifecycle
# ──────────────────────────────────────────────────────────────────────────────
async def service_status(sid: str) -> dict:
    """Liveness via process + port probe (immune to HTTP queue blocking)."""
    svc = SERVICES[sid]
    cmd = (
        f"PID=$(pgrep -f {shlex.quote(svc['binary'])} | head -1); "
        f"PORT=$(lsof -iTCP:{svc['port']} -sTCP:LISTEN -P -n 2>/dev/null | "
        f"  awk 'NR==2 {{print $2}}'); "
        f"MODEL=''; "
        f"if [ -n \"$PID\" ]; then "
        f"  MODEL=$(ps -p $PID -o command= 2>/dev/null | "
        f"          sed -nE 's/.*--model[ =]+([^ ]+).*/\\1/p'); "
        f"fi; "
        f"echo \"pid=$PID port_pid=$PORT model=$MODEL\""
    )
    pid: Optional[int] = None
    port_listen = False
    loaded: Optional[str] = None
    try:
        rc, out, _ = await _ssh(svc["ssh"], cmd, 8)
        for tok in (out or "").strip().split():
            if tok.startswith("pid=") and tok != "pid=":
                try: pid = int(tok.split("=", 1)[1])
                except: pass
            elif tok.startswith("port_pid=") and tok != "port_pid=":
                port_listen = True
            elif tok.startswith("model=") and tok != "model=":
                loaded = tok.split("=", 1)[1] or None
    except Exception:
        pass

    # Selected model (state file, optional)
    selected: Optional[str] = None
    if svc.get("state_file"):
        try:
            rc, out, _ = await _ssh(svc["ssh"], f"cat {shlex.quote(svc['state_file'])} 2>/dev/null || true")
            selected = (out or "").strip() or None
        except Exception:
            pass

    running = (pid is not None and port_listen)
    return {
        "id": sid,
        "name": svc["name"],
        "description": svc["description"],
        "host_id": svc["host_id"],
        "ssh": svc["ssh"],
        "port": svc["port"],
        "running": running,
        "pid": pid,
        "port_listen": port_listen,
        "loaded_model": loaded,
        "selected_model": selected,
        "kind": svc["kind"],
    }


async def service_kickstart(sid: str) -> dict:
    """Kill the binary and rerun the watchdog. The watchdog respawns the
    service with the current `selected_model` (or its hardcoded default)."""
    svc = SERVICES[sid]
    cmd = (
        f"pkill -9 -f {shlex.quote(svc['binary'])} 2>/dev/null ; "
        f"sleep 2 ; "
        f"{shlex.quote(svc['watchdog'])}"
    )
    rc, out, err = await _ssh(svc["ssh"], cmd, 25)
    return {"ok": rc == 0, "stdout": (out or "")[-300:], "stderr": (err or "")[-300:]}


async def service_stop(sid: str, unload: bool = True) -> dict:
    """Kill the binary. If `unload=True` AND the service has a state_file,
    swap it to '__unloaded__' so the cron watchdog stops respawning it."""
    svc = SERVICES[sid]
    cmds = [f"pkill -9 -f {shlex.quote(svc['binary'])} 2>/dev/null"]
    if unload and svc.get("state_file"):
        # backup current selected model first
        if svc.get("prev_state_file"):
            cmds.append(
                f"cp -f {shlex.quote(svc['state_file'])} {shlex.quote(svc['prev_state_file'])} 2>/dev/null || true"
            )
        cmds.append(f"echo __unloaded__ > {shlex.quote(svc['state_file'])}")
    cmds.append("sleep 1 ; true")
    rc, out, err = await _ssh(svc["ssh"], " ; ".join(cmds), 10)
    return {"ok": rc == 0, "stdout": (out or "")[-300:], "stderr": (err or "")[-300:]}


async def service_start(sid: str) -> dict:
    """Resume the service. If a previous model was saved (prev_state_file),
    restore it; then kickstart."""
    svc = SERVICES[sid]
    cmds = []
    if svc.get("state_file") and svc.get("prev_state_file"):
        cmds.append(
            f"if [ -s {shlex.quote(svc['prev_state_file'])} ]; then "
            f"  cp -f {shlex.quote(svc['prev_state_file'])} {shlex.quote(svc['state_file'])} ; "
            f"fi"
        )
    cmds.append(f"pkill -9 -f {shlex.quote(svc['binary'])} 2>/dev/null || true")
    cmds.append("sleep 2")
    cmds.append(shlex.quote(svc["watchdog"]))
    rc, out, err = await _ssh(svc["ssh"], " ; ".join(cmds), 25)
    return {"ok": rc == 0, "stdout": (out or "")[-300:], "stderr": (err or "")[-300:]}


async def service_set_model(sid: str, model_path: str) -> dict:
    """Write the model path to the service's state file. Caller must kickstart
    to apply. Services without state_file (mlx-coder) don't support live model
    switching — model is hardcoded in their watchdog."""
    svc = SERVICES[sid]
    if not svc.get("state_file"):
        raise HTTPException(400, f"service '{sid}' has no live model switching (hardcoded in watchdog)")
    cmd = f"echo {shlex.quote(model_path)} > {shlex.quote(svc['state_file'])}"
    rc, out, err = await _ssh(svc["ssh"], cmd, 5)
    return {"ok": rc == 0, "stderr": (err or "")[-200:]}


async def service_list_models(sid: str) -> list[dict]:
    """Scan candidate dirs on the host for model directories."""
    svc = SERVICES[sid]
    if not svc.get("models_dirs"):
        return []
    bases = svc["models_dirs"]
    inner = (
        "for base in " + " ".join(shlex.quote(b) for b in bases) + " ; do "
        "  [ -d \"$base\" ] || continue ; "
        "  for d in \"$base\"/*/ \"$base\"/*/*/; do "
        "    [ -f \"$d/config.json\" ] || continue ; "
        "    echo \"$d\" ; "
        "  done ; "
        "done 2>/dev/null"
    )
    cmd = "bash -c " + shlex.quote(inner)
    try:
        rc, out, _ = await _ssh(svc["ssh"], cmd, 20)
    except Exception:
        return []
    paths = [line.rstrip("/") for line in (out or "").splitlines() if line.strip()]
    if not paths:
        return []
    # Read all configs in a single SSH call.
    cat_cmd = " ; ".join(
        f"echo '===PATH:{p}===' && cat {shlex.quote(p)}/config.json 2>/dev/null ; echo"
        for p in paths
    )
    try:
        rc, raw, _ = await _ssh(svc["ssh"], cat_cmd, 30)
    except Exception:
        return []
    models: list[dict] = []
    current_path: Optional[str] = None
    buf: list[str] = []

    def flush():
        nonlocal buf, current_path
        if current_path and buf:
            try:
                cfg = json.loads("".join(buf))
            except Exception:
                cfg = {}
            has_vision = ("vision_config" in cfg or "image_token_id" in cfg)
            models.append({
                "id": current_path,
                "name": current_path.split("/")[-1],
                "model_type": cfg.get("model_type"),
                "has_vision": has_vision,
            })
        buf = []

    for line in (raw or "").splitlines():
        if line.startswith("===PATH:") and line.endswith("==="):
            flush()
            current_path = line[len("===PATH:"):-3]
        else:
            buf.append(line)
    flush()
    return models


async def service_host_info(sid: str) -> dict:
    """RAM, latency, uptime on the host."""
    svc = SERVICES[sid]
    t0 = time.time()
    cmd = (
        "vm_stat 2>/dev/null | head -20 ; echo '===UPTIME===' ; "
        "uptime ; echo '===HOSTNAME===' ; hostname"
    )
    info = {"reachable": False, "latency_ms": None,
            "hostname": None, "uptime": None,
            "free_mb": None, "wired_mb": None, "compressed_mb": None}
    try:
        rc, out, _ = await _ssh(svc["ssh"], cmd, 6)
        info["latency_ms"] = round((time.time() - t0) * 1000, 1)
        if rc == 0:
            info["reachable"] = True
            sections = (out or "").split("===")
            vm = sections[0] if sections else ""
            page_size = 16384  # macOS default on Apple Silicon
            for line in vm.splitlines():
                if "page size of" in line:
                    try:
                        page_size = int(line.split("page size of")[1].split()[0])
                    except Exception: pass
                if line.startswith("Pages free:"):
                    pages = int(line.split(":")[1].strip().rstrip("."))
                    info["free_mb"] = round(pages * page_size / 1024 / 1024)
                elif line.startswith("Pages wired down:"):
                    pages = int(line.split(":")[1].strip().rstrip("."))
                    info["wired_mb"] = round(pages * page_size / 1024 / 1024)
                elif line.startswith("Pages occupied by compressor:"):
                    pages = int(line.split(":")[1].strip().rstrip("."))
                    info["compressed_mb"] = round(pages * page_size / 1024 / 1024)
            for i, sec in enumerate(sections):
                if "UPTIME" in sec and i + 1 < len(sections):
                    info["uptime"] = sections[i + 1].strip().splitlines()[0] if sections[i + 1].strip() else None
                if "HOSTNAME" in sec and i + 1 < len(sections):
                    info["hostname"] = sections[i + 1].strip().splitlines()[0] if sections[i + 1].strip() else None
    except Exception:
        pass
    return info


async def service_reboot_host(sid: str) -> dict:
    """Reboot the host. Tries `sudo /sbin/shutdown -r now` (NOPASSWD recommended),
    falls back to osascript."""
    svc = SERVICES[sid]
    cmds = [
        ("sudo", "sudo -n /sbin/shutdown -r now 2>&1"),
        ("osascript", "osascript -e 'tell application \"System Events\" to restart'"),
    ]
    for method, cmd in cmds:
        try:
            rc, out, err = await _ssh(svc["ssh"], cmd, 5)
            if rc == 0:
                return {"ok": True, "method": method, "stdout": (out or "")[-200:]}
        except Exception as e:
            continue
    return {"ok": False, "method": "failed",
            "error": "all reboot methods failed (NOPASSWD sudo not configured?)"}


# ──────────────────────────────────────────────────────────────────────────────
# Recent activity: small ring buffer of probed /v1/models latency
# ──────────────────────────────────────────────────────────────────────────────
_activity_log: deque = deque(maxlen=200)


async def _probe_http_latency(sid: str) -> Optional[float]:
    """Probe the service's /health or /v1/models for round-trip latency."""
    svc = SERVICES[sid]
    url = f"http://{svc['ssh'].split('@')[1]}:{svc['port']}/v1/models"
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(url)
        if r.status_code in (200, 401):
            return round((time.time() - t0) * 1000, 1)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    sys.stderr.write(f"[services] Odyssai Services starting; managing {list(SERVICES.keys())}\n")
    if ODYSSAI_SERVICES_TOKEN:
        sys.stderr.write(f"[services] admin token required (length {len(ODYSSAI_SERVICES_TOKEN)})\n")
    else:
        sys.stderr.write("[services] WARNING: ODYSSAI_SERVICES_TOKEN unset — admin routes open\n")
    yield


app = FastAPI(title="Odyssai Services (odyssai.eu)", lifespan=lifespan)

# Document Producer (mirror of the Docling parser): renders model output —
# markdown -> real .docx (pandoc), JSON spec -> .xlsx (openpyxl) — under
# /render/*. Optional import so the sidecar still boots if the render deps
# aren't present in a given environment.
try:
    from render import build_router as _build_render_router
    app.include_router(_build_render_router())
except Exception as _render_err:  # pragma: no cover
    import sys as _sys
    _sys.stderr.write(f"[render] router not mounted ({_render_err})\n")


@app.middleware("http")
async def _admin_token_middleware(request: Request, call_next):
    if not ODYSSAI_SERVICES_TOKEN:
        return await call_next(request)
    path = request.url.path
    if not path.startswith("/admin/"):
        return await call_next(request)
    token = None
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.query_params.get("token")
    if token != ODYSSAI_SERVICES_TOKEN:
        return JSONResponse(
            {"detail": "missing or invalid admin token"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)


# Public routes
@app.get("/")
async def dashboard():
    if not DASHBOARD_FILE.exists():
        return HTMLResponse("<h1>dashboard.html missing</h1>", status_code=500)
    return HTMLResponse(DASHBOARD_FILE.read_text())


@app.get("/health")
async def health():
    return {"status": "ok", "services": list(SERVICES.keys())}


@app.get("/.well-known/inference-engine.json")
async def well_known():
    """Companion (and others) can probe this to recognize the service kind.
    Not an inference engine itself — declares its admin scope."""
    return {
        "name": "Odyssai Services",
        "vendor": "odyssai.eu",
        "version": "0.1.0",
        "kind": "service-admin",   # vs Odysseus which is "inference-engine"
        "api_compat": [],
        "auth": {
            "required": bool(ODYSSAI_SERVICES_TOKEN),
            "scheme": "bearer",
            "scope": "/admin/*",
            "public_routes": ["/health", "/.well-known/*"],
        },
        "managed_services": [
            {"id": sid, "name": s["name"], "kind": s["kind"],
             "host_id": s["host_id"], "port": s["port"]}
            for sid, s in SERVICES.items()
        ],
    }


# Admin routes
@app.get("/admin/services")
async def list_services():
    statuses = await asyncio.gather(*[service_status(sid) for sid in SERVICES])
    host_info = await asyncio.gather(*[service_host_info(sid) for sid in SERVICES])
    latencies = await asyncio.gather(*[_probe_http_latency(sid) for sid in SERVICES])
    out = []
    for s, h, lat in zip(statuses, host_info, latencies):
        s["host"] = h
        s["http_latency_ms"] = lat
        out.append(s)
    return {"data": out, "count": len(out)}


@app.get("/admin/services/{sid}")
async def get_service(sid: str):
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    status = await service_status(sid)
    status["host"] = await service_host_info(sid)
    status["http_latency_ms"] = await _probe_http_latency(sid)
    return status


@app.post("/admin/services/{sid}/restart")
async def post_restart(sid: str):
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    return await service_kickstart(sid)


@app.post("/admin/services/{sid}/start")
async def post_start(sid: str):
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    return await service_start(sid)


@app.post("/admin/services/{sid}/stop")
async def post_stop(sid: str):
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    return await service_stop(sid, unload=True)


@app.get("/admin/services/{sid}/models")
async def get_models(sid: str):
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    return {"data": await service_list_models(sid)}


class LoadRequest(BaseModel):
    model: str


@app.post("/admin/services/{sid}/load")
async def post_load(sid: str, req: LoadRequest):
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    set_result = await service_set_model(sid, req.model)
    if not set_result.get("ok"):
        raise HTTPException(500, f"set_model failed: {set_result.get('stderr')}")
    kick = await service_kickstart(sid)
    return {"ok": kick.get("ok", False), "set_model": set_result, "kickstart": kick}


@app.post("/admin/services/{sid}/host/reboot")
async def post_reboot_host(sid: str):
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    return await service_reboot_host(sid)


@app.get("/admin/services/{sid}/logs")
async def get_logs(sid: str, lines: int = 100, follow: bool = False):
    """Tail logs. With follow=true, returns SSE stream."""
    if sid not in SERVICES:
        raise HTTPException(404, f"unknown service {sid}")
    svc = SERVICES[sid]
    logs = svc.get("log_files", [])
    if not logs:
        return {"data": []}

    if not follow:
        # one-shot tail
        cmds = []
        for lf in logs:
            cmds.append(f"echo '=== {lf} ==='")
            cmds.append(f"tail -n {int(lines)} {shlex.quote(lf)} 2>/dev/null || true")
        rc, out, _ = await _ssh(svc["ssh"], " ; ".join(cmds), 10)
        return {"data": (out or "").splitlines()}

    async def gen() -> AsyncIterator[bytes]:
        cmd = " ; ".join(
            f"tail -n {int(lines)} -F {shlex.quote(lf)} 2>/dev/null"
            for lf in logs
        )
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
            svc["ssh"], cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                yield b"data: " + line.rstrip() + b"\n\n"
        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────────────────────
# Bench (stress + sweep) — talks to any OpenAI-compat engine (Odysseus,
# mlx-vlm direct, …). Internal tooling, lives here in Odyssai Services
# rather than cluttering Odysseus.
# ──────────────────────────────────────────────────────────────────────────────
from bench_stress import (
    run_stress as _bench_run_stress,
    run_sweep as _bench_run_sweep,
    list_stress_history as _bench_list_stress,
    get_stress_run as _bench_get_stress,
    delete_stress_run as _bench_delete_stress,
    stress_run_to_markdown as _bench_stress_md,
    list_sweep_history as _bench_list_sweep,
    get_sweep as _bench_get_sweep,
    delete_sweep as _bench_delete_sweep,
    sweep_to_markdown as _bench_sweep_md,
)
from bench_stability import (
    run_stability as _bench_run_stability,
    list_stability_history as _bench_list_stability,
    get_stability_run as _bench_get_stability,
    delete_stability_run as _bench_delete_stability,
    stability_run_to_markdown as _bench_stability_md,
)
from bench_endurance import (
    run_endurance as _bench_run_endurance,
    list_endurance_history as _bench_list_endurance,
    get_endurance_run as _bench_get_endurance,
    delete_endurance_run as _bench_delete_endurance,
    get_endurance_samples as _bench_get_endurance_samples,
)
from bench_directed import (
    run_directed as _bench_run_directed,
    list_directed_history as _bench_list_directed,
    get_directed_run as _bench_get_directed,
    delete_directed_run as _bench_delete_directed,
    directed_run_to_markdown as _bench_directed_md,
)

# Default engine endpoint (Odysseus on the same host).
DEFAULT_ENGINE_URL = os.environ.get(
    "ODYSSAI_ENGINE_URL", "http://host.docker.internal:8000"
)

# Active in-flight bench runs — {run_id: {queue: asyncio.Queue, abort: Event, kind: 'stress'|'sweep'}}
_bench_active: dict[str, dict] = {}


class BenchEnduranceRequest(BaseModel):
    endpoint: Optional[str] = None
    model: str                           # single alias
    concurrency: int = 1
    prompt: str = "Tell me about the ocean in three sentences."
    maxTokens: int = 150
    temperature: float = 0.7
    headers: Optional[dict] = None


class BenchStressRequest(BaseModel):
    endpoint: Optional[str] = None       # defaults to DEFAULT_ENGINE_URL
    cluster: Optional[str] = None        # label only
    models: list[str]                    # one or more aliases (argo, hades, or:claude-haiku, vlm:qwen3.6-35b…)
    concurrency: int = 2
    total: int = 10
    prompt: str = "Tell me about the ocean in three sentences."
    maxTokens: int = 150
    temperature: float = 0.7
    headers: Optional[dict] = None       # e.g. {"authorization": "Bearer crew_..."}


class BenchStabilityRequest(BaseModel):
    endpoint: Optional[str] = None       # defaults to DEFAULT_ENGINE_URL
    cluster: str = "argo"                # argo | hades | nautilus
    model: str                           # full model path or HF id
    nodes: int = 1                       # Argo only — number of nodes
    iterations: int = 5
    prompt: str = "Write a short haiku about a Mac Studio cluster."
    maxTokens: int = 200
    headers: Optional[dict] = None       # e.g. {"authorization": "Bearer crew_..."}


class BenchSweepRequest(BaseModel):
    type: str                            # "concurrency" | "repeat" | "custom"
    endpoint: Optional[str] = None
    cluster: Optional[str] = None
    models: list[str]
    prompt: str = "Tell me about the ocean in three sentences."
    maxTokens: int = 150
    temperature: float = 0.7
    # Defaults inherited by each run (overridable by `configs[]` in custom mode)
    concurrency: int = 2
    total: int = 10
    # Sweep params
    values: Optional[list[int]] = None    # for type=concurrency
    repeats: Optional[int] = None         # for type=repeat
    configs: Optional[list[dict]] = None  # for type=custom
    totalPerRun: Optional[int] = None
    headers: Optional[dict] = None


class BenchDirectedRequest(BaseModel):
    """Directed benchmark — runs an ordered prompt set against one model,
    persists each answer alongside the prompt set for later analysis."""
    endpoint: Optional[str] = None       # defaults to DEFAULT_ENGINE_URL
    model: str                            # alias Odysseus (e.g. or:kimi-k-2.7-code)
    test_repo: Optional[str] = None       # default ODYSSAI_BENCH_DEFAULT_TEST_REPO
    maxTokens: int = 1024
    temperature: float = 0.0
    concurrency: int = 1
    headers: Optional[dict] = None


def _make_event_emitter(queue: asyncio.Queue):
    async def emit(name: str, data: dict) -> None:
        await queue.put((name, data))
    return emit


async def _drain_to_sse(queue: asyncio.Queue, abort: asyncio.Event) -> AsyncIterator[bytes]:
    """Drain events from a queue and yield as SSE. All events are emitted on
    the default `message` channel with `{type, payload}` JSON — avoids the
    browser quirks around `addEventListener` on event names containing colons.

    Sentinel `(None, None)` ends the stream."""
    while True:
        try:
            ev = await asyncio.wait_for(queue.get(), timeout=30.0)
        except asyncio.TimeoutError:
            yield b": keepalive\n\n"
            continue
        if ev[0] is None:
            return
        name, data = ev
        try:
            wrapped = json.dumps({"type": name, "payload": data}, default=str)
        except Exception:
            wrapped = json.dumps({"type": name, "payload": {"_serialize_error": True}})
        yield ("data: " + wrapped + "\n\n").encode()


@app.post("/admin/bench/stress")
async def admin_bench_stress(req: BenchStressRequest):
    """Kick off a stress run, return {run_id}. Client follows via SSE."""
    if not req.models:
        raise HTTPException(400, "models[] required (at least one)")
    endpoint = req.endpoint or DEFAULT_ENGINE_URL
    cfg = {
        "endpoint": endpoint,
        "cluster": req.cluster,
        "models": req.models,
        "concurrency": req.concurrency,
        "total": req.total,
        "prompt": req.prompt,
        "maxTokens": req.maxTokens,
        "temperature": req.temperature,
        "headers": req.headers,
    }
    queue: asyncio.Queue = asyncio.Queue()
    abort = asyncio.Event()
    emit = _make_event_emitter(queue)

    async def runner():
        try:
            record = await _bench_run_stress(cfg, emit, abort)
            await queue.put(("done", {"runId": record["id"], "record": record}))
        except Exception as e:
            await queue.put(("error", {"error": str(e)}))
        finally:
            await queue.put((None, None))

    task = asyncio.create_task(runner())
    # Wait briefly to capture the run_id from the first 'progress' event
    first = await asyncio.wait_for(queue.get(), timeout=10.0)
    run_id = (first[1] or {}).get("runId")
    if not run_id:
        raise HTTPException(500, "failed to obtain run_id")
    # Put the event back so the SSE consumer sees it
    pre = asyncio.Queue()
    await pre.put(first)
    # Bridge: pre then queue
    async def bridge():
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.001)
            except asyncio.TimeoutError:
                break
            await pre.put(ev)
    await bridge()
    # Chain queue → pre
    async def relay():
        while True:
            ev = await queue.get()
            await pre.put(ev)
            if ev[0] is None:
                return
    asyncio.create_task(relay())

    _bench_active[run_id] = {"queue": pre, "abort": abort, "kind": "stress", "task": task}
    return {"runId": run_id, "stream": f"/admin/bench/stress/{run_id}/stream"}


@app.get("/admin/bench/stress/{run_id}/stream")
async def admin_bench_stress_stream(run_id: str):
    """SSE follow of an in-flight stress run."""
    state = _bench_active.get(run_id)
    if not state:
        # Maybe the run is already done; return final record once.
        record = _bench_get_stress(run_id)
        if not record:
            raise HTTPException(404, f"unknown run {run_id}")
        async def replay() -> AsyncIterator[bytes]:
            payload = json.dumps({"type": "done", "payload": {"runId": run_id, "record": record}}, default=str)
            yield ("data: " + payload + "\n\n").encode()
        return StreamingResponse(replay(), media_type="text/event-stream")
    return StreamingResponse(
        _drain_to_sse(state["queue"], state["abort"]),
        media_type="text/event-stream",
    )


@app.post("/admin/bench/stress/{run_id}/abort")
async def admin_bench_stress_abort(run_id: str):
    state = _bench_active.get(run_id)
    if not state:
        raise HTTPException(404, f"unknown run {run_id}")
    state["abort"].set()
    return {"ok": True, "aborting": run_id}


@app.get("/admin/bench/stress/runs")
async def admin_bench_stress_runs():
    return {"data": _bench_list_stress()}


@app.get("/admin/bench/stress/runs/{run_id}")
async def admin_bench_stress_get(run_id: str):
    r = _bench_get_stress(run_id)
    if not r:
        raise HTTPException(404, f"unknown run {run_id}")
    return r


@app.get("/admin/bench/stress/runs/{run_id}/markdown")
async def admin_bench_stress_md(run_id: str):
    r = _bench_get_stress(run_id)
    if not r:
        raise HTTPException(404, f"unknown run {run_id}")
    return PlainTextResponse(_bench_stress_md(r), media_type="text/markdown")


@app.delete("/admin/bench/stress/runs/{run_id}")
async def admin_bench_stress_delete(run_id: str):
    if not _bench_delete_stress(run_id):
        raise HTTPException(404, f"unknown run {run_id}")
    return {"ok": True}


# ── Sweep endpoints ──────────────────────────────────────────────────────
@app.post("/admin/bench/sweep")
async def admin_bench_sweep(req: BenchSweepRequest):
    if req.type not in ("concurrency", "repeat", "custom"):
        raise HTTPException(400, f"unknown sweep type: {req.type}")
    endpoint = req.endpoint or DEFAULT_ENGINE_URL
    sweep_cfg = req.model_dump(exclude_none=True)

    queue: asyncio.Queue = asyncio.Queue()
    abort = asyncio.Event()
    emit = _make_event_emitter(queue)

    async def runner():
        try:
            record = await _bench_run_sweep(sweep_cfg, emit, endpoint, abort)
            await queue.put(("sweep:done", {"sweepId": record["id"], "sweep": record}))
        except Exception as e:
            await queue.put(("error", {"error": str(e)}))
        finally:
            await queue.put((None, None))

    task = asyncio.create_task(runner())
    first = await asyncio.wait_for(queue.get(), timeout=10.0)
    sweep_id = (first[1] or {}).get("sweepId")
    if not sweep_id:
        raise HTTPException(500, "failed to obtain sweep_id")

    pre: asyncio.Queue = asyncio.Queue()
    await pre.put(first)
    async def bridge():
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=0.001)
            except asyncio.TimeoutError:
                break
            await pre.put(ev)
    await bridge()
    async def relay():
        while True:
            ev = await queue.get()
            await pre.put(ev)
            if ev[0] is None:
                return
    asyncio.create_task(relay())

    _bench_active[sweep_id] = {"queue": pre, "abort": abort, "kind": "sweep", "task": task}
    return {"sweepId": sweep_id, "stream": f"/admin/bench/sweep/{sweep_id}/stream"}


@app.get("/admin/bench/sweep/{sweep_id}/stream")
async def admin_bench_sweep_stream(sweep_id: str):
    state = _bench_active.get(sweep_id)
    if not state:
        record = _bench_get_sweep(sweep_id)
        if not record:
            raise HTTPException(404, f"unknown sweep {sweep_id}")
        async def replay() -> AsyncIterator[bytes]:
            payload = json.dumps({"type": "sweep:done", "payload": {"sweepId": sweep_id, "sweep": record}}, default=str)
            yield ("data: " + payload + "\n\n").encode()
        return StreamingResponse(replay(), media_type="text/event-stream")
    return StreamingResponse(
        _drain_to_sse(state["queue"], state["abort"]),
        media_type="text/event-stream",
    )


@app.post("/admin/bench/sweep/{sweep_id}/abort")
async def admin_bench_sweep_abort(sweep_id: str):
    state = _bench_active.get(sweep_id)
    if not state:
        raise HTTPException(404, f"unknown sweep {sweep_id}")
    state["abort"].set()
    return {"ok": True, "aborting": sweep_id}


@app.get("/admin/bench/sweep/runs")
async def admin_bench_sweep_runs():
    return {"data": _bench_list_sweep()}


@app.get("/admin/bench/sweep/runs/{sweep_id}")
async def admin_bench_sweep_get(sweep_id: str):
    s = _bench_get_sweep(sweep_id)
    if not s:
        raise HTTPException(404, f"unknown sweep {sweep_id}")
    return s


@app.get("/admin/bench/sweep/runs/{sweep_id}/markdown")
async def admin_bench_sweep_md(sweep_id: str):
    s = _bench_get_sweep(sweep_id)
    if not s:
        raise HTTPException(404, f"unknown sweep {sweep_id}")
    return PlainTextResponse(_bench_sweep_md(s), media_type="text/markdown")


@app.delete("/admin/bench/sweep/runs/{sweep_id}")
async def admin_bench_sweep_delete(sweep_id: str):
    if not _bench_delete_sweep(sweep_id):
        raise HTTPException(404, f"unknown sweep {sweep_id}")
    return {"ok": True}


# ── Stability endpoints ──────────────────────────────────────────────────
# LOAD → CHAT → UNLOAD repro loop. Same SSE + history pattern as stress.

@app.post("/admin/bench/stability")
async def admin_bench_stability(req: BenchStabilityRequest):
    """Kick off a stability run, return {runId}. Client follows via SSE."""
    if not req.model:
        raise HTTPException(400, "model required")
    if req.iterations < 1 or req.iterations > 200:
        raise HTTPException(400, "iterations must be in 1..200")
    if req.nodes < 1 or req.nodes > 8:
        raise HTTPException(400, "nodes must be in 1..8")
    cfg = {
        "endpoint": req.endpoint or DEFAULT_ENGINE_URL,
        "cluster": req.cluster,
        "model": req.model,
        "nodes": req.nodes,
        "iterations": req.iterations,
        "prompt": req.prompt,
        "maxTokens": req.maxTokens,
        "headers": req.headers,
    }
    queue: asyncio.Queue = asyncio.Queue()
    abort = asyncio.Event()
    emit = _make_event_emitter(queue)

    async def runner():
        try:
            record = await _bench_run_stability(cfg, emit, abort)
            await queue.put(("done", {"runId": record["id"], "record": record}))
        except Exception as e:
            await queue.put(("error", {"error": str(e)}))
        finally:
            await queue.put((None, None))

    task = asyncio.create_task(runner())
    # Wait briefly to capture the run_id from the first 'runStart' event.
    first = await asyncio.wait_for(queue.get(), timeout=10.0)
    run_id = (first[1] or {}).get("runId")
    if not run_id:
        raise HTTPException(500, "failed to obtain run_id")
    # Bridge: put the first event back so SSE consumer sees it, then chain.
    pre: asyncio.Queue = asyncio.Queue()
    await pre.put(first)

    async def relay():
        while True:
            ev = await queue.get()
            await pre.put(ev)
            if ev[0] is None:
                return
    asyncio.create_task(relay())

    _bench_active[run_id] = {"queue": pre, "abort": abort,
                              "kind": "stability", "task": task}
    return {"runId": run_id, "stream": f"/admin/bench/stability/{run_id}/stream"}


@app.get("/admin/bench/stability/{run_id}/stream")
async def admin_bench_stability_stream(run_id: str):
    state = _bench_active.get(run_id)
    if not state:
        record = _bench_get_stability(run_id)
        if not record:
            raise HTTPException(404, f"unknown run {run_id}")
        async def replay() -> AsyncIterator[bytes]:
            payload = json.dumps(
                {"type": "done", "payload": {"runId": run_id, "record": record}},
                default=str,
            )
            yield ("data: " + payload + "\n\n").encode()
        return StreamingResponse(replay(), media_type="text/event-stream")
    return StreamingResponse(
        _drain_to_sse(state["queue"], state["abort"]),
        media_type="text/event-stream",
    )


@app.post("/admin/bench/stability/{run_id}/abort")
async def admin_bench_stability_abort(run_id: str):
    state = _bench_active.get(run_id)
    if not state:
        raise HTTPException(404, f"unknown run {run_id}")
    state["abort"].set()
    return {"ok": True, "aborting": run_id}


@app.get("/admin/bench/stability/runs")
async def admin_bench_stability_runs():
    return {"data": _bench_list_stability()}


@app.get("/admin/bench/stability/runs/{run_id}")
async def admin_bench_stability_get(run_id: str):
    r = _bench_get_stability(run_id)
    if not r:
        raise HTTPException(404, f"unknown run {run_id}")
    return r


@app.get("/admin/bench/stability/runs/{run_id}/markdown")
async def admin_bench_stability_md(run_id: str):
    r = _bench_get_stability(run_id)
    if not r:
        raise HTTPException(404, f"unknown run {run_id}")
    return PlainTextResponse(_bench_stability_md(r), media_type="text/markdown")


@app.delete("/admin/bench/stability/runs/{run_id}")
async def admin_bench_stability_delete(run_id: str):
    if not _bench_delete_stability(run_id):
        raise HTTPException(404, f"unknown run {run_id}")
    return {"ok": True}


# ── Endurance endpoints ──────────────────────────────────────────────────────

@app.post("/admin/bench/endurance")
async def admin_bench_endurance(req: BenchEnduranceRequest):
    """Start an endurance run. Runs until /stop is called. Returns {runId, stream}."""
    if not req.model:
        raise HTTPException(400, "model required")
    cfg = {
        "endpoint": req.endpoint or DEFAULT_ENGINE_URL,
        "model": req.model,
        "concurrency": req.concurrency,
        "prompt": req.prompt,
        "maxTokens": req.maxTokens,
        "temperature": req.temperature,
        "headers": req.headers,
    }
    queue: asyncio.Queue = asyncio.Queue()
    abort = asyncio.Event()
    emit = _make_event_emitter(queue)

    async def runner():
        try:
            record = await _bench_run_endurance(cfg, emit, abort)
            await queue.put(("done", {"runId": record["id"], "record": record}))
        except Exception as e:
            await queue.put(("error", {"error": str(e)}))
        finally:
            await queue.put((None, None))

    task = asyncio.create_task(runner())
    # The runner emits "start" first — extract run_id from it.
    first = await asyncio.wait_for(queue.get(), timeout=10.0)
    run_id = (first[1] or {}).get("runId")
    if not run_id:
        raise HTTPException(500, "failed to obtain run_id")

    pre: asyncio.Queue = asyncio.Queue()
    await pre.put(first)

    async def relay():
        while True:
            ev = await queue.get()
            await pre.put(ev)
            if ev[0] is None:
                return
    asyncio.create_task(relay())

    _bench_active[run_id] = {"queue": pre, "abort": abort, "kind": "endurance", "task": task}
    return {"runId": run_id, "stream": f"/admin/bench/endurance/{run_id}/stream"}


@app.get("/admin/bench/endurance/{run_id}/stream")
async def admin_bench_endurance_stream(run_id: str):
    state = _bench_active.get(run_id)
    if not state:
        record = _bench_get_endurance(run_id)
        if not record:
            raise HTTPException(404, f"unknown run {run_id}")
        async def replay() -> AsyncIterator[bytes]:
            payload = json.dumps(
                {"type": "done", "payload": {"runId": run_id, "record": record}},
                default=str,
            )
            yield ("data: " + payload + "\n\n").encode()
        return StreamingResponse(replay(), media_type="text/event-stream")
    return StreamingResponse(
        _drain_to_sse(state["queue"], state["abort"]),
        media_type="text/event-stream",
    )


@app.post("/admin/bench/endurance/{run_id}/stop")
async def admin_bench_endurance_stop(run_id: str):
    """Graceful stop: finish in-flight requests, then save record."""
    state = _bench_active.get(run_id)
    if not state:
        raise HTTPException(404, f"unknown run {run_id}")
    state["abort"].set()
    return {"ok": True, "stopping": run_id}


@app.get("/admin/bench/endurance/runs")
async def admin_bench_endurance_runs():
    return {"data": _bench_list_endurance()}


@app.get("/admin/bench/active")
async def admin_bench_all_active():
    """Return all currently running bench runs (any kind)."""
    return {"runs": [
        {"runId": rid, "kind": state["kind"]}
        for rid, state in _bench_active.items()
    ]}


@app.get("/admin/bench/endurance/{run_id}/samples")
async def admin_bench_endurance_samples(run_id: str):
    return {"data": _bench_get_endurance_samples(run_id)}


@app.delete("/admin/bench/endurance/runs/{run_id}")
async def admin_bench_endurance_delete(run_id: str):
    if not _bench_delete_endurance(run_id):
        raise HTTPException(404, f"unknown run {run_id}")
    return {"ok": True}


# ── Directed benchmark endpoints ────────────────────────────────────────
# Sends an ordered prompt set (from {test_repo}/prompt/) to a single model,
# persists each answer in {test_repo}/answer/{run_id}/{model_slug}/.
# Pattern follows stress/stability SSE bridge.


@app.post("/admin/bench/directed")
async def admin_bench_directed(req: BenchDirectedRequest):
    """Kick off a directed run. Returns {runId, stream}."""
    if not req.model:
        raise HTTPException(400, "model required")
    cfg = {
        "endpoint": req.endpoint or DEFAULT_ENGINE_URL,
        "model": req.model,
        "test_repo": req.test_repo,
        "maxTokens": req.maxTokens,
        "temperature": req.temperature,
        "concurrency": req.concurrency,
        "headers": req.headers,
    }
    # Pre-validate so we surface a clean error before starting the run task
    from bench_directed import _resolve_test_repo, _validate_test_repo
    try:
        _validate_test_repo(_resolve_test_repo(cfg))
    except ValueError as e:
        raise HTTPException(400, str(e))

    queue: asyncio.Queue = asyncio.Queue()
    abort = asyncio.Event()
    emit = _make_event_emitter(queue)

    async def runner():
        try:
            record = await _bench_run_directed(cfg, emit, abort)
            await queue.put(("runDone", {"runId": record["id"], "record": record}))
        except Exception as e:
            await queue.put(("error", {"error": str(e)}))
        finally:
            await queue.put((None, None))

    task = asyncio.create_task(runner())
    # The runner emits runStart first — extract run_id from it.
    first = await asyncio.wait_for(queue.get(), timeout=10.0)
    run_id = (first[1] or {}).get("runId")
    if not run_id:
        raise HTTPException(500, "failed to obtain run_id")

    pre: asyncio.Queue = asyncio.Queue()
    await pre.put(first)

    async def relay():
        while True:
            ev = await queue.get()
            await pre.put(ev)
            if ev[0] is None:
                return
    asyncio.create_task(relay())

    _bench_active[run_id] = {"queue": pre, "abort": abort,
                              "kind": "directed", "task": task}
    return {"runId": run_id, "stream": f"/admin/bench/directed/{run_id}/stream"}


@app.get("/admin/bench/directed/{run_id}/stream")
async def admin_bench_directed_stream(run_id: str):
    state = _bench_active.get(run_id)
    if not state:
        record = _bench_get_directed(run_id)
        if not record:
            raise HTTPException(404, f"unknown run {run_id}")
        async def replay() -> AsyncIterator[bytes]:
            payload = json.dumps(
                {"type": "runDone", "payload": {"runId": run_id, "record": record}},
                default=str,
            )
            yield ("data: " + payload + "\n\n").encode()
        return StreamingResponse(replay(), media_type="text/event-stream")
    return StreamingResponse(
        _drain_to_sse(state["queue"], state["abort"]),
        media_type="text/event-stream",
    )


@app.post("/admin/bench/directed/{run_id}/abort")
async def admin_bench_directed_abort(run_id: str):
    state = _bench_active.get(run_id)
    if not state:
        raise HTTPException(404, f"unknown run {run_id}")
    state["abort"].set()
    return {"ok": True, "aborting": run_id}


@app.get("/admin/bench/directed/runs")
async def admin_bench_directed_runs():
    return {"data": _bench_list_directed()}


@app.get("/admin/bench/directed/runs/{run_id}")
async def admin_bench_directed_get(run_id: str):
    r = _bench_get_directed(run_id)
    if not r:
        raise HTTPException(404, f"unknown run {run_id}")
    return r


@app.get("/admin/bench/directed/runs/{run_id}/markdown")
async def admin_bench_directed_md(run_id: str):
    r = _bench_get_directed(run_id)
    if not r:
        raise HTTPException(404, f"unknown run {run_id}")
    return PlainTextResponse(_bench_directed_md(r), media_type="text/markdown")


@app.delete("/admin/bench/directed/runs/{run_id}")
async def admin_bench_directed_delete(run_id: str):
    if not _bench_delete_directed(run_id):
        raise HTTPException(404, f"unknown run {run_id}")
    return {"ok": True}


@app.get("/admin/bench/clusters")
async def admin_bench_clusters():
    """Proxy through to Odysseus's `GET /admin/clusters` so the dashboard
    populates the cluster selector from the engine's own truth instead of
    hardcoding ids. Returns the upstream payload verbatim ({data: [...]}),
    or an empty list on failure."""
    url = f"{DEFAULT_ENGINE_URL.rstrip('/')}/admin/clusters"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            if not r.is_success:
                return {"engine": DEFAULT_ENGINE_URL,
                        "error": f"HTTP {r.status_code}", "data": []}
            return {"engine": DEFAULT_ENGINE_URL, **r.json()}
    except Exception as e:
        return {"engine": DEFAULT_ENGINE_URL, "error": str(e), "data": []}


@app.get("/admin/bench/targets")
async def admin_bench_targets():
    """Probe the configured engine and return its model list, so the UI
    can populate the model selector for bench runs."""
    url = f"{DEFAULT_ENGINE_URL.rstrip('/')}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            if not r.is_success:
                return {"engine": DEFAULT_ENGINE_URL, "error": f"HTTP {r.status_code}",
                        "data": []}
            d = r.json()
            models = []
            for m in d.get("data", []):
                caps = m.get("x_odyssai") or {}
                models.append({
                    "id": m.get("id"),
                    "pool": caps.get("pool"),
                    "backend": caps.get("backend"),
                    "loaded": caps.get("loaded"),
                    "supports_vision": caps.get("supports_vision"),
                    "supports_tools": caps.get("supports_tools"),
                })
            return {"engine": DEFAULT_ENGINE_URL, "data": models}
    except Exception as e:
        return {"engine": DEFAULT_ENGINE_URL, "error": str(e), "data": []}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
