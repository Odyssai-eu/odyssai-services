"""Stress + Sweep benchmark for Odyssai engines.

Ported from `exoscopy-starbase/server/bench-stress.js`. Talks to any
OpenAI-compatible engine (Odysseus, mlx-vlm, OpenRouter directly, …)
via streaming chat completions. Captures latency percentiles and
aggregate throughput, persists run history as JSON in /app/data/.

Sweep variants: ramp concurrency, repeat identical, or custom config grid.

Usage from api.py:

    from bench_stress import (
        run_stress, run_sweep,
        list_stress_history, get_stress_run, delete_stress_run, stress_run_to_markdown,
        list_sweep_history, get_sweep, delete_sweep, sweep_to_markdown,
    )
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx


DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", "/app/data"))
HISTORY_FILE = DATA_DIR / "bench-stress-history.json"
SWEEP_HISTORY_FILE = DATA_DIR / "bench-sweep-history.json"

MAX_HISTORY = 100        # keep at most this many runs
MAX_SWEEP_HISTORY = 50

# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pct(sorted_vals: list[float], p: int) -> Optional[float]:
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int((p / 100) * len(sorted_vals)))
    return sorted_vals[i]


def _aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    ttfts = sorted(r["ttftSec"] for r in ok if r.get("ttftSec") is not None)
    totals = sorted(r["totalSec"] for r in ok)
    speeds = sorted(r["speed"] for r in ok if (r.get("speed") or 0) > 0)
    tokens = sum((r.get("completionTokens") or 0) for r in ok)

    starts = [r["startedAtMs"] for r in results if r.get("startedAtMs")]
    ends = [r["startedAtMs"] + r["totalSec"] * 1000 for r in ok]
    wall_start = min(starts) if starts else time.time() * 1000
    wall_end = max(ends) if ends else wall_start
    wall_sec = (wall_end - wall_start) / 1000

    # Per model breakdown
    per_model: dict[str, dict] = {}
    for r in ok:
        m = r["modelId"]
        if m not in per_model:
            per_model[m] = {"n": 0, "ttft": [], "total": [], "tokens": 0, "speeds": []}
        per_model[m]["n"] += 1
        if r.get("ttftSec") is not None:
            per_model[m]["ttft"].append(r["ttftSec"])
        per_model[m]["total"].append(r["totalSec"])
        per_model[m]["tokens"] += r.get("completionTokens") or 0
        if (r.get("speed") or 0) > 0:
            per_model[m]["speeds"].append(r["speed"])

    per_model_agg = []
    for model, s in per_model.items():
        ttft_sorted = sorted(s["ttft"])
        total_sorted = sorted(s["total"])
        speed_sorted = sorted(s["speeds"])
        # If any result for this model carried a resolved label, surface it
        # alongside the raw id so 'argo' is visibly 'argo — Qwen3.5-122B'.
        # We pick the most common label seen (sessions where the alias was
        # swapped mid-run leave both labels present — keep the majority one).
        labels = [r.get("modelLabel") for r in ok
                  if r.get("modelId") == model and r.get("modelLabel")]
        label = max(set(labels), key=labels.count) if labels else model
        per_model_agg.append({
            "model": model,
            "modelLabel": label,
            "reqs": s["n"], "tokens": s["tokens"],
            "ttftP50": _pct(ttft_sorted, 50), "ttftP90": _pct(ttft_sorted, 90),
            "totalP50": _pct(total_sorted, 50), "totalP90": _pct(total_sorted, 90),
            "speedP50": _pct(speed_sorted, 50), "speedP90": _pct(speed_sorted, 90),
        })

    return {
        "total": len(results),
        "success": len(ok),
        "failed": len(failed),
        "wallSec": wall_sec,
        "aggregateTokPerSec": (tokens / wall_sec) if wall_sec > 0 else 0,
        "totalCompletionTokens": tokens,
        "ttft": {
            "p50": _pct(ttfts, 50), "p90": _pct(ttfts, 90),
            "p95": _pct(ttfts, 95), "p99": _pct(ttfts, 99),
            "min": ttfts[0] if ttfts else None,
            "max": ttfts[-1] if ttfts else None,
        },
        "totalTime": {
            "p50": _pct(totals, 50), "p90": _pct(totals, 90),
            "p95": _pct(totals, 95), "p99": _pct(totals, 99),
            "min": totals[0] if totals else None,
            "max": totals[-1] if totals else None,
        },
        "speedPerReq": {
            "p50": _pct(speeds, 50), "p90": _pct(speeds, 90),
            "min": speeds[0] if speeds else None,
            "max": speeds[-1] if speeds else None,
        },
        "perModel": per_model_agg,
        "failedDetails": [
            {"reqId": f.get("reqId"), "model": f.get("modelId"), "error": f.get("error")}
            for f in failed
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single streamed completion
# ──────────────────────────────────────────────────────────────────────────────
async def _run_one(
    endpoint: str, model_id: str, prompt: str,
    max_tokens: int, temperature: float, req_id: int,
    headers_extra: Optional[dict] = None,
) -> dict:
    t0 = time.time()
    first_token_at: Optional[float] = None
    content_parts: list[str] = []
    chunks = 0
    prompt_tokens = 0
    completion_tokens = 0
    error: Optional[str] = None

    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    headers = {"content-type": "application/json"}
    if headers_extra:
        headers.update(headers_extra)

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None)) as client:
            async with client.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code >= 400:
                    text = (await r.aread()).decode("utf-8", "ignore")
                    raise RuntimeError(f"HTTP {r.status_code}: {text[:200]}")

                buf = ""
                async for chunk_bytes in r.aiter_bytes():
                    buf += chunk_bytes.decode("utf-8", "ignore")
                    lines = buf.split("\n")
                    buf = lines.pop() or ""
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            j = json.loads(data)
                        except Exception:
                            continue
                        delta = (j.get("choices") or [{}])[0].get("delta", {}).get("content")
                        if delta:
                            if first_token_at is None:
                                first_token_at = time.time()
                            content_parts.append(delta)
                            chunks += 1
                        usage = j.get("usage") or {}
                        if usage.get("prompt_tokens"):
                            prompt_tokens = usage["prompt_tokens"]
                        if usage.get("completion_tokens"):
                            completion_tokens = usage["completion_tokens"]
    except Exception as e:
        error = str(e)[:300]

    t1 = time.time()
    total = t1 - t0
    ttft = (first_token_at - t0) if first_token_at else None
    gen_time = (t1 - first_token_at) if first_token_at else 0
    tokens = completion_tokens or chunks
    speed = (tokens / gen_time) if (gen_time > 0 and tokens > 0) else 0
    content = "".join(content_parts)

    return {
        "reqId": req_id, "modelId": model_id,
        "success": error is None, "error": error,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)) + "Z",
        "startedAtMs": t0 * 1000,
        "totalSec": total, "ttftSec": ttft, "genSec": gen_time,
        "promptTokens": prompt_tokens, "completionTokens": tokens,
        "chunks": chunks, "speed": speed,
        "contentPreview": content[:120],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Stress run
# ──────────────────────────────────────────────────────────────────────────────
EventCallback = Callable[[str, dict], Awaitable[None]]


async def _resolve_model_labels(
    endpoint: str, model_ids: list[str], headers_extra: Optional[dict] = None,
) -> dict[str, str]:
    """Snapshot the alias→concrete-name map at run start.

    Calls /v1/models on the bench endpoint once and, for each requested
    model id, returns a friendly label that expands pool aliases (e.g.
    'argo') to 'argo — Qwen3.5-122B-A10B-MLX-9bit' using `x_odyssai.alias_for`
    or `x_concrete`. Models that aren't aliases (or that the endpoint
    doesn't enrich) get the raw id back.

    The map is captured at the start of the run, NOT per-request — so if
    the operator hot-swaps the loaded model mid-bench, the history still
    shows the model that was actually under test for the bulk of requests.
    """
    out: dict[str, str] = {mid: mid for mid in model_ids}
    if not endpoint:
        return out
    headers = {"content-type": "application/json"}
    if headers_extra:
        headers.update(headers_extra)
    url = f"{endpoint.rstrip('/')}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            r = await client.get(url, headers=headers)
            if not r.is_success:
                return out
            data = r.json().get("data") or []
    except Exception:
        return out
    by_id: dict[str, dict] = {}
    for m in data:
        mid = (m.get("id") or "").lower()
        if mid:
            by_id[mid] = m
    for orig in model_ids:
        m = by_id.get(orig.lower())
        if not m:
            continue
        caps = m.get("x_odyssai") or {}
        concrete = caps.get("alias_for") or m.get("x_concrete")
        if not concrete or concrete == orig:
            continue
        # Strip the filesystem path — keep only the model basename.
        basename = concrete.split("/")[-1] or concrete
        if basename and basename != orig:
            out[orig] = f"{orig} — {basename}"
    return out


async def run_stress(
    cfg: dict,
    on_event: EventCallback,
    abort_event: Optional[asyncio.Event] = None,
) -> dict:
    endpoint = cfg["endpoint"]
    models = cfg["models"]
    concurrency = int(cfg.get("concurrency", 2))
    total = int(cfg.get("total", 10))
    prompt = cfg["prompt"]
    max_tokens = int(cfg.get("maxTokens", 150))
    temperature = float(cfg.get("temperature", 0.7))
    headers_extra = cfg.get("headers") or None

    run_id = str(uuid.uuid4())
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    t0 = time.time()

    # Snapshot model labels (alias → 'argo — Qwen3.5-122B…') so the history
    # row, perModel table, and request details all show what actually served
    # the bench instead of just 'argo'.
    model_labels = await _resolve_model_labels(endpoint, models, headers_extra)

    await on_event("progress", {"runId": run_id, "inFlight": 0, "completed": 0, "total": total})

    queue: list[int] = list(range(1, total + 1))
    results: list[dict] = []
    in_flight = 0
    completed = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal in_flight, completed
        while True:
            if abort_event and abort_event.is_set():
                return
            async with lock:
                if not queue:
                    return
                req_id = queue.pop(0)
                in_flight += 1
            model_id = models[(req_id - 1) % len(models)]
            await on_event("progress", {"runId": run_id, "inFlight": in_flight, "completed": completed, "total": total})

            r = await _run_one(endpoint, model_id, prompt, max_tokens, temperature, req_id, headers_extra)
            # Stamp the resolved label per-result so the dashboard can show
            # 'argo — Qwen3.5-122B' rows in the requests table directly,
            # without needing to also load the run's modelLabels map.
            r["modelLabel"] = model_labels.get(model_id, model_id)

            async with lock:
                in_flight -= 1
                completed += 1
                results.append(r)
            await on_event("item", {"runId": run_id, "item": r, "completed": completed, "total": total})
            await on_event("progress", {"runId": run_id, "inFlight": in_flight, "completed": completed, "total": total})

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers, return_exceptions=True)

    wall_sec = time.time() - t0
    agg = _aggregate(results)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"

    record = {
        "id": run_id,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "wallSec": wall_sec,
        "config": {
            "endpoint": endpoint,
            "cluster": cfg.get("cluster"),
            "models": models,
            # Snapshot of alias→'alias — concrete' labels at run start, used
            # by the dashboard for history rows and per-model breakdown.
            "modelLabels": model_labels,
            "concurrency": concurrency,
            "total": total,
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
        "aggregate": agg,
        "requests": results,
        "aborted": bool(abort_event and abort_event.is_set()),
    }

    _save_stress_run(record)
    await on_event("done", {"runId": run_id, "record": record})
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────
def _build_run_configs(sweep_cfg: dict) -> list[dict]:
    base = {
        "cluster": sweep_cfg.get("cluster"),
        "models": sweep_cfg["models"],
        "prompt": sweep_cfg["prompt"],
        "maxTokens": sweep_cfg.get("maxTokens", 150),
        "temperature": sweep_cfg.get("temperature", 0.7),
        "total": sweep_cfg.get("total", 10),
        "concurrency": sweep_cfg.get("concurrency", 2),
    }

    sweep_type = sweep_cfg.get("type")
    if sweep_type == "concurrency":
        values = sweep_cfg.get("values") or [1, 2, 4, 8]
        per_run = sweep_cfg.get("totalPerRun")
        return [
            {**base, "concurrency": c,
             "total": per_run if per_run else max(c * 5, 10),
             "label": f"c={c}"}
            for c in values
        ]
    if sweep_type == "repeat":
        n = max(1, min(20, int(sweep_cfg.get("repeats") or 3)))
        return [{**base, "label": f"run {i + 1}/{n}"} for i in range(n)]
    if sweep_type == "custom":
        configs = sweep_cfg.get("configs") or []
        out = []
        for i, c in enumerate(configs):
            out.append({
                "cluster": c.get("cluster", sweep_cfg.get("cluster")),
                "models": c.get("models", sweep_cfg["models"]),
                "prompt": c.get("prompt", sweep_cfg["prompt"]),
                "maxTokens": c.get("maxTokens", sweep_cfg.get("maxTokens", 150)),
                "temperature": c.get("temperature", sweep_cfg.get("temperature", 0.7)),
                "concurrency": c.get("concurrency", 2),
                "total": c.get("total", (c.get("concurrency", 2)) * 5),
                "label": c.get("label", f"custom {i + 1}"),
            })
        return out
    raise ValueError(f"Unknown sweep type: {sweep_type}")


async def run_sweep(
    sweep_cfg: dict,
    on_event: EventCallback,
    endpoint: str,
    abort_event: Optional[asyncio.Event] = None,
) -> dict:
    sweep_id = str(uuid.uuid4())
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    t0 = time.time()

    run_configs = _build_run_configs(sweep_cfg)
    await on_event("sweep:start", {
        "sweepId": sweep_id, "type": sweep_cfg["type"],
        "runCount": len(run_configs),
        "runs": [{"label": r["label"], "concurrency": r["concurrency"], "total": r["total"]}
                 for r in run_configs],
    })

    runs: list[dict] = []
    for i, rc in enumerate(run_configs):
        if abort_event and abort_event.is_set():
            break
        await on_event("run:start", {
            "sweepId": sweep_id, "runIdx": i, "label": rc["label"], "config": rc,
        })

        async def fwd(name: str, data: dict) -> None:
            # Forward inner-run events with runIdx prefix
            if name in ("progress", "item"):
                await on_event(name, {"sweepId": sweep_id, "runIdx": i, **data})
            elif name == "error":
                await on_event("error", {"sweepId": sweep_id, "runIdx": i, **data})

        rec = await run_stress({**rc, "endpoint": endpoint}, fwd, abort_event)
        runs.append({"label": rc["label"], **rec})
        await on_event("run:done", {
            "sweepId": sweep_id, "runIdx": i, "label": rc["label"], "record": rec,
        })

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    wall_sec = time.time() - t0

    comparison = []
    for r in runs:
        a = r["aggregate"]
        # Prefer the resolved label per model (set by run_stress) over the
        # raw alias. Fall back to the legacy basename-of-path display for
        # entries that resolved to nothing (non-Odysseus engines).
        model_labels = (r["config"].get("modelLabels") or {})
        comparison.append({
            "label": r["label"],
            "concurrency": r["config"]["concurrency"],
            "total": r["config"]["total"],
            "models": [
                model_labels.get(m) or m.split("/")[-1]
                for m in r["config"]["models"]
            ],
            "wallSec": r["wallSec"],
            "success": a["success"],
            "failed": a["failed"],
            "aggTps": a["aggregateTokPerSec"],
            "ttftP50": a["ttft"].get("p50"),
            "ttftP90": a["ttft"].get("p90"),
            "ttftP95": a["ttft"].get("p95"),
            "totalP50": a["totalTime"].get("p50"),
            "totalP90": a["totalTime"].get("p90"),
            "speedP50": a["speedPerReq"].get("p50"),
            "speedP90": a["speedPerReq"].get("p90"),
        })

    sweep_record = {
        "id": sweep_id,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "wallSec": wall_sec,
        "type": sweep_cfg["type"],
        "baseConfig": {
            "cluster": sweep_cfg.get("cluster"),
            "models": sweep_cfg["models"],
            # Pull alias resolution from the first run's snapshot. All
            # child runs share the same endpoint, so the labels are stable
            # across a sweep unless someone reloaded mid-flight.
            "modelLabels": (runs[0]["config"].get("modelLabels")
                            if runs else {}) or {},
            "prompt": sweep_cfg["prompt"],
            "maxTokens": sweep_cfg.get("maxTokens", 150),
            "temperature": sweep_cfg.get("temperature", 0.7),
        },
        "params": {
            "values": sweep_cfg.get("values"),
            "repeats": sweep_cfg.get("repeats"),
            "configs": sweep_cfg.get("configs"),
            "totalPerRun": sweep_cfg.get("totalPerRun"),
        },
        "runs": runs,
        "comparison": comparison,
        "aborted": bool(abort_event and abort_event.is_set()),
    }

    _save_sweep(sweep_record)
    await on_event("sweep:done", {"sweepId": sweep_id, "sweep": sweep_record})
    return sweep_record


# ──────────────────────────────────────────────────────────────────────────────
# Persistence (JSON files)
# ──────────────────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> list:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        print(f"[bench] load {path.name} failed: {e}", flush=True)
    return []


def _save_json(path: Path, data: list) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"[bench] save {path.name} failed: {e}", flush=True)


def _save_stress_run(record: dict) -> None:
    hist = _load_json(HISTORY_FILE)
    hist.insert(0, record)
    while len(hist) > MAX_HISTORY:
        hist.pop()
    _save_json(HISTORY_FILE, hist)


def _save_sweep(record: dict) -> None:
    hist = _load_json(SWEEP_HISTORY_FILE)
    hist.insert(0, record)
    while len(hist) > MAX_SWEEP_HISTORY:
        hist.pop()
    _save_json(SWEEP_HISTORY_FILE, hist)


def list_stress_history() -> list[dict]:
    hist = _load_json(HISTORY_FILE)
    return [{
        "id": r["id"], "startedAt": r["startedAt"], "finishedAt": r["finishedAt"],
        "wallSec": r["wallSec"], "config": r["config"],
        "summary": {
            "total": r["aggregate"]["total"],
            "success": r["aggregate"]["success"],
            "failed": r["aggregate"]["failed"],
            "aggTps": r["aggregate"]["aggregateTokPerSec"],
            "ttftP50": r["aggregate"]["ttft"].get("p50"),
            "speedP50": r["aggregate"]["speedPerReq"].get("p50"),
        },
        "aborted": r.get("aborted", False),
    } for r in hist]


def get_stress_run(run_id: str) -> Optional[dict]:
    for r in _load_json(HISTORY_FILE):
        if r["id"] == run_id:
            return r
    return None


def delete_stress_run(run_id: str) -> bool:
    hist = _load_json(HISTORY_FILE)
    new = [r for r in hist if r["id"] != run_id]
    if len(new) == len(hist):
        return False
    _save_json(HISTORY_FILE, new)
    return True


def list_sweep_history() -> list[dict]:
    hist = _load_json(SWEEP_HISTORY_FILE)
    out = []
    for s in hist:
        comp = s.get("comparison") or []
        best = max((c.get("aggTps") or 0 for c in comp), default=0)
        worst = min((c.get("aggTps") or 0 for c in comp), default=0)
        out.append({
            "id": s["id"], "startedAt": s["startedAt"], "finishedAt": s["finishedAt"],
            "wallSec": s["wallSec"], "type": s["type"],
            "baseConfig": s.get("baseConfig", {}),
            "runCount": len(s.get("runs") or []),
            "aborted": s.get("aborted", False),
            "aggBestTps": best, "aggWorstTps": worst,
        })
    return out


def get_sweep(sweep_id: str) -> Optional[dict]:
    for s in _load_json(SWEEP_HISTORY_FILE):
        if s["id"] == sweep_id:
            return s
    return None


def delete_sweep(sweep_id: str) -> bool:
    hist = _load_json(SWEEP_HISTORY_FILE)
    new = [s for s in hist if s["id"] != sweep_id]
    if len(new) == len(hist):
        return False
    _save_json(SWEEP_HISTORY_FILE, new)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Markdown export
# ──────────────────────────────────────────────────────────────────────────────
def _fmt(n: Optional[float], unit: str = "s", decimals: int = 2) -> str:
    if n is None:
        return "—"
    return f"{n:.{decimals}f}{unit}"


def stress_run_to_markdown(r: dict) -> str:
    if not r:
        return "# Stress run not found\n"
    a = r["aggregate"]
    cfg = r["config"]
    lines = []
    lines.append(f"# Stress run — {r['startedAt']}")
    lines.append("")
    lines.append(f"**Cluster**: `{cfg.get('cluster','—')}` · **Endpoint**: `{cfg['endpoint']}`")
    labels = cfg.get("modelLabels") or {}
    lines.append(
        "**Models**: "
        + ", ".join("`" + (labels.get(m) or m) + "`" for m in cfg["models"])
    )
    lines.append(f"**Concurrency**: {cfg['concurrency']} · **Total**: {cfg['total']} · "
                 f"**MaxTokens**: {cfg['maxTokens']} · **Temp**: {cfg['temperature']}")
    lines.append(f"**Wall**: {_fmt(r['wallSec'], 's', 1)} · **Success**: {a['success']}/{a['total']}")
    lines.append(f"**Aggregate throughput**: **{_fmt(a['aggregateTokPerSec'], ' tk/s', 1)}**")
    lines.append("")
    lines.append("> " + cfg["prompt"].replace("\n", "\n> "))
    lines.append("")
    lines.append("## Latency percentiles")
    lines.append("")
    lines.append("| Metric | P50 | P90 | P95 | P99 | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| TTFT  | {_fmt(a['ttft']['p50'])} | {_fmt(a['ttft']['p90'])} | {_fmt(a['ttft']['p95'])} | {_fmt(a['ttft']['p99'])} | {_fmt(a['ttft']['min'])} | {_fmt(a['ttft']['max'])} |")
    lines.append(f"| Total | {_fmt(a['totalTime']['p50'])} | {_fmt(a['totalTime']['p90'])} | {_fmt(a['totalTime']['p95'])} | {_fmt(a['totalTime']['p99'])} | {_fmt(a['totalTime']['min'])} | {_fmt(a['totalTime']['max'])} |")
    lines.append(f"| Speed/req | {_fmt(a['speedPerReq']['p50'], ' tk/s', 1)} | {_fmt(a['speedPerReq']['p90'], ' tk/s', 1)} | — | — | {_fmt(a['speedPerReq']['min'], ' tk/s', 1)} | {_fmt(a['speedPerReq']['max'], ' tk/s', 1)} |")
    lines.append("")
    if len(a.get("perModel") or []) > 1:
        lines.append("## Per-model breakdown")
        lines.append("")
        lines.append("| Model | Reqs | TTFT p50 | Speed p50 | Tokens |")
        lines.append("|---|---|---|---|---|")
        for m in a["perModel"]:
            label = m.get("modelLabel") or m['model'].split('/')[-1]
            lines.append(f"| `{label}` | {m['reqs']} | {_fmt(m['ttftP50'])} | {_fmt(m['speedP50'], ' tk/s', 1)} | {m['tokens']} |")
        lines.append("")
    if a.get("failedDetails"):
        lines.append("## Failures")
        lines.append("")
        for f in a["failedDetails"]:
            lines.append(f"- `{f.get('model')}` req {f.get('reqId')}: {f.get('error')}")
        lines.append("")
    return "\n".join(lines)


def sweep_to_markdown(s: dict) -> str:
    if not s:
        return "# Sweep not found\n"
    lines = []
    lines.append(f"# Bench Sweep — {s['startedAt']}")
    lines.append("")
    lines.append(f"**Type**: `{s['type']}`")
    base = s.get("baseConfig", {})
    if base:
        lines.append(f"**Cluster**: `{base.get('cluster','—')}`")
        base_labels = base.get("modelLabels") or {}
        lines.append(
            "**Models**: "
            + ", ".join(
                "`" + (base_labels.get(m) or m) + "`"
                for m in (base.get("models") or [])
            )
        )
        lines.append(f"**MaxTokens**: {base.get('maxTokens')} · **Temperature**: {base.get('temperature')}")
    lines.append(f"**Runs**: {len(s.get('runs') or [])}{' (aborted)' if s.get('aborted') else ''}")
    lines.append(f"**Total wall**: {_fmt(s['wallSec'], 's', 1)}")
    lines.append("")
    if base.get("prompt"):
        lines.append("**Prompt**")
        lines.append("")
        lines.append("> " + base["prompt"].replace("\n", "\n> "))
        lines.append("")

    lines.append("## Comparison")
    lines.append("")
    lines.append("| # | Label | C | N | Wall | Success | **Agg tk/s** | TTFT p50 | TTFT p90 | Total p50 | Speed p50 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(s.get("comparison") or []):
        lines.append(
            f"| {i + 1} | {c['label']} | {c['concurrency']} | {c['total']} | "
            f"{_fmt(c['wallSec'], 's', 1)} | {c['success']}/{c['success'] + c['failed']} | "
            f"**{_fmt(c['aggTps'], ' tk/s', 1)}** | {_fmt(c.get('ttftP50'))} | "
            f"{_fmt(c.get('ttftP90'))} | {_fmt(c.get('totalP50'))} | "
            f"{_fmt(c.get('speedP50'), ' tk/s', 1)} |"
        )
    lines.append("")

    comp = s.get("comparison") or []
    if len(comp) > 1:
        best = max(comp, key=lambda c: c.get("aggTps") or 0)
        lines.append(f"**Best aggregate throughput**: `{best['label']}` at **{_fmt(best['aggTps'], ' tk/s', 1)}**")
        lines.append("")

    lines.append("## Individual runs")
    lines.append("")
    for i, r in enumerate(s.get("runs") or []):
        cfg = r["config"]
        a = r["aggregate"]
        lines.append(f"### {i + 1}. {r['label']} (concurrency={cfg['concurrency']}, total={cfg['total']})")
        lines.append("")
        lines.append(f"- Wall: {_fmt(r['wallSec'], 's', 1)} · Success: {a['success']}/{a['total']} · "
                     f"**Agg: {_fmt(a['aggregateTokPerSec'], ' tk/s', 1)}**")
        lines.append("")
        lines.append("| Metric | P50 | P90 | P95 | P99 | min | max |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(f"| TTFT  | {_fmt(a['ttft']['p50'])} | {_fmt(a['ttft']['p90'])} | {_fmt(a['ttft']['p95'])} | {_fmt(a['ttft']['p99'])} | {_fmt(a['ttft']['min'])} | {_fmt(a['ttft']['max'])} |")
        lines.append(f"| Total | {_fmt(a['totalTime']['p50'])} | {_fmt(a['totalTime']['p90'])} | {_fmt(a['totalTime']['p95'])} | {_fmt(a['totalTime']['p99'])} | {_fmt(a['totalTime']['min'])} | {_fmt(a['totalTime']['max'])} |")
        lines.append(f"| Speed/req | {_fmt(a['speedPerReq']['p50'], ' tk/s', 1)} | {_fmt(a['speedPerReq']['p90'], ' tk/s', 1)} | — | — | {_fmt(a['speedPerReq']['min'], ' tk/s', 1)} | {_fmt(a['speedPerReq']['max'], ' tk/s', 1)} |")
        lines.append("")
    return "\n".join(lines)
