#!/usr/bin/env python3
"""Stability / RDMA repro kit for Odysseus pools.

Runs N iterations of LOAD → CHAT → UNLOAD against a cluster and captures
the per-iteration outcome with enough detail to file a clean upstream
issue against MLX / JACCL when it goes wrong.

This is the bench Sophie's 2026-05-18 audit asked for ("stability
benchmark, pas seulement speed benchmark"). The 'reboot Mac to recover'
ladder we live with today is invisible in throughput-only benchmarks —
this one's score is success rate + memory released + no-reboot completion.

Usage:
    python3 bench_stability.py \
        --endpoint http://192.168.86.141:8000 \
        --cluster  argo \
        --model    /Volumes/models/odysseus/.../some-model \
        --nodes    3 \
        --iterations 10 \
        --out-prefix bench-stability-2026-05-19

Outputs:
    <prefix>.json     full per-iteration log
    <prefix>.md       human-readable report

Per iteration, captures:
    - load_ok, load_s
    - chat_ok, chat_ttft_s, chat_tokens
    - unload_ok, sweep_warnings, post_unload_wired_gb (per node)
    - degraded_after — whether the cluster ended the iteration flagged

The Mac nodes are probed via Odysseus's own /admin/nodes/telemetry
(it already does the vm_stat round-trip we'd need) so the bench has no
SSH side-channel — everything flows through the engine API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

try:
    import httpx
except ImportError:
    print("httpx required: pip install httpx", file=sys.stderr)
    sys.exit(1)


# ─── Persistence ─────────────────────────────────────────────────────────
# Mirrors bench_stress's layout — same DATA_DIR, dedicated history file.
DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", "/app/data"))
STABILITY_HISTORY_FILE = DATA_DIR / "bench-stability-history.json"
MAX_STABILITY_HISTORY = 100

EventCallback = Callable[[str, dict], Awaitable[None]]


async def _noop_emit(_name: str, _data: dict) -> None:
    return None


# ─── Helpers ─────────────────────────────────────────────────────────────


async def _get(client: httpx.AsyncClient, url: str,
               headers: Optional[dict] = None) -> Any:
    r = await client.get(url, timeout=30.0, headers=headers)
    r.raise_for_status()
    return r.json()


async def _post(client: httpx.AsyncClient, url: str, payload: dict | None = None,
                timeout: float = 600.0, headers: Optional[dict] = None) -> Any:
    r = await client.post(url, json=(payload or {}), timeout=timeout,
                          headers=headers)
    if r.status_code >= 400:
        # Don't raise — we want to capture the error structure in the report.
        try:
            return {"_http_status": r.status_code, **r.json()}
        except Exception:
            return {"_http_status": r.status_code, "_text": r.text[:500]}
    return r.json()


async def _telemetry_snapshot(client: httpx.AsyncClient,
                              endpoint: str, hosts: list[str],
                              headers: Optional[dict] = None) -> list[dict]:
    """Per-host wired memory + RDMA + runner-count snapshot. Filtered to the
    hosts we care about for the current cluster."""
    try:
        data = await _get(client,
                          f"{endpoint}/admin/nodes/telemetry?fresh=true",
                          headers=headers)
    except Exception as e:
        return [{"error": str(e)}]
    out: list[dict] = []
    for h in data.get("hosts") or []:
        if h.get("id") not in hosts:
            continue
        out.append({
            "host": h.get("id"),
            "wired_gb": round((h.get("ram_wired_bytes") or 0) / 1024**3, 1),
            "free_gb": round(
                ((h.get("ram_total_bytes") or 0) - (h.get("ram_used_bytes") or 0))
                / 1024**3, 1,
            ),
            "runner_count": h.get("runner_count"),
            "rdma_active": sum(
                1 for p in (h.get("rdma_ports") or [])
                if p.get("state") == "ACTIVE"
            ),
            "rdma_total": len(h.get("rdma_ports") or []),
        })
    return out


# ─── Per-iteration run ───────────────────────────────────────────────────


async def _one_iteration(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    cluster: str,
    model: str,
    nodes: int,
    iteration: int,
    chat_prompt: str,
    chat_max_tokens: int,
    hosts: list[str],
    headers: Optional[dict] = None,
) -> dict:
    """Run one full LOAD → CHAT → UNLOAD cycle."""
    result: dict = {"iteration": iteration}
    pre = await _telemetry_snapshot(client, endpoint, hosts, headers=headers)
    result["pre"] = pre

    # ── LOAD ──────────────────────────────────────────────────────────
    t0 = time.time()
    load_resp = await _post(
        client,
        f"{endpoint}/admin/{cluster}/load",
        {"model": model, "nodes": nodes, "force": True},
        timeout=600.0,
        headers=headers,
    )
    result["load"] = {
        "ok": load_resp.get("loaded") is True,
        "elapsed_s": round(time.time() - t0, 2),
        "load_s": load_resp.get("load_s"),
        "error": (load_resp.get("detail")
                  if not load_resp.get("loaded") else None),
    }
    if not result["load"]["ok"]:
        result["chat"] = {"ok": False, "skipped": "load_failed"}
    else:
        # ── CHAT ──────────────────────────────────────────────────────
        t1 = time.time()
        chat_resp = await _post(
            client,
            f"{endpoint}/v1/chat/completions",
            {
                "model": cluster,
                "messages": [{"role": "user", "content": chat_prompt}],
                "max_tokens": chat_max_tokens,
                "stream": False,
            },
            timeout=300.0,
            headers=headers,
        )
        elapsed = time.time() - t1
        choice = (chat_resp.get("choices") or [{}])[0]
        usage = chat_resp.get("usage") or {}
        result["chat"] = {
            "ok": "error" not in chat_resp and bool(choice.get("message")),
            "elapsed_s": round(elapsed, 2),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "ttft_proxy_s": chat_resp.get("x_mlx_cluster", {}).get("ttft_s"),
            "tps": chat_resp.get("x_mlx_cluster", {}).get("tps"),
            "error": chat_resp.get("detail")
                     if "error" in chat_resp else None,
        }

    # ── UNLOAD ────────────────────────────────────────────────────────
    t2 = time.time()
    unload_resp = await _post(
        client,
        f"{endpoint}/admin/{cluster}/unload",
        {},
        timeout=120.0,
        headers=headers,
    )
    result["unload"] = {
        "ok": unload_resp.get("loaded") is False,
        "elapsed_s": round(time.time() - t2, 2),
        "sweep_warnings": (unload_resp.get("sweep") or {}).get("warnings") or [],
    }

    post = await _telemetry_snapshot(client, endpoint, hosts, headers=headers)
    result["post"] = post

    # Check cluster degraded flag — if set by the sweep / load failure,
    # subsequent iterations would be refused by the degraded gate.
    try:
        st = await _get(client, f"{endpoint}/admin/{cluster}/status",
                        headers=headers)
        result["degraded_after"] = bool(st.get("degraded"))
    except Exception:
        result["degraded_after"] = None

    return result


# ─── Driver ──────────────────────────────────────────────────────────────


async def run_stability(
    cfg: dict,
    on_event: EventCallback = _noop_emit,
    abort_event: Optional[asyncio.Event] = None,
) -> dict:
    """Execute the LOAD → CHAT → UNLOAD loop and persist the result.

    `cfg` keys (all optional except where noted):
        endpoint        engine URL (default ODYSSAI_ENGINE_URL env var)
        cluster         "argo" | "hades" | "nautilus"  (default "argo")
        model           full model path or HF id (required)
        nodes           int — Argo only (default 1)
        iterations      int (default 5)
        prompt          chat prompt
        maxTokens       int (default 200)
        headers         optional dict of headers (e.g. crew auth)

    `on_event(name, data)` is called for these names:
        runStart       {runId, total}
        iterationStart {runId, i, total}
        iterationDone  {runId, i, total, iteration}   # full per-iter record
        progress       {runId, completed, total}
        done           {runId, record}                # whole record
    """
    endpoint = cfg.get("endpoint") or os.environ.get(
        "ODYSSAI_ENGINE_URL", "http://host.docker.internal:8000"
    )
    cluster = cfg.get("cluster", "argo")
    model = cfg["model"]
    nodes = int(cfg.get("nodes", 1))
    total_iters = int(cfg.get("iterations", 5))
    prompt = cfg.get("prompt", "Write a short haiku about a Mac Studio cluster.")
    max_tokens = int(cfg.get("maxTokens", 200))
    headers = cfg.get("headers") or None

    run_id = str(uuid.uuid4())
    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started)) + "Z"

    await on_event("runStart", {"runId": run_id, "total": total_iters,
                                 "cluster": cluster, "model": model,
                                 "nodes": nodes})

    iterations: list[dict] = []
    aborted_degraded = False
    aborted_user = False
    cluster_resolve_error: Optional[str] = None

    async with httpx.AsyncClient() as client:
        # Resolve cluster hosts for telemetry filtering.
        try:
            cluster_def = await _get(
                client, f"{endpoint}/admin/clusters/{cluster}", headers=headers,
            )
        except Exception as e:
            cluster_resolve_error = f"failed to resolve cluster: {e}"
            cluster_def = {}

        hosts = [n["host"] for n in (cluster_def.get("nodes") or [])]

        # Safety: refuse to run when the cluster is already degraded.
        already_degraded = bool(cluster_def.get("degraded"))

        if not cluster_resolve_error and hosts and not already_degraded:
            for i in range(1, total_iters + 1):
                if abort_event and abort_event.is_set():
                    aborted_user = True
                    break
                await on_event(
                    "iterationStart",
                    {"runId": run_id, "i": i, "total": total_iters},
                )
                try:
                    rec = await _one_iteration(
                        client,
                        endpoint=endpoint,
                        cluster=cluster,
                        model=model,
                        nodes=nodes,
                        iteration=i,
                        chat_prompt=prompt,
                        chat_max_tokens=max_tokens,
                        hosts=hosts,
                        headers=headers,
                    )
                except Exception as e:
                    rec = {"iteration": i, "fatal": str(e)}
                iterations.append(rec)
                await on_event(
                    "iterationDone",
                    {"runId": run_id, "i": i, "total": total_iters,
                     "iteration": rec},
                )
                await on_event(
                    "progress",
                    {"runId": run_id, "completed": i, "total": total_iters},
                )
                # Bail early when degraded — continuing in a sick state would
                # just hammer the same broken QPs.
                if rec.get("degraded_after"):
                    aborted_degraded = True
                    break

    n_load_ok = sum(1 for r in iterations if r.get("load", {}).get("ok"))
    n_chat_ok = sum(1 for r in iterations if r.get("chat", {}).get("ok"))
    n_unload_ok = sum(1 for r in iterations if r.get("unload", {}).get("ok"))
    n_sweep_warn = sum(
        1 for r in iterations if r.get("unload", {}).get("sweep_warnings")
    )

    record = {
        "id": run_id,
        "startedAt": started_at,
        "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
        "wallSec": round(time.time() - started, 1),
        "config": {
            "endpoint": endpoint,
            "cluster": cluster,
            "model": model,
            "nodes": nodes,
            "iterations": total_iters,
            "prompt": prompt,
            "maxTokens": max_tokens,
        },
        "iterations_total": total_iters,
        "iterations_completed": len(iterations),
        "aborted_degraded": aborted_degraded,
        "aborted_user": aborted_user,
        "cluster_resolve_error": cluster_resolve_error,
        "cluster_already_degraded": already_degraded,
        "iterations": iterations,
        "aggregate": {
            "load_success_rate":  n_load_ok / max(1, len(iterations)),
            "chat_success_rate":  n_chat_ok / max(1, len(iterations)),
            "unload_success_rate": n_unload_ok / max(1, len(iterations)),
            "iterations_with_sweep_warnings": n_sweep_warn,
        },
    }

    _save_stability_run(record)
    await on_event("done", {"runId": run_id, "record": record})
    return record


# Legacy CLI helper: keeps the argparse entry point working without the
# refactor leaking into all the call sites. Bridges argparse namespace
# → cfg dict.
async def _run_from_cli(args: argparse.Namespace) -> dict:
    cfg = {
        "endpoint": args.endpoint,
        "cluster": args.cluster,
        "model": args.model,
        "nodes": args.nodes,
        "iterations": args.iterations,
        "prompt": args.prompt,
        "maxTokens": args.max_tokens,
    }

    last_progress = {"done": 0}

    async def cli_emit(name: str, data: dict) -> None:
        if name == "iterationStart":
            print(f"[bench-stability] iteration {data['i']}/{data['total']}")
        elif name == "progress":
            last_progress["done"] = data.get("completed", 0)

    return await run_stability(cfg, cli_emit)


# ─── Persistence helpers ────────────────────────────────────────────────


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str))


def _save_stability_run(record: dict) -> None:
    hist = _load_json(STABILITY_HISTORY_FILE)
    hist.insert(0, record)
    while len(hist) > MAX_STABILITY_HISTORY:
        hist.pop()
    _save_json(STABILITY_HISTORY_FILE, hist)


def list_stability_history() -> list[dict]:
    """Return a compact summary per run (no per-iteration detail) for the
    history table. The full record is fetched on demand."""
    out = []
    for r in _load_json(STABILITY_HISTORY_FILE):
        cfg = r.get("config") or {}
        agg = r.get("aggregate") or {}
        out.append({
            "id": r.get("id"),
            "startedAt": r.get("startedAt"),
            "finishedAt": r.get("finishedAt"),
            "wallSec": r.get("wallSec"),
            "cluster": cfg.get("cluster"),
            "model": cfg.get("model"),
            "nodes": cfg.get("nodes"),
            "iterations_total": r.get("iterations_total"),
            "iterations_completed": r.get("iterations_completed"),
            "aborted_degraded": r.get("aborted_degraded"),
            "aborted_user": r.get("aborted_user"),
            "load_success_rate": agg.get("load_success_rate"),
            "chat_success_rate": agg.get("chat_success_rate"),
            "unload_success_rate": agg.get("unload_success_rate"),
            "iterations_with_sweep_warnings":
                agg.get("iterations_with_sweep_warnings"),
        })
    return out


def get_stability_run(run_id: str) -> Optional[dict]:
    for r in _load_json(STABILITY_HISTORY_FILE):
        if r.get("id") == run_id:
            return r
    return None


def delete_stability_run(run_id: str) -> bool:
    hist = _load_json(STABILITY_HISTORY_FILE)
    new = [r for r in hist if r.get("id") != run_id]
    if len(new) == len(hist):
        return False
    _save_json(STABILITY_HISTORY_FILE, new)
    return True


def stability_run_to_markdown(record: dict) -> str:
    """Format a stability record as a Markdown report. Mirrors
    bench_stress.stress_run_to_markdown for parity in the dashboard."""
    cfg = record.get("config") or {}
    a = record.get("aggregate") or {}
    lines: list[str] = []
    started = record.get("startedAt") or "?"
    lines.append(f"# Stability bench — {started}")
    lines.append("")
    lines.append(f"- **Endpoint**: `{cfg.get('endpoint', '?')}`")
    lines.append(f"- **Cluster**: `{cfg.get('cluster', '?')}` × "
                 f"{cfg.get('nodes', '?')} nodes")
    model_short = (cfg.get("model") or "?").rsplit("/", 1)[-1]
    lines.append(f"- **Model**: `{model_short}`")
    lines.append(f"- **Iterations**: {record.get('iterations_completed', 0)}"
                 f"/{record.get('iterations_total', 0)}"
                 f"{' (aborted on degraded state)' if record.get('aborted_degraded') else ''}"
                 f"{' (aborted by user)' if record.get('aborted_user') else ''}")
    lines.append(f"- **Wall**: {record.get('wallSec', 0)} s")
    if record.get("cluster_resolve_error"):
        lines.append(f"- **Setup error**: `{record['cluster_resolve_error']}`")
    if record.get("cluster_already_degraded"):
        lines.append("- **Setup**: cluster was already degraded — refused to run")
    lines.append("")
    lines.append("## Success rates")
    lines.append("")
    lines.append(f"- Load:   {a.get('load_success_rate', 0)*100:.1f} %")
    lines.append(f"- Chat:   {a.get('chat_success_rate', 0)*100:.1f} %")
    lines.append(f"- Unload: {a.get('unload_success_rate', 0)*100:.1f} %")
    lines.append(f"- Iterations with wired-memory leak warning: "
                 f"{a.get('iterations_with_sweep_warnings', 0)}")
    lines.append("")
    lines.append("## Per-iteration details")
    lines.append("")
    lines.append("| # | load_s | chat_s | unload_s | sweep warn | degraded after |")
    lines.append("|---|---|---|---|---|---|")
    for r in record.get("iterations") or []:
        ls = r.get("load", {}).get("elapsed_s", "—")
        cs = r.get("chat", {}).get("elapsed_s", "—")
        us = r.get("unload", {}).get("elapsed_s", "—")
        sw = len(r.get("unload", {}).get("sweep_warnings") or [])
        deg = "🔴 yes" if r.get("degraded_after") else "✓ no"
        lines.append(f"| {r.get('iteration')} | {ls} | {cs} | {us} | {sw} | {deg} |")
    lines.append("")
    has_anomalies = any(
        r.get("fatal") or r.get("unload", {}).get("sweep_warnings")
        for r in record.get("iterations") or []
    )
    if has_anomalies:
        lines.append("## Anomalies")
        lines.append("")
        for r in record.get("iterations") or []:
            if r.get("fatal"):
                lines.append(f"- iter {r['iteration']}: fatal — `{r['fatal']}`")
            for w in r.get("unload", {}).get("sweep_warnings") or []:
                lines.append(f"- iter {r['iteration']}: sweep warning — {w}")
        lines.append("")
    return "\n".join(lines)


# Backwards-compat shim for the CLI path that still wants to write the
# Markdown to a file.
def write_markdown(record: dict, path: Path) -> None:
    path.write_text(stability_run_to_markdown(record))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint", default="http://192.168.86.141:8000")
    p.add_argument("--cluster", default="argo", choices=["argo", "hades", "nautilus"])
    p.add_argument("--model", required=True,
                   help="Full model path or HF id, e.g. /Volumes/models/odysseus/...")
    p.add_argument("--nodes", type=int, default=1,
                   help="Argo only — number of nodes to load on")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--prompt",
                   default="Write a short haiku about a Mac Studio cluster.",
                   help="Chat prompt sent each iteration")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--out-prefix", default="bench-stability")
    args = p.parse_args()

    print(f"[bench-stability] {args.iterations} iterations against {args.endpoint} "
          f"cluster={args.cluster} model={args.model.split('/')[-1]} "
          f"nodes={args.nodes}")
    record = asyncio.run(_run_from_cli(args))
    json_path = Path(f"{args.out_prefix}.json")
    md_path = Path(f"{args.out_prefix}.md")
    json_path.write_text(json.dumps(record, indent=2))
    write_markdown(record, md_path)
    print(f"[bench-stability] wrote {json_path} + {md_path}")
    a = record["aggregate"]
    print(f"[bench-stability] load OK {a['load_success_rate']*100:.0f}% · "
          f"chat OK {a['chat_success_rate']*100:.0f}% · "
          f"unload OK {a['unload_success_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
