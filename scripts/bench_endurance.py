"""Endurance benchmark — runs continuously until stopped.

No fixed total. Each worker loops: fire one inference, record metrics,
repeat. Calling stop sets the abort event; workers finish their current
in-flight request then exit. Final record is saved to history.

Usage from api.py:

    from bench_endurance import (
        run_endurance,
        list_endurance_history, get_endurance_run, delete_endurance_run,
    )
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Awaitable, Callable, Optional

from bench_stress import _run_one


DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", "/app/data"))
HISTORY_FILE = DATA_DIR / "bench-endurance-history.json"
SAMPLES_DIR = DATA_DIR / "endurance-samples"
MAX_HISTORY = 50
SAMPLE_INTERVAL = 60  # seconds between tk/s samples

EventCallback = Callable[[str, dict], Awaitable[None]]


# ── History helpers ──────────────────────────────────────────────────────────

def _load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text("utf-8"))
    except Exception:
        return []


def _save_history(records: list) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(records[-MAX_HISTORY:], indent=2), "utf-8")


def list_endurance_history() -> list:
    return list(reversed(_load_history()))


def get_endurance_run(run_id: str) -> Optional[dict]:
    for r in _load_history():
        if r.get("id") == run_id:
            return r
    return None


def delete_endurance_run(run_id: str) -> bool:
    records = _load_history()
    filtered = [r for r in records if r.get("id") != run_id]
    if len(filtered) == len(records):
        return False
    _save_history(filtered)
    # Also remove samples file if present
    samples_file = SAMPLES_DIR / f"{run_id}.jsonl"
    if samples_file.exists():
        samples_file.unlink()
    return True


# ── Samples helpers ──────────────────────────────────────────────────────────

def _append_sample(run_id: str, sample: dict) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLES_DIR / f"{run_id}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample) + "\n")


def get_endurance_samples(run_id: str) -> list:
    f = SAMPLES_DIR / f"{run_id}.jsonl"
    if not f.exists():
        return []
    samples = []
    for line in f.read_text("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                samples.append(json.loads(line))
            except Exception:
                pass
    return samples


# ── Runner ───────────────────────────────────────────────────────────────────

async def run_endurance(
    cfg: dict,
    on_event: EventCallback,
    abort_event: asyncio.Event,
) -> dict:
    endpoint = cfg["endpoint"]
    model = cfg["model"]
    concurrency = max(1, int(cfg.get("concurrency", 1)))
    prompt = cfg.get("prompt", "Tell me about the ocean in three sentences.")
    max_tokens = int(cfg.get("maxTokens", 150))
    temperature = float(cfg.get("temperature", 0.7))
    headers_extra = cfg.get("headers") or None

    run_id = str(uuid.uuid4())
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    t0 = time.time()

    inferences = 0
    prompt_tokens = 0
    completion_tokens = 0
    errors = 0
    req_counter = 0
    last_sample_t = t0
    lock = asyncio.Lock()

    # First event — lets api.py extract the run_id before streaming starts.
    await on_event("start", {"runId": run_id, "startedAt": started_at})

    async def worker():
        nonlocal inferences, prompt_tokens, completion_tokens, errors, req_counter, last_sample_t
        while not abort_event.is_set():
            async with lock:
                req_counter += 1
                req_id = req_counter

            result = await _run_one(
                endpoint, model, prompt, max_tokens, temperature,
                req_id, headers_extra,
            )

            sample_to_emit = None
            async with lock:
                if result.get("success"):
                    inferences += 1
                    prompt_tokens += result.get("promptTokens") or 0
                    completion_tokens += result.get("completionTokens") or 0
                else:
                    errors += 1

                now = time.time()
                uptime = now - t0
                tps = completion_tokens / uptime if uptime > 0 else 0

                if now - last_sample_t >= SAMPLE_INTERVAL:
                    last_sample_t = now
                    sample_to_emit = {
                        "t": int(now),
                        "tps": round(tps, 2),
                        "inferences": inferences,
                        "errors": errors,
                    }

            uptime = time.time() - t0
            tps = completion_tokens / uptime if uptime > 0 else 0
            await on_event("tick", {
                "runId": run_id,
                "inferences": inferences,
                "promptTokens": prompt_tokens,
                "completionTokens": completion_tokens,
                "totalTokens": prompt_tokens + completion_tokens,
                "tkPerSec": round(tps, 2),
                "uptimeSec": round(uptime, 1),
                "errors": errors,
                "lastSuccess": result.get("success", False),
                "lastError": result.get("error") if not result.get("success") else None,
            })

            if sample_to_emit:
                _append_sample(run_id, sample_to_emit)
                await on_event("sample", sample_to_emit)

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers, return_exceptions=True)

    wall_sec = time.time() - t0
    stopped_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    total_attempts = inferences + errors
    tps = completion_tokens / wall_sec if wall_sec > 0 else 0

    record = {
        "id": run_id,
        "startedAt": started_at,
        "stoppedAt": stopped_at,
        "uptimeSec": round(wall_sec, 1),
        "config": {
            "endpoint": endpoint,
            "model": model,
            "concurrency": concurrency,
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
        "inferences": inferences,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "totalTokens": prompt_tokens + completion_tokens,
        "avgTkPerSec": round(tps, 2),
        "errors": errors,
        "errorRate": round(errors / total_attempts, 4) if total_attempts > 0 else 0,
    }

    history = _load_history()
    history.append(record)
    _save_history(history)

    return record
