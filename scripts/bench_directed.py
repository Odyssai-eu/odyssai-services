"""Directed benchmark — sends a set of ordered prompts to a model, records
each answer, and persists them alongside the prompt set for later analysis.

Structure expected in the test repo:
  {test_repo}/
    instructions.md       # system message sent with every prompt
    prompt/               # .md or .txt files, ordered by filename
    answer/               # created on first run
      {run_id}/
        {model_slug}/
          01_*.md
          02_*.md
          _run.json

Pattern follows bench_stress.py (SSE events, history JSON, etc).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

import httpx


DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", "/app/data"))
HISTORY_FILE = DATA_DIR / "directed-history.json"
MAX_HISTORY = 100


# ──────────────────────────────────────────────────────────────────────────────
# History helpers (same pattern as bench_stress)
# ──────────────────────────────────────────────────────────────────────────────
def _load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text()) or []
    except Exception:
        return []


def _save_history(records: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(records, indent=2, default=str))


def list_directed_history() -> list[dict]:
    return _load_history()


def get_directed_run(run_id: str) -> Optional[dict]:
    for r in _load_history():
        if r["id"] == run_id:
            return r
    return None


def delete_directed_run(run_id: str) -> bool:
    hist = _load_history()
    new = [r for r in hist if r["id"] != run_id]
    if len(new) == len(hist):
        return False
    _save_history(new)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Test repo helpers
# ──────────────────────────────────────────────────────────────────────────────
def _slugify(s: str) -> str:
    """Model names like 'or:kimi-k-2.7-code' or paths with slashes become fs-safe."""
    return re.sub(r"[^\w\-]+", "-", s).strip("-").lower()


def _list_prompts(prompts_dir: Path) -> list[tuple[str, str, str]]:
    """Return sorted [(stem, filename, content), ...] from prompts dir."""
    if not prompts_dir.is_dir():
        return []
    files = []
    for p in sorted(prompts_dir.iterdir()):
        if p.suffix.lower() in (".md", ".txt") and not p.name.startswith("."):
            files.append((p.stem, p.name, p.read_text(encoding="utf-8", errors="replace")))
    return files


def _resolve_test_repo(cfg: dict) -> Path:
    """Determine the test repo path from config or env default."""
    return Path(cfg.get("test_repo") or os.environ.get("ODYSSAI_BENCH_DEFAULT_TEST_REPO", ""))


def _validate_test_repo(test_dir: Path) -> tuple[Path, Path, list[tuple[str, str, str]]]:
    """Validate the test repo structure and return (instructions_path, prompts_dir, prompts).
    Raises ValueError on failure."""
    prompts_dir = test_dir / "prompt"
    instructions_path = test_dir / "instructions.md"

    if not test_dir.is_dir():
        raise ValueError(f"test repo not found: {test_dir}")
    if not instructions_path.is_file():
        raise ValueError(f"instructions.md missing in {test_dir}")
    prompts = _list_prompts(prompts_dir)
    if not prompts:
        raise ValueError(f"no prompts (*.md, *.txt) found in {prompts_dir}")
    return instructions_path, prompts_dir, prompts


# ──────────────────────────────────────────────────────────────────────────────
# Core: run one non-streamed completion
# ──────────────────────────────────────────────────────────────────────────────
async def _run_one(
    endpoint: str,
    model: str,
    system_content: str,
    prompt_content: str,
    max_tokens: int,
    temperature: float,
    headers_extra: Optional[dict] = None,
) -> dict:
    """POST a single chat completion (non-streaming) and return the result."""
    t0 = time.time()
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    headers = {"content-type": "application/json"}
    if headers_extra:
        headers.update(headers_extra)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt_content},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    error: Optional[str] = None
    content = ""
    usage = {}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            r = await client.post(url, headers=headers, json=body)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            j = r.json()
            choice = (j.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            usage = j.get("usage") or {}
    except Exception as e:
        error = str(e)[:300]

    elapsed = time.time() - t0
    return {
        "success": error is None,
        "error": error,
        "content": content,
        "latencyMs": round(elapsed * 1000, 1),
        "promptTokens": usage.get("prompt_tokens"),
        "completionTokens": usage.get("completion_tokens"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
EventCallback = Callable[[str, dict], Awaitable[None]]


async def run_directed(
    cfg: dict,
    on_event: EventCallback,
    abort_event: Optional[asyncio.Event] = None,
) -> dict:
    """Run the directed benchmark. Returns the record dict."""
    endpoint = cfg.get("endpoint") or os.environ.get("ODYSSAI_ENGINE_URL", "http://host.docker.internal:8000")
    model = cfg["model"]
    max_tokens = int(cfg.get("maxTokens", 1024))
    temperature = float(cfg.get("temperature", 0.0))
    concurrency = int(cfg.get("concurrency", 1))
    headers_extra = cfg.get("headers") or None

    test_repo = _resolve_test_repo(cfg)
    instructions_path, prompts_dir, prompts = _validate_test_repo(test_repo)

    # Read instructions as system message
    system_content = instructions_path.read_text(encoding="utf-8", errors="replace")

    run_id = str(uuid.uuid4())
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    t0 = time.time()
    model_slug = _slugify(model)

    # Create answers dir
    answers_dir = test_repo / "answer" / run_id / model_slug
    answers_dir.mkdir(parents=True, exist_ok=True)

    await on_event("runStart", {
        "runId": run_id,
        "model": model,
        "total": len(prompts),
        "testRepo": str(test_repo),
    })

    results: list[dict] = []
    queue: list[tuple[int, str, str, str]] = [
        (i, stem, fname, content) for i, (stem, fname, content) in enumerate(prompts, 1)
    ]
    lock = asyncio.Lock()
    completed = 0
    in_flight = 0

    async def worker():
        nonlocal completed, in_flight
        while True:
            if abort_event and abort_event.is_set():
                return
            async with lock:
                if not queue:
                    return
                idx, stem, fname, prompt_content = queue.pop(0)
                in_flight += 1
            await on_event("progress", {
                "runId": run_id, "inFlight": in_flight,
                "completed": completed, "total": len(prompts),
            })
            await on_event("promptStart", {
                "runId": run_id, "index": idx, "stem": stem,
                "file": fname,
            })

            result = await _run_one(
                endpoint, model, system_content, prompt_content,
                max_tokens, temperature, headers_extra,
            )

            # Write answer file
            answer_file = answers_dir / f"{stem}.md"
            if result["success"]:
                answer_file.write_text(result["content"], encoding="utf-8")
            else:
                answer_file.write_text(
                    f"# ERROR\n\n{result['error']}\n",
                    encoding="utf-8",
                )

            async with lock:
                in_flight -= 1
                completed += 1
                results.append({
                    "index": idx,
                    "stem": stem,
                    "file": str(answer_file),
                    "success": result["success"],
                    "error": result["error"],
                    "latencyMs": result["latencyMs"],
                    "promptTokens": result["promptTokens"],
                    "completionTokens": result["completionTokens"],
                    "contentPreview": result["content"][:200],
                })

            await on_event("promptDone", {
                "runId": run_id,
                "index": idx,
                "stem": stem,
                "status": "ok" if result["success"] else "error",
                "latencyMs": result["latencyMs"],
                "error": result["error"],
                "outputFile": str(answer_file),
            })
            await on_event("progress", {
                "runId": run_id, "inFlight": in_flight,
                "completed": completed, "total": len(prompts),
            })

    workers = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
    await asyncio.gather(*workers, return_exceptions=True)

    wall_sec = time.time() - t0
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"

    total = len(prompts)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    errors = [{"index": r["index"], "stem": r["stem"], "error": r["error"]}
              for r in results if not r["success"]]

    record = {
        "id": run_id,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "wallSec": wall_sec,
        "config": {
            "endpoint": endpoint,
            "model": model,
            "modelSlug": model_slug,
            "testRepo": str(test_repo),
            "maxTokens": max_tokens,
            "temperature": temperature,
            "concurrency": concurrency,
            "instructionsPreview": system_content[:200],
        },
        "result": {
            "total": total,
            "passed": passed,
            "failed": failed,
        },
        "errors": errors,
        "results": results,
        "answersDir": str(answers_dir),
        "aborted": bool(abort_event and abort_event.is_set()),
    }

    # Write _run.json alongside answers
    run_meta = answers_dir / "_run.json"
    run_meta.write_text(json.dumps({
        "id": run_id,
        "model": model,
        "startedAt": started_at,
        "finishedAt": finished_at,
        "total": total,
        "passed": passed,
        "failed": failed,
        "answers": [{"index": r["index"], "stem": r["stem"],
                      "file": r["stem"] + ".md",
                      "success": r["success"], "error": r["error"]}
                     for r in results],
    }, indent=2, default=str), encoding="utf-8")

    # Save to history
    hist = _load_history()
    hist.insert(0, record)
    if len(hist) > MAX_HISTORY:
        hist = hist[:MAX_HISTORY]
    _save_history(hist)

    await on_event("runDone", {"runId": run_id, "record": record})
    return record


# ──────────────────────────────────────────────────────────────────────────────
# Markdown export
# ──────────────────────────────────────────────────────────────────────────────
def directed_run_to_markdown(r: dict) -> str:
    if not r:
        return "# Directed run not found\n"
    cfg = r["config"]
    res = r["result"]
    lines = []
    lines.append(f"# Directed benchmark — {r['startedAt']}")
    lines.append("")
    lines.append(f"**Model**: `{cfg['model']}`")
    lines.append(f"**Test repo**: `{cfg['testRepo']}`")
    lines.append(f"**Total**: {res['total']} · **Passed**: {res['passed']} · **Failed**: {res['failed']}")
    lines.append(f"**Wall**: {r['wallSec']:.1f}s{ ' (aborted)' if r.get('aborted') else ''}")
    lines.append(f"**Answers dir**: `{r['answersDir']}`")
    lines.append("")
    lines.append("## Instructions preview")
    lines.append("")
    lines.append("> " + (cfg.get("instructionsPreview") or "").replace("\n", "\n> "))
    lines.append("")
    lines.append("## Results")
    lines.append("")
    for rr in r.get("results") or []:
        status = "✓" if rr["success"] else "✗"
        lines.append(f"- {status} **{rr['index']:02d}. {rr['stem']}** "
                     f"— {rr['latencyMs']:.0f}ms"
                     f"{' · error: ' + rr['error'] if rr.get('error') else ''}")
    lines.append("")
    if r.get("errors"):
        lines.append("## Errors")
        lines.append("")
        for e in r["errors"]:
            lines.append(f"- `{e['stem']}`: {e['error']}")
        lines.append("")
    return "\n".join(lines)