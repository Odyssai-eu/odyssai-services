# CLAUDE.md — Odyssai Services

> Part of **odyssai.eu** family : Odysseus (engine), Companion (client), 
> **Odyssai Services** (this — sidecar cockpit). Created 2026-05-13 to
> separate non-distributed services from Odysseus.

## Scope

This service is a thin orchestrator over SSH for two local services:

- **mlx-vlm** on ultra-96A (`192.168.86.49:8080`)
- **mlx-coder** on max-64 (`192.168.86.50:8086`)

It does NOT do any compute. It does NOT serve `/v1/chat/completions`. The
actual model inference happens on the target hosts (mlx_vlm.server,
mlx_lm.server). This is purely an admin/control plane.

For the inference itself, clients route through LiteLLM (`192.168.86.44:4000`)
which calls those services directly.

## Architecture

```
Browser → Dashboard (port 8001)
   ↓
FastAPI orchestrator (container `odyssai-services` on mini-i3)
   ↓ SSH
ultra-96A:8080 (mlx_vlm.server)  &  max-64:8086 (mlx_lm.server)
```

## Watchdog convention on managed hosts

Each managed Mac runs a cron @minute watchdog that respawns the service if
not running. The watchdog reads a state file for the model path (mlx-vlm
only — mlx-coder hardcodes its model).

| Service | Binary | Watchdog | State file |
|---|---|---|---|
| mlx-vlm | `mlx_vlm.server` | `~/bin/mlx-vlm-watchdog.sh` | `~/.mlx-vlm-current-model` |
| mlx-coder | `mlx_lm.server` | `~/bin/mlx-coder-watchdog.sh` | hardcoded (Qwen2.5-1.5B) |

To "stop" a service: set state file to `__unloaded__` + kill the process.
The watchdog respects the marker and doesn't respawn.

## Adding a new managed service

Edit `scripts/api.py`, add an entry to `SERVICES = {...}` :

```python
"new-service": {
    "name": "...",
    "description": "...",
    "host_id": "...",
    "ssh": "admin@192.168.86.X",
    "port": 0,
    "binary": "...",           # what pgrep -f matches
    "models_dirs": ["..."],
    "state_file": "...",        # or None if hardcoded
    "watchdog": "...",
    "log_files": ["..."],
    "kind": "...",
}
```

Rebuild + redeploy.

## Deploy commands

```bash
# Hot reload (same container, no rebuild)
scp scripts/api.py admin@192.168.86.141:/tmp/
ssh admin@192.168.86.141 'docker cp /tmp/api.py odyssai-services:/app/api.py'
ssh admin@192.168.86.141 'docker restart odyssai-services'

# Same for dashboard.html (no restart needed — served static)

# Full rebuild
scp -r . admin@192.168.86.141:~/odyssai-services-build/
ssh admin@192.168.86.141 'cd ~/odyssai-services-build && docker compose up -d --build'
```

## Auth

`ODYSSAI_SERVICES_TOKEN` env var on the container. Same pattern as Odysseus.
Public routes : `/`, `/health`, `/.well-known/*`.
Bearer routes : everything under `/admin/*`.

## SSH access from container

The container mounts `${HOME}/.ssh:/root/.ssh:ro` on mini-i3. The container
uses root, so `/root/.ssh/id_*` are the actual `admin@mini-i3` keys mounted
in. Same setup as Odysseus.

## Things this is NOT

- Not an inference engine (no `/v1/chat/completions`)
- Not a product (internal cockpit for the operator)
- Not Companion's responsibility (separation of concerns)
- Not auto-discovery (services explicitly listed in code)

## Directed benchmark

5e bench dans la famille (`bench_directed.py`). Différent des 4 benchs de perf :
envoie une liste ordonnée de prompts (lus sur disque) à UN modèle, écrit chaque
réponse dans un dossier pour analyse ultérieure.

**Structure attendue** dans `{test_repo}/` :
```
{test_repo}/
├── instructions.md      # lu UNE fois, envoyé comme system message à chaque prompt
├── prompt/              # .md / .txt triés par nom de fichier (01_xxx, 02_xxx, …)
└── answer/              # créé au 1er run
    └── {run_id}/{model_slug}/
        ├── 01_xxx.md
        └── _run.json
```

**Path par défaut** : `/Volumes/ext_512/workplace/benchs/{test_name}` (overridable
via le champ `test_repo` du formulaire UI, ou `ODYSSAI_BENCH_DEFAULT_TEST_REPO`).
Le container monte `/Volumes/ext_512` en `rw` (cf `docker-compose.yml`).

**Routes** : `/admin/bench/directed/*` (run, stream, abort, runs, runs/{id},
runs/{id}/markdown, delete). SSE events : `runStart`, `promptStart`,
`promptDone`, `progress`, `runDone`.

**Bouton "Analyse"** dans l'UI est volontairement **disabled** — projet futur.
