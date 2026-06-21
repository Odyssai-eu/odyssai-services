# Stability bench — 2026-05-19T13:29:14Z

- **Endpoint**: `http://192.168.86.141:8000`
- **Cluster**: `argo` × 2 nodes
- **Model**: `Hy3-preview-MLX-9bit`
- **Iterations**: 5/5
- **Wall**: 180.1 s

## Success rates

- Load:   100.0 %
- Chat:   100.0 %
- Unload: 100.0 %
- Iterations with wired-memory leak warning: 0

## Per-iteration details

| # | load_s | chat_s | unload_s | sweep warn | degraded after |
|---|---|---|---|---|---|
| 1 | 30.6 | 3.91 | 1.45 | 0 | ✓ no |
| 2 | 28.95 | 5.97 | 1.39 | 0 | ✓ no |
| 3 | 28.87 | 4.82 | 1.48 | 0 | ✓ no |
| 4 | 28.91 | 4.24 | 1.4 | 0 | ✓ no |
| 5 | 28.48 | 5.58 | 1.4 | 0 | ✓ no |
