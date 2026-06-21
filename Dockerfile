# Odyssai Services (odyssai.eu) — admin cockpit for non-distributed services
# Manages mlx-vlm (ultra-96A) and mlx-coder (max-64) via SSH.
# Runs alongside Odysseus on the same Docker host, different port (8001).

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole scripts/ tree: api.py imports bench_stress / bench_stability /
# bench_endurance, and dashboard.html is referenced via DASHBOARD_FILE.
# Copying only api.py (as the previous Dockerfile did) silently broke boot
# the moment someone rebuilt the image -- bench_* were missing from /app,
# and the import crashed the container into a restart loop.
COPY scripts/ /app/

ENV DASHBOARD_FILE=/app/dashboard.html \
    API_HOST=0.0.0.0 \
    API_PORT=8001 \
    PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["python", "/app/api.py"]
