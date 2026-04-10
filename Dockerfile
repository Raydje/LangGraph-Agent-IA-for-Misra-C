# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ──────────────────────────────────────────────────────────────────────────────
# Pin by digest for reproducible builds across all environments and CI runs.
# To update: docker pull python:3.12.3-slim && docker inspect python:3.12.3-slim --format='{{index .RepoDigests 0}}'
FROM python:3.12.3-slim@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63 AS builder

WORKDIR /app

# Install system build dependencies for C-extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a user-local directory for clean copy into runner
RUN pip install --no-cache-dir --user -r requirements.txt


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runner
# ──────────────────────────────────────────────────────────────────────────────
# Same base image and digest as builder — single source of truth.
FROM python:3.12.3-slim@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63 AS runner

# OCI image annotations — injected at build time by CI via --build-arg.
# Enables full traceability: registry UI, docker inspect, and audit tooling.
# Usage: docker build \
#   --build-arg VCS_REF=$(git rev-parse HEAD) \
#   --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
#   --build-arg VERSION=1.0.0 .
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.title="MISRA C:2023 Compliance Agent" \
      org.opencontainers.image.description="Autonomous regulatory compliance analysis using a LangGraph multi-agent system." \
      org.opencontainers.image.source="https://github.com/${VCS_REF}"

WORKDIR /app

# curl: required for the HEALTHCHECK below (not present in slim)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Prevent .pyc files polluting the layer; ensure logs stream unbuffered to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user before any file copies
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy source code with correct ownership in a single layer
# (avoids a chown copy-on-write duplicate that doubles the layer size)
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Docker engine and orchestrators (ECS, K8s liveness probes) use this to
# determine container health. start-period gives uvicorn time to initialise
# all services (MongoDB, Pinecone, Redis) before checks begin.
HEALTHCHECK --interval=5s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
