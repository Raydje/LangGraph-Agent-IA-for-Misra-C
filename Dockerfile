# Stage 1: Builder
FROM python:3.12.3-slim AS builder
WORKDIR /app
# Install system build dependencies for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# Install dependencies into a local directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runner
FROM python:3.12.3-slim AS runner
WORKDIR /app
# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
# Copy installed dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
# Copy source code
COPY . .
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
