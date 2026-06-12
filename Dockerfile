FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: Install CPU-only PyTorch BEFORE requirements ───────────────────────
# Pinned to 2.5.1+cpu — same major.minor as the working local venv (2.5.1+cu121),
# just without CUDA. sentence-transformers 5.2.3 was built against 2.5.x and
# uses APIs that don't exist in older torch releases.
# Must come BEFORE requirements.docker.txt so pip sees torch as already satisfied
# and never pulls the CUDA wheel (~2.5 GB) as a transitive dependency.
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ── Step 2: Install project dependencies ───────────────────────────────────────
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# ── Step 3: Copy source code ───────────────────────────────────────────────────
# Explicit directory copies (not COPY . .) so build context stays clean
# and large files excluded by .dockerignore are never even considered.
COPY agent/    agent/
COPY api/      api/
COPY hammer/   hammer/
COPY pipeline/ pipeline/
COPY rag/      rag/
COPY sentries/ sentries/
COPY training/ training/

# ── Runtime config ─────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
