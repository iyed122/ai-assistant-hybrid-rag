#!/usr/bin/env python3
"""
training_pipeline_router.py
════════════════════════════
FastAPI router for the training pipeline.

Endpoints:
  POST /training/prepare         — Dataset preparation (returns stats)
  POST /training/run             — Start training (SSE stream of logs)
  POST /training/run/stop        — Kill running training subprocess
  GET  /training/runs            — List MLflow runs
  POST /training/promote/{id}    — Promote a run to Production
  GET  /training/model/current   — Current production model info
  GET  /training/config          — Get default training config
  POST /training/config          — Save training config

Mount in main.py:
  from api.training_pipeline_router import router as pipeline_router
  app.include_router(pipeline_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("training_pipeline_router")

# ── Project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Router ──────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/training", tags=["training-pipeline"])

# ── Global: running training subprocess ─────────────────────────────────────
_training_process: Optional[subprocess.Popen] = None


# ── Request models ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    method:               str   = "qlora"    # qlora | dpo | sequential
    base_model:           str   = ""         # empty = use default from .env
    lora_rank:            int   = 16
    lora_alpha:           int   = 32
    learning_rate:        float = 2e-4
    epochs:               int   = 3
    batch_size:           int   = 1
    gradient_accumulation: int  = 8
    max_seq_length:       int   = 1024
    dpo_beta:             float = 0.1


class PromoteRequest(BaseModel):
    run_id: str


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/prepare — Dataset preparation
# ═════════════════════════════════════════════════════════════════════════════

@router.post("/prepare")
async def prepare_dataset():
    """
    Pull from MongoDB, deduplicate, split 85/15, format, snapshot.
    Returns dataset stats before committing to training.
    """
    try:
        from training.pipeline import prepare_datasets
        stats = prepare_datasets()
        return JSONResponse(stats)
    except Exception as e:
        logger.error("prepare failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/run — Start training with SSE stream
# ═════════════════════════════════════════════════════════════════════════════

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/run")
async def start_training(req: TrainRequest):
    """
    Spawn training as a subprocess and stream stdout/stderr back as SSE.

    The training script writes JSON log lines to stdout which we parse
    and forward. Same SSE pattern as the /chat/stream endpoint.
    """
    global _training_process

    if _training_process and _training_process.poll() is None:
        raise HTTPException(
            status_code=409,
            detail="A training run is already in progress. Stop it first.",
        )

    # Write config to disk so the subprocess picks it up
    from training.pipeline import TrainConfig, TRAINING_DIR

    config = TrainConfig(
        method=req.method,
        lora_rank=req.lora_rank,
        lora_alpha=req.lora_alpha,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        batch_size=req.batch_size,
        gradient_accumulation=req.gradient_accumulation,
        max_seq_length=req.max_seq_length,
        dpo_beta=req.dpo_beta,
    )
    if req.base_model:
        config.base_model = req.base_model

    config_path = TRAINING_DIR / "run_config.json"
    config.save(config_path)

    # Spawn the training script
    cmd = [
        sys.executable, "-u",  # unbuffered
        str(ROOT / "training" / "pipeline.py"),
        "train",
        "--method", req.method,
        "--config", str(config_path),
    ]

    logger.info("Spawning training: %s", " ".join(cmd))

    async def stream_training():
        global _training_process

        yield _sse({"type": "status", "message": "Starting training...", "config": json.loads(json.dumps(req.dict()))})

        try:
            _training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            step_count = 0
            for line in iter(_training_process.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue

                # Try to parse structured log lines
                if "step=" in line and "epoch=" in line:
                    # Parse training metrics from log lines
                    step_count += 1
                    yield _sse({
                        "type":    "log",
                        "step":    step_count,
                        "message": line,
                        "raw":     line,
                    })
                elif line.startswith("{"):
                    try:
                        data = json.loads(line)
                        yield _sse({"type": "metrics", **data})
                    except json.JSONDecodeError:
                        yield _sse({"type": "log", "message": line})
                else:
                    yield _sse({"type": "log", "message": line})

                # Yield control to the event loop
                await asyncio.sleep(0)

            _training_process.wait()
            exit_code = _training_process.returncode

            if exit_code == 0:
                yield _sse({"type": "complete", "message": "Training finished successfully", "exit_code": 0})
            elif exit_code < 0 or exit_code == 1:
                yield _sse({"type": "stopped", "message": "Training stopped by user", "exit_code": exit_code})
            else:
                yield _sse({"type": "error", "message": f"Training failed (exit code {exit_code})", "exit_code": exit_code})

        except Exception as e:
            logger.error("Training stream error: %s", e, exc_info=True)
            yield _sse({"type": "error", "message": str(e)})
        finally:
            _training_process = None

    return StreamingResponse(
        stream_training(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "Connection":                  "keep-alive",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin":  "*",
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/run/stop — Kill running training
# ═════════════════════════════════════════════════════════════════════════════

@router.post("/run/stop")
async def stop_training():
    """Gracefully kill the running training subprocess."""
    global _training_process

    if _training_process is None or _training_process.poll() is not None:
        return JSONResponse({"ok": True, "message": "No training running"})

    logger.info("Stopping training process (PID %d)", _training_process.pid)
    _training_process.terminate()

    # Give it 10s to clean up, then kill
    try:
        _training_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _training_process.kill()

    _training_process = None
    return JSONResponse({"ok": True, "message": "Training stopped"})


# ═════════════════════════════════════════════════════════════════════════════
# GET /training/runs — List MLflow runs
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/runs")
async def get_runs():
    """List all training runs from MLflow with metrics."""
    try:
        from training.pipeline import list_runs
        runs = list_runs()
        return JSONResponse({"runs": runs, "total": len(runs)})
    except Exception as e:
        logger.warning("list_runs failed: %s", e)
        return JSONResponse({"runs": [], "total": 0})


# ═════════════════════════════════════════════════════════════════════════════
# POST /training/promote/{run_id} — Promote to Production
# ═════════════════════════════════════════════════════════════════════════════

@router.post("/promote/{run_id}")
async def promote_run(run_id: str):
    """
    Promote a training run to Production in MLflow registry.
    Weaver automatically picks up the new adapter on next request.
    Optionally runs Hammer eval on held-out set.
    """
    try:
        from training.pipeline import promote_model, hammer_eval_after_promote

        result = promote_model(run_id)

        # Auto-eval with Hammer
        try:
            hammer = hammer_eval_after_promote(run_id)
            result["hammer_eval"] = hammer
        except Exception as e:
            logger.warning("Hammer post-promote eval failed: %s", e)
            result["hammer_eval"] = {"error": str(e)}

        return JSONResponse(result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("promote failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ═════════════════════════════════════════════════════════════════════════════
# GET /training/model/current — Current production model
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/model/current")
async def get_current_model():
    """What version is Weaver currently using."""
    try:
        from training.pipeline import get_active_adapter, MODEL_REGISTRY, MLFLOW_URI
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()

        versions = client.get_latest_versions(MODEL_REGISTRY, stages=["Production"])
        if versions:
            v = versions[0]
            return JSONResponse({
                "active":      True,
                "model_name":  MODEL_REGISTRY,
                "version":     v.version,
                "run_id":      v.run_id,
                "stage":       v.current_stage,
                "source":      v.source,
                "created":     v.creation_timestamp,
            })
        else:
            return JSONResponse({
                "active":      False,
                "model_name":  MODEL_REGISTRY,
                "message":     "No Production model — using base model",
            })
    except Exception as e:
        return JSONResponse({
            "active":  False,
            "message": f"MLflow unavailable: {e}",
        })


# ═════════════════════════════════════════════════════════════════════════════
# GET /training/config — Default config
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/config")
async def get_default_config():
    """Return the default training config (for the UI form)."""
    from training.pipeline import TrainConfig
    from dataclasses import asdict
    return JSONResponse(asdict(TrainConfig()))
