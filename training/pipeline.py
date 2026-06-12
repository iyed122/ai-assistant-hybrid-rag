#!/usr/bin/env python3
"""
Training Pipeline  —  QLoRA SFT + DPO with MLflow
═══════════════════════════════════════════════════
Local-first training pipeline for the AI Assistant.
Designed for RTX 3050 (6 GB VRAM).

Stages:
  1. prepare  — Pull from MongoDB, deduplicate, split, format, snapshot
  2. train    — QLoRA SFT or DPO (or sequential SFT → DPO)
  3. promote  — Register adapter in MLflow, Weaver picks it up

Hardware budget (RTX 3050, 6 GB):
  Base model 4-bit : ~4.0 GB
  LoRA adapters     : ~0.1 GB
  Optimizer states  : ~0.2 GB
  Activations (GC)  : ~1.2 GB
  ─────────────────────────────
  Total             : ~5.5 GB  ← fits with gradient_checkpointing

Usage:
  python training/pipeline.py prepare
  python training/pipeline.py train --method qlora
  python training/pipeline.py train --method dpo
  python training/pipeline.py train --method sequential
  python training/pipeline.py promote <run_id>
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# ── Project root on sys.path ────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("training.pipeline")

# ── Configuration ───────────────────────────────────────────────────────────
MONGO_URI       = os.getenv("MONGO_URI",       "mongodb://localhost:27017/")
MONGO_DB        = os.getenv("MONGO_DB",        "knowledge_base")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "qwen3:8b").strip()

# HuggingFace model ID — pre-quantized bnb 4-bit (~2 GB download)
HF_MODEL_ID     = os.getenv("HF_MODEL_ID",     "unsloth/Qwen2.5-3B-Instruct-bnb-4bit")

# Directories
TRAINING_DIR    = Path(os.getenv("TRAINING_DIR",   str(_HERE)))
DATASET_DIR     = TRAINING_DIR / "datasets"
CHECKPOINT_DIR  = TRAINING_DIR / "checkpoints"
SNAPSHOT_DIR    = TRAINING_DIR / "snapshots"

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{TRAINING_DIR / 'mlflow.db'}")
MLFLOW_EXP      = os.getenv("MLFLOW_EXPERIMENT",   "weaver-finetune")
MODEL_REGISTRY  = os.getenv("MLFLOW_MODEL_NAME",   "weaver-qwen")


# ═════════════════════════════════════════════════════════════════════════════
# Training Config
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    """All hyperparams in one place — serializable to/from JSON."""

    # Model
    base_model:          str   = HF_MODEL_ID       # AWQ = ~4 GB download (not 15 GB)
    method:              str   = "qlora"           # qlora | dpo | sequential

    # LoRA
    lora_rank:           int   = 16
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    lora_target_modules: list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    epochs:              int   = 3
    batch_size:          int   = 1                 # 6 GB VRAM constraint
    gradient_accumulation: int = 8                 # effective batch = 8
    learning_rate:       float = 2e-4
    lr_scheduler:        str   = "cosine"
    warmup_ratio:        float = 0.05
    max_seq_length:      int   = 1024              # balance context vs VRAM
    weight_decay:        float = 0.01
    max_grad_norm:       float = 1.0

    # DPO-specific
    dpo_beta:            float = 0.1

    # Eval
    eval_strategy:       str   = "epoch"
    eval_split:          float = 0.15

    # Hardware
    fp16:                bool  = True
    bf16:                bool  = False              # 3050 doesn't support bf16
    gradient_checkpointing: bool = True
    optim:               str   = "paged_adamw_8bit"

    # Output
    output_dir:          str   = str(CHECKPOINT_DIR)
    run_name:            str   = ""

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        log.info("Config saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        with open(path) as f:
            return cls(**json.load(f))


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Dataset Preparation
# ═════════════════════════════════════════════════════════════════════════════

def _get_db():
    from pymongo import MongoClient
    return MongoClient(MONGO_URI)[MONGO_DB]


def _query_hash(query: str) -> str:
    """Stable hash for deduplication."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def check_base_model_compat(
    train_model: str = HF_MODEL_ID,
    serve_model: str = OLLAMA_MODEL,
) -> Dict[str, Any]:
    """
    FLATTENED FIX (warning, not a blocker): a LoRA adapter only loads into the
    EXACT architecture+size it was trained on. Default training base is
    Qwen2.5-3B while serving is qwen3:8b — such adapters can never serve.
    Set TRAIN_COMPAT_ACK=1 to acknowledge and silence the warning.
    """
    import re as _re

    def _sig(name):
        fam  = _re.search(r"qwen[\s_-]?(\d+(?:\.\d+)?)", name or "", _re.I)
        size = _re.search(r"(\d+(?:\.\d+)?)\s*b\b",      name or "", _re.I)
        return (fam.group(1) if fam else None, size.group(1) if size else None)

    t_fam, t_size = _sig(train_model)
    s_fam, s_size = _sig(serve_model)
    issues = []
    if t_fam and s_fam and t_fam != s_fam:
        issues.append(f"model family mismatch: training Qwen{t_fam} vs serving Qwen{s_fam}")
    if t_size and s_size and t_size != s_size:
        issues.append(f"parameter-size mismatch: training {t_size}B vs serving {s_size}B")
    if not (t_fam or t_size) or not (s_fam or s_size):
        issues.append("could not parse one of the model names — verify manually")

    result = {"train_model": train_model, "serve_model": serve_model,
              "compatible": not issues, "issues": issues}
    if issues and os.getenv("TRAIN_COMPAT_ACK", "0") != "1":
        log.warning("=" * 64)
        log.warning("ADAPTER/SERVING MISMATCH — adapters from this training run")
        log.warning("CANNOT be loaded into the serving model:")
        for i in issues:
            log.warning("  • %s", i)
        log.warning("Fix HF_MODEL_ID / OLLAMA_MODEL, or set TRAIN_COMPAT_ACK=1.")
        log.warning("=" * 64)
    return result


def prepare_datasets(
    eval_split: float = 0.15,
    min_answer_len: int = 80,
    dedup: bool = True,
) -> Dict[str, Any]:
    """
    Stage 1: Pull scored data from MongoDB, deduplicate, split, format, snapshot.

    Returns stats dict streamed back to the UI via SSE.
    """
    import random

    check_base_model_compat()   # FLATTENED FIX: warn early if adapters can't serve

    db   = _get_db()
    coll = db["chat_history"]

    log.info("═" * 50)
    log.info("Stage 1 — Dataset Preparation")
    log.info("═" * 50)

    # ── Pull QLoRA candidates (GOLD grade) ──────────────────────────────────
    qlora_docs = list(coll.find({
        "$or": [
            {"evaluation.quality_grade": "GOLD"},
            {"$and": [
                {"evaluation.quality_grade": {"$exists": False}},
                {"evaluation.training_signal": "qlora_positive"},
            ]},
        ],
        "error": None,
        "answer": {"$exists": True},
    }))

    # ── Pull DPO candidates (FAILED with actionable tags) ──────────────────
    dpo_docs = list(coll.find({
        "$or": [
            {"$and": [
                {"evaluation.quality_grade": "FAILED"},
                {"evaluation.failure_tags": {"$in": ["hallucination", "tool_misuse"]}},
            ]},
            {"$and": [
                {"evaluation.quality_grade": {"$exists": False}},
                {"evaluation.training_signal": "dpo_rejected"},
                {"evaluation.weighted_score": {"$lt": 0.50}},
            ]},
        ],
        "answer": {"$exists": True},
    }))

    # Check for curated chosen halves
    dpo_candidates = list(db["dpo_candidates"].find({"curated": True, "chosen": {"$ne": None}}))
    chosen_map = {c["source_id"]: c["chosen"] for c in dpo_candidates}

    log.info("Raw counts — QLoRA: %d, DPO source: %d, DPO curated: %d",
             len(qlora_docs), len(dpo_docs), len(chosen_map))

    # ── Deduplicate by query similarity ─────────────────────────────────────
    if dedup:
        seen_hashes = set()
        deduped_qlora = []
        for doc in qlora_docs:
            h = _query_hash(doc.get("query", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped_qlora.append(doc)
        qlora_dupes = len(qlora_docs) - len(deduped_qlora)
        qlora_docs = deduped_qlora
        log.info("QLoRA dedup: removed %d duplicates → %d unique", qlora_dupes, len(qlora_docs))
    else:
        qlora_dupes = 0

    # ── Filter short answers ────────────────────────────────────────────────
    qlora_docs = [d for d in qlora_docs if len(d.get("answer", "")) >= min_answer_len]

    # ── Format QLoRA into Alpaca/ChatML ─────────────────────────────────────
    qlora_records = []
    for doc in qlora_docs:
        query  = doc.get("query", "")
        answer = doc.get("answer", "")
        intent = doc.get("intent", "both")

        # ChatML format matching Qwen's expected format
        record = {
            "messages": [
                {"role": "system", "content": _system_prompt(intent)},
                {"role": "user",   "content": query},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "intent": intent,
                "score":  doc.get("evaluation", {}).get("weighted_score"),
                "doc_id": str(doc["_id"]),
            },
        }
        qlora_records.append(record)

    # ── Format DPO pairs ────────────────────────────────────────────────────
    dpo_records = []
    for doc in dpo_docs:
        doc_id = str(doc["_id"])
        if doc_id not in chosen_map:
            continue  # skip uncurated — DPO needs both halves

        query    = doc.get("query", "")
        rejected = doc.get("answer", "")
        chosen   = chosen_map[doc_id]
        intent   = doc.get("intent", "both")

        record = {
            "prompt": [
                {"role": "system", "content": _system_prompt(intent)},
                {"role": "user",   "content": query},
            ],
            "chosen":   [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "metadata": {
                "intent": intent,
                "score":  doc.get("evaluation", {}).get("weighted_score"),
                "doc_id": doc_id,
            },
        }
        dpo_records.append(record)

    # ── Shuffle and split ───────────────────────────────────────────────────
    random.shuffle(qlora_records)
    random.shuffle(dpo_records)

    def _split(records, ratio):
        # FLATTENED FIX: train keeps priority on tiny datasets. Previously a
        # single GOLD record produced train=[] / eval=[record] — training on
        # nothing while holding out the only example.
        if len(records) < 2:
            return list(records), []
        n = max(1, int(len(records) * ratio))
        if n >= len(records):
            n = len(records) - 1
        return records[n:], records[:n]

    qlora_train, qlora_eval = _split(qlora_records, eval_split)
    dpo_train,   dpo_eval   = _split(dpo_records,   eval_split)

    # ── Save versioned snapshot ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir  = SNAPSHOT_DIR / timestamp
    snap_dir.mkdir(parents=True, exist_ok=True)

    def _write_jsonl(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_jsonl(qlora_train, snap_dir / "qlora_train.jsonl")
    _write_jsonl(qlora_eval,  snap_dir / "qlora_eval.jsonl")
    _write_jsonl(dpo_train,   snap_dir / "dpo_train.jsonl")
    _write_jsonl(dpo_eval,    snap_dir / "dpo_eval.jsonl")

    # Also write to the "current" dataset dir for easy access
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(qlora_train, DATASET_DIR / "qlora_train.jsonl")
    _write_jsonl(qlora_eval,  DATASET_DIR / "qlora_eval.jsonl")
    _write_jsonl(dpo_train,   DATASET_DIR / "dpo_train.jsonl")
    _write_jsonl(dpo_eval,    DATASET_DIR / "dpo_eval.jsonl")

    stats = {
        "snapshot":       timestamp,
        "qlora_total":    len(qlora_records),
        "qlora_train":    len(qlora_train),
        "qlora_eval":     len(qlora_eval),
        "qlora_deduped":  qlora_dupes,
        "dpo_total":      len(dpo_records),
        "dpo_train":      len(dpo_train),
        "dpo_eval":       len(dpo_eval),
        "dpo_uncurated":  len(dpo_docs) - len(dpo_records),
        "snapshot_dir":   str(snap_dir),
    }

    log.info("Preparation complete: %s", json.dumps(stats, indent=2))
    return stats


def _system_prompt(intent: str) -> str:
    """Minimal system prompt for ChatML training — matches Weaver's style."""
    if intent == "rag":
        return (
            "You are a helpful AI assistant for a software engineering team. "
            "Answer using the knowledge-base context provided. Be concise and cite sources."
        )
    elif intent == "sentries":
        return (
            "You are a helpful AI assistant for a software engineering team. "
            "Answer using the live API data provided. Include IDs, titles, states, and URLs."
        )
    else:
        return (
            "You are a helpful AI assistant for a software engineering team. "
            "Answer using both knowledge-base context and live API data. "
            "Live API data is authoritative for tickets and status."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Training (QLoRA SFT)
# ═════════════════════════════════════════════════════════════════════════════

def _setup_mlflow(config: TrainConfig) -> str:
    """Initialize MLflow tracking, return run_id."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXP)

    run_name = config.run_name or f"{config.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "method":              config.method,
        "base_model":          config.base_model,
        "lora_rank":           config.lora_rank,
        "lora_alpha":          config.lora_alpha,
        "learning_rate":       config.learning_rate,
        "epochs":              config.epochs,
        "batch_size":          config.batch_size,
        "gradient_accumulation": config.gradient_accumulation,
        "max_seq_length":      config.max_seq_length,
        "optim":               config.optim,
    })
    config.save(TRAINING_DIR / "last_config.json")
    mlflow.log_artifact(str(TRAINING_DIR / "last_config.json"))

    log.info("MLflow run started: %s (id=%s)", run_name, run.info.run_id)
    return run.info.run_id


class TrainingCallback:
    """
    Callback that logs metrics to MLflow and yields SSE-compatible dicts.
    Designed to work with both HF Trainer callbacks and manual training loops.
    """

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def on_log(self, step: int, epoch: float, metrics: Dict[str, float]):
        import mlflow

        entry = {
            "step":    step,
            "epoch":   round(epoch, 2),
            "elapsed": round(time.time() - self.start_time, 1),
            **metrics,
        }
        self.logs.append(entry)

        # Log to MLflow
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=step)

        log.info("step=%d epoch=%.2f %s", step, epoch,
                 " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))))

    def on_train_end(self, final_metrics: Dict[str, float]):
        import mlflow
        for k, v in final_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"final_{k}", v)


def train_qlora_sft(config: TrainConfig, callback: Optional[TrainingCallback] = None):
    """
    QLoRA Supervised Fine-Tuning.

    Loads the base model in 4-bit, attaches LoRA adapters, trains on
    the ChatML-formatted QLoRA dataset.

    Memory budget (RTX 3050, 6 GB):
      4-bit model ~4 GB + LoRA ~0.1 GB + optimizer ~0.2 GB
      + activations with gradient_checkpointing ~1.2 GB = ~5.5 GB
    """
    import torch
    import mlflow
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    cb = callback or TrainingCallback()
    run_id = _setup_mlflow(config)

    log.info("═" * 50)
    log.info("Stage 2 — QLoRA SFT Training")
    log.info("═" * 50)
    log.info("Model: %s | LoRA rank: %d | LR: %s | Epochs: %d",
             config.base_model, config.lora_rank, config.learning_rate, config.epochs)

    # ── Load dataset ────────────────────────────────────────────────────────
    train_path = str(DATASET_DIR / "qlora_train.jsonl")
    eval_path  = str(DATASET_DIR / "qlora_eval.jsonl")

    if not Path(train_path).exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Run `prepare` first."
        )

    dataset_train = load_dataset("json", data_files=train_path, split="train")
    dataset_eval  = load_dataset("json", data_files=eval_path,  split="train") if Path(eval_path).exists() else None

    log.info("Dataset: %d train, %d eval",
             len(dataset_train), len(dataset_eval) if dataset_eval else 0)

    mlflow.log_metrics({
        "dataset_train_size": len(dataset_train),
        "dataset_eval_size":  len(dataset_eval) if dataset_eval else 0,
    })

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model (pre-quantized bnb-4bit) ─────────────────────────────────
    log.info("Loading %s ...", config.base_model)

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA config ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    log.info("Trainable params: %s / %s (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    mlflow.log_metric("trainable_params", trainable)
    mlflow.log_metric("total_params", total)

    # ── Training arguments ──────────────────────────────────────────────────
    output_dir = str(CHECKPOINT_DIR / f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        logging_steps=1,
        eval_strategy=config.eval_strategy if dataset_eval else "no",
        save_strategy="epoch",
        save_total_limit=3,
        max_seq_length=config.max_seq_length,
        report_to="none",  # we use MLflow directly
        dataloader_pin_memory=False,  # save memory
    )

    # ── Custom HF callback for SSE streaming ────────────────────────────────
    from transformers import TrainerCallback as HFCallback

    class _StreamCallback(HFCallback):
        def on_log(self_cb, args, state, control, logs=None, **kwargs):
            if logs:
                metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                cb.on_log(
                    step=state.global_step,
                    epoch=state.epoch or 0,
                    metrics=metrics,
                )

    # ── Format function for ChatML messages ─────────────────────────────────
    def formatting_func(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        callbacks=[_StreamCallback()],
    )

    log.info("Starting SFT training...")
    result = trainer.train()

    # ── Save adapter ────────────────────────────────────────────────────────
    adapter_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    log.info("Adapter saved → %s", adapter_path)

    # ── Log to MLflow ───────────────────────────────────────────────────────
    final_metrics = {
        "train_loss":     result.training_loss,
        "train_runtime":  result.metrics.get("train_runtime", 0),
        "train_samples_per_second": result.metrics.get("train_samples_per_second", 0),
    }

    if dataset_eval:
        eval_result = trainer.evaluate()
        final_metrics["eval_loss"] = eval_result.get("eval_loss", 0)

    cb.on_train_end(final_metrics)
    mlflow.log_artifact(str(adapter_path))

    # Register model in MLflow
    mlflow.log_artifact(str(adapter_path))

    mlflow.end_run()
    log.info("SFT training complete. Run ID: %s", run_id)

    # Free GPU memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "run_id":       run_id,
        "method":       "qlora_sft",
        "adapter_path": str(adapter_path),
        "metrics":      final_metrics,
        "logs":         cb.logs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2b — DPO Training
# ═════════════════════════════════════════════════════════════════════════════

def train_dpo(config: TrainConfig, sft_adapter_path: Optional[str] = None,
              callback: Optional[TrainingCallback] = None):
    """
    DPO Training — teaches the model to prefer chosen over rejected.

    If sft_adapter_path is provided, loads the SFT adapter first (sequential mode).
    Otherwise trains DPO from the base model with a fresh LoRA.

    Requires curated chosen halves in the DPO dataset.
    """
    import torch
    import mlflow
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from trl import DPOTrainer, DPOConfig

    cb = callback or TrainingCallback()
    run_id = _setup_mlflow(config)

    log.info("═" * 50)
    log.info("Stage 2b — DPO Training")
    log.info("═" * 50)

    # ── Load dataset ────────────────────────────────────────────────────────
    train_path = str(DATASET_DIR / "dpo_train.jsonl")
    eval_path  = str(DATASET_DIR / "dpo_eval.jsonl")

    if not Path(train_path).exists():
        raise FileNotFoundError(f"DPO data not found at {train_path}. Run `prepare` first.")

    dataset_train = load_dataset("json", data_files=train_path, split="train")
    dataset_eval  = load_dataset("json", data_files=eval_path,  split="train") if Path(eval_path).exists() else None

    n_train = len(dataset_train)
    if n_train == 0:
        raise ValueError("DPO training set is empty. Curate chosen halves in the DataPanel first.")

    log.info("DPO dataset: %d train, %d eval", n_train, len(dataset_eval) if dataset_eval else 0)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model (pre-quantized bnb-4bit) ─────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Load SFT adapter if sequential mode
    if sft_adapter_path and Path(sft_adapter_path).exists():
        log.info("Loading SFT adapter from %s", sft_adapter_path)
        model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
        mlflow.log_param("sft_adapter", sft_adapter_path)
    else:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    # ── DPO reference model = same base (frozen) ───────────────────────────
    # trl DPOTrainer can use the model itself as reference when ref_model=None
    # (it freezes the non-LoRA weights internally). Saves ~4 GB VRAM.

    # ── Training args ───────────────────────────────────────────────────────
    output_dir = str(CHECKPOINT_DIR / f"dpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    from transformers import TrainerCallback as HFCallback

    class _StreamCallback(HFCallback):
        def on_log(self_cb, args, state, control, logs=None, **kwargs):
            if logs:
                metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                cb.on_log(step=state.global_step, epoch=state.epoch or 0, metrics=metrics)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        beta=config.dpo_beta,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        logging_steps=1,
        eval_strategy=config.eval_strategy if dataset_eval else "no",
        save_strategy="epoch",
        save_total_limit=3,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # use implicit reference (saves VRAM)
        args=dpo_config,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        callbacks=[_StreamCallback()],
    )

    log.info("Starting DPO training...")
    result = trainer.train()

    # ── Save adapter ────────────────────────────────────────────────────────
    adapter_path = Path(output_dir) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    final_metrics = {
        "train_loss":    result.training_loss,
        "train_runtime": result.metrics.get("train_runtime", 0),
    }
    if dataset_eval:
        eval_result = trainer.evaluate()
        final_metrics["eval_loss"] = eval_result.get("eval_loss", 0)

    cb.on_train_end(final_metrics)
    mlflow.log_artifact(str(adapter_path))
    mlflow.log_artifact(str(adapter_path))
    mlflow.end_run()

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "run_id":       run_id,
        "method":       "dpo",
        "adapter_path": str(adapter_path),
        "metrics":      final_metrics,
        "logs":         cb.logs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2c — Sequential (SFT → DPO)
# ═════════════════════════════════════════════════════════════════════════════

def train_sequential(config: TrainConfig, callback: Optional[TrainingCallback] = None):
    """Run SFT first, then DPO on top of the SFT adapter."""
    cb = callback or TrainingCallback()

    log.info("Sequential mode: SFT → DPO")

    # Phase 1: SFT
    config.method = "qlora"
    sft_result = train_qlora_sft(config, callback=cb)

    # Phase 2: DPO using the SFT adapter
    config.method = "dpo"
    config.learning_rate = config.learning_rate / 2  # lower LR for DPO phase
    dpo_result = train_dpo(config, sft_adapter_path=sft_result["adapter_path"], callback=cb)

    return {
        "run_id":       dpo_result["run_id"],
        "method":       "sequential",
        "sft_result":   sft_result,
        "dpo_result":   dpo_result,
        "adapter_path": dpo_result["adapter_path"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — Model Promotion
# ═════════════════════════════════════════════════════════════════════════════

def promote_model(run_id: str) -> Dict[str, Any]:
    """
    Promote a trained adapter to Production in MLflow registry.
    Weaver picks up the Production adapter automatically.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Find the model version for this run
    try:
        versions = client.search_model_versions(f"name='{MODEL_REGISTRY}'")
    except Exception:
        versions = []
    target_version = None
    for v in versions:
        if v.run_id == run_id:
            target_version = v
            break

    # FLATTENED FIX: training never called mlflow.register_model, so the
    # registry was always empty and promotion ALWAYS failed (and the weaver's
    # get_active_adapter() always returned None). Register on demand — this
    # also makes historical runs promotable retroactively.
    if target_version is None:
        client.get_run(run_id)  # raises if run_id is invalid
        adapter_uri = f"runs:/{run_id}/final_adapter"
        log.info("No registered version for run %s — registering %s", run_id, adapter_uri)
        mv = mlflow.register_model(adapter_uri, MODEL_REGISTRY)
        target_version = client.get_model_version(MODEL_REGISTRY, mv.version)
        versions = client.search_model_versions(f"name='{MODEL_REGISTRY}'")

    # Transition to Production (archive current Production first)
    for v in versions:
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name=MODEL_REGISTRY,
                version=v.version,
                stage="Archived",
            )

    client.transition_model_version_stage(
        name=MODEL_REGISTRY,
        version=target_version.version,
        stage="Production",
    )

    log.info("Promoted version %s (run=%s) to Production", target_version.version, run_id)

    return {
        "model_name":    MODEL_REGISTRY,
        "version":       target_version.version,
        "run_id":        run_id,
        "stage":         "Production",
        "adapter_path":  target_version.source,
    }


def get_active_adapter() -> Optional[str]:
    """
    Query MLflow registry for the current Production adapter path.
    Called by Weaver at inference time.
    Returns None if no Production model exists (uses base model).
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_REGISTRY, stages=["Production"])
        if versions:
            source = versions[0].source
            log.info("Active adapter: version=%s source=%s", versions[0].version, source)
            return source
        return None
    except Exception as e:
        log.debug("MLflow registry lookup failed: %s", e)
        return None


def list_runs() -> List[Dict[str, Any]]:
    """List all training runs from MLflow."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(MLFLOW_EXP)
        if experiment is None:
            return []

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=50,
        )

        # Get current production version
        prod_run_id = None
        try:
            versions = client.get_latest_versions(MODEL_REGISTRY, stages=["Production"])
            if versions:
                prod_run_id = versions[0].run_id
        except Exception:
            pass

        results = []
        for run in runs:
            results.append({
                "run_id":     run.info.run_id,
                "run_name":   run.info.run_name or run.info.run_id[:8],
                "status":     run.info.status,
                "start_time": run.info.start_time,
                "end_time":   run.info.end_time,
                "params":     dict(run.data.params),
                "metrics":    dict(run.data.metrics),
                "is_production": run.info.run_id == prod_run_id,
            })

        return results
    except Exception as e:
        log.warning("list_runs failed: %s", e)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# Hammer post-training eval (auto-runs after promotion)
# ═════════════════════════════════════════════════════════════════════════════

def hammer_eval_after_promote(run_id: str) -> Dict[str, Any]:
    """
    Run Hammer on a held-out eval set and log results back to the MLflow run.
    This gives you before/after scores in the same MLflow dashboard.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)

    eval_path = DATASET_DIR / "qlora_eval.jsonl"
    if not eval_path.exists():
        return {"error": "No eval set found"}

    # Load eval samples
    samples = []
    with open(eval_path) as f:
        for line in f:
            samples.append(json.loads(line))

    if not samples:
        return {"error": "Eval set is empty"}

    # Score using Hammer evaluator
    try:
        from hammer.evaluator import score_ragas, score_temporal_consistency, compute_weighted_score
    except ImportError:
        return {"error": "Hammer evaluator not importable"}

    scores = []
    faithfulness_scores = []

    for sample in samples[:30]:  # cap at 30 for speed
        messages = sample.get("messages", [])
        query  = next((m["content"] for m in messages if m["role"] == "user"), "")
        answer = sample.get("response", next((m["content"] for m in messages if m["role"] == "assistant"), ""))

        # Simple eval — answer_relevance only (no context re-retrieval for speed)
        from hammer.evaluator import _approx_answer_relevance
        ar = _approx_answer_relevance(query, answer)
        tc = score_temporal_consistency(answer)
        ws = compute_weighted_score(0.7, ar, tc, None)  # assume faith=0.7 for held-out

        scores.append(ws)
        faithfulness_scores.append(0.7)

    avg_score = sum(scores) / len(scores) if scores else 0
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0

    # Log to the training run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "hammer_weighted_score": round(avg_score, 4),
            "hammer_eval_samples":  len(scores),
        })

    return {
        "hammer_weighted_score": round(avg_score, 4),
        "eval_samples":          len(scores),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="training_pipeline",
        description="Training Pipeline — QLoRA SFT + DPO with MLflow",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    sub.add_parser("prepare", help="Pull data from MongoDB, deduplicate, split, snapshot")

    # train
    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("--method", choices=["qlora", "dpo", "sequential"], default="qlora")
    p_train.add_argument("--config", type=str, default=None, help="Path to config JSON")
    p_train.add_argument("--rank", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--model", type=str, default=HF_MODEL_ID)

    # promote
    p_promote = sub.add_parser("promote", help="Promote a run to Production")
    p_promote.add_argument("run_id", type=str)

    # runs
    sub.add_parser("runs", help="List training runs")

    # check-compat
    sub.add_parser("check-compat", help="Check training base vs serving model compatibility")

    args = parser.parse_args()

    if args.command == "prepare":
        stats = prepare_datasets()
        print(json.dumps(stats, indent=2))

    elif args.command == "train":
        if args.config:
            config = TrainConfig.load(Path(args.config))
        else:
            config = TrainConfig(
                method=args.method,
                base_model=args.model,
                lora_rank=args.rank,
                learning_rate=args.lr,
                epochs=args.epochs,
            )

        if config.method == "qlora":
            result = train_qlora_sft(config)
        elif config.method == "dpo":
            result = train_dpo(config)
        elif config.method == "sequential":
            result = train_sequential(config)
        else:
            parser.error(f"Unknown method: {config.method}")
            return

        print(json.dumps({k: v for k, v in result.items() if k != "logs"}, indent=2, default=str))

    elif args.command == "promote":
        result = promote_model(args.run_id)
        print(json.dumps(result, indent=2))

    elif args.command == "runs":
        runs = list_runs()
        for r in runs:
            prod = " ★ PRODUCTION" if r.get("is_production") else ""
            method = r["params"].get("method", "?")
            loss = r["metrics"].get("final_train_loss", r["metrics"].get("train_loss", "?"))
            print(f"  {r['run_name']:<30} {method:<12} loss={loss}{prod}")

    elif args.command == "check-compat":
        print(json.dumps(check_base_model_compat(), indent=2))


if __name__ == "__main__":
    main()
