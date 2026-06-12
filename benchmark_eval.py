#!/usr/bin/env python3
"""
Benchmarking Questions + Ground Truth + F1 Evaluation
======================================================
Every question is grounded in the data from ingest_pipeline.py.
No question asks about something that wasn't ingested.

Usage:
    # Interactive labeling mode — label responses manually
    python benchmark_eval.py label

    # Evaluate a set of model responses against ground truth
    python benchmark_eval.py evaluate --responses responses.jsonl

    # Print all benchmark questions (for running against the system)
    python benchmark_eval.py questions

    # Full F1 report from saved labels
    python benchmark_eval.py report

Output files:
    ground_truth_labels.jsonl   — your manual labels (correct/incorrect per failure mode)
    f1_report.json              — precision, recall, F1 per failure mode and overall
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK QUESTIONS
# Each question has:
#   id            — unique identifier
#   question      — the query to send to your system
#   intent        — rag | sentries | both
#   source        — which documents/issues must appear in a correct answer
#   required_facts— list of facts that MUST appear in a correct answer
#   forbidden     — phrases that indicate a failure (refusal, hallucination etc.)
#   failure_modes — which failure modes this question tests
#
# Failure modes tracked:
#   refusal       — model says "I don't have access" when it does
#   fabrication   — model invents IDs, URLs, or facts not in context
#   wrong_count   — model states wrong number of items
#   missing_key_fact — correct topic but omits a required specific fact
#   correct       — answer is fully correct
# ══════════════════════════════════════════════════════════════════════════════

BENCHMARK_QUESTIONS = [

    # ── RAG QUESTIONS (answered from Confluence pages) ─────────────────────

    {
        "id": "RAG-01",
        "question": "How does the OAuth2 token refresh work in this system?",
        "intent": "rag",
        "source_documents": ["AUTH-oauth", "ECOM-arch"],
        "required_facts": [
            "AuthMiddleware checks JWT exp claim before each request",
            "60 seconds",
            "POST /auth/refresh",
            "redirect to login",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot access",
            "I don't have real-time",
            "as an AI",
        ],
        "failure_modes_tested": ["refusal", "missing_key_fact"],
        "notes": "AUTH-oauth page has the exact 60-second buffer and redirect-to-login edge case.",
    },
    {
        "id": "RAG-02",
        "question": "What are the PostgreSQL connection pool settings and how do I investigate pool exhaustion?",
        "intent": "rag",
        "source_documents": ["ECOM-db-runbook"],
        "required_facts": [
            "max_size': 50",
            "80%",
            "pg_stat_activity",
            "Grafana",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "refusal"],
        "notes": "DB runbook has exact pool config (50, was 20) and the step-by-step investigation queries.",
    },
    {
        "id": "RAG-03",
        "question": "How is Redis caching implemented for the product catalog? What are the TTLs?",
        "intent": "rag",
        "source_documents": ["ECOM-caching"],
        "required_facts": [
            "5 min",
            "10 min",
            "cache-aside",
            "80%",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact"],
        "notes": "Caching page has exact TTLs: list=5min, detail=10min, and the cache-aside code pattern.",
    },
    {
        "id": "RAG-04",
        "question": "Why do Stripe webhooks fail intermittently and how is it fixed?",
        "intent": "rag",
        "source_documents": ["ECOM-stripe"],
        "required_facts": [
            "logging middleware",
            "request.body()",
            "raw_body",
            "request.state",
            "construct_event",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "fabrication"],
        "notes": "Stripe guide explains the exact root cause: middleware consuming the stream before webhook handler.",
    },
    {
        "id": "RAG-05",
        "question": "How do I set up my local development environment for the ecommerce platform?",
        "intent": "rag",
        "source_documents": ["ECOM-onboarding"],
        "required_facts": [
            "docker compose",
            ".env",
            "uvicorn",
            "localhost:8000",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["refusal", "missing_key_fact"],
        "notes": "Onboarding guide has the exact 4-step setup sequence.",
    },
    {
        "id": "RAG-06",
        "question": "What delivery channels does the notification service support?",
        "intent": "rag",
        "source_documents": ["NOTIF-arch"],
        "required_facts": [
            "WebSocket",
            "Email",
            "Push",
            "FCM",
            "Redis pub/sub",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact"],
        "notes": "NOTIF-arch page lists all 3 channels: WebSocket, Email, Push (FCM).",
    },
    {
        "id": "RAG-07",
        "question": "What is the TOTP MFA enrolment flow?",
        "intent": "rag",
        "source_documents": ["AUTH-mfa"],
        "required_facts": [
            "POST /auth/mfa/enrol",
            "QR code",
            "6-digit",
            "pyotp",
            "AES-256",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact"],
        "notes": "MFA guide has the exact 4-step enrolment flow and the pyotp + AES-256 encryption detail.",
    },
    {
        "id": "RAG-08",
        "question": "What is the dead-letter queue retry policy for failed notification deliveries?",
        "intent": "rag",
        "source_documents": ["NOTIF-dlq"],
        "required_facts": [
            "30 seconds",
            "2 minutes",
            "10 minutes",
            "PagerDuty",
            "notif:dlq",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "fabrication"],
        "notes": "DLQ page has exact retry intervals and the Redis sorted set structure.",
    },
    {
        "id": "RAG-09",
        "question": "What are the API rate limits per user tier?",
        "intent": "rag",
        "source_documents": ["ECOM-rate-limit"],
        "required_facts": [
            "Anonymous",
            "30 req/min",
            "Free user",
            "100 req/min",
            "Premium",
            "500 req/min",
            "429",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "wrong_count"],
        "notes": "Rate limiting policy page has all 4 tiers with exact limits.",
    },
    {
        "id": "RAG-10",
        "question": "Why was WebSocket + Redis pub/sub chosen for real-time order updates? What alternatives were rejected?",
        "intent": "rag",
        "source_documents": ["ECOM-realtime-adr"],
        "required_facts": [
            "Client polling",
            "Rejected",
            "Server-Sent Events",
            "gRPC",
            "bidirectional",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact"],
        "notes": "Real-Time ADR has all 4 alternatives with rejection reasons.",
    },
    {
        "id": "RAG-11",
        "question": "What authentication events are logged and for how long are they retained?",
        "intent": "rag",
        "source_documents": ["AUTH-audit"],
        "required_facts": [
            "90 days",
            "login_success",
            "login_failure",
            "mfa_enrolled",
            "PagerDuty",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact"],
        "notes": "Auth audit policy has all events, 90-day retention, and alerting thresholds.",
    },

    # ── SENTRIES QUESTIONS (answered from Jira / GitLab live data) ─────────

    {
        "id": "SENT-01",
        "question": "Show me all open issues in ecommerce-backend",
        "intent": "sentries",
        "source_documents": ["eb-1","eb-2","eb-3","eb-4","eb-5","eb-6","eb-7","eb-8"],
        "required_facts": [
            "ecommerce-backend",
            "#1",
            "#7",
            "OAuth2",
            "SQL injection",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot access",
            "I don't have real-time",
            "as an AI",
            "no issues found",
        ],
        "failure_modes_tested": ["refusal", "wrong_count", "fabrication"],
        "notes": "8 open GitLab issues exist in ecommerce-backend. System must not say 'no access'.",
    },
    {
        "id": "SENT-02",
        "question": "What is the status of ECOM-12?",
        "intent": "sentries",
        "source_documents": ["ECOM-12"],
        "required_facts": [
            "ECOM-12",
            "OAuth2",
            "Open",
            "Critical",
            "token refresh",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
            "not found",
        ],
        "failure_modes_tested": ["refusal", "fabrication"],
        "notes": "ECOM-12 is in Jira: Open, Critical priority, OAuth2 token refresh issue.",
    },
    {
        "id": "SENT-03",
        "question": "List all merge requests in auth-service",
        "intent": "sentries",
        "source_documents": ["as-mr-1"],
        "required_facts": [
            "auth-service",
            "!1",
            "/tokens/introspect",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot access",
            "no merge requests",
        ],
        "failure_modes_tested": ["refusal", "wrong_count"],
        "notes": "1 MR exists in auth-service: !1 feat /tokens/introspect. Must not say 'no access'.",
    },
    {
        "id": "SENT-04",
        "question": "Show me all critical Jira issues in the ECOM project",
        "intent": "sentries",
        "source_documents": ["ECOM-9","ECOM-12","ECOM-23"],
        "required_facts": [
            "ECOM-9",
            "ECOM-12",
            "ECOM-23",
            "Critical",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["refusal", "wrong_count", "fabrication"],
        "notes": "3 Critical issues in ECOM: ECOM-9 (SQL injection), ECOM-12 (OAuth), ECOM-23 (DB pool).",
    },
    {
        "id": "SENT-05",
        "question": "What open issues exist in the auth-service GitLab project?",
        "intent": "sentries",
        "source_documents": ["as-1","as-2","as-3"],
        "required_facts": [
            "auth-service",
            "#1",
            "#2",
            "#3",
            "token",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot access",
        ],
        "failure_modes_tested": ["refusal", "wrong_count"],
        "notes": "3 open GitLab issues exist in auth-service. Must list all 3.",
    },
    {
        "id": "SENT-06",
        "question": "Show me the memory leak issue in ecommerce-backend",
        "intent": "sentries",
        "source_documents": ["ECOM-34","eb-6"],
        "required_facts": [
            "ECOM-34",
            "memory leak",
            "Pillow",
            "BytesIO",
            "In Progress",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
            "not found",
        ],
        "failure_modes_tested": ["refusal", "fabrication"],
        "notes": "ECOM-34 is In Progress, High priority. Root cause: Pillow BytesIO buffer retention.",
    },
    {
        "id": "SENT-07",
        "question": "List merge requests in notification-service",
        "intent": "sentries",
        "source_documents": ["ns-mr-1","ns-mr-2"],
        "required_facts": [
            "notification-service",
            "!1",
            "!2",
            "SendGrid",
            "Redis",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot access",
            "no merge requests",
        ],
        "failure_modes_tested": ["refusal", "wrong_count"],
        "notes": "2 MRs exist: !1 async SendGrid, !2 Redis WS registry.",
    },
    {
        "id": "SENT-08",
        "question": "Show me all security-related open issues across all projects",
        "intent": "sentries",
        "source_documents": ["ECOM-9","AUTH-7","AUTH-12","eb-7","as-1","as-2"],
        "required_facts": [
            "SQL injection",
            "security",
            "ECOM-9",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["refusal", "wrong_count", "fabrication"],
        "notes": "Multiple security-labelled issues across ECOM and AUTH projects.",
    },

    # ── BOTH QUESTIONS (need Confluence docs + Jira/GitLab live data) ──────

    {
        "id": "BOTH-01",
        "question": "What is the OAuth2 token refresh issue and what is the current fix status including the MR?",
        "intent": "both",
        "source_documents": ["ECOM-12","AUTH-7","AUTH-oauth","eb-mr-1"],
        "required_facts": [
            "ECOM-12",
            "AUTH-7",
            "exp claim",
            "60 seconds",
            "MR !1",
            "fix/oauth-token-refresh",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["refusal", "missing_key_fact", "fabrication"],
        "notes": "Multi-source: Jira issue ECOM-12, GitLab MR eb-mr-1 (fix/oauth-token-refresh), and AUTH-oauth Confluence page.",
    },
    {
        "id": "BOTH-02",
        "question": "Tell me about the Stripe webhook problem — open Jira ticket, GitLab issue, and the architecture docs",
        "intent": "both",
        "source_documents": ["ECOM-31","eb-4","eb-mr-5","ECOM-stripe"],
        "required_facts": [
            "ECOM-31",
            "logging middleware",
            "request.body()",
            "5%",
            "construct_event",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "refusal"],
        "notes": "Cross-source: ECOM-31 Jira, eb-4 GitLab issue, eb-mr-5 fix MR, ECOM-stripe Confluence.",
    },
    {
        "id": "BOTH-03",
        "question": "Memory leak in the image resize worker — what's the open issue status and what does the fix involve?",
        "intent": "both",
        "source_documents": ["ECOM-34","eb-6"],
        "required_facts": [
            "ECOM-34",
            "In Progress",
            "BytesIO",
            "context manager",
            "12 hours",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "refusal"],
        "notes": "ECOM-34 is In Progress. Fix involves explicit context managers for BytesIO and Image.",
    },
    {
        "id": "BOTH-04",
        "question": "What is the real-time notification architecture and what are the current known issues?",
        "intent": "both",
        "source_documents": ["NOTIF-arch","ECOM-realtime-adr","NOTIF-9","NOTIF-4"],
        "required_facts": [
            "Redis pub/sub",
            "WebSocket",
            "NOTIF-9",
            "NOTIF-4",
            "Kubernetes pods",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "fabrication"],
        "notes": "Confluence ADR + NOTIF-arch + two open Jira issues (NOTIF-4 SendGrid, NOTIF-9 multi-pod).",
    },
    {
        "id": "BOTH-05",
        "question": "What is the database connection pool situation — the incident ticket, fix status, and runbook?",
        "intent": "both",
        "source_documents": ["ECOM-23","eb-2","ECOM-db-runbook"],
        "required_facts": [
            "ECOM-23",
            "In Progress",
            "50",
            "80%",
            "Grafana",
            "pg_stat_activity",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "wrong_count"],
        "notes": "ECOM-23 (In Progress), eb-2 GitLab issue, and ECOM-db-runbook with pool=50 and investigation steps.",
    },
    {
        "id": "BOTH-06",
        "question": "Show me the onboarding guide and also what issues a new engineer should know about right now",
        "intent": "both",
        "source_documents": ["ECOM-onboarding","ECOM-12","ECOM-23","ECOM-31","NOTIF-9"],
        "required_facts": [
            "docker compose",
            "ECOM-12",
            "ECOM-23",
            "ECOM-31",
            "NOTIF-9",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "refusal"],
        "notes": "Onboarding Confluence page + current active Jira issues listed in it.",
    },
    {
        "id": "BOTH-07",
        "question": "Explain the rate limiting architecture and show me the current open rate limiting tickets",
        "intent": "both",
        "source_documents": ["ECOM-rate-limit","ECOM-18","AUTH-12","eb-3"],
        "required_facts": [
            "429",
            "ECOM-18",
            "AUTH-12",
            "X-RateLimit",
            "Retry-After",
        ],
        "forbidden_phrases": [
            "I don't have access",
            "I cannot",
        ],
        "failure_modes_tested": ["missing_key_fact", "fabrication"],
        "notes": "Rate limit Confluence page + ECOM-18 (500 vs 429 bug) + AUTH-12 (login no rate limit).",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# FAILURE MODE LABELS
# ══════════════════════════════════════════════════════════════════════════════

FAILURE_MODES = [
    "refusal",        # model says it can't access data when it can
    "fabrication",    # model invents IDs, URLs, facts not in context
    "wrong_count",    # model states wrong number of results
    "missing_key_fact", # correct topic but omits a required specific fact
]


# ══════════════════════════════════════════════════════════════════════════════
# AUTOMATIC SCORING (pre-labeling pass)
# Checks required_facts and forbidden_phrases against a response
# Returns a preliminary label — human should validate
# ══════════════════════════════════════════════════════════════════════════════

def auto_score(question: Dict, response: str) -> Dict:
    """
    Automatic preliminary scoring based on string matching.
    Returns a dict of detected issues — NOT a final label.
    Human validation is required before these go into F1 computation.
    """
    response_lower = response.lower()
    issues = {}

    # Check for refusal phrases
    refusal_detected = any(p.lower() in response_lower for p in question["forbidden_phrases"])
    issues["refusal_detected"] = refusal_detected

    # Check for required facts
    missing_facts = [
        fact for fact in question["required_facts"]
        if fact.lower() not in response_lower
    ]
    issues["missing_facts"] = missing_facts
    issues["missing_key_fact_detected"] = len(missing_facts) > 0

    # Note: fabrication and wrong_count require human judgment
    issues["fabrication_detected"] = None   # human must label
    issues["wrong_count_detected"] = None   # human must label

    # Preliminary correctness
    issues["preliminary_correct"] = (not refusal_detected) and (len(missing_facts) == 0)

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# F1 COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Returns (precision, recall, f1)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def compute_f1_report(labels: List[Dict]) -> Dict:
    """
    Compute per-failure-mode and overall F1 from a list of label dicts.

    Each label dict must have:
        question_id: str
        failure_mode: str  (one of FAILURE_MODES or "correct")
        ground_truth: bool  (True = this failure DID occur)
        predicted: bool     (True = system flagged this as occurring)
    """
    per_mode = defaultdict(lambda: {"tp":0,"fp":0,"fn":0,"tn":0})

    for label in labels:
        mode = label["failure_mode"]
        gt   = label["ground_truth"]
        pred = label["predicted"]
        if gt and pred:
            per_mode[mode]["tp"] += 1
        elif not gt and pred:
            per_mode[mode]["fp"] += 1
        elif gt and not pred:
            per_mode[mode]["fn"] += 1
        else:
            per_mode[mode]["tn"] += 1

    report = {}
    all_tp = all_fp = all_fn = 0

    for mode, counts in per_mode.items():
        p, r, f1 = compute_f1(counts["tp"], counts["fp"], counts["fn"])
        report[mode] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f1, 4),
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "tn": counts["tn"],
        }
        all_tp += counts["tp"]
        all_fp += counts["fp"]
        all_fn += counts["fn"]

    overall_p, overall_r, overall_f1 = compute_f1(all_tp, all_fp, all_fn)
    report["OVERALL"] = {
        "precision": round(overall_p, 4),
        "recall":    round(overall_r, 4),
        "f1":        round(overall_f1, 4),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
    }
    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_questions(_args):
    """Print all benchmark questions as a JSONL file for running against the system."""
    output_file = "benchmark_questions.jsonl"
    with open(output_file, "w") as f:
        for q in BENCHMARK_QUESTIONS:
            f.write(json.dumps({
                "id": q["id"],
                "question": q["question"],
                "intent": q["intent"],
                "required_facts": q["required_facts"],
                "source_documents": q["source_documents"],
                "failure_modes_tested": q["failure_modes_tested"],
                "notes": q["notes"],
            }) + "\n")

    print(f"\nWrote {len(BENCHMARK_QUESTIONS)} questions to {output_file}")
    print("\nSummary:")
    by_intent = defaultdict(list)
    for q in BENCHMARK_QUESTIONS:
        by_intent[q["intent"]].append(q["id"])
    for intent, ids in sorted(by_intent.items()):
        print(f"  {intent:10s}: {len(ids)} questions ({', '.join(ids)})")

    print("\nTo run against your system:")
    print("  while IFS= read -r line; do")
    print('    question=$(echo "$line" | jq -r .question)')
    print('    id=$(echo "$line" | jq -r .id)')
    print('    echo "{\"id\":\"$id\",\"response\":\"$(curl -s -X POST http://localhost:8000/chat -d \"{\\\"message\\\":\\\"$question\\\"}\" | jq -r .answer)\"}" >> responses.jsonl')
    print("  done < benchmark_questions.jsonl")


def cmd_evaluate(args):
    """Auto-score responses from a JSONL file and save preliminary labels."""
    if not os.path.exists(args.responses):
        print(f"File not found: {args.responses}")
        return

    # Build lookup
    q_by_id = {q["id"]: q for q in BENCHMARK_QUESTIONS}

    preliminary_labels = []
    with open(args.responses) as f:
        for line in f:
            rec = json.loads(line.strip())
            qid = rec.get("id")
            response = rec.get("response", "")
            q = q_by_id.get(qid)
            if not q:
                log.warning("Unknown question id: %s", qid)
                continue

            issues = auto_score(q, response)
            preliminary_labels.append({
                "question_id": qid,
                "intent": q["intent"],
                "response_excerpt": response[:200],
                "auto": issues,
                "human_labels": {
                    "refusal":          issues["refusal_detected"],
                    "missing_key_fact": issues["missing_key_fact_detected"],
                    "fabrication":      None,   # ← fill manually
                    "wrong_count":      None,   # ← fill manually
                    "correct":          issues["preliminary_correct"],
                },
                "labelled_at": None,
                "notes": "",
            })

    out_file = "preliminary_labels.jsonl"
    with open(out_file, "w") as f:
        for rec in preliminary_labels:
            f.write(json.dumps(rec) + "\n")

    print(f"\nWrote {len(preliminary_labels)} preliminary labels to {out_file}")
    print("\nAuto-detected issues:")
    refusals = sum(1 for r in preliminary_labels if r["auto"]["refusal_detected"])
    missing  = sum(1 for r in preliminary_labels if r["auto"]["missing_key_fact_detected"])
    print(f"  Refusals detected:      {refusals}/{len(preliminary_labels)}")
    print(f"  Missing key facts:      {missing}/{len(preliminary_labels)}")
    print(f"  Fabrication/wrong_count: requires human review")
    print(f"\nOpen {out_file} and set fabrication/wrong_count to true/false for each response.")
    print(f"Then run:  python {__file__} report --labels {out_file}")


def cmd_report(args):
    """Compute F1 report from a labelled JSONL file."""
    label_file = getattr(args, "labels", "preliminary_labels.jsonl")
    if not os.path.exists(label_file):
        print(f"File not found: {label_file}")
        return

    # Expand labels into per-failure-mode rows
    rows = []
    with open(label_file) as f:
        for line in f:
            rec = json.loads(line.strip())
            hl = rec.get("human_labels", {})
            qid = rec["question_id"]
            q = next((q for q in BENCHMARK_QUESTIONS if q["id"] == qid), None)
            if not q:
                continue
            for mode in FAILURE_MODES:
                gt = bool(hl.get(mode))
                # predicted = whether auto-scoring flagged it
                if mode == "refusal":
                    pred = bool(rec["auto"]["refusal_detected"])
                elif mode == "missing_key_fact":
                    pred = bool(rec["auto"]["missing_key_fact_detected"])
                else:
                    pred = bool(hl.get(mode))  # human provides this
                rows.append({
                    "question_id": qid,
                    "failure_mode": mode,
                    "ground_truth": gt,
                    "predicted": pred,
                })

    if not rows:
        print("No labelled data found.")
        return

    report = compute_f1_report(rows)

    print("\n" + "═" * 60)
    print("F1 EVALUATION REPORT")
    print("═" * 60)
    print(f"{'Mode':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("─" * 60)
    for mode in FAILURE_MODES + ["OVERALL"]:
        if mode not in report:
            continue
        r = report[mode]
        print(f"{mode:<20} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5}")
    print("═" * 60)

    out_file = "f1_report.json"
    with open(out_file, "w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(),
            "n_responses": len(set(r["question_id"] for r in rows)),
            "per_mode": report,
        }, f, indent=2)
    print(f"\nFull report saved to {out_file}")


def cmd_label(_args):
    """Interactive labeling — walks through each question for human review."""
    print("\n" + "=" * 60)
    print("INTERACTIVE LABELING MODE")
    print("=" * 60)
    print("For each question, you will enter:")
    print("  The system's response (paste and press Enter twice)")
    print("  Then label each failure mode: y=yes, n=no, s=skip")
    print()

    labels = []
    for q in BENCHMARK_QUESTIONS:
        print(f"\n{'─' * 60}")
        print(f"Question: [{q['id']}] ({q['intent'].upper()})")
        print(f"  {q['question']}")
        print(f"Required facts: {', '.join(q['required_facts'][:3])} ...")
        print()
        print("Paste system response (press Enter twice when done):")

        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        response = "\n".join(lines[:-1] if lines and lines[-1] == "" else lines)

        auto = auto_score(q, response)
        print(f"\nAuto-detected: refusal={auto['refusal_detected']}, "
              f"missing_facts={auto['missing_facts'][:2]}")

        hl = {}
        for mode in FAILURE_MODES:
            default = "y" if (
                (mode == "refusal" and auto["refusal_detected"]) or
                (mode == "missing_key_fact" and auto["missing_key_fact_detected"])
            ) else "n"
            val = input(f"  {mode}? [y/n/s, default={default}]: ").strip().lower() or default
            if val == "s":
                hl[mode] = None
            else:
                hl[mode] = (val == "y")

        hl["correct"] = not any(v for v in hl.values() if v is True)
        labels.append({
            "question_id": q["id"],
            "intent": q["intent"],
            "response_excerpt": response[:200],
            "auto": auto,
            "human_labels": hl,
            "labelled_at": datetime.utcnow().isoformat(),
        })

        out_file = "ground_truth_labels.jsonl"
        with open(out_file, "w") as f:
            for rec in labels:
                f.write(json.dumps(rec) + "\n")
        print(f"  Saved ({len(labels)}/{len(BENCHMARK_QUESTIONS)})")

    print(f"\n✓ All {len(labels)} responses labelled. Run report:")
    print(f"  python {__file__} report --labels ground_truth_labels.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark evaluation with F1 metrics")
    sub = parser.add_subparsers(dest="cmd")

    p_questions = sub.add_parser("questions", help="Export all benchmark questions to JSONL")
    p_questions.set_defaults(func=cmd_questions)

    p_evaluate = sub.add_parser("evaluate", help="Auto-score responses from JSONL file")
    p_evaluate.add_argument("--responses", default="responses.jsonl")
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_report = sub.add_parser("report", help="Compute F1 report from labelled JSONL")
    p_report.add_argument("--labels", default="preliminary_labels.jsonl")
    p_report.set_defaults(func=cmd_report)

    p_label = sub.add_parser("label", help="Interactive labeling mode")
    p_label.set_defaults(func=cmd_label)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        # Default: print question summary
        print(f"\n{len(BENCHMARK_QUESTIONS)} benchmark questions loaded:")
        for q in BENCHMARK_QUESTIONS:
            print(f"  [{q['id']}] ({q['intent']:8s}) {q['question'][:70]}")
        print("\nCommands: questions | evaluate | report | label")
