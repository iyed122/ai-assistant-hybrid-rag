#!/usr/bin/env python3
"""
Base Sentry
Shared utilities: rate limiting, pagination, structured responses, logging.
"""

import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Generator
from dataclasses import dataclass, field

# ──────────────────────────────────────────────
# Structured response envelope
# ──────────────────────────────────────────────

@dataclass
class SentryResult:
    """
    Unified response envelope returned by every sentry operation.
    The Weaver (and any other consumer) always receives this shape.
    """
    success: bool
    source: str                        # 'gitlab' | 'jira' | 'confluence'
    operation: str                     # e.g. 'get_issues', 'get_mr_diff'
    data: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "operation": self.operation,
            "count": len(self.data),
            "data": self.data,
            "meta": self.meta,
            "error": self.error,
            "fetched_at": self.fetched_at,
        }

    def __repr__(self):
        status = "✓" if self.success else "✗"
        return (
            f"SentryResult({status} source={self.source!r} "
            f"op={self.operation!r} count={len(self.data)} "
            f"error={self.error!r})"
        )


# ──────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────

class RateLimiter:
    """Token-bucket style rate limiter (simple sliding window)."""

    def __init__(self, calls_per_second: float = 5.0):
        self.min_interval = 1.0 / calls_per_second
        self._last_call: float = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


# ──────────────────────────────────────────────
# Base sentry
# ──────────────────────────────────────────────

class BaseSentry:
    """
    Abstract base for all API sentries.

    Subclasses must implement:
        - _check_credentials()    → raises ValueError if creds missing
        - _build_session()        → sets up self.session (requests.Session)

    Provides:
        - paginate_offset()    cursor/offset-based pagination helper
        - paginate_cursor()    link-header / next-URL pagination helper
        - _ok()                build a success SentryResult
        - _err()               build an error SentryResult
    """

    SOURCE = "base"          # override in subclass

    def __init__(self, rate_limit: float = 5.0):
        self.logger = logging.getLogger(f"sentry.{self.SOURCE}")
        self.rate_limiter = RateLimiter(calls_per_second=rate_limit)
        self._check_credentials()
        self._build_session()

    # ── Must be implemented ──────────────────

    def _check_credentials(self):
        raise NotImplementedError

    def _build_session(self):
        raise NotImplementedError

    # ── Pagination helpers ───────────────────

    def paginate_offset(
        self,
        fetch_fn,           # callable(start_at, max_results) → (items, total)
        start_at: int = 0,
        page_size: int = 50,
        max_items: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Offset/startAt-style pagination (Jira pattern).
        `fetch_fn` must return a tuple: (list_of_items, total_count).
        """
        all_items: List[Dict[str, Any]] = []
        current = start_at

        while True:
            self.rate_limiter.wait()
            items, total = fetch_fn(current, page_size)

            if not items:
                break

            all_items.extend(items)
            current += len(items)

            self.logger.debug(
                "paginate_offset: fetched %d/%d (total=%d)",
                len(all_items), max_items, total,
            )

            if current >= total or len(all_items) >= max_items:
                break

        return all_items[:max_items]

    def paginate_cursor(
        self,
        fetch_fn,           # callable(url_or_params) → (items, next_url_or_None)
        initial_url_or_params,
        max_items: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Cursor/next-link pagination (Confluence, GitLab REST pattern).
        `fetch_fn` must return (list_of_items, next_cursor_or_None).
        """
        all_items: List[Dict[str, Any]] = []
        cursor = initial_url_or_params

        while cursor is not None:
            self.rate_limiter.wait()
            items, cursor = fetch_fn(cursor)

            if not items:
                break

            all_items.extend(items)

            self.logger.debug(
                "paginate_cursor: fetched %d so far (limit=%d)",
                len(all_items), max_items,
            )

            if len(all_items) >= max_items:
                break

        return all_items[:max_items]

    # ── Result builders ──────────────────────

    def _ok(
        self,
        operation: str,
        data: List[Dict[str, Any]],
        **meta_kwargs,
    ) -> SentryResult:
        return SentryResult(
            success=True,
            source=self.SOURCE,
            operation=operation,
            data=data,
            meta=meta_kwargs,
        )

    def _err(
        self,
        operation: str,
        error: str,
        **meta_kwargs,
    ) -> SentryResult:
        self.logger.error("[%s] %s → %s", self.SOURCE, operation, error)
        return SentryResult(
            success=False,
            source=self.SOURCE,
            operation=operation,
            error=error,
            meta=meta_kwargs,
        )
