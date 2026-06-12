#!/usr/bin/env python3
"""
Confluence API Sentry
─────────────────────
Deterministic, real-time fetcher for Confluence data.
No LLM involved — pure programmatic retrieval.

Supported operations
────────────────────
  get_spaces           – list all accessible spaces
  get_pages            – pages in a space (with cursor pagination)
  get_page             – single page: title, body, version, ancestors
  search_pages         – CQL full-text search across all spaces
  get_page_comments    – inline + footer comments for a page
  get_recent_pages     – pages modified in the last N days
  get_child_pages      – child pages under a parent page ID
"""

import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests

from sentries.base_sentry import BaseSentry, SentryResult

load_dotenv()

ATLASSIAN_EMAIL     = os.getenv("ATLASSIAN_EMAIL", "")
ATLASSIAN_API_TOKEN = os.getenv("ATLASSIAN_API_TOKEN", "")
ATLASSIAN_DOMAIN    = os.getenv("ATLASSIAN_DOMAIN", "")


class ConfluenceSentry(BaseSentry):
    """Real-time Confluence data fetcher using Confluence REST API v2."""

    SOURCE = "confluence"

    # ── Setup ────────────────────────────────

    def _check_credentials(self):
        missing = [
            name for name, val in [
                ("ATLASSIAN_EMAIL",     ATLASSIAN_EMAIL),
                ("ATLASSIAN_API_TOKEN", ATLASSIAN_API_TOKEN),
                ("ATLASSIAN_DOMAIN",    ATLASSIAN_DOMAIN),
            ] if not val
        ]
        if missing:
            raise ValueError(
                f"Missing Confluence credentials: {', '.join(missing)}. "
                "Add them to your .env file."
            )

    def _build_session(self):
        self.base_v2  = ATLASSIAN_DOMAIN.rstrip("/") + "/wiki/api/v2"
        self.base_v1  = ATLASSIAN_DOMAIN.rstrip("/") + "/wiki/rest/api"
        self.domain   = ATLASSIAN_DOMAIN.rstrip("/")
        self.session  = requests.Session()
        self.session.auth = (ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)
        self.session.headers.update({"Accept": "application/json"})

    # ── Internal helpers ─────────────────────

    def _get(self, path: str, params: Dict = None, base: str = None) -> requests.Response:
        self.rate_limiter.wait()
        root = (base or self.base_v2).rstrip("/")
        url  = f"{root}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp

    def _paginate_v2(
        self,
        path:      str,
        params:    Dict,
        max_items: int = 200,
        base:      str = None,
    ) -> List[Dict]:
        """
        Confluence v2 API uses cursor-based pagination via _links.next.
        """
        all_items: List[Dict] = []
        params = {**params, "limit": 250}
        current_url = None

        # First request
        self.rate_limiter.wait()
        root = (base or self.base_v2).rstrip("/")
        url  = f"{root}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        all_items.extend(data.get("results", []))
        next_link = data.get("_links", {}).get("next")

        while next_link and len(all_items) < max_items:
            self.rate_limiter.wait()
            # next_link is a relative path like /wiki/api/v2/pages?cursor=XXX
            full_next = self.domain + next_link
            resp = self.session.get(full_next, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            all_items.extend(data.get("results", []))
            next_link = data.get("_links", {}).get("next")

        return all_items[:max_items]

    @staticmethod
    def _storage_to_text(storage_html: str) -> str:
        """
        Very lightweight HTML/storage-format → plain text conversion.
        Strips tags, decodes common entities, collapses whitespace.
        """
        if not storage_html:
            return ""
        # Remove XML/HTML tags
        text = re.sub(r"<[^>]+>", " ", storage_html)
        # Decode common entities
        for entity, char in [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
            ("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"),
        ]:
            text = text.replace(entity, char)
        # Collapse whitespace
        return re.sub(r"\s+", " ", text).strip()

    # ── Public operations ────────────────────

    def get_spaces(self, limit: int = 50) -> SentryResult:
        """
        List all accessible Confluence spaces.

        Returns name, key, type, description, and homepage URL.
        """
        op = "get_spaces"
        try:
            raw = self._paginate_v2("spaces", {"limit": 50}, max_items=limit)
            items = [
                {
                    "id":          s["id"],
                    "key":         s.get("key"),
                    "name":        s.get("name"),
                    "type":        s.get("type"),
                    "status":      s.get("status"),
                    "url":         self.domain + s.get("_links", {}).get("webui", ""),
                }
                for s in raw
            ]
            return self._ok(op, items, total=len(items))
        except Exception as exc:
            return self._err(op, str(exc))

    def get_pages(
        self,
        space_key: str,
        title_contains: Optional[str] = None,
        limit: int = 50,
    ) -> SentryResult:
        """
        Fetch pages from a specific Confluence space.

        Args:
            space_key:       Space key (e.g. "PLAT", "ONBOARD")
            title_contains:  Optional filter — only pages whose title contains this
            limit:           Max pages
        """
        op = "get_pages"
        try:
            # First get the space ID from the key
            spaces_resp = self._get("spaces", {"keys": space_key, "limit": 1})
            spaces = spaces_resp.json().get("results", [])
            if not spaces:
                return self._err(op, f"Space with key '{space_key}' not found")

            space_id = spaces[0]["id"]

            params: Dict[str, Any] = {"space-id": space_id, "body-format": "storage"}
            raw = self._paginate_v2("pages", params, max_items=limit * 2)

            items = []
            for p in raw:
                title = p.get("title", "")
                if title_contains and title_contains.lower() not in title.lower():
                    continue

                # Get body if available in response, else minimal entry
                body_storage = p.get("body", {}).get("storage", {}).get("value", "")
                plain_text   = self._storage_to_text(body_storage)

                items.append({
                    "id":          p["id"],
                    "title":       title,
                    "space_key":   space_key,
                    "status":      p.get("status"),
                    "version":     (p.get("version") or {}).get("number"),
                    "created_at":  p.get("createdAt"),
                    "text":        plain_text[:1000],
                    "url":         self.domain + p.get("_links", {}).get("webui", ""),
                })

                if len(items) >= limit:
                    break

            return self._ok(op, items, space_key=space_key, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), space_key=space_key)

    def get_page(
        self,
        page_id:          str,
        include_body:     bool = True,
        include_comments: bool = False,
    ) -> SentryResult:
        """
        Fetch a single Confluence page with full content.

        Args:
            page_id:          Numeric page ID
            include_body:     Include the page body as plain text
            include_comments: Also fetch footer comments
        """
        op = "get_page"
        try:
            # Confluence v2: only body-format is a valid query param here
            params: Dict[str, Any] = {}
            if include_body:
                params["body-format"] = "storage"

            resp = self._get(f"pages/{page_id}", params)
            p = resp.json()

            body_storage = p.get("body", {}).get("storage", {}).get("value", "")
            plain_text   = self._storage_to_text(body_storage) if include_body else ""

            # Fetch version separately (lightweight call)
            version_num = None
            try:
                ver_resp  = self._get(f"pages/{page_id}/versions", {"limit": 1})
                ver_data  = ver_resp.json().get("results", [])
                version_num = ver_data[0].get("number") if ver_data else None
            except Exception:
                pass

            item: Dict[str, Any] = {
                "id":        p["id"],
                "title":     p.get("title"),
                "status":    p.get("status"),
                "version":   version_num,
                "created_at":p.get("createdAt"),
                "space_id":  p.get("spaceId"),
                "text":      plain_text[:3000],
                "url":       self.domain + p.get("_links", {}).get("webui", ""),
            }

            if include_comments:
                comments_result = self.get_page_comments(page_id)
                item["comments"] = comments_result.data

            return self._ok(op, [item], page_id=page_id)

        except Exception as exc:
            return self._err(op, str(exc), page_id=page_id)

    def get_page_comments(self, page_id: str) -> SentryResult:
        """
        Fetch footer comments for a page.

        Args:
            page_id: Numeric Confluence page ID
        """
        op = "get_page_comments"
        try:
            raw = self._paginate_v2(
                f"pages/{page_id}/footer-comments",
                {"body-format": "atlas_doc_format"},
                max_items=200,
            )
            items = []
            for c in raw:
                body = c.get("body", {}).get("atlas_doc_format", {}).get("value", "") or ""
                # Quick text extraction from atlas_doc JSON string
                if isinstance(body, str):
                    plain = re.sub(r'"type":"text","text":"([^"]+)"', r"\1 ", body)
                    plain = re.sub(r'"[^"]+":"[^"]*"', "", plain)
                    plain = re.sub(r"[{}\[\],:]+", " ", plain).strip()[:400]
                else:
                    plain = str(body)[:400]
                items.append({
                    "id":         c["id"],
                    "created_at": c.get("createdAt"),
                    "text":       plain,
                    "url":        self.domain + c.get("_links", {}).get("webui", ""),
                })
            return self._ok(op, items, page_id=page_id, total=len(items))
        except Exception as exc:
            return self._err(op, str(exc), page_id=page_id)

    def search_pages(
        self,
        query:     str,
        space_key: Optional[str] = None,
        limit:     int = 20,
    ) -> SentryResult:
        """
        Full-text CQL search across Confluence pages.

        Searches both title and body text so that pages are found even when
        the query terms appear only in the title (common for doc pages).
        Falls back to title-only search if the full-text search returns nothing.

        Args:
            query:     Search text (natural language — cleaned before CQL)
            space_key: Optionally restrict to one space
            limit:     Max results
        """
        op = "search_pages"
        try:
            # Sanitise: strip CQL-unsafe characters from the query term
            safe_query = re.sub(r'["\\\[\]{}()]', ' ', query).strip()
            safe_query = re.sub(r'\s+', ' ', safe_query)

            # Primary CQL: title OR text — catches pages where terms appear in
            # either field. "text ~" alone misses pages whose relevant content
            # is only in the title (e.g. "AWS stack 1.16.0 release notes").
            cql = f'(title ~ "{safe_query}" OR text ~ "{safe_query}") AND type = "page"'
            if space_key:
                cql += f' AND space.key = "{space_key}"'
            cql += " ORDER BY lastModified DESC"

            params = {
                "cql":    cql,
                "limit":  min(limit, 50),
                "expand": "space,history,excerpt",
            }
            resp = self._get("content/search", params=params, base=self.base_v1)
            data = resp.json()
            results = data.get("results", [])

            # Fallback: if no results and query is multi-word, try each significant
            # word individually joined by AND in title only (CQL is strict about
            # multi-word phrase matching in large Confluence instances)
            if not results and " " in safe_query:
                words = [w for w in safe_query.split() if len(w) > 3]
                if words:
                    title_clauses = " AND ".join(f'title ~ "{w}"' for w in words[:4])
                    cql_fb = f'({title_clauses}) AND type = "page"'
                    if space_key:
                        cql_fb += f' AND space.key = "{space_key}"'
                    cql_fb += " ORDER BY lastModified DESC"
                    try:
                        resp_fb = self._get(
                            "content/search",
                            params={"cql": cql_fb, "limit": min(limit, 50), "expand": "space,history,excerpt"},
                            base=self.base_v1,
                        )
                        results = resp_fb.json().get("results", [])
                    except Exception:
                        pass  # keep empty results, don't mask primary error

            items = []
            for result in results:
                excerpt = result.get("excerpt", "")
                items.append({
                    "id":          result["id"],
                    "title":       result.get("title"),
                    "type":        result.get("type"),
                    "space_key":   (result.get("space") or {}).get("key"),
                    "space_name":  (result.get("space") or {}).get("name"),
                    "excerpt":     excerpt[:500],
                    "last_updated":(result.get("history") or {}).get("lastUpdated", {}).get("when"),
                    "url":         self.domain + result.get("_links", {}).get("webui", ""),
                })

            return self._ok(op, items, query=query, space_key=space_key, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), query=query)

    def get_recent_pages(
        self,
        space_key: Optional[str] = None,
        days:      int = 7,
        limit:     int = 30,
    ) -> SentryResult:
        """
        Get pages modified in the last N days.

        Args:
            space_key: Optionally restrict to one space
            days:      Look-back window in days
            limit:     Max pages
        """
        op = "get_recent_pages"
        try:
            # CQL date format: "yyyy-MM-dd" (no time component — Confluence Cloud requirement)
            since = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            cql   = f'type = "page" AND lastModified >= "{since}"'
            if space_key:
                cql += f' AND space.key = "{space_key}"'
            cql += " ORDER BY lastModified DESC"

            params = {"cql": cql, "limit": min(limit, 50), "expand": "space,history"}
            resp   = self._get("content/search", params=params, base=self.base_v1)
            data   = resp.json()

            items = [
                {
                    "id":          r["id"],
                    "title":       r.get("title"),
                    "space_key":   (r.get("space") or {}).get("key"),
                    "last_updated":(r.get("history") or {}).get("lastUpdated", {}).get("when"),
                    "url":         self.domain + r.get("_links", {}).get("webui", ""),
                }
                for r in data.get("results", [])
            ]

            return self._ok(op, items, days=days, space_key=space_key, total=len(items))

        except Exception as exc:
            return self._err(op, str(exc), days=days)

    def get_child_pages(
        self,
        parent_page_id: str,
        limit: int = 50,
    ) -> SentryResult:
        """
        List all direct child pages under a parent page.

        Args:
            parent_page_id: Parent page ID
            limit:          Max child pages
        """
        op = "get_child_pages"
        try:
            raw = self._paginate_v2(
                f"pages/{parent_page_id}/children",
                {},
                max_items=limit,
            )
            items = [
                {
                    "id":        p["id"],
                    "title":     p.get("title"),
                    "status":    p.get("status"),
                    "version":   (p.get("version") or {}).get("number"),
                    "url":       self.domain + p.get("_links", {}).get("webui", ""),
                }
                for p in raw
            ]
            return self._ok(op, items, parent_page_id=parent_page_id, total=len(items))
        except Exception as exc:
            return self._err(op, str(exc), parent_page_id=parent_page_id)