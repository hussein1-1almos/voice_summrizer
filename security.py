"""
Protection against common web attacks, shared by the HTTP and WebSocket code.

- Security headers on every response: Content-Security-Policy (XSS),
  frame-ancestors / X-Frame-Options (clickjacking), nosniff, Referrer-Policy,
  Permissions-Policy, and HSTS when served over HTTPS.
- Same-origin check for state-changing requests and WebSocket handshakes that
  carry the session cookie (CSRF and cross-site WebSocket hijacking).
- Request body size limits (denial of service with huge bodies).
- An in-memory rate limiter used for sign-in (password guessing).

Behind a reverse proxy, run uvicorn with --proxy-headers and
--forwarded-allow-ips=<proxy ip> so client IPs and https are seen correctly,
and keep the proxy forwarding the original Host header.

Environment:
    ALLOWED_ORIGINS  extra allowed origins, comma separated
                     (e.g. "https://notes.example.com"); same-host is always allowed
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from urllib.parse import urlsplit

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

SESSION_COOKIE = "cn_session"
EXTRA_ORIGINS = {o.strip().rstrip("/").lower() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()}

MAX_UPLOAD_BODY = 105 * 1024 * 1024   # /transcribe: 100 MB audio + form overhead
MAX_API_BODY = 6 * 1024 * 1024        # everything else (largest is a .md import)
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


# ============================================================
# Origin checks
# ============================================================

def origin_allowed(origin: str | None, host: str | None) -> bool:
    """True when the Origin header names this same site (or an allowed extra origin)."""
    if not origin or origin == "null":
        return False
    origin = origin.rstrip("/").lower()
    if origin in EXTRA_ORIGINS:
        return True
    return bool(host) and urlsplit(origin).netloc == host.lower()


def request_origin(headers) -> str | None:
    origin = headers.get("origin")
    if origin:
        return origin
    # Some browsers omit Origin on same-origin requests; fall back to Referer.
    referer = headers.get("referer")
    if referer:
        parts = urlsplit(referer)
        return f"{parts.scheme}://{parts.netloc}"
    return None


# ============================================================
# Rate limiting
# ============================================================

class RateLimiter:
    """Sliding-window counter per key, kept in memory (one server process)."""

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque:
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        return hits

    def retry_after(self, key: str) -> int:
        """Seconds until the key may try again; 0 if it is not limited."""
        now = time.monotonic()
        with self._lock:
            hits = self._recent(key, now)
            if len(hits) < self.limit:
                return 0
            return max(1, int(hits[0] + self.window - now) + 1)

    def hit(self, key: str):
        now = time.monotonic()
        with self._lock:
            self._recent(key, now).append(now)
            if len(self._hits) > 50_000:  # keep memory bounded under a flood
                for stale in [k for k, v in self._hits.items() if not v or v[-1] <= now - self.window]:
                    del self._hits[stale]

    def reset(self, key: str):
        with self._lock:
            self._hits.pop(key, None)


# ============================================================
# Middleware
# ============================================================

def _csp(request: Request) -> str:
    host = request.headers.get("host", "")
    ws = f"wss://{host}" if request.url.scheme == "https" else f"ws://{host} wss://{host}"
    return "; ".join([
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data:",
        f"connect-src 'self' {ws}",
        "media-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if request.method not in SAFE_METHODS:
            # Cross-site request forgery: a request that carries the session
            # cookie must come from a page on this site.
            if SESSION_COOKIE in request.cookies and not origin_allowed(
                request_origin(request.headers), request.headers.get("host")
            ):
                return JSONResponse({"detail": "Cross-site request blocked."}, status_code=403)

            limit = MAX_UPLOAD_BODY if path == "/transcribe" else MAX_API_BODY
            length = request.headers.get("content-length")
            if length is None:
                if request.headers.get("transfer-encoding"):
                    return JSONResponse({"detail": "Content-Length is required."}, status_code=411)
            elif not length.isdigit() or int(length) > limit:
                return JSONResponse({"detail": "Request is too large."}, status_code=413)

        response = await call_next(request)

        headers = response.headers
        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "no-referrer")
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        headers.setdefault("Permissions-Policy", "microphone=(self), display-capture=(self), camera=(), geolocation=()")
        headers.setdefault("Content-Security-Policy", _csp(request))
        if request.url.scheme == "https":
            headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        if path.startswith("/api/"):
            headers.setdefault("Cache-Control", "no-store")
        return response


def install(app: FastAPI):
    app.add_middleware(SecurityMiddleware)
