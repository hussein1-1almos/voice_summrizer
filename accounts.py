"""
User accounts and saved notes, stored in SQLite.

Passwords are hashed with scrypt (standard library). Sign-in creates a random
session token that is sent as an HttpOnly cookie; only its SHA-256 hash is
stored, so a leaked database does not expose usable sessions.

Each upload or live session becomes a note owned by the signed-in user:
title, source, language, transcript, summary and the ticked task state.

Environment:
    NOTES_DB            SQLite file path (default: notes.db next to this file)
    ALLOW_REGISTRATION  "1" to let visitors create accounts (default: off;
                        create accounts with "python manage.py create-user")
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel

import security

DB_PATH = os.getenv("NOTES_DB", str(Path(__file__).with_name("notes.db")))
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "0") == "1"
SESSION_COOKIE = security.SESSION_COOKIE
SESSION_SECONDS = 30 * 24 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 256  # hashing very long inputs is wasted CPU for an attacker to exploit

# Password guessing: every sign-in attempt counts against the IP; failures
# also count against the username, which locks it for a while.
LOGIN_IP_LIMIT = security.RateLimiter(limit=20, window_seconds=5 * 60)
LOGIN_USER_FAILURES = security.RateLimiter(limit=5, window_seconds=15 * 60)
REGISTER_IP_LIMIT = security.RateLimiter(limit=5, window_seconds=60 * 60)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _too_many(seconds: int):
    minutes = max(1, round(seconds / 60))
    raise HTTPException(
        status_code=429,
        detail=f"Too many attempts. Try again in {minutes} minute{'s' if minutes != 1 else ''}.",
        headers={"Retry-After": str(seconds)},
    )


# ============================================================
# Database
# ============================================================

@contextmanager
def db():
    # One short-lived connection per operation keeps this safe to call
    # from request threads, the event loop and to_thread workers alike.
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                title_locked INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                language TEXT NOT NULL,
                transcript TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                checks TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS notes_user_updated ON notes(user_id, updated_at DESC);
        """)
        # status: "processing" (upload job queued/running), "live" (recording),
        # "ready", or "failed" (see error).
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(notes)")}
        if "status" not in columns:
            conn.execute("ALTER TABLE notes ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'")
        if "error" not in columns:
            conn.execute("ALTER TABLE notes ADD COLUMN error TEXT NOT NULL DEFAULT ''")
        # Jobs do not survive a restart; say so instead of showing them as running forever.
        conn.execute(
            "UPDATE notes SET status = 'failed', error = ? WHERE status = 'processing'",
            ("The server restarted before this recording finished processing. Please upload it again.",),
        )
        conn.execute(
            "UPDATE notes SET status = 'ready', error = ? WHERE status = 'live'",
            ("The server restarted during this live session; the end of the session and the final summary may be missing.",),
        )
        conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (time.time(),))


# ============================================================
# Passwords and sessions
# ============================================================

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return hmac.compare_digest(digest.hex(), digest_hex)


# Verifying against a dummy hash keeps "unknown user" as slow as "wrong password".
_DUMMY_HASH = hash_password(secrets.token_hex(8))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with db() as conn:
        conn.execute(
            "INSERT INTO auth_sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_token_hash(token), user_id, time.time() + SESSION_SECONDS),
        )
    return token


def user_for_token(token: str | None) -> dict | None:
    if not token:
        return None
    with db() as conn:
        row = conn.execute(
            """SELECT users.id, users.username FROM auth_sessions
               JOIN users ON users.id = auth_sessions.user_id
               WHERE token_hash = ? AND expires_at > ?""",
            (_token_hash(token), time.time()),
        ).fetchone()
    return dict(row) if row else None


def require_user(request: Request) -> dict:
    user = user_for_token(request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(status_code=401, detail="Please sign in.")
    return user


def websocket_user(websocket: WebSocket) -> dict | None:
    # Browsers send cookies on cross-site WebSocket handshakes and CORS does not
    # apply, so the Origin must be checked here (cross-site WebSocket hijacking).
    if not security.origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        print(f"Blocked WebSocket from origin {websocket.headers.get('origin')!r}")
        return None
    return user_for_token(websocket.cookies.get(SESSION_COOKIE))


def _set_session_cookie(response: Response, request: Request, token: str):
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=SESSION_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


# ============================================================
# Notes
# ============================================================

def _title_from_summary(summary: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*$", summary or "", re.M)
    return match.group(1)[:200] if match else None


def create_note(user_id: int, title: str, source: str, language: str,
                transcript: str = "", summary: str = "", status: str = "ready") -> int:
    now = time.time()
    with db() as conn:
        cursor = conn.execute(
            """INSERT INTO notes (user_id, title, source, language, transcript, summary, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, _title_from_summary(summary) or title, source, language,
             transcript, summary or "", status, now, now),
        )
        return cursor.lastrowid


def save_note(note_id: int, transcript: str | None = None, summary: str | None = None,
              status: str | None = None, error: str | None = None):
    """Update a note's transcript, summary and/or status. A summary "# Title"
    becomes the note title unless the user renamed it."""
    fields, values = [], []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if transcript is not None:
        fields.append("transcript = ?")
        values.append(transcript)
    if summary is not None:
        fields.append("summary = ?")
        values.append(summary)
        title = _title_from_summary(summary)
        if title:
            fields.append("title = CASE WHEN title_locked THEN title ELSE ? END")
            values.append(title)
    if not fields:
        return
    fields.append("updated_at = ?")
    values.append(time.time())
    with db() as conn:
        conn.execute(f"UPDATE notes SET {', '.join(fields)} WHERE id = ?", (*values, note_id))


def count_notes(user_id: int, status: str) -> int:
    with db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM notes WHERE user_id = ? AND status = ?", (user_id, status)
        ).fetchone()[0]


def _note_out(row: sqlite3.Row, full: bool) -> dict:
    note = {
        "id": row["id"],
        "title": row["title"],
        "source": row["source"],
        "language": row["language"],
        "status": row["status"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if full:
        note.update(
            transcript=row["transcript"],
            summary=row["summary"],
            checks=json.loads(row["checks"] or "{}"),
        )
    else:
        note["preview"] = (row["transcript"] or "")[:160]
    return note


def split_exported_markdown(markdown: str) -> tuple[str, str]:
    """Split a file made by the page's "Download .md" into (summary, transcript).
    A Markdown file without a "## Full transcript" section is all summary."""
    text = markdown.replace("\r\n", "\n").lstrip("﻿").strip()
    match = re.search(r"^##\s+(Full transcript|النص الكامل)\s*$", text, re.M | re.I)
    if not match:
        return text, ""
    summary = re.sub(r"\n-{3,}\s*$", "", text[:match.start()].rstrip()).strip()
    transcript = text[match.end():].strip()
    # Drop the "---\n_Exported ..._" footer the page appends.
    transcript = re.sub(r"\n-{3,}\s*\n_Exported[^\n]*_\s*$", "", transcript).strip()
    if transcript.startswith("_No transcript available._"):
        transcript = ""
    if summary.startswith("_No summary available._"):
        summary = ""
    return summary, transcript


def import_markdown(user_id: int, markdown: str, title: str = "Imported notes") -> int:
    summary, transcript = split_exported_markdown(markdown)
    language = "ar" if re.search(r"[؀-ۿ]", summary + transcript) else "en"
    return create_note(user_id, title=title, source="import", language=language,
                       transcript=transcript, summary=summary)


def validate_credentials(username: str, password: str) -> str:
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-32 characters: letters, numbers, _ . -")
    _check_password_length(password)
    return username


def _check_password_length(password: str):
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")


def create_user(username: str, password: str) -> int:
    """Raises ValueError for invalid input and sqlite3.IntegrityError if taken."""
    username = validate_credentials(username, password)
    with db() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), time.time()),
        )
        return cursor.lastrowid


def find_user(username: str) -> dict | None:
    with db() as conn:
        row = conn.execute(
            "SELECT id, username FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    return dict(row) if row else None


def set_password(user_id: int, password: str):
    """Change a password and sign the user out everywhere."""
    _check_password_length(password)
    with db() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(password), user_id))
        conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))


def _owned_note(conn, note_id: int, user: dict) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM notes WHERE id = ? AND user_id = ?", (note_id, user["id"])
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Note not found.")
    return row


# ============================================================
# API
# ============================================================

class Credentials(BaseModel):
    username: str
    password: str


class NotePatch(BaseModel):
    title: str | None = None
    checks: dict[str, bool] | None = None


class NoteImport(BaseModel):
    markdown: str
    title: str = "Imported notes"


def build_accounts_router() -> APIRouter:
    init_db()
    router = APIRouter(prefix="/api")

    @router.get("/config")
    def config():
        return {"allow_registration": ALLOW_REGISTRATION}

    @router.post("/register")
    def register(body: Credentials, request: Request, response: Response):
        if not ALLOW_REGISTRATION:
            raise HTTPException(status_code=403, detail="Creating new accounts is disabled on this server.")
        ip = _client_ip(request)
        if wait := REGISTER_IP_LIMIT.retry_after(ip):
            _too_many(wait)
        REGISTER_IP_LIMIT.hit(ip)
        username = body.username.strip()
        try:
            user_id = create_user(username, body.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="That username is already taken.")
        _set_session_cookie(response, request, create_session(user_id))
        return {"id": user_id, "username": username}

    @router.post("/login")
    def login(body: Credentials, request: Request, response: Response):
        ip = _client_ip(request)
        user_key = body.username.strip().lower()[:64]
        wait = max(LOGIN_IP_LIMIT.retry_after(ip), LOGIN_USER_FAILURES.retry_after(user_key))
        if wait:
            _too_many(wait)
        LOGIN_IP_LIMIT.hit(ip)

        row = None
        password_ok = False
        if len(body.password) <= MAX_PASSWORD_LENGTH:
            with db() as conn:
                row = conn.execute(
                    "SELECT id, username, password_hash FROM users WHERE username = ?",
                    (body.username.strip(),),
                ).fetchone()
            password_ok = verify_password(body.password, row["password_hash"] if row else _DUMMY_HASH)
        if row is None or not password_ok:
            LOGIN_USER_FAILURES.hit(user_key)
            print(f"Failed sign-in for {user_key!r} from {ip}")
            raise HTTPException(status_code=401, detail="Wrong username or password.")
        LOGIN_USER_FAILURES.reset(user_key)
        _set_session_cookie(response, request, create_session(row["id"]))
        return {"id": row["id"], "username": row["username"]}

    @router.post("/logout")
    def logout(request: Request, response: Response):
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            with db() as conn:
                conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),))
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @router.get("/me")
    def me(user: dict = Depends(require_user)):
        return user

    @router.get("/notes")
    def list_notes(user: dict = Depends(require_user)):
        with db() as conn:
            rows = conn.execute(
                "SELECT * FROM notes WHERE user_id = ? ORDER BY updated_at DESC LIMIT 200",
                (user["id"],),
            ).fetchall()
        return [_note_out(row, full=False) for row in rows]

    @router.post("/notes/import")
    def import_note(body: NoteImport, user: dict = Depends(require_user)):
        if not body.markdown.strip():
            raise HTTPException(status_code=400, detail="The Markdown file is empty.")
        if len(body.markdown) > 5_000_000:
            raise HTTPException(status_code=413, detail="The Markdown file is too large.")
        return {"id": import_markdown(user["id"], body.markdown, body.title.strip()[:200] or "Imported notes")}

    @router.get("/notes/{note_id}")
    def get_note(note_id: int, user: dict = Depends(require_user)):
        with db() as conn:
            return _note_out(_owned_note(conn, note_id, user), full=True)

    @router.patch("/notes/{note_id}")
    def patch_note(note_id: int, body: NotePatch, user: dict = Depends(require_user)):
        with db() as conn:
            _owned_note(conn, note_id, user)
            if body.title is not None:
                title = body.title.strip()[:200]
                if not title:
                    raise HTTPException(status_code=400, detail="Title cannot be empty.")
                conn.execute("UPDATE notes SET title = ?, title_locked = 1 WHERE id = ?", (title, note_id))
            if body.checks is not None:
                # Ticking tasks is not an edit of the content, so updated_at stays.
                conn.execute("UPDATE notes SET checks = ? WHERE id = ?", (json.dumps(body.checks), note_id))
        return {"ok": True}

    @router.delete("/notes/{note_id}")
    def delete_note(note_id: int, user: dict = Depends(require_user)):
        with db() as conn:
            _owned_note(conn, note_id, user)
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return {"ok": True}

    return router
