"""
User accounts and saved notes, stored in SQLite.

Passwords are hashed with scrypt (standard library). Sign-in creates a random
session token that is sent as an HttpOnly cookie; only its SHA-256 hash is
stored, so a leaked database does not expose usable sessions.

Each upload or live session becomes a note owned by the signed-in user:
title, source, language, transcript, summary and the ticked task state.

Environment:
    NOTES_DB            SQLite file path (default: notes.db next to this file)
    ALLOW_REGISTRATION  "0" to disable creating new accounts (default: "1")
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

DB_PATH = os.getenv("NOTES_DB", str(Path(__file__).with_name("notes.db")))
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "1") != "0"
SESSION_COOKIE = "cn_session"
SESSION_SECONDS = 30 * 24 * 3600
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 8


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
    return user_for_token(websocket.cookies.get(SESSION_COOKIE))


def _set_session_cookie(response: Response, request: Request, token: str):
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=SESSION_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )


# ============================================================
# Notes
# ============================================================

def _title_from_summary(summary: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*$", summary or "", re.M)
    return match.group(1)[:200] if match else None


def create_note(user_id: int, title: str, source: str, language: str,
                transcript: str = "", summary: str = "") -> int:
    now = time.time()
    with db() as conn:
        cursor = conn.execute(
            """INSERT INTO notes (user_id, title, source, language, transcript, summary, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, _title_from_summary(summary) or title, source, language,
             transcript, summary or "", now, now),
        )
        return cursor.lastrowid


def save_note(note_id: int, transcript: str | None = None, summary: str | None = None):
    """Update a note's transcript and/or summary. A summary "# Title" becomes
    the note title unless the user renamed it."""
    fields, values = [], []
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


def _note_out(row: sqlite3.Row, full: bool) -> dict:
    note = {
        "id": row["id"],
        "title": row["title"],
        "source": row["source"],
        "language": row["language"],
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
        username = body.username.strip()
        if not USERNAME_RE.match(username):
            raise HTTPException(status_code=400, detail="Username must be 3-32 characters: letters, numbers, _ . -")
        if len(body.password) < MIN_PASSWORD_LENGTH:
            raise HTTPException(status_code=400, detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        try:
            with db() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, hash_password(body.password), time.time()),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="That username is already taken.")
        _set_session_cookie(response, request, create_session(user_id))
        return {"id": user_id, "username": username}

    @router.post("/login")
    def login(body: Credentials, request: Request, response: Response):
        with db() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (body.username.strip(),),
            ).fetchone()
        password_ok = verify_password(body.password, row["password_hash"] if row else _DUMMY_HASH)
        if row is None or not password_ok:
            raise HTTPException(status_code=401, detail="Wrong username or password.")
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
