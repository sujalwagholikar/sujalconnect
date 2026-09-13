"""
SujalConnect - store.py
=========================
Serverless-safe persistence layer, backed by Upstash Redis (installed from
the Vercel Marketplace). Replaces two things that used to be local JSON
files on disk:

    1. server.py's UserStore  (users.json)      -> users, sessions, likes, history
    2. music.py's disk caches (metadata/album)  -> long-lived song + album cache

WHY REDIS, AND WHY *THIS* CLIENT:
Vercel Functions are ephemeral -- each cold start gets a fresh, empty
filesystem, and concurrent invocations may run in entirely separate
instances that never share local state. A local JSON file (as this project
originally used) silently loses data or corrupts itself under any real
traffic. Upstash's REST API (rather than the raw Redis TCP protocol) is
used here specifically because it works over plain HTTPS with no
persistent socket -- exactly what a stateless serverless function needs,
and it avoids adding a native/binary Redis client dependency.

CONFIGURATION:
When you add the "Upstash Redis" integration from the Vercel Marketplace to
this project, Vercel automatically injects these two environment variables
into your deployment (no manual copy-pasting needed):

    UPSTASH_REDIS_REST_URL
    UPSTASH_REDIS_REST_TOKEN

If they are missing (e.g. running locally without setting them up), this
module transparently falls back to a local file-backed store at
/tmp/sujalconnect_fallback.json so the app still runs end-to-end -- with
the same durability caveats as before (fine for local dev, not for prod).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

import requests

log = logging.getLogger("sujalconnect.store")

UPSTASH_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").rstrip("/")
UPSTASH_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_ENABLED = bool(UPSTASH_URL and UPSTASH_TOKEN)

_HEADERS = {"Authorization": f"Bearer {UPSTASH_TOKEN}"} if REDIS_ENABLED else {}
_TIMEOUT = 5  # seconds -- Upstash REST calls are typically <50ms


class _LocalFallbackStore:
    """
    Dead-simple JSON-file key/value store used ONLY when no Upstash
    credentials are configured (local development). Not safe for
    concurrent/multi-instance production use -- see module docstring.
    """

    def __init__(self):
        self._path = "/tmp/sujalconnect_fallback.json"
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            self._data = {}
        log.warning(
            "UPSTASH_REDIS_REST_URL/TOKEN not set -- using an in-memory/"
            "local-file fallback store. Fine for local dev; DO NOT rely on "
            "this in production on Vercel (no cross-instance durability)."
        )

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except Exception as e:
            log.warning("fallback store save failed: %s", e)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None):
        with self._lock:
            self._data[key] = value
            self._save()

    def delete(self, key: str):
        with self._lock:
            self._data.pop(key, None)
            self._save()

    def hget(self, key: str, field: str) -> Optional[str]:
        with self._lock:
            return (self._data.get(key) or {}).get(field)

    def hgetall(self, key: str) -> dict:
        with self._lock:
            return dict(self._data.get(key) or {})

    def hset(self, key: str, field: str, value: str):
        with self._lock:
            self._data.setdefault(key, {})[field] = value
            self._save()


_fallback = None if REDIS_ENABLED else _LocalFallbackStore()


def _upstash(*parts: str) -> Any:
    """
    Calls the Upstash Redis REST API with a single command, given as
    positional string parts, e.g. _upstash("SET", "foo", "bar").
    Docs: https://upstash.com/docs/redis/features/restapi
    """
    url = f"{UPSTASH_URL}/" + "/".join(_quote(p) for p in parts)
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("result")


def _quote(part: str) -> str:
    from urllib.parse import quote

    return quote(part, safe="")


# --------------------------------------------------------------------------
# Public key/value API (used directly by music.py for song/album caches)
# --------------------------------------------------------------------------
def kv_get(key: str) -> Optional[str]:
    try:
        if REDIS_ENABLED:
            return _upstash("GET", key)
        return _fallback.get(key)
    except Exception as e:
        log.warning("kv_get(%s) failed: %s", key, e)
        return None


def kv_set(key: str, value: str, ex: Optional[int] = None):
    try:
        if REDIS_ENABLED:
            if ex:
                _upstash("SET", key, value, "EX", str(ex))
            else:
                _upstash("SET", key, value)
        else:
            _fallback.set(key, value, ex=ex)
    except Exception as e:
        log.warning("kv_set(%s) failed: %s", key, e)


def kv_get_json(key: str) -> Optional[Any]:
    raw = kv_get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def kv_set_json(key: str, value: Any, ex: Optional[int] = None):
    kv_set(key, json.dumps(value, ensure_ascii=False), ex=ex)


# --------------------------------------------------------------------------
# Hash helpers (used for the user-profile object: one Redis hash per user)
# --------------------------------------------------------------------------
def hash_get_all(key: str) -> dict:
    try:
        if REDIS_ENABLED:
            flat = _upstash("HGETALL", key) or []
            return dict(zip(flat[0::2], flat[1::2]))
        return _fallback.hgetall(key)
    except Exception as e:
        log.warning("hash_get_all(%s) failed: %s", key, e)
        return {}


def hash_set(key: str, field: str, value: str):
    try:
        if REDIS_ENABLED:
            _upstash("HSET", key, field, value)
        else:
            _fallback.hset(key, field, value)
    except Exception as e:
        log.warning("hash_set(%s,%s) failed: %s", key, field, e)


# --------------------------------------------------------------------------
# UserStore -- drop-in replacement for the old JSON-file UserStore in
# server.py, with the exact same method names/signatures so server.py needs
# no logic changes beyond swapping the import.
# --------------------------------------------------------------------------
class UserStore:
    """
    Redis-backed user store. Each user is one JSON blob under
    `user:{session_id}`, plus a `username_index:{lowercased_username}` ->
    session_id lookup key so username uniqueness checks don't require a
    full table scan (which Redis/Upstash has no cheap way to do anyway).
    """

    def _key(self, session_id: str) -> str:
        return f"sujal:user:{session_id}"

    def _username_key(self, username: str) -> str:
        return f"sujal:username:{username.lower()}"

    def get(self, session_id: str) -> Optional[dict]:
        if not session_id:
            return None
        return kv_get_json(self._key(session_id))

    def get_by_username(self, username: str) -> Optional[tuple[str, dict]]:
        session_id = kv_get(self._username_key(username))
        if not session_id:
            return None
        user = self.get(session_id)
        if not user:
            return None
        return session_id, user

    def create(self, username: str) -> tuple[str, dict]:
        import uuid

        session_id = str(uuid.uuid4())
        user = {
            "username": username,
            "created_at": time.time(),
            "onboarded": False,
            "preferences": {"genres": [], "language": "any"},
            "history": [],
            "liked": [],
        }
        kv_set_json(self._key(session_id), user)
        kv_set(self._username_key(username), session_id)
        return session_id, user

    def update(self, session_id: str, **fields):
        user = self.get(session_id)
        if user is None:
            return
        old_username = user.get("username")
        user.update(fields)
        kv_set_json(self._key(session_id), user)
        new_username = fields.get("username")
        if new_username and new_username != old_username:
            kv_set(self._username_key(new_username), session_id)

    def add_history(self, session_id: str, song_id: str, max_len: int = 200):
        user = self.get(session_id)
        if not user:
            return
        user["history"] = [h for h in user.get("history", []) if h["id"] != song_id]
        user["history"].insert(0, {"id": song_id, "played_at": time.time()})
        user["history"] = user["history"][:max_len]
        kv_set_json(self._key(session_id), user)

    def toggle_like(self, session_id: str, song_id: str) -> bool:
        user = self.get(session_id)
        if user is None:
            raise KeyError("no such session")
        liked = set(user.get("liked", []))
        if song_id in liked:
            liked.remove(song_id)
            is_liked = False
        else:
            liked.add(song_id)
            is_liked = True
        user["liked"] = list(liked)
        kv_set_json(self._key(session_id), user)
        return is_liked
