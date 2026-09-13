"""
SujalConnect - server.py
==========================
FastAPI backend that bridges the frontend (index.html) with the
Music Intelligence Engine (music.py).

Features:
    - Serves index.html + static assets (single-file deploy friendly)
    - /api/search           search songs by query
    - /api/stream/{id}      resolve + return a fresh playable stream URL
    - /api/proxy-audio/{id} proxies the actual audio bytes (avoids CORS /
                             hot-linking / expiry issues -- the <audio> tag
                             just points at us, we handle YouTube's URL)
    - /api/trending         home-feed / discovery feed
    - /api/related/{id}     "Up Next" queue for autoplay
    - /api/song/{id}        metadata only (no network re-resolve)
    - User handling:
        /api/users/register       simple username-based session (no password
                                   needed for a demo project, but structured
                                   so real auth can be swapped in later)
        /api/users/preferences    save onboarding genre preferences
        /api/users/me             fetch current profile + preferences
        /api/users/history        listen history (recently played)
        /api/users/like/{id}      like / unlike a song
        /api/users/liked          list liked songs

Run locally:
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Deployed on Vercel:
    Vercel imports this module's `app` object directly (via api/index.py) as
    a single Fluid-compute Function. There is no `uvicorn.run()` call made
    in that environment -- Vercel's Python runtime handles the ASGI serving.

PERSISTENCE ON VERCEL:
    This originally used a local users.json file as a zero-dependency demo
    "database". On Vercel, function instances don't share (or reliably
    keep) a local filesystem, so that file would silently reset/corrupt
    under real traffic. User accounts, sessions, likes, and history are now
    stored in Upstash Redis via store.py -- see that module's docstring for
    setup details (Vercel Marketplace > Upstash Redis integration).
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

import music  # our engine (music.py) -- module-level `engine` singleton
import store  # Redis (Upstash)-backed persistence -- see store.py

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [server.py] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sujalconnect.server")

BASE_DIR = Path(__file__).resolve().parent
SESSION_COOKIE_NAME = "sujal_session"

AVAILABLE_GENRES = list(music.MusicEngine.GENRE_SEED_QUERIES.keys())

# --------------------------------------------------------------------------
# User store -- Redis-backed (store.UserStore), safe across many Vercel
# function instances. Falls back to a local /tmp file automatically when
# Upstash env vars aren't set (e.g. local dev) -- see store.py.
# --------------------------------------------------------------------------
users = store.UserStore()


# --------------------------------------------------------------------------
# Pydantic request/response models
# --------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=40)


class PreferencesRequest(BaseModel):
    genres: list[str] = Field(default_factory=list)
    language: str = "any"


class LikeRequest(BaseModel):
    song_id: str


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(
    title="SujalConnect Music API",
    description="Backend powering SujalConnect -- search, stream, and discover music.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Session helper
# --------------------------------------------------------------------------
def get_or_create_session(request: Request, response: Response) -> tuple[str, dict]:
    """
    Resolves the current user's session. If none exists, creates an
    anonymous "Guest" user transparently so the app works instantly without
    forcing a hard login wall -- registering later just renames the guest.

    NOTE: we read the cookie directly off `request.cookies` (rather than
    FastAPI's `Cookie(...)` parameter-injection) because this helper is
    called manually from inside route bodies rather than being used as a
    route parameter / Depends() itself -- `Cookie(...)` markers are only
    resolved by FastAPI's dependency-injection system when the function is
    invoked *by* FastAPI, not when we call it ourselves.
    """
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    user = users.get(session_id) if session_id else None

    if not user:
        session_id, user = users.create(username=f"Guest{str(uuid.uuid4())[:6]}")
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            samesite="lax",
        )
    return session_id, user


# --------------------------------------------------------------------------
# Root + static frontend
# --------------------------------------------------------------------------
# ON VERCEL: index.html / album.html / playlist.html live in /public at the
# project root and are served directly from Vercel's CDN at "/",
# "/album.html", and "/playlist.html" -- no Python code runs for those
# requests at all (faster, and doesn't spend function execution time).
# These routes exist ONLY as a fallback for local `uvicorn` development
# (where there is no CDN/public-folder magic) and as a safety net in case
# a future change accidentally shadows the CDN mapping.
# This module is deployed at api/index.py, so the project root (where
# /public lives) is one directory up. When run locally as ./server.py
# instead (BASE_DIR already the project root), fall back to BASE_DIR/public.
_candidate = (BASE_DIR.parent / "public")
PUBLIC_DIR = _candidate if _candidate.exists() else (BASE_DIR / "public")
INDEX_FILE = PUBLIC_DIR / "index.html"
ALBUM_FILE = PUBLIC_DIR / "album.html"
PLAYLIST_FILE = PUBLIC_DIR / "playlist.html"


def _serve_html_file(path: Path, label: str) -> HTMLResponse:
    if path.exists():
        return HTMLResponse(path.read_text(encoding="utf-8"))
    return HTMLResponse(
        f"<h1>SujalConnect</h1><p>{label} not found (expected at {path}). "
        f"On Vercel this route should normally never be hit -- {label} is "
        f"served straight from the CDN's /public folder.</p>",
        status_code=404,
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return _serve_html_file(INDEX_FILE, "index.html")


# NOTE: kept as a fallback (see block comment above) for the original
# "Not Found" JSON page bug -- without an explicit route, FastAPI's default
# 404 JSON handler used to answer for GET /album.html and /playlist.html
# whenever they weren't otherwise resolved. Query params (?id=...&title=...)
# are untouched since they're read client-side via URLSearchParams anyway.
@app.get("/album.html", response_class=HTMLResponse)
async def serve_album_page():
    return _serve_html_file(ALBUM_FILE, "album.html")


@app.get("/playlist.html", response_class=HTMLResponse)
async def serve_playlist_page():
    return _serve_html_file(PLAYLIST_FILE, "playlist.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine_stats": music.engine.stats(), "time": time.time()}


@app.get("/api/genres")
async def get_genres():
    """List of genres the onboarding portal can offer as preference chips."""
    return {"genres": AVAILABLE_GENRES}


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
@app.get("/api/search")
async def search_songs(q: str = Query(..., min_length=1), limit: int = 24):
    results = music.engine.search(q, limit=limit)
    return {"query": q, "count": len(results), "results": [s.to_public_dict() for s in results]}


# --------------------------------------------------------------------------
# Metadata only (fast, no network hit if cached)
# --------------------------------------------------------------------------
@app.get("/api/song/{video_id}")
async def get_song(video_id: str):
    song = music.engine.get_song_metadata(video_id)
    if not song:
        # not seen before -- do a full resolve as a fallback
        song = music.engine.get_stream_url(video_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    return song.to_public_dict()


# --------------------------------------------------------------------------
# Stream URL resolution (used by the player right before playback)
# --------------------------------------------------------------------------
@app.get("/api/stream/{video_id}")
async def get_stream(video_id: str, request: Request, response: Response):
    session_id, _user = get_or_create_session(request, response)
    song = music.engine.get_stream_url(video_id)
    if not song or not song.stream_url:
        raise HTTPException(status_code=404, detail="Could not resolve a playable stream for this song")

    users.add_history(session_id, video_id)

    data = song.to_public_dict(include_stream=False)
    # We give the frontend OUR proxy URL, not the raw googlevideo URL --
    # this sidesteps CORS issues, URL expiry mid-playback edge cases, and
    # keeps YouTube's real URL off the wire to the browser.
    data["playback_url"] = f"/api/proxy-audio/{video_id}"
    return data


# --------------------------------------------------------------------------
# Audio proxy -- streams actual bytes with Range support for seeking
# --------------------------------------------------------------------------
@app.get("/api/proxy-audio/{video_id}")
async def proxy_audio(video_id: str, request: Request):
    song = music.engine.get_stream_url(video_id)
    if not song or not song.stream_url:
        raise HTTPException(status_code=404, detail="Stream unavailable")

    upstream_headers = {}
    range_header = request.headers.get("range")
    if range_header:
        upstream_headers["Range"] = range_header

    try:
        upstream = requests.get(
            song.stream_url,
            headers=upstream_headers,
            stream=True,
            timeout=15,
        )
    except requests.RequestException as e:
        log.warning("Upstream fetch failed for %s (%s), re-resolving once...", video_id, e)
        # URL might have just expired -- force a fresh resolve and retry once
        with music.engine._lock:  # noqa: SLF001 (internal reuse is fine, same package)
            cached = music.engine._song_cache.get(video_id)
            if cached:
                cached.stream_fetched_at = 0.0
        song = music.engine.get_stream_url(video_id)
        if not song or not song.stream_url:
            raise HTTPException(status_code=502, detail="Upstream audio source unavailable")
        try:
            upstream = requests.get(song.stream_url, headers=upstream_headers, stream=True, timeout=15)
        except requests.RequestException:
            raise HTTPException(status_code=502, detail="Upstream audio source unavailable")

    def iter_bytes():
        try:
            for chunk in upstream.iter_content(chunk_size=64 * 1024):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    passthrough_headers = {}
    for h in ("Content-Length", "Content-Range", "Accept-Ranges", "Content-Type"):
        if h in upstream.headers:
            passthrough_headers[h] = upstream.headers[h]
    passthrough_headers.setdefault("Accept-Ranges", "bytes")
    passthrough_headers.setdefault("Content-Type", "audio/mp4")
    passthrough_headers["Cache-Control"] = "no-store"

    status_code = upstream.status_code if upstream.status_code in (200, 206) else 200
    return StreamingResponse(iter_bytes(), status_code=status_code, headers=passthrough_headers)


# --------------------------------------------------------------------------
# Trending / discovery feed
# --------------------------------------------------------------------------
@app.get("/api/trending")
async def trending(genre: Optional[str] = None, limit: int = 24):
    results = music.engine.trending(genre=genre, limit=limit)
    return {"genre": genre, "results": [s.to_public_dict() for s in results]}


# --------------------------------------------------------------------------
# ALBUM-BASED discovery -- what the homepage rows actually render.
# --------------------------------------------------------------------------
# /api/trending and /api/feed (below) return individual SONGS, which is
# correct for things like search results or "up next" queues, but is wrong
# for homepage rows that are supposed to look and behave like Spotify's
# "Popular albums" / "Made for you" shelves: each tile there needs to be a
# real album (its own cover, its own title, its own artist line, and a full
# tracklist waiting behind it) -- not a single song mislabeled as an album.
#
# These two endpoints return real, multi-track Album objects (lightweight
# summary form -- see Album.to_summary_dict) built via music.engine's album
# discovery methods, so the frontend can render genuine album tiles that
# open straight into /api/album/{id} with zero extra resolution needed.
# --------------------------------------------------------------------------
@app.get("/api/albums/trending")
async def trending_albums(genre: Optional[str] = None, limit: int = 12):
    albums = music.engine.trending_albums(genre=genre, limit=limit)
    return {"genre": genre, "results": [a.to_summary_dict() for a in albums]}


@app.get("/api/albums/feed")
async def personalized_album_feed(request: Request, response: Response, limit_per_genre: int = 6):
    session_id, user = get_or_create_session(request, response)
    genres = (user.get("preferences") or {}).get("genres") or []

    if not genres:
        albums = music.engine.trending_albums(limit=12)
        return {"personalized": False, "results": [a.to_summary_dict() for a in albums]}

    albums = music.engine.albums_for_genres(genres, limit_per_genre=limit_per_genre)
    return {"personalized": True, "genres": genres, "results": [a.to_summary_dict() for a in albums]}


# --------------------------------------------------------------------------
# Related songs -> autoplay "Up Next" queue
# --------------------------------------------------------------------------
@app.get("/api/related/{video_id}")
async def related(video_id: str, limit: int = 15):
    results = music.engine.related_songs(video_id, limit=limit)
    return {"seed": video_id, "results": [s.to_public_dict() for s in results]}


# --------------------------------------------------------------------------
# ALBUMS -- Spotify-style expanded album view
# --------------------------------------------------------------------------
# The frontend (album.html) calls one of these two endpoints when the user
# clicks a song/card:
#
#   GET /api/album/by-song/{video_id}   <- normal entry point. Given the
#       clicked song's id, resolves (and, on first hit, *builds*) the real
#       album it belongs to, loads every track on the backend via
#       music.engine, and returns the full ordered tracklist ready to play.
#
#   GET /api/album/{album_id}           <- direct lookup once the frontend
#       already knows the album id (e.g. navigating between albums, or a
#       cached URL), avoiding the by-song resolution work entirely.
#
# Both are backed by the same MusicEngine album cache, so repeat opens of
# the same album are instant and never re-hit YouTube unless the cached
# tracklist has expired (see ALBUM_TTL_SECONDS in music.py).
# --------------------------------------------------------------------------
@app.get("/api/album/by-song/{video_id}")
async def album_by_song(video_id: str, limit: int = 50):
    album = music.engine.get_album_for_song(video_id, limit=limit)
    if not album:
        raise HTTPException(status_code=404, detail="Could not resolve an album for this song")
    return album.to_public_dict(music.engine)


@app.get("/api/album/{album_id}")
async def album_by_id(album_id: str, limit: int = 50):
    album = music.engine.get_album_by_id(album_id, limit=limit)
    if not album or not album.track_ids:
        raise HTTPException(status_code=404, detail="Album not found")
    return album.to_public_dict(music.engine)


@app.get("/api/album/{album_id}/preload")
async def album_preload(album_id: str, limit: int = 5):
    """
    Warms the stream-URL cache for the first `limit` tracks of an album in
    the background threadpool as soon as the album view opens, so the first
    few "play" clicks feel instant instead of waiting on a fresh yt-dlp
    resolve. Fire-and-forget from the frontend right after loading the
    tracklist; the response just reports what got warmed.
    """
    album = music.engine.get_album_by_id(album_id)
    if not album:
        raise HTTPException(status_code=404, detail="Album not found")

    warmed = []
    for vid in album.track_ids[:limit]:
        song = music.engine.get_song_metadata(vid)
        if song and song.stream_url and (time.time() - song.stream_fetched_at) < music.STREAM_URL_TTL_SECONDS:
            warmed.append(vid)
            continue
        # resolve synchronously here -- endpoint already runs in FastAPI's
        # threadpool (sync def would be needed for true blocking-safety, but
        # get_stream_url is fast enough per-call and the whole point is to
        # do this eagerly right when the album opens, off the play-button's
        # critical path).
        resolved = music.engine.get_stream_url(vid)
        if resolved and resolved.stream_url:
            warmed.append(vid)

    return {"album_id": album_id, "warmed": warmed, "requested": album.track_ids[:limit]}


# --------------------------------------------------------------------------
# Personalized home feed (built from onboarding preferences)
# --------------------------------------------------------------------------
@app.get("/api/feed")
async def personalized_feed(request: Request, response: Response):
    session_id, user = get_or_create_session(request, response)
    genres = (user.get("preferences") or {}).get("genres") or []

    if not genres:
        results = music.engine.trending(limit=24)
        return {"personalized": False, "results": [s.to_public_dict() for s in results]}

    results = music.engine.recommendations_for_genres(genres, limit_per_genre=8)
    return {"personalized": True, "genres": genres, "results": [s.to_public_dict() for s in results]}


# --------------------------------------------------------------------------
# User handling
# --------------------------------------------------------------------------
@app.post("/api/users/register")
async def register(body: RegisterRequest, request: Request, response: Response):
    """
    Registers/renames the current session's user. Since this project has
    no password requirement, "registering" simply claims a display name on
    top of the already-existing anonymous session -- so history/likes/
    preferences accumulated as a guest carry over seamlessly.
    """
    session_id, user = get_or_create_session(request, response)

    existing = users.get_by_username(body.username)
    if existing and existing[0] != session_id:
        raise HTTPException(status_code=409, detail="Username already taken")

    users.update(session_id, username=body.username)
    user = users.get(session_id)
    return {"session_created": True, "user": _public_user(user)}


@app.get("/api/users/me")
async def me(request: Request, response: Response):
    session_id, user = get_or_create_session(request, response)
    return _public_user(user)


@app.post("/api/users/preferences")
async def set_preferences(body: PreferencesRequest, request: Request, response: Response):
    session_id, _user = get_or_create_session(request, response)
    valid_genres = [g for g in body.genres if g.lower() in AVAILABLE_GENRES] or body.genres
    users.update(
        session_id,
        onboarded=True,
        preferences={"genres": valid_genres, "language": body.language},
    )
    user = users.get(session_id)
    return {"saved": True, "user": _public_user(user)}


@app.get("/api/users/history")
async def history(request: Request, response: Response, limit: int = 50):
    session_id, user = get_or_create_session(request, response)
    hist = (user.get("history") or [])[:limit]
    ids = [h["id"] for h in hist]
    songs = {s.id: s for s in music.engine.bulk_get(ids)}
    enriched = []
    for h in hist:
        song = songs.get(h["id"])
        if song:
            entry = song.to_public_dict()
            entry["played_at"] = h["played_at"]
            enriched.append(entry)
    return {"history": enriched}


@app.post("/api/users/like")
async def like_song(body: LikeRequest, request: Request, response: Response):
    session_id, _user = get_or_create_session(request, response)
    try:
        is_liked = users.toggle_like(session_id, body.song_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"song_id": body.song_id, "liked": is_liked}


@app.get("/api/users/liked")
async def liked_songs(request: Request, response: Response):
    session_id, user = get_or_create_session(request, response)
    ids = user.get("liked") or []
    songs = music.engine.bulk_get(ids)
    # resolve any liked songs we don't have cached metadata for yet
    missing = [i for i in ids if i not in {s.id for s in songs}]
    for mid in missing:
        s = music.engine.get_stream_url(mid)
        if s:
            songs.append(s)
    return {"results": [s.to_public_dict() for s in songs]}


def _public_user(user: dict) -> dict:
    return {
        "username": user["username"],
        "onboarded": user.get("onboarded", False),
        "preferences": user.get("preferences", {}),
        "liked_count": len(user.get("liked", [])),
        "history_count": len(user.get("history", [])),
    }


# --------------------------------------------------------------------------
# Global error handler -> always return clean JSON, never a raw 500 trace
# --------------------------------------------------------------------------
@app.exception_handler(Exception)
async def all_exceptions_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
