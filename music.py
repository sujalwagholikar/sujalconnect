"""
SujalConnect - music.py
========================
Core Music Intelligence Engine.

Responsibilities:
    - Search for songs (YouTube Music search via yt-dlp's `ytsearch`)
    - Extract playable, direct audio stream URLs (highest quality available)
    - Fetch rich metadata: title, artist, duration, thumbnail/poster, album-ish info
    - In-memory + on-disk caching so repeat requests are instant and we don't
      hammer YouTube (and so previously played songs act as "preloaded" songs)
    - Basic recommendation / "related songs" engine used for autoplay-next
      and the onboarding "genre preference" flow
    - Thread-safe, since FastAPI (server.py) will call into this from an
      async threadpool for many concurrent users

This file has NO web-framework code in it on purpose -- server.py is the
only thing that talks HTTP. music.py is pure Python "business logic" so it
can be reused, tested, or swapped independently.

Dependencies (put these in requirements.txt):
    yt-dlp
    requests
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    from yt_dlp import YoutubeDL
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "yt-dlp is not installed. Run:  pip install -U yt-dlp\n"
        f"Original error: {e}"
    )

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None

try:
    import store as _store  # Redis (Upstash)-backed cross-instance cache
except ImportError:  # pragma: no cover
    _store = None

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [music.py] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sujalconnect.music")


# --------------------------------------------------------------------------
# Paths / persistent cache
# --------------------------------------------------------------------------
# NOTE ON VERCEL: the project source directory is a read-only bundle at
# runtime, so we never write cache files next to this module. Instead we use
# /tmp, which Vercel Functions provide as writable scratch space (up to
# 512MB by default) -- but /tmp is LOCAL TO ONE FUNCTION INSTANCE and is
# wiped on cold start, so this is purely a same-instance speed optimization,
# never a source of truth. Cross-instance/cross-restart caching (the thing
# that actually matters for cost/perf under real traffic) is handled by
# Redis in store.py, which MusicEngine consults first (see get_stream_url,
# get_song_metadata, search) before ever falling through to yt-dlp.
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("SUJAL_CACHE_DIR", "/tmp/sujalconnect_cache"))
try:
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
except OSError:
    CACHE_DIR = Path("/tmp")
METADATA_CACHE_FILE = CACHE_DIR / "metadata_cache.json"
TRENDING_CACHE_FILE = CACHE_DIR / "trending_cache.json"
ALBUM_CACHE_FILE = CACHE_DIR / "album_cache.json"

# Stream URLs from YouTube expire (~6 hours typically). We cache metadata
# (poster, title, artist, duration, video id) essentially forever, but we
# always re-resolve the actual playable stream URL if it's older than this:
STREAM_URL_TTL_SECONDS = 60 * 60 * 3  # 3 hours, safely inside YouTube's window
METADATA_TTL_SECONDS = 60 * 60 * 24 * 14  # 2 weeks
ALBUM_TTL_SECONDS = 60 * 60 * 24 * 7  # 1 week -- album tracklists rarely change


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------
@dataclass
class Song:
    """A single playable track, fully described for the frontend player."""

    id: str  # youtube video id -- our canonical song id
    title: str
    artist: str
    duration: int  # seconds
    duration_str: str
    poster: str  # high-res thumbnail / "album art"
    stream_url: Optional[str] = None
    stream_fetched_at: float = 0.0
    audio_format: str = "m4a"
    bitrate: Optional[float] = None
    source: str = "youtube"
    view_count: Optional[int] = None
    genre_hint: Optional[str] = None
    album_id: Optional[str] = None  # canonical album this track belongs to (if known)
    album_name: Optional[str] = None
    track_number: Optional[int] = None

    def to_public_dict(self, include_stream: bool = False) -> dict:
        """Dict shape sent to the frontend. Stream URL only included on demand."""
        d = {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "duration": self.duration,
            "duration_str": self.duration_str,
            "poster": self.poster,
            "audio_format": self.audio_format,
            "bitrate": self.bitrate,
            "source": self.source,
            "album_id": self.album_id,
            "album_name": self.album_name,
            "track_number": self.track_number,
        }
        if include_stream:
            d["stream_url"] = self.stream_url
        return d


@dataclass
class Album:
    """
    A real album/EP/playlist grouping, resolved from YouTube Music, with its
    full ordered tracklist -- this is what powers the Spotify-style expanded
    album view (album.html): click any song -> see + play every track that
    actually belongs to that release, not just "related" songs.
    """

    id: str  # YouTube Music browse id (e.g. "MPREb_xxxxx") or synthetic fallback id
    title: str
    artist: str
    poster: str = ""
    year: Optional[str] = None
    track_ids: list[str] = field(default_factory=list)  # ordered song ids
    fetched_at: float = 0.0
    is_synthetic: bool = False  # True when we couldn't find a real album and
    # built a best-effort "single + related" pseudo-album instead

    def to_public_dict(self, engine: "MusicEngine") -> dict:
        tracks = engine.bulk_get(self.track_ids)
        by_id = {t.id: t for t in tracks}
        ordered = [by_id[tid] for tid in self.track_ids if tid in by_id]
        total_duration = sum((t.duration or 0) for t in ordered)
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "poster": self.poster or (ordered[0].poster if ordered else ""),
            "year": self.year,
            "is_synthetic": self.is_synthetic,
            "track_count": len(ordered),
            "total_duration": total_duration,
            "tracks": [t.to_public_dict() for t in ordered],
        }

    def to_summary_dict(self) -> dict:
        """
        Lightweight dict for homepage album grids ("Made for you", "Popular
        right now") -- no per-track resolution needed, just enough to
        render an album tile (cover, title, artist, track count) and to
        deep-link straight into /api/album/{id} when clicked.
        """
        return {
            "id": self.id,
            "title": self.title,
            "artist": self.artist,
            "poster": self.poster,
            "year": self.year,
            "is_synthetic": self.is_synthetic,
            "track_count": len(self.track_ids),
        }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fmt_duration(seconds: Optional[int]) -> str:
    if not seconds:
        return "0:00"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


_ARTIST_SPLIT_RE = re.compile(r"\s*[-–—|]\s*")
_JUNK_RE = re.compile(
    r"\(?\b(official( music)? video|official audio|lyrics?( video)?|"
    r"visualizer|hd|4k|full song|audio|mv|prod\.?.*?)\b\)?",
    re.IGNORECASE,
)


def _clean_title_artist(raw_title: str, uploader: str) -> tuple[str, str]:
    """
    YouTube titles are messy: 'Artist - Song (Official Video)'.
    Try to split into a clean (title, artist) pair. Falls back to uploader.
    """
    cleaned = _JUNK_RE.sub("", raw_title).strip(" -|()[]").strip()
    parts = _ARTIST_SPLIT_RE.split(cleaned, maxsplit=1)
    if len(parts) == 2 and 0 < len(parts[0]) < 60:
        artist, title = parts[0].strip(), parts[1].strip()
    else:
        artist, title = uploader or "Unknown Artist", cleaned or raw_title
    # Strip "- Topic" suffix YouTube auto-generates for music channels
    artist = re.sub(r"\s*-\s*Topic$", "", artist).strip()
    return (title or raw_title, artist or "Unknown Artist")


def _slugify_album_key(artist: str, album: str) -> str:
    """Deterministic synthetic album id when YouTube gives us an album *name*
    but no browse id (common on flat search results) -- lets us still group
    tracks from the same album consistently across requests/sessions."""
    raw = f"{(artist or '').strip().lower()}::{(album or '').strip().lower()}"
    raw = re.sub(r"[^a-z0-9:]+", "-", raw).strip("-")
    return f"alb_{raw}" if raw.strip("-:") else ""


def _best_thumbnail(thumbnails: list[dict]) -> str:
    if not thumbnails:
        return ""
    # yt-dlp usually returns them sorted ascending by preference/size already,
    # but we sort defensively by resolution area when available.
    def area(t):
        return (t.get("width") or 0) * (t.get("height") or 0)

    try:
        best = max(thumbnails, key=area)
        return best.get("url", thumbnails[-1].get("url", ""))
    except Exception:
        return thumbnails[-1].get("url", "")


# --------------------------------------------------------------------------
# Core engine
# --------------------------------------------------------------------------
class MusicEngine:
    """
    Thread-safe wrapper around yt-dlp providing search, stream resolution,
    metadata caching, trending/discovery, and simple recommendations.
    """

    # Common yt-dlp options shared across all extractions.
    _BASE_OPTS = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "extract_flat": False,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "socket_timeout": 15,
        "source_address": "0.0.0.0",
        # Prefer m4a (AAC) since it's broadly seekable/streamable in <audio>
        # tags across browsers without extra transcoding, while still HQ.
        "format": (
            "bestaudio[ext=m4a][abr<=256]/"
            "bestaudio[ext=m4a]/"
            "bestaudio[acodec^=mp4a]/"
            "bestaudio/best"
        ),
        "extractor_args": {
            "youtube": {
                # 'android'/'ios' clients are far less likely to be throttled
                # or blocked than 'web', and usually return direct googlevideo
                # URLs that work great in an HTML5 <audio> element.
                "player_client": ["android", "web"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._song_cache: dict[str, Song] = {}
        self._search_cache: dict[str, tuple[float, list[str]]] = {}  # query -> (ts, [song_ids])
        self._album_cache: dict[str, Album] = {}  # album_id -> Album
        self._song_to_album: dict[str, str] = {}  # song_id -> album_id (fast reverse lookup)
        self._load_persistent_cache()
        self._load_album_cache()
        log.info(
            "MusicEngine initialized. %d songs, %d albums preloaded from disk cache.",
            len(self._song_cache), len(self._album_cache),
        )

    # ---------------------------------------------------------------- I/O
    #
    # SERVERLESS NOTE: on Vercel, each function instance starts with an
    # empty in-memory `_song_cache`/`_album_cache`. Rather than trying to
    # bulk-load "everything" up front (which doesn't scale and isn't how
    # Redis is meant to be used), we now cache per-song and per-album under
    # individual Redis keys, fetched lazily the first time this instance
    # needs them (see _cache_song / _persist_album, and the read-through
    # helpers get_song_metadata / get_album_by_id below). This means:
    #   - First request for a song on a cold instance -> yt-dlp resolve.
    #   - Every subsequent request for that song, from ANY instance,
    #     anywhere -> a single fast Redis GET, no yt-dlp call at all.
    # The old bulk load/persist-everything-to-one-file methods are kept as
    # harmless no-ops (still called from __init__) so the rest of the class
    # doesn't need restructuring.
    def _load_persistent_cache(self):
        pass  # replaced by lazy, per-key Redis reads (see _cache_song)

    def _persist_cache(self):
        pass  # replaced by lazy, per-key Redis writes (see _cache_song)

    def _load_album_cache(self):
        pass  # replaced by lazy, per-key Redis reads (see get_album_by_id)

    def _persist_album_cache(self):
        pass  # replaced by lazy, per-key Redis writes (see _cache_album)

    @staticmethod
    def _song_redis_key(song_id: str) -> str:
        return f"sujal:song:{song_id}"

    @staticmethod
    def _album_redis_key(album_id: str) -> str:
        return f"sujal:album:{album_id}"

    def _redis_get_song(self, song_id: str) -> Optional["Song"]:
        if not _store:
            return None
        data = _store.kv_get_json(self._song_redis_key(song_id))
        if not data:
            return None
        try:
            data.pop("stream_url", None)  # never trust a cached stream URL -- may have expired
            data["stream_fetched_at"] = 0.0
            return Song(**data)
        except Exception:
            return None

    def _redis_put_song(self, song: "Song"):
        if not _store:
            return
        try:
            data = {**asdict(song), "stream_url": None}
            _store.kv_set_json(
                self._song_redis_key(song.id), data, ex=METADATA_TTL_SECONDS
            )
        except Exception as e:
            log.warning("redis put song failed for %s: %s", song.id, e)

    def _redis_get_album(self, album_id: str) -> Optional["Album"]:
        if not _store:
            return None
        data = _store.kv_get_json(self._album_redis_key(album_id))
        if not data:
            return None
        try:
            return Album(**data)
        except Exception:
            return None

    def _redis_put_album(self, album: "Album"):
        if not _store:
            return
        try:
            _store.kv_set_json(
                self._album_redis_key(album.id), asdict(album), ex=ALBUM_TTL_SECONDS
            )
        except Exception as e:
            log.warning("redis put album failed for %s: %s", album.id, e)

    def _persist_album_cache_async(self, album: Optional["Album"] = None):
        """
        Write-through to Redis for a single album (or, if not given, every
        album currently in memory -- used only as a fallback for older call
        sites). Run in a background thread so the HTTP response isn't held
        up by the Redis round-trip.
        """
        if album is not None:
            threading.Thread(target=self._redis_put_album, args=(album,), daemon=True).start()
            return
        with self._lock:
            albums = list(self._album_cache.values())
        for a in albums:
            threading.Thread(target=self._redis_put_album, args=(a,), daemon=True).start()

    # ----------------------------------------------------------- internals
    def _ydl(self, extra_opts: Optional[dict] = None) -> YoutubeDL:
        opts = dict(self._BASE_OPTS)
        if extra_opts:
            opts.update(extra_opts)
        return YoutubeDL(opts)

    def _song_from_info(self, info: dict) -> Song:
        vid = info.get("id")
        raw_title = info.get("title") or "Unknown Title"
        uploader = info.get("uploader") or info.get("channel") or ""
        title, artist = _clean_title_artist(raw_title, uploader)
        duration = int(info.get("duration") or 0)
        poster = _best_thumbnail(info.get("thumbnails") or []) or info.get("thumbnail", "")
        if not poster and vid:
            poster = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

        stream_url = info.get("url")
        abr = info.get("abr")
        ext = info.get("ext", "m4a")

        # yt-dlp surfaces YouTube Music release metadata on many extractions:
        # `album` (name), `track` (clean track title), `artists`/`artist`,
        # `track_number`, `release_year`. Not always present for plain
        # YouTube (non-Music) uploads, in which case these stay None and the
        # album resolver falls back to a YT Music search.
        album_name = info.get("album") or None
        track_number = info.get("track_number")
        if album_name:
            artists_list = info.get("artists") or ([artist] if artist else [])
            album_artist = (artists_list[0] if artists_list else artist) or artist
            album_id = _slugify_album_key(album_artist, album_name)
            if info.get("track"):
                title = info["track"]
        else:
            album_id = None

        song = Song(
            id=vid,
            title=title,
            artist=artist,
            duration=duration,
            duration_str=_fmt_duration(duration),
            poster=poster,
            stream_url=stream_url,
            stream_fetched_at=time.time() if stream_url else 0.0,
            audio_format=ext,
            bitrate=abr,
            view_count=info.get("view_count"),
            album_id=album_id,
            album_name=album_name,
            track_number=track_number,
        )
        return song

    def _cache_song(self, song: Song):
        with self._lock:
            existing = self._song_cache.get(song.id) or self._redis_get_song(song.id)
            if existing and not song.stream_url:
                song.stream_url = existing.stream_url
                song.stream_fetched_at = existing.stream_fetched_at
            # Never let a partial/flat re-fetch erase album info we already
            # resolved for this song (e.g. via full extraction or the album
            # resolver itself, which is a stronger signal than search flat entries).
            if existing and existing.album_id and not song.album_id:
                song.album_id = existing.album_id
                song.album_name = existing.album_name
                song.track_number = song.track_number or existing.track_number
            self._song_cache[song.id] = song
            if song.album_id:
                self._song_to_album[song.id] = song.album_id
        # Write-through to Redis so other/future instances skip yt-dlp too.
        self._redis_put_song(song)

    # ------------------------------------------------------------- search
    def search(self, query: str, limit: int = 20) -> list[Song]:
        """
        Search YouTube for `query` and return lightweight Song objects
        (metadata + poster, stream_url resolved lazily on play for speed).
        """
        query = (query or "").strip()
        if not query:
            return []

        cache_key = f"{query.lower()}::{limit}"
        with self._lock:
            cached = self._search_cache.get(cache_key)
        if cached and (time.time() - cached[0] < 60 * 30):
            with self._lock:
                return [self._song_cache[sid] for sid in cached[1] if sid in self._song_cache]

        search_term = f"ytsearch{limit}:{query} audio"
        log.info("Searching YouTube for: %r", query)
        try:
            with self._ydl({"extract_flat": "in_playlist"}) as ydl:
                result = ydl.extract_info(search_term, download=False)
        except Exception as e:
            log.error("Search failed for %r: %s", query, e)
            return []

        entries = (result or {}).get("entries") or []
        songs: list[Song] = []
        for entry in entries:
            if not entry:
                continue
            vid = entry.get("id")
            if not vid:
                continue
            # flat search entries lack full thumbnails sometimes; normalize
            raw_title = entry.get("title") or "Unknown Title"
            uploader = entry.get("uploader") or entry.get("channel") or ""
            title, artist = _clean_title_artist(raw_title, uploader)
            duration = int(entry.get("duration") or 0)
            thumbs = entry.get("thumbnails") or []
            poster = _best_thumbnail(thumbs) or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

            # Flat search entries occasionally carry album metadata too
            # (YT Music-sourced results) -- grab it opportunistically so we
            # can group tracks without a full per-video extraction.
            album_name = entry.get("album") or None
            album_id = None
            if album_name:
                artists_list = entry.get("artists") or ([artist] if artist else [])
                album_artist = (artists_list[0] if artists_list else artist) or artist
                album_id = _slugify_album_key(album_artist, album_name)

            song = Song(
                id=vid,
                title=title,
                artist=artist,
                duration=duration,
                duration_str=_fmt_duration(duration),
                poster=poster,
                view_count=entry.get("view_count"),
                album_id=album_id,
                album_name=album_name,
                track_number=entry.get("track_number"),
            )
            self._cache_song(song)
            songs.append(song)

        with self._lock:
            self._search_cache[cache_key] = (time.time(), [s.id for s in songs])

        self._persist_cache_async()
        return songs

    def _persist_cache_async(self):
        threading.Thread(target=self._persist_cache, daemon=True).start()

    # --------------------------------------------------------- resolution
    def get_stream_url(self, video_id: str) -> Optional[Song]:
        """
        Resolve (or re-resolve if stale) the direct playable stream URL for
        a given video id, returning the fully-populated Song. This is the
        function server.py calls right before playback so the URL is fresh.
        """
        with self._lock:
            cached = self._song_cache.get(video_id)

        needs_fetch = (
            cached is None
            or not cached.stream_url
            or (time.time() - cached.stream_fetched_at) > STREAM_URL_TTL_SECONDS
        )

        if not needs_fetch:
            return cached

        url = f"https://www.youtube.com/watch?v={video_id}"
        log.info("Resolving stream for video_id=%s", video_id)
        try:
            with self._ydl() as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            log.error("Failed to resolve stream for %s: %s", video_id, e)
            return cached  # return whatever we had (maybe just metadata, no stream)

        fresh = self._song_from_info(info)
        # Preserve any nicer cleaned title/artist we might already have cached
        if cached:
            fresh.genre_hint = cached.genre_hint
            if cached.album_id and not fresh.album_id:
                fresh.album_id = cached.album_id
                fresh.album_name = cached.album_name
                fresh.track_number = fresh.track_number or cached.track_number
        self._cache_song(fresh)
        self._persist_cache_async()
        return fresh

    def get_song_metadata(self, video_id: str) -> Optional[Song]:
        with self._lock:
            hit = self._song_cache.get(video_id)
        if hit:
            return hit
        # Not in this instance's memory yet -- check Redis before giving up
        # (caller falls back to a full yt-dlp resolve only if this is None).
        redis_hit = self._redis_get_song(video_id)
        if redis_hit:
            with self._lock:
                self._song_cache[video_id] = redis_hit
        return redis_hit

    def bulk_get(self, video_ids: list[str]) -> list[Song]:
        found: list[Song] = []
        missing: list[str] = []
        with self._lock:
            for v in video_ids:
                if v in self._song_cache:
                    found.append(self._song_cache[v])
                else:
                    missing.append(v)
        for v in missing:
            redis_hit = self._redis_get_song(v)
            if redis_hit:
                with self._lock:
                    self._song_cache[v] = redis_hit
                found.append(redis_hit)
        # Preserve caller's requested order
        by_id = {s.id: s for s in found}
        return [by_id[v] for v in video_ids if v in by_id]

    # ------------------------------------------------------------ trending
    def trending(self, genre: Optional[str] = None, limit: int = 24) -> list[Song]:
        """
        "Preload" style trending/discovery feed. If genre is given, biases
        the search query toward it (used for onboarding preferences).
        """
        query = f"{genre} popular songs 2026" if genre else "top trending songs 2026"
        return self.search(query, limit=limit)

    # -------------------------------------------------------- recommender
    GENRE_SEED_QUERIES = {
        "pop": ["pop hits", "top 40 pop"],
        "hiphop": ["hip hop hits", "rap bangers"],
        "rock": ["rock anthems", "classic rock hits"],
        "electronic": ["edm hits", "house music mix"],
        "bollywood": ["bollywood hits", "hindi songs 2026"],
        "lofi": ["lofi chill beats", "study lofi"],
        "jazz": ["smooth jazz", "jazz classics"],
        "classical": ["classical masterpieces", "piano classics"],
        "indie": ["indie hits", "indie folk"],
        "rnb": ["r&b hits", "soul classics"],
    }

    def recommendations_for_genres(self, genres: list[str], limit_per_genre: int = 8) -> list[Song]:
        """Used right after onboarding to build the user's first home feed."""
        results: list[Song] = []
        seen_ids: set[str] = set()
        for g in genres:
            queries = self.GENRE_SEED_QUERIES.get(g.lower(), [f"{g} music hits"])
            for q in queries[:1]:
                for song in self.search(q, limit=limit_per_genre):
                    if song.id not in seen_ids:
                        song.genre_hint = g
                        seen_ids.add(song.id)
                        results.append(song)
        random.shuffle(results)
        return results

    def related_songs(self, video_id: str, limit: int = 15) -> list[Song]:
        """
        Autoplay-next / "Up Next" queue generator.
        Strategy: use the currently cached song's cleaned artist + a keyword
        from its title to find similar tracks, excluding the seed itself and
        recently repeated artists-only spam.
        """
        seed = self.get_song_metadata(video_id)
        if not seed:
            seed = self.get_stream_url(video_id)
        if not seed:
            return self.trending(limit=limit)

        candidates: list[Song] = []
        seen_ids = {video_id}

        # 1. More from the same artist
        for song in self.search(f"{seed.artist} songs", limit=8):
            if song.id not in seen_ids:
                candidates.append(song)
                seen_ids.add(song.id)

        # 2. Similar vibe using genre hint or a generic "mix" query
        mix_query = f"{seed.artist} {seed.title} mix" if seed.genre_hint is None else f"{seed.genre_hint} music mix"
        for song in self.search(mix_query, limit=10):
            if song.id not in seen_ids:
                candidates.append(song)
                seen_ids.add(song.id)

        random.shuffle(candidates)
        return candidates[:limit]

    # ---------------------------------------------------------------
    # ALBUM ENGINE
    # ---------------------------------------------------------------
    # Spotify-style "click a song -> open its real album with every track
    # from that release, loaded and playable" behaviour. Strategy, cheapest
    # to most expensive:
    #
    #   1. Song already has album_id cached (from a prior full extraction
    #      or YT Music-flavoured search result) -> just look up the Album.
    #   2. No album_id yet -> do a full (non-flat) extraction of the video
    #      to read its real `album`/`artists`/`track` metadata, which
    #      YouTube attaches to Content-ID music uploads and YT Music tracks.
    #   3. Still nothing (plain YouTube upload with no Music metadata) ->
    #      search YouTube Music directly for "<artist> <album-guess> album"
    #      style queries and try to detect an actual album playlist page.
    #   4. Total failure -> build a "synthetic" pseudo-album (marked
    #      is_synthetic=True) out of the seed track + its related songs, so
    #      the UI never shows an empty state. This preserves the previous
    #      behaviour as a graceful fallback rather than removing it.
    # ---------------------------------------------------------------

    def get_album_for_song(self, video_id: str, limit: int = 50) -> Optional[Album]:
        """Resolve the real album a track belongs to, given its video id."""
        song = self.get_song_metadata(video_id) or self.get_stream_url(video_id)
        if not song:
            return None

        # Step 1: already known
        with self._lock:
            known_album_id = self._song_to_album.get(video_id) or song.album_id
        if known_album_id:
            album = self.get_album_by_id(known_album_id, limit=limit)
            if album and album.track_ids:
                return album

        # Step 2: full extraction to pick up album/artists/track fields that
        # flat search doesn't carry.
        if not song.album_name:
            resolved = self.get_stream_url(video_id)  # forces full extraction
            if resolved and resolved.album_id:
                song = resolved

        if song.album_id and song.album_name:
            album = self._resolve_real_album(song)
            if album and len(album.track_ids) > 1:
                return album

        # Step 3: YouTube Music album search using cleaned artist/title as a
        # heuristic guess (handles uploads that never carried Music metadata
        # but still clearly belong to a known release).
        guessed = self._search_album_by_artist_title(song)
        if guessed and len(guessed.track_ids) > 1:
            return guessed

        # Step 4: graceful synthetic fallback -- never leave the user with
        # a blank "album" page.
        return self._build_synthetic_album(song, limit=limit)

    def get_album_by_id(self, album_id: str, limit: int = 50) -> Optional[Album]:
        with self._lock:
            cached = self._album_cache.get(album_id)
        if not cached:
            cached = self._redis_get_album(album_id)
            if cached:
                with self._lock:
                    self._album_cache[album_id] = cached
        fresh_enough = cached and (time.time() - cached.fetched_at) < ALBUM_TTL_SECONDS
        if fresh_enough and cached.track_ids:
            return cached
        if cached and not cached.is_synthetic:
            # try to refresh a real album's tracklist; on failure keep old data
            refreshed = self._fetch_album_tracklist(cached, limit=limit)
            if refreshed:
                return refreshed
            return cached
        return cached

    def _resolve_real_album(self, song: Song) -> Optional[Album]:
        """
        Given a Song that already has album_name (+ album_id) populated,
        find/build the Album by searching YT Music for that exact album and
        pulling its full ordered tracklist.
        """
        album_id = song.album_id
        with self._lock:
            existing = self._album_cache.get(album_id)
        if existing and existing.track_ids and (time.time() - existing.fetched_at) < ALBUM_TTL_SECONDS:
            return existing

        album = Album(
            id=album_id,
            title=song.album_name,
            artist=song.artist,
            poster=song.poster,
        )
        return self._fetch_album_tracklist(album, seed_song=song)

    def _fetch_album_tracklist(self, album: Album, seed_song: Optional[Song] = None, limit: int = 50) -> Optional[Album]:
        """
        Runs the actual YouTube Music search for `"<artist> <album>" album`
        and extracts every track it can find, in release order where
        possible. This is what makes the album view genuinely load *all*
        songs of the clicked album, not just the one song + guesses.
        """
        query = f"{album.artist} {album.title} album"
        log.info("Resolving album tracklist for %r by %r", album.title, album.artist)

        candidate_songs: list[Song] = []
        seen_ids: set[str] = set()

        def _add_candidate(s: Optional[Song]):
            if s and s.id not in seen_ids:
                candidate_songs.append(s)
                seen_ids.add(s.id)

        try:
            # First try to find a YouTube Music "album" / "playlist" result
            # directly -- yt-dlp can expand actual YT Music album pages
            # (browse ids starting with MPRE/OLAK5uy...) into their full
            # ordered track entries when given the right URL shape.
            probe_query = f"https://music.youtube.com/search?q={query.replace(' ', '+')}"
            with self._ydl({"extract_flat": "in_playlist", "playlistend": limit}) as ydl:
                info = ydl.extract_info(probe_query, download=False)
            entries = (info or {}).get("entries") or []
            # If the search itself surfaced an actual album/playlist entry,
            # yt-dlp nests entries-of-entries; flatten one level defensively.
            flat_entries = []
            for e in entries:
                if not e:
                    continue
                if e.get("_type") == "playlist" and e.get("entries"):
                    flat_entries.extend(e["entries"])
                else:
                    flat_entries.append(e)

            for idx, e in enumerate(flat_entries[:limit], start=1):
                vid = e.get("id")
                if not vid:
                    continue
                raw_title = e.get("title") or "Unknown Title"
                uploader = e.get("uploader") or e.get("channel") or album.artist
                title, artist = _clean_title_artist(raw_title, uploader)
                duration = int(e.get("duration") or 0)
                poster = _best_thumbnail(e.get("thumbnails") or []) or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
                s = Song(
                    id=vid,
                    title=title,
                    artist=artist or album.artist,
                    duration=duration,
                    duration_str=_fmt_duration(duration),
                    poster=poster,
                    view_count=e.get("view_count"),
                    album_id=album.id,
                    album_name=album.title,
                    track_number=e.get("track_number") or idx,
                )
                self._cache_song(s)
                _add_candidate(s)
        except Exception as e:
            log.warning("Direct album probe failed for %r: %s", query, e)

        # Fallback / supplement: plain search for the album name, which
        # reliably surfaces the individual uploaded tracks of most releases
        # even when a clean YT Music playlist page isn't extractable.
        if len(candidate_songs) < 2:
            for s in self.search(f"{album.artist} {album.title}", limit=limit):
                # Only keep songs that are plausibly from this album: either
                # explicitly tagged with this album_id, or same-artist songs
                # if we found nothing better (keeps it from being empty).
                if s.album_id == album.id or (not s.album_id and s.artist.lower() == album.artist.lower()):
                    _add_candidate(s)

        if seed_song:
            _add_candidate(seed_song)
            # keep the seed track first if it wasn't already at the front
            if candidate_songs and candidate_songs[0].id != seed_song.id:
                candidate_songs = [seed_song] + [c for c in candidate_songs if c.id != seed_song.id]

        if not candidate_songs:
            return None

        # Tag every candidate track with this album so future lookups
        # (get_album_for_song) short-circuit straight to step 1, and collect
        # the final ordered id list.
        ordered_ids: list[str] = []
        with self._lock:
            for s in candidate_songs:
                cached_song = self._song_cache.get(s.id, s)
                cached_song.album_id = album.id
                if not cached_song.album_name:
                    cached_song.album_name = album.title
                self._song_cache[s.id] = cached_song
                self._song_to_album[s.id] = album.id
                ordered_ids.append(s.id)

        album.track_ids = ordered_ids
        album.fetched_at = time.time()
        with self._lock:
            self._album_cache[album.id] = album
        self._persist_album_cache_async(album)
        self._persist_cache_async()
        return album

    def _search_album_by_artist_title(self, song: Song) -> Optional[Album]:
        """
        Heuristic album search for tracks with no album metadata at all --
        guesses that the track's own title might *be* the album/EP name
        (common for singles) and searches YT Music for a same-named release
        by the same artist. Cheap, best-effort, skipped quickly on failure.
        """
        if not song.artist or song.artist == "Unknown Artist":
            return None
        guess_id = _slugify_album_key(song.artist, song.title)
        if not guess_id:
            return None

        with self._lock:
            cached = self._album_cache.get(guess_id)
        if cached and cached.track_ids:
            return cached

        album = Album(id=guess_id, title=song.title, artist=song.artist, poster=song.poster)
        result = self._fetch_album_tracklist(album, seed_song=song)
        # Only accept this guess if it actually produced a multi-track
        # release; otherwise it's just the single song and step 4 (synthetic)
        # is a more honest fallback.
        if result and len(result.track_ids) > 1:
            return result
        return None

    def _build_synthetic_album(self, seed: Song, limit: int = 30) -> Album:
        """
        Best-effort "pseudo-album" for tracks where no real album grouping
        could be found anywhere -- preserves the previous app behaviour
        (seed track + related songs) so the expanded view is never empty,
        but is clearly flagged `is_synthetic` so the frontend can label it
        appropriately (e.g. "Similar tracks" instead of "Album").
        """
        album_id = f"single_{seed.id}"
        with self._lock:
            existing = self._album_cache.get(album_id)
        if existing and existing.track_ids and (time.time() - existing.fetched_at) < ALBUM_TTL_SECONDS:
            return existing

        track_ids = [seed.id]
        seen = {seed.id}
        for s in self.related_songs(seed.id, limit=limit - 1):
            if s.id not in seen:
                track_ids.append(s.id)
                seen.add(s.id)
        if len(track_ids) < 6:
            for s in self.trending(limit=20):
                if s.id not in seen:
                    track_ids.append(s.id)
                    seen.add(s.id)

        album = Album(
            id=album_id,
            title=seed.title,
            artist=seed.artist,
            poster=seed.poster,
            track_ids=track_ids,
            fetched_at=time.time(),
            is_synthetic=True,
        )
        with self._lock:
            self._album_cache[album_id] = album
        self._persist_album_cache_async(album)
        return album

    # ---------------------------------------------------------------
    # ALBUM DISCOVERY -- powers the *homepage* grids ("Made for you",
    # "Popular right now", genre rows). This is distinct from
    # get_album_for_song() above: that resolves the one album belonging to
    # an already-known track; these methods go the other way -- they find
    # a set of real *albums* directly (each with its own cover, title, and
    # artist) so the home page can show actual album tiles instead of
    # individual song cards mislabeled as albums.
    # ---------------------------------------------------------------
    ALBUM_DISCOVERY_QUERIES = {
        None: ["top albums 2026", "best albums of the year"],
        "pop": ["top pop albums 2026", "best pop albums"],
        "hiphop": ["top hip hop albums 2026", "best rap albums"],
        "rock": ["top rock albums", "classic rock albums"],
        "electronic": ["top edm albums", "best electronic albums"],
        "bollywood": ["top bollywood albums 2026", "best hindi albums"],
        "lofi": ["best lofi albums", "lofi album compilations"],
        "jazz": ["classic jazz albums", "best jazz albums"],
        "classical": ["classical music albums", "best classical albums"],
        "indie": ["top indie albums", "best indie albums"],
        "rnb": ["top r&b albums", "best soul albums"],
    }

    def _discover_albums_for_query(self, query: str, limit: int, seen_album_ids: set[str]) -> list[Album]:
        """
        Runs one discovery query, and for every distinct album name it
        surfaces, resolves the *real* full tracklist via the same
        _fetch_album_tracklist() machinery used by the single-song album
        resolver -- so a homepage album tile opens to a genuinely complete,
        already-cached tracklist rather than a single placeholder track.
        """
        albums: list[Album] = []
        candidates = self.search(query, limit=limit * 3)
        for song in candidates:
            if len(albums) >= limit:
                break
            # Prefer songs that already carry real album metadata (from a
            # full extraction or a YT-Music-flavoured search result).
            if song.album_id and song.album_name:
                if song.album_id in seen_album_ids:
                    continue
                with self._lock:
                    cached = self._album_cache.get(song.album_id)
                if cached and cached.track_ids:
                    album = cached
                else:
                    album = self._resolve_real_album(song)
            else:
                # No album metadata on this particular song -- take a
                # best-effort guess using its own title as a candidate
                # album/EP name by the same artist. Skipped quickly if it
                # doesn't pan out into a genuine multi-track release.
                album = self._search_album_by_artist_title(song)

            if not album or len(album.track_ids) < 2 or album.id in seen_album_ids:
                continue
            seen_album_ids.add(album.id)
            albums.append(album)
        return albums

    def discover_albums(self, query: str, limit: int = 12) -> list[Album]:
        """General-purpose album search, e.g. for a future /api/albums/search."""
        return self._discover_albums_for_query(query, limit, set())

    def trending_albums(self, genre: Optional[str] = None, limit: int = 12) -> list[Album]:
        """
        Homepage "Popular right now" row, but built from real albums instead
        of individual songs. Tries a couple of query phrasings and keeps
        going until `limit` distinct real albums are found (or gives up
        gracefully with however many it managed).
        """
        queries = self.ALBUM_DISCOVERY_QUERIES.get(
            (genre or "").lower() if genre else None,
            [f"{genre} albums 2026", f"best {genre} albums"] if genre else self.ALBUM_DISCOVERY_QUERIES[None],
        )
        seen: set[str] = set()
        albums: list[Album] = []
        for q in queries:
            if len(albums) >= limit:
                break
            albums.extend(self._discover_albums_for_query(q, limit - len(albums), seen))
        random.shuffle(albums)
        return albums[:limit]

    def albums_for_genres(self, genres: list[str], limit_per_genre: int = 6) -> list[Album]:
        """Used to build a personalized "Made for you" row of real albums
        after onboarding, mirroring recommendations_for_genres() but
        returning Album objects instead of individual Song objects."""
        results: list[Album] = []
        seen: set[str] = set()
        for g in genres:
            queries = self.ALBUM_DISCOVERY_QUERIES.get(g.lower(), [f"{g} albums 2026"])
            for q in queries[:1]:
                found = self._discover_albums_for_query(q, limit_per_genre, seen)
                results.extend(found)
        random.shuffle(results)
        return results

    # ------------------------------------------------------------ housekeeping
    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_songs": len(self._song_cache),
                "cached_searches": len(self._search_cache),
                "cached_albums": len(self._album_cache),
            }


# --------------------------------------------------------------------------
# Module-level singleton -- server.py imports and reuses this one instance
# so the cache is genuinely shared across all requests/users (preloading).
# --------------------------------------------------------------------------
engine = MusicEngine()


# --------------------------------------------------------------------------
# CLI smoke test:  python music.py "song name"
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Blinding Lights The Weeknd"
    print(f"\nSearching for: {q}\n" + "-" * 50)
    found = engine.search(q, limit=5)
    for i, s in enumerate(found, 1):
        print(f"{i}. {s.title} — {s.artist} [{s.duration_str}]")
        print(f"   poster: {s.poster}")

    if found:
        print("\nResolving stream for first result...")
        resolved = engine.get_stream_url(found[0].id)
        print("Stream URL (first 100 chars):", (resolved.stream_url or "")[:100], "...")

        print("\nRelated / Up Next songs:")
        for s in engine.related_songs(found[0].id, limit=5):
            print(" -", s.title, "—", s.artist)

        print("\nResolving full ALBUM for this track (Spotify-style expand)...")
        album = engine.get_album_for_song(found[0].id, limit=30)
        if album:
            tag = "(synthetic / best-effort)" if album.is_synthetic else "(real album)"
            print(f"Album: {album.title} — {album.artist} {tag}")
            for i, s in enumerate(engine.bulk_get(album.track_ids), 1):
                print(f"  {i:>2}. {s.title} — {s.artist} [{s.duration_str}]")
        else:
            print("Could not resolve an album for this track.")
