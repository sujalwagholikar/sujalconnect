# SujalConnect — Vercel deployment

This project has been restructured to deploy on Vercel (Fluid Compute,
Python/FastAPI runtime).

## What changed from the original project

| Before | Now | Why |
|---|---|---|
| `server.py` at project root | `api/index.py` | Vercel's Python runtime looks for a FastAPI `app` at `api/index.py` (or a few other supported entrypoints). |
| `index.html`, `album.html`, `playlist.html` next to `server.py` | Moved into `public/` | Vercel serves `public/` straight from its CDN — faster than routing HTML through the Python function, and it's automatic (zero config). |
| `data/users.json` (local file) | Upstash Redis, via `store.py` | Vercel functions don't have a durable, shared local filesystem. A local JSON file resets on every cold start and corrupts under concurrent writes. Redis fixes both. |
| `cache/*.json` (local files, in `music.py`) | Per-key Redis cache (`store.py`) + `/tmp` as a same-instance speedup only | Same reasoning — `/tmp` is fine as a scratch pad but is never shared across instances or guaranteed to survive between requests. |
| `app.mount("/static", ...)` with `mkdir()` on import | Removed (unused; `public/` covers static needs) | The project's filesystem is read-only at runtime except `/tmp`; a `mkdir()` outside `/tmp` would fail. |

No frontend code changed. No API route paths changed. No feature was removed.

## One-time setup

### 1. Push this project to a Git repo (GitHub/GitLab/Bitbucket)

Vercel deploys from a connected Git repository (recommended) or via the CLI.

### 2. Import the project into Vercel

- Go to vercel.com/new and import the repo.
- Framework preset: Vercel will auto-detect Python/FastAPI. No build command needed.
- Leave the root directory as the repo root (where `vercel.json` lives).

### 3. Add Upstash Redis (for persistent users/likes/history/song-cache)

1. In your Vercel project dashboard, go to Storage -> Marketplace Database Providers -> find Upstash -> Redis.
2. Create a new Redis database (the free tier — 256MB, 500k commands/month — is enough to start).
3. Connect it to this project. Vercel automatically injects two environment variables into your deployment:
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`
4. Redeploy (or it will auto-redeploy after the integration is connected).

If you skip this step, the app still runs — `store.py` falls back to a
local `/tmp` file automatically — but user accounts, likes, and history will
reset unpredictably across requests. Fine for a quick demo, not for real use.

### 4. Deploy

If using Git integration: push to your main branch, Vercel deploys automatically.

If using the CLI instead:
```bash
npm i -g vercel
vercel login
vercel        # preview deploy
vercel --prod # production deploy
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: point at real Upstash Redis locally too, otherwise it uses
# the /tmp fallback automatically.
export UPSTASH_REDIS_REST_URL="..."
export UPSTASH_REDIS_REST_TOKEN="..."

uvicorn api.index:app --reload --port 8000
```

Or with the Vercel CLI (closer to production behavior):
```bash
vercel dev
```

## Things worth knowing before you get real traffic

- `/api/proxy-audio/{id}` streams audio through the function. This works
  on Fluid Compute (long-lived streaming responses are supported), but
  every second of playback is billed function execution time. At low
  traffic this is a non-issue; at scale, consider a caching/edge layer in
  front of it.
- yt-dlp calls out to YouTube on every stream resolve (stream URLs expire
  every few hours by design — this can't be cached away). This is the
  slowest part of any request path; `maxDuration` is set to 60s in
  `vercel.json` to give it headroom. Search/metadata results ARE cached in
  Redis and skip yt-dlp entirely on a cache hit.
- This relies on scraping YouTube via yt-dlp, which is against YouTube's
  Terms of Service and can break at any time if YouTube changes its
  player/API internals — that risk exists independent of which host you
  deploy to.
