#!/usr/bin/env python3
"""
Shadertoy public-API ingester for the OmniTrace shader-LM corpus.

Pulls every API-visible shader (authors who opted into "public + API"), caches the
raw JSON, and extracts prompt->code training records. Stdlib-only (urllib) so it runs
anywhere with zero installs. Polite by default: rate-limited, retry/backoff, resumable.

WHY THIS EXISTS
  The overnight run showed the bottleneck is DATA, not compute: 47.6M tokens (~22k
  shaders21k docs) is far too little for the from-scratch arm, and its 0/8 compile@1
  was partly a PROMPT-FORMAT MISMATCH. This tool (a) ~3-4x's the corpus via the API
  and (b) writes records in the SAME instruction format the eval prompts use
  ("// Shader: <name>") so the model actually learns prompt->code conditioning.

GET A KEY (one-time, ~30s, requires your Shadertoy login — I can't do this for you):
  1. Sign in at https://www.shadertoy.com
  2. Open https://www.shadertoy.com/myapps  -> "Create new app"
  3. Name it (e.g. "omnitrace-research"), copy the generated App key.
  4. export SHADERTOY_KEY=<key>     (or pass --key <key>)

USAGE
  python3 shadertoy_fetch.py --out ~/shader_data/api            # full pull
  python3 shadertoy_fetch.py --out ~/shader_data/api --limit 200 --workers 4   # smoke test
  # then build the training corpus (dedupes against shaders21k by code hash):
  python3 shadertoy_fetch.py --out ~/shader_data/api --extract-only \
      --dedupe-against ~/shader_data/codes

API ENDPOINTS (v1)
  list all ids : https://www.shadertoy.com/api/v1/shaders?key=KEY
  one shader   : https://www.shadertoy.com/api/v1/shaders/{id}?key=KEY
  search       : https://www.shadertoy.com/api/v1/shaders/query/{terms}?key=KEY
"""
import argparse, hashlib, json, os, sys, time, threading, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://www.shadertoy.com/api/v1"


# --------------------------------------------------------------------------- net
class RateLimiter:
    """Token-ish limiter: spaces requests at most `rps` per second across threads."""
    def __init__(self, rps):
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            sleep_for = self.next_t - now
            self.next_t = max(now, self.next_t) + self.min_interval
        if sleep_for > 0:
            time.sleep(sleep_for)


def get_json(url, limiter, retries=5, timeout=30):
    """GET with backoff on 429/5xx/transient errors. Returns parsed JSON or raises."""
    last = None
    for attempt in range(retries):
        limiter.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "omnitrace-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 30))  # exponential backoff, capped
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GET failed after {retries} retries: {url} ({last})")


# ---------------------------------------------------------------------- fetching
def fetch_id_list(key, limiter, out_dir):
    """Return all API-visible shader IDs (cached to ids.json for resume)."""
    cache = os.path.join(out_dir, "ids.json")
    if os.path.exists(cache):
        ids = json.load(open(cache))
        print(f"[ids] resumed {len(ids)} ids from cache", flush=True)
        return ids
    url = f"{API}/shaders?key={urllib.parse.quote(key)}"
    data = get_json(url, limiter)
    ids = data.get("Results", [])
    if not ids:
        raise SystemExit(f"[ids] API returned no results — bad key? raw: {str(data)[:200]}")
    json.dump(ids, open(cache, "w"))
    print(f"[ids] {data.get('Shaders', len(ids))} shaders available; saved {len(ids)} ids", flush=True)
    return ids


def fetch_one(sid, key, limiter, json_dir):
    """Download one shader's JSON to json_dir/{sid}.json (skip if present)."""
    path = os.path.join(json_dir, f"{sid}.json")
    if os.path.exists(path) and os.path.getsize(path) > 2:
        return "skip"
    url = f"{API}/shaders/{urllib.parse.quote(sid)}?key={urllib.parse.quote(key)}"
    try:
        data = get_json(url, limiter)
    except Exception as e:
        return f"err:{e}"
    if "Shader" not in data:
        return f"err:no-shader-field"  # often a privacy/permission case
    json.dump(data["Shader"], open(path, "w"))
    return "ok"


def fetch_all(key, ids, json_dir, workers, limiter, limit=0):
    os.makedirs(json_dir, exist_ok=True)
    todo = ids[:limit] if limit else ids
    counts = {"ok": 0, "skip": 0, "err": 0}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, s, key, limiter, json_dir): s for s in todo}
        for f in as_completed(futs):
            r = f.result()
            counts["ok" if r == "ok" else "skip" if r == "skip" else "err"] += 1
            done += 1
            if done % 200 == 0 or done == len(todo):
                print(f"[fetch] {done}/{len(todo)}  ok={counts['ok']} "
                      f"skip={counts['skip']} err={counts['err']}", flush=True)
    return counts


# -------------------------------------------------------------------- extraction
def code_of(shader):
    """Concatenate renderpass code (common buffers first, Image last) -> full GLSL."""
    passes = shader.get("renderpass", [])
    order = {"common": 0, "buffer": 1, "cubemap": 2, "sound": 3, "image": 4}
    passes = sorted(passes, key=lambda p: order.get(p.get("type", ""), 2))
    return "\n\n".join(p.get("code", "") for p in passes if p.get("code", "").strip())


def record_of(shader):
    """Build a prompt->code training record in the eval's instruction format.

    Format (matches eval prompt '// Shader: <name>' so the model learns to condition):
        // Shader: <name>
        // <description first line>
        // tags: a, b, c
        <glsl code>
    """
    info = shader.get("info", {})
    name = (info.get("name") or "").strip().replace("\n", " ")
    desc = (info.get("description") or "").strip().splitlines()
    desc = desc[0].strip() if desc else ""
    tags = ", ".join(t for t in info.get("tags", []) if t)[:200]
    code = code_of(shader)
    if not name or len(code) < 40:
        return None, None
    header = f"// Shader: {name}\n"
    if desc:
        header += f"// {desc[:200]}\n"
    if tags:
        header += f"// tags: {tags}\n"
    return code, header + code


def extract(json_dir, texts_dir, dedupe_dirs=None, min_len=40, min_likes=0):
    """Turn cached JSON into one .txt record per shader, deduped by code hash."""
    os.makedirs(texts_dir, exist_ok=True)
    seen = set()
    # Seed the dedupe set with existing corpora (e.g. shaders21k codes/).
    for d in (dedupe_dirs or []):
        if not d or not os.path.isdir(d):
            continue
        n = 0
        for fn in os.listdir(d):
            try:
                txt = open(os.path.join(d, fn), errors="replace").read()
                seen.add(hashlib.sha1(txt.encode("utf-8", "replace")).hexdigest())
                n += 1
            except Exception:
                pass
        print(f"[dedupe] seeded {n} hashes from {d}", flush=True)

    kept = dup = thin = 0
    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    for fn in files:
        try:
            shader = json.load(open(os.path.join(json_dir, fn)))
        except Exception:
            thin += 1
            continue
        if min_likes and int(shader.get("info", {}).get("likes", 0)) < min_likes:
            continue
        code, record = record_of(shader)
        if not record or len(code) < min_len:
            thin += 1
            continue
        h = hashlib.sha1(code.encode("utf-8", "replace")).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        open(os.path.join(texts_dir, fn.replace(".json", ".txt")), "w").write(record + "\n")
        kept += 1
    print(f"[extract] kept={kept} dup={dup} thin/bad={thin} -> {texts_dir}", flush=True)
    return kept


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description="Shadertoy API ingester for OmniTrace shader-LM")
    ap.add_argument("--key", default=os.environ.get("SHADERTOY_KEY", ""),
                    help="Shadertoy app key (or set SHADERTOY_KEY)")
    ap.add_argument("--out", default=os.path.expanduser("~/shader_data/api"),
                    help="output root (json/ + texts/ + ids.json)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rps", type=float, default=8.0, help="max requests/sec (be polite)")
    ap.add_argument("--limit", type=int, default=0, help="cap #shaders (0=all; for smoke tests)")
    ap.add_argument("--min-likes", type=int, default=0, help="quality filter on extract")
    ap.add_argument("--dedupe-against", action="append", default=[],
                    help="existing corpus dir(s) to dedupe code against (repeatable)")
    ap.add_argument("--extract-only", action="store_true",
                    help="skip fetching; just (re)extract texts/ from cached json/")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json_dir = os.path.join(args.out, "json")
    texts_dir = os.path.join(args.out, "texts")
    limiter = RateLimiter(args.rps)

    if not args.extract_only:
        if not args.key:
            sys.exit("ERROR: no key. Get one at https://www.shadertoy.com/myapps "
                     "then --key <key> or export SHADERTOY_KEY=<key>.")
        ids = fetch_id_list(args.key, limiter, args.out)
        t0 = time.time()
        counts = fetch_all(args.key, ids, json_dir, args.workers, limiter, args.limit)
        print(f"[fetch] done in {time.time()-t0:.0f}s: {counts}", flush=True)

    kept = extract(json_dir, texts_dir, args.dedupe_against, min_likes=args.min_likes)
    print(f"\nDONE. {kept} new prompt->code records in {texts_dir}\n"
          f"Next: concat with existing texts/, retokenize to .npy, retrain.", flush=True)


if __name__ == "__main__":
    main()
