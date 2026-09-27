"""TMDB fetching with an on-disk cache.

    python3 pipeline/tmdb.py training   # every MovieLens tag-genome film (for training)
    python3 pipeline/tmdb.py pool       # the current game pool: top films per year + now playing

Needs TMDB_API_KEY in the environment or in .env at the repo root.
Each movie is cached as cache/tmdb/<id>.json in a compact form.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "tmdb"
API = "https://api.themoviedb.org/3"
TV_MOVIE = 10770


def api_key():
    key = os.environ.get("TMDB_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TMDB_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("Set TMDB_API_KEY (environment or .env)")
    return key


KEY = api_key()


def get(path, **params):
    params["api_key"] = KEY
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 or e.code >= 500:
                time.sleep(1 + attempt * 2)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f"gave up on {path}")


def compact(d):
    crew = d.get("credits", {}).get("crew", [])
    cast = sorted(d.get("credits", {}).get("cast", []), key=lambda c: c.get("order", 99))
    return {
        "id": d["id"],
        "imdb": d.get("imdb_id") or "",
        "title": d.get("title") or "",
        "release": d.get("release_date") or "",
        "genres": [g["name"] for g in d.get("genres", [])],
        "overview": d.get("overview") or "",
        "tagline": d.get("tagline") or "",
        "keywords": [k["name"] for k in d.get("keywords", {}).get("keywords", [])],
        "directors": [[c["id"], c["name"]] for c in crew if c.get("job") == "Director"],
        "cast": [[c["id"], c["name"]] for c in cast[:5]],
        "votes": d.get("vote_count", 0),
        "score": d.get("vote_average", 0),
        "popularity": d.get("popularity", 0),
        "poster": d.get("poster_path") or "",
        "runtime": d.get("runtime") or 0,
        "lang": d.get("original_language") or "",
    }


def fetch_movie(tmdb_id, max_age_days=None):
    f = CACHE / f"{tmdb_id}.json"
    if f.exists():
        age = (time.time() - f.stat().st_mtime) / 86400
        if max_age_days is None or age < max_age_days:
            return json.loads(f.read_text())
    d = get(f"/movie/{tmdb_id}", append_to_response="keywords,credits")
    if d is None:
        return None
    m = compact(d)
    f.write_text(json.dumps(m))
    return m


def fetch_many(ids, max_age_days=None, workers=24):
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = list(dict.fromkeys(int(i) for i in ids))
    out, t0 = {}, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, m in enumerate(ex.map(lambda i: fetch_movie(i, max_age_days), ids), 1):
            if m:
                out[m["id"]] = m
            if n % 1000 == 0:
                print(f"  {n}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def pool_config():
    return json.loads((ROOT / "config.json").read_text())["pool"]


def per_year_quota(y, cfg):
    quota = 0
    for start, n in cfg["per_year"]:
        if y >= start:
            quota = n
    return round(quota * cfg.get("scale", 1))


def discover_pool():
    """Most-voted films for every release year, plus what's in theaters now."""
    cfg = pool_config()
    this_year = date.today().year
    picks = {}

    def year_ids(y):
        quota = per_year_quota(y, cfg)
        recent = y >= this_year - 1
        ids, page = [], 1
        while len(ids) < quota:
            r = get("/discover/movie", primary_release_year=y, page=page,
                    sort_by="popularity.desc" if y == this_year else "vote_count.desc",
                    **{"vote_count.gte": cfg["min_votes_recent"] if recent else cfg["min_votes"],
                       "with_runtime.gte": cfg.get("min_runtime", 0),
                       "without_genres": TV_MOVIE, "include_adult": "false"})
            ids += [m["id"] for m in (r or {}).get("results", [])]
            if not r or page >= min(r.get("total_pages", 0), 500):
                break
            page += 1
        return ids[:quota]

    with ThreadPoolExecutor(16) as ex:
        for ids in ex.map(year_ids, range(cfg["first_year"], this_year + 1)):
            for i in ids:
                picks[i] = False

    now_playing = set()
    for page in (1, 2, 3):
        r = get("/movie/now_playing", region="US", page=page)
        for m in (r or {}).get("results", []):
            if m.get("vote_count", 0) >= cfg["now_playing_min_votes"]:
                now_playing.add(m["id"])
                picks[m["id"]] = True
    for i in now_playing:
        picks[i] = True
    return picks


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pool"
    if mode == "training":
        import pandas as pd
        raw = ROOT / "raw" / "ml-25m"
        links = pd.read_csv(raw / "links.csv").dropna(subset=["tmdbId"])
        genome_ids = pd.read_csv(raw / "genome-scores.csv", usecols=["movieId"])["movieId"].unique()
        ids = links[links.movieId.isin(genome_ids)].tmdbId.astype(int).tolist()
        print(f"fetching {len(ids)} training films")
        got = fetch_many(ids)
        print(f"cached {len(got)}")
    elif mode == "pool":
        picks = discover_pool()
        print(f"pool: {len(picks)} films, {sum(picks.values())} now playing")
        # Recent films change fast (votes, keywords); refresh them weekly.
        got = fetch_many(picks, max_age_days=6)
        # Now-playing films skip discover's runtime filter; unknown runtimes (0) stay.
        short = {i for i in got if 0 < got[i]["runtime"] < pool_config().get("min_runtime", 0)}
        pool = [{"id": i, "nowPlaying": np_} for i, np_ in picks.items() if i in got and i not in short]
        if short:
            print(f"dropped {len(short)} films under the minimum runtime")
        (ROOT / "cache" / "pool.json").write_text(json.dumps(
            {"built": date.today().isoformat(), "films": pool}))
        print(f"wrote cache/pool.json ({len(pool)} films)")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
