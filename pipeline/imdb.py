"""IMDb ratings from IMDb's bulk dataset (one file, no API key, refreshed daily by IMDb).

    python3 pipeline/imdb.py     # download if the cached copy is over a day old

Used by build.py for the quality score: IMDb has far more votes than TMDB, and
TMDB ratings of new releases run high (about +0.9 in their first four months,
+0.6 for the rest of the first year, measured September 2026).
Non-commercial use: https://developer.imdb.com/non-commercial-datasets/
"""
import csv
import gzip
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "cache" / "title.ratings.tsv.gz"
URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
MAX_AGE = 86400


def fetch():
    if FILE.exists() and time.time() - FILE.stat().st_mtime < MAX_AGE:
        return
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".part")
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(URL, tmp)
            tmp.replace(FILE)
            return
        except OSError:
            time.sleep(2 + attempt * 5)
    if not FILE.exists():
        raise RuntimeError(f"could not download {URL}")
    print("IMDb download failed; using the cached copy")


def ratings():
    """{imdb id: (average rating, number of votes)}"""
    fetch()
    with gzip.open(FILE, "rt", newline="") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        return {t: (float(r), int(n)) for t, r, n in rows}


if __name__ == "__main__":
    print(f"{len(ratings())} titles rated")
