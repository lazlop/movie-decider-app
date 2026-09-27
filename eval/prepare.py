"""One-time data for eval/sim.py, from MovieLens 25M (raw/) and the training TMDB cache.

    .venv/bin/python eval/prepare.py

Writes eval/cache/ (git-ignored):
  ratings.csv    600 MovieLens users with 300+ ratings of current pool films: user, tmdb, rating
  film_mean.csv  every film's average MovieLens rating over all users: tmdb, mean
  oof.npz        out-of-fold predicted taste vectors for every film MovieLens scored,
                 so the eval can pretend known films were predicted and check the result

Films are keyed by TMDB id, so the files stay valid as the weekly pool changes. Rerun
after a retrain (the taste space changes) or to sample users from a newer pool.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import features as F  # noqa: E402
import train as TR  # noqa: E402

OUT = ROOT / "eval" / "cache"
USERS = 600
MIN_RATED = 300
FOLDS = 5


def pool_ids():
    s = (ROOT / "docs" / "movies.js").read_text()
    data = json.loads(s[len("window.MOVIE_DATA="):s.rindex(";")])
    col = data["fields"].index("tmdb")
    return {r[col] for r in data["movies"]}


def ratings():
    links = pd.read_csv(TR.RAW / "links.csv").dropna(subset=["tmdbId"])
    tmdb_of = dict(zip(links.movieId, links.tmdbId.astype(int)))
    print("reading ratings...")
    r = pd.read_csv(TR.RAW / "ratings.csv", usecols=[0, 1, 2],
                    dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32})
    r["tmdb"] = r.movieId.map(tmdb_of)
    r = r.dropna(subset=["tmdb"]).astype({"tmdb": np.int64}).drop_duplicates(["userId", "tmdb"])
    r.groupby("tmdb").rating.mean().rename("mean").to_csv(OUT / "film_mean.csv")

    pool = r[r.tmdb.isin(pool_ids())]
    counts = pool.userId.value_counts()
    eligible = counts[counts >= MIN_RATED].index
    users = np.random.default_rng(1).choice(eligible, USERS, replace=False)
    pool[pool.userId.isin(users)].rename(columns={"userId": "user"})[["user", "tmdb", "rating"]] \
        .to_csv(OUT / "ratings.csv", index=False)
    print(f"{len(eligible)} users rated {MIN_RATED}+ pool films; kept {USERS}")


def out_of_fold():
    films, G, tag_names = TR.load()
    Z, _, _, _ = TR.targets(films, G, tag_names)
    ids = np.array([m["id"] for m in films])
    emb = np.load(TR.CACHE / "emb_train.npz")
    assert np.array_equal(emb["ids"], ids), "run pipeline/train.py first"
    emb = emb["emb"]
    meta = json.loads((TR.MODEL / "meta.json").read_text())
    kw_index = {k: i for i, k in enumerate(meta["keywords"])}
    fold = np.random.default_rng(0).integers(0, FOLDS, len(films))
    Zp = np.zeros_like(Z)
    for k in range(FOLDS):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        ftr = [films[i] for i in tr]
        prof = F.person_profiles(ftr, Z[tr])
        Xa = TR.build_X(ftr, emb[tr], kw_index, prof, Z[tr])
        Xt = TR.build_X([films[i] for i in te], emb[te], kw_index, prof, None)
        Zp[te] = TR.Ridge(meta["alpha"]).fit(Xa, Z[tr]).predict(Xt)
    r2 = 1 - ((Z - Zp) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum()
    np.savez_compressed(OUT / "oof.npz", tmdb=ids, zpred=Zp.astype(np.float32))
    print(f"out-of-fold predictions for {len(ids)} films (R² {r2:.2f})")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ratings()
    out_of_fold()
