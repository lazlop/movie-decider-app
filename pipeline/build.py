"""Build docs/movies.js from the current TMDB pool and the trained model.

    python3 pipeline/tmdb.py pool && .venv/bin/python pipeline/build.py

Films that MovieLens scored keep their real tag-genome profile; every other
film gets a profile predicted from its TMDB plot, keywords, genres, director
and leads (see train.py). This is what the weekly GitHub Action runs.
"""
import json
import re
from datetime import date
from pathlib import Path

import numpy as np

import features as F

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
MODEL = ROOT / "model"
OUT = ROOT / "docs" / "movies.js"
TOP_TAGS = 6


def load_model():
    meta = json.loads((MODEL / "meta.json").read_text())
    p = np.load(MODEL / "predictor.npz")
    g = np.load(MODEL / "genome.npz")
    profiles = (
        {int(i): s.astype(np.float64) for i, s in zip(p["dir_ids"], p["dir_sum"])},
        {int(i): int(n) for i, n in zip(p["dir_ids"], p["dir_n"])},
        {int(i): s.astype(np.float64) for i, s in zip(p["cast_ids"], p["cast_sum"])},
        {int(i): int(n) for i, n in zip(p["cast_ids"], p["cast_n"])},
    )
    genome = {int(t): (z.astype(np.float64), tt.astype(np.float64) / 255)
              for t, z, tt in zip(g["tmdb"], g["Z"], g["T"])}
    return meta, p, profiles, genome


def predict(films, meta, p, profiles):
    dims = meta["dims"]
    kw_index = {k: i for i, k in enumerate(meta["keywords"])}
    emb = F.encode(films)
    X = np.hstack([
        np.array([F.meta_features(m) for m in films]),
        np.array([F.keyword_features(m, kw_index) for m in films]),
        emb,
        np.array([F.person_features(m, profiles, dims) for m in films]),
    ])
    Y = ((X - p["mx"]) / p["sx"]) @ p["W"].astype(np.float64) + p["my"]
    return Y[:, :dims], np.clip(Y[:, dims:], 0, 1)


def zscore(a):
    return (a - a.mean()) / (a.std() + 1e-9)


def display_title(t):
    return re.sub(r"\s+", " ", t).strip()


def main():
    today = date.today()
    pool = json.loads((CACHE / "pool.json").read_text())
    now_playing = {f["id"] for f in pool["films"] if f["nowPlaying"]}
    films = []
    for f in pool["films"]:
        m = json.loads((CACHE / "tmdb" / f"{f['id']}.json").read_text())
        if not m["release"] or m["release"] > today.isoformat():
            continue                                    # unreleased
        if not m["overview"] and not m["keywords"]:
            continue
        films.append(m)
    films.sort(key=lambda m: -m["votes"])
    print(f"{len(films)} released films in the pool")

    meta, p, profiles, genome = load_model()
    dims, vocab = meta["dims"], meta["vocab"]

    real = [m["id"] in genome for m in films]
    Z = np.zeros((len(films), dims))
    T = np.zeros((len(films), len(vocab)))
    for i, m in enumerate(films):
        if real[i]:
            Z[i], T[i] = genome[m["id"]]
    todo = [i for i, r in enumerate(real) if not r]
    print(f"{len(films) - len(todo)} with MovieLens profiles, predicting {len(todo)}")
    if todo:
        Zp, Tp = predict([films[i] for i in todo], meta, p, profiles)
        # Ridge predictions are shrunk toward the middle; restore the spread of
        # real profiles so predicted films can still be anyone's top pick.
        real_idx = [i for i, r in enumerate(real) if r]
        stretch = np.clip(Z[real_idx].std(0) / (Zp.std(0) + 1e-9), 1, 1.8)
        Zp = Zp.mean(0) + (Zp - Zp.mean(0)) * stretch
        Z[todo], T[todo] = Zp, Tp

    # Quality prior: shrunk TMDB score plus votes relative to films of the same era
    # (recent films have had less time to collect votes).
    votes = np.array([m["votes"] for m in films], dtype=np.float64)
    score = np.array([m["score"] for m in films])
    bayes = (votes * score + 300 * score.mean()) / (votes + 300)
    years = np.array([F.year_of(m) for m in films])
    lv = np.log(votes + 1)
    rel = np.array([lv[i] - np.median(lv[np.abs(years - years[i]) <= 2]) for i in range(len(films))])
    quality = 0.6 * zscore(bayes) + 0.4 * zscore(rel)

    # How each display tag moves with each taste dimension (for "you're leaning toward…").
    Tz = (T - T.mean(0)) / (T.std(0) + 1e-9)
    affinity = (Tz.T @ zscore(Z)) / len(films)

    people, person_index = [], {}

    def pid(pair):
        i, name = pair
        if i not in person_index:
            person_index[i] = len(people)
            people.append(name)
        return person_index[i]

    tag_mean = T.mean(0)
    out = []
    for i, m in enumerate(films):
        top, stems = [], set()
        for t in np.argsort(-(T[i] - tag_mean)):
            stem = re.sub(r"[^a-z]", "", vocab[t])[:4]
            if stem not in stems:
                top.append(int(t))
                stems.add(stem)
            if len(top) == TOP_TAGS:
                break
        out.append([
            display_title(m["title"]),
            m["release"],
            "|".join(m["genres"]),
            m["imdb"],
            m["id"],
            int(m["votes"]),
            round(float(m["score"]), 1),
            round(float(quality[i]), 2),
            [round(float(v), 2) for v in Z[i]],
            top,
            [pid(d) for d in m["directors"][:2]],
            [pid(c) for c in m["cast"][:F.N_CAST]],
            m["poster"],
            0 if real[i] else 1,
            1 if m["id"] in now_playing else 0,
            int(m["runtime"]),
        ])

    data = {
        "source": "TMDB + MovieLens 25M tag genome",
        "built": today.isoformat(),
        "dims": dims,
        "vocab": vocab,
        "affinity": [[round(float(v), 3) for v in row] for row in affinity],
        "people": people,
        "fields": ["title", "release", "genres", "imdb", "tmdb", "votes", "score", "quality",
                   "vec", "tags", "directors", "cast", "poster", "predicted", "nowPlaying", "runtime"],
        "movies": out,
    }
    OUT.write_text("window.MOVIE_DATA=" + json.dumps(data, separators=(",", ":"), ensure_ascii=False) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {len(out)} films, {len(people)} people)")


if __name__ == "__main__":
    main()
