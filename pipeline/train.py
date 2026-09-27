"""Learn to predict MovieLens tag-genome scores from TMDB data.

    .venv/bin/python pipeline/train.py

Needs raw/ml-25m (MovieLens 25M) and the TMDB cache from `pipeline/tmdb.py training`.
Writes model/ (committed, so the weekly build does not need MovieLens) and prints
a held-out evaluation comparing feature sets.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
import imdb as I

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "ml-25m"
CACHE = ROOT / "cache"
MODEL = ROOT / "model"

DIMS = 24
VOCAB = 300
KW_MIN = 25          # keywords used as one-hot features must appear this often
PCA_MIN_VOTES = 150  # PCA basis is fit on films popular enough to be in the game

# Tags that judge quality or describe the dataset rather than the movie.
BLOCK = re.compile(
    r"^(good|great|excellent|awesome|bad|awful|boring|brilliant|masterpiece|"
    r"classic|cool|fun|funny|entertaining|dumb|stupid|overrated|underrated|"
    r"original|interesting|enjoyable|amazing|beautiful|predictable|mediocre|"
    r"disappointing|pointless|pretentious|weird|imdb.*|afi.*|oscar.*|best .*|"
    r"better than.*|book was better|criterion|easily confused.*|"
    r"free to download|golden palm|good .*|great .*|bad .*|excellent .*|"
    r"exceptional acting|very good|visually appealing|so bad it's funny|"
    r"crappy sequel|remake|sequel|sequels|franchise|nudity.*|"
    r"catastrophe|cute!|boring!|stylized|quotable|highly quotable|"
    r"directorial debut|drama|comedy|action|adventure|horror|thriller|"
    r"animation|animated|documentary|fantasy|romance|sci-fi|scifi|"
    r"science fiction|mystery|crime|musical|western|war|children|family|"
    r"cinematography|dialogue|story|storytelling|plot|plot twist|script|"
    r"acting|ending|effects|special effects|soundtrack|music|visuals|"
    r"splatter|goretastic|reflective|affectionate|touching|harsh|"
    r"notable soundtrack|oscar winner|chick flick|girlie movie|"
    r"guilty pleasure|must see|worth watching|watch the credits|"
    r"saturn award.*|pg-13|r|series|visual|humor|humorous|violent|"
    r"based on book|adapted from:book|fun movie|funniest movies|love|"
    r"stupid as hell|long|adapted from:comic|adapted from:game)$"
)
RENAME = {"mindfuck": "mind-bending", "dystopic future": "dystopian future",
          "distopia": "dystopia", "sci fi": "sci-fi"}


def load():
    links = pd.read_csv(RAW / "links.csv").dropna(subset=["tmdbId"])
    tags = pd.read_csv(RAW / "genome-tags.csv")
    print("reading genome scores...")
    scores = pd.read_csv(RAW / "genome-scores.csv")
    genome = scores.pivot(index="movieId", columns="tagId", values="relevance")
    tag_names = tags.set_index("tagId").loc[genome.columns, "tag"].tolist()

    tmdb_of = links.set_index("movieId").tmdbId.astype(int)
    films, rows = [], []
    for mid in genome.index:
        if mid not in tmdb_of.index:
            continue
        f = CACHE / "tmdb" / f"{tmdb_of[mid]}.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        if not m["overview"] and not m["keywords"]:
            continue
        films.append(m)
        rows.append(mid)
    G = genome.loc[rows].to_numpy(np.float64)
    # A few MovieLens ids map to the same TMDB film; keep the first.
    seen, keep = set(), []
    for i, m in enumerate(films):
        if m["id"] not in seen:
            seen.add(m["id"])
            keep.append(i)
    films = [films[i] for i in keep]
    F.add_imdb_scores(films, I.ratings())
    return films, G[keep], tag_names


def targets(films, G, tag_names):
    votes = np.array([m["votes"] for m in films])
    basis = votes >= PCA_MIN_VOTES
    mu = G[basis].mean(0)
    _, _, Vt = np.linalg.svd(G[basis] - mu, full_matrices=False)
    comps = Vt[:DIMS]
    Z = (G - mu) @ comps.T
    scale = Z[basis].std(0)
    Z = Z / scale                                   # unit variance on the game-like films

    usable = np.array([not BLOCK.match(t) for t in tag_names])
    vocab = np.argsort(-(G[basis].std(0) * usable))[:VOCAB]
    names = [RENAME.get(tag_names[i], tag_names[i]) for i in vocab]
    return Z, G[:, vocab], names, basis


def build_X(films, emb, kw_index, profiles, Z=None, blocks=("meta", "kw", "emb", "people")):
    parts = []
    if "meta" in blocks:
        parts.append(np.array([F.meta_features(m) for m in films]))
    if "kw" in blocks:
        parts.append(np.array([F.keyword_features(m, kw_index) for m in films]))
    if "emb" in blocks:
        parts.append(emb)
    if "people" in blocks:
        parts.append(np.array([F.person_features(m, profiles, DIMS, None if Z is None else Z[i])
                               for i, m in enumerate(films)]))
    return np.hstack(parts)


class Ridge:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, Y):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-6
        self.my = Y.mean(0)
        Xs = (X - self.mx) / self.sx
        A = Xs.T @ Xs + self.alpha * np.eye(Xs.shape[1])
        self.W = np.linalg.solve(A, Xs.T @ (Y - self.my))
        return self

    def predict(self, X):
        return ((X - self.mx) / self.sx) @ self.W + self.my


def metrics(Zt, Zp, Tt, Tp):
    r2 = 1 - ((Zt - Zp) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum()
    tc = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(Tt, Tp)])

    def nn(Z):
        Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        S = Zn @ Zn.T
        np.fill_diagonal(S, -np.inf)
        return np.argsort(-S, axis=1)[:, :10]

    a, b = nn(Zt), nn(Zp)
    overlap = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(a, b)])
    return r2, tc, overlap


def main():
    films, G, tag_names = load()
    print(f"{len(films)} films with both MovieLens genome and TMDB data")
    Z, T, vocab_names, basis = targets(films, G, tag_names)
    Y = np.hstack([Z, T])

    emb_file = CACHE / "emb_train.npz"
    ids = np.array([m["id"] for m in films])
    if emb_file.exists() and np.array_equal(np.load(emb_file)["ids"], ids):
        emb = np.load(emb_file)["emb"]
    else:
        print("encoding text (one-time)...")
        emb = F.encode(films)
        np.savez(emb_file, ids=ids, emb=emb)

    kw_counts = pd.Series([k for m in films for k in m["keywords"]]).value_counts()
    kw_list = kw_counts[kw_counts >= KW_MIN].index.tolist()
    kw_index = {k: i for i, k in enumerate(kw_list)}
    print(f"{len(kw_list)} keywords used as features")

    # ---- held-out evaluation ----
    rng = np.random.default_rng(0)
    test = rng.random(len(films)) < 0.2
    tr, te = np.where(~test)[0], np.where(test)[0]
    ftr, fte = [films[i] for i in tr], [films[i] for i in te]
    prof_tr = F.person_profiles(ftr, Z[tr])
    years = np.array([F.year_of(m) for m in films])
    game_like = basis[te]
    recent = game_like & (years[te] >= 2012)

    configs = [
        ("genres + year etc.", ("meta",)),
        ("+ TMDB keywords", ("meta", "kw")),
        ("+ text encoder", ("meta", "kw", "emb")),
        ("+ directors & leads", ("meta", "kw", "emb", "people")),
    ]
    print(f"\nheld-out test: {len(te)} films ({game_like.sum()} popular, {recent.sum()} popular from 2012+)")
    print(f"{'features':24} {'subset':14} {'R² taste':>9} {'tag corr':>9} {'top-10 nbrs':>12}")
    results = []
    # Regularization is chosen on a validation split carved out of the training films.
    val = rng.random(len(tr)) < 0.15
    tr2, va = tr[~val], tr[val]
    ftr2, fva = [films[i] for i in tr2], [films[i] for i in va]
    prof_tr2 = F.person_profiles(ftr2, Z[tr2])
    va_like = basis[va]
    for name, blocks in configs:
        Xa = build_X(ftr2, emb[tr2], kw_index, prof_tr2, Z[tr2], blocks)
        Xv = build_X(fva, emb[va], kw_index, prof_tr2, None, blocks)
        alpha = max((30, 100, 300, 1000, 3000, 10000, 30000), key=lambda a: metrics(
            Z[va][va_like], Ridge(a).fit(Xa, Y[tr2]).predict(Xv)[va_like, :DIMS],
            T[va][va_like][:, :2], T[va][va_like][:, :2])[0])
        Xtr = build_X(ftr, emb[tr], kw_index, prof_tr, Z[tr], blocks)
        Xte = build_X(fte, emb[te], kw_index, prof_tr, None, blocks)
        P = Ridge(alpha).fit(Xtr, Y[tr]).predict(Xte)
        s = metrics(Z[te][game_like], P[game_like, :DIMS], T[te][game_like], P[game_like, DIMS:])
        r = metrics(Z[te][recent], P[recent, :DIMS], T[te][recent], P[recent, DIMS:])
        print(f"{name:24} {'popular':14} {s[0]:9.2f} {s[1]:9.2f} {s[2]:12.0%}   (alpha {alpha})")
        print(f"{'':24} {'2012+':14} {r[0]:9.2f} {r[1]:9.2f} {r[2]:12.0%}")
        results.append({"features": name, "alpha": alpha, "popular": s, "recent": r})
    print(f"(random neighbors would overlap ~{10 / game_like.sum():.1%}; "
          f"recent-subset neighbors are found within a smaller set, so they overlap more by chance)")

    # ---- final model on everything ----
    alpha = results[-1]["alpha"]
    profiles = F.person_profiles(films, Z)
    X = build_X(films, emb, kw_index, profiles, Z)
    model = Ridge(alpha).fit(X, Y)

    MODEL.mkdir(exist_ok=True)
    d_sum, d_n, c_sum, c_n = profiles
    np.savez_compressed(
        MODEL / "predictor.npz",
        mx=model.mx.astype(np.float32), sx=model.sx.astype(np.float32),
        my=model.my.astype(np.float32), W=model.W.astype(np.float16),
        dir_ids=np.array(list(d_sum), dtype=np.int64),
        dir_sum=np.array(list(d_sum.values()), dtype=np.float32),
        dir_n=np.array([d_n[p] for p in d_sum], dtype=np.int32),
        cast_ids=np.array(list(c_sum), dtype=np.int64),
        cast_sum=np.array(list(c_sum.values()), dtype=np.float32),
        cast_n=np.array([c_n[p] for p in c_sum], dtype=np.int32),
    )
    np.savez_compressed(
        MODEL / "genome.npz",
        tmdb=ids, Z=Z.astype(np.float16), T=np.round(T * 255).astype(np.uint8),
    )
    (MODEL / "meta.json").write_text(json.dumps({
        "encoder": F.ENCODER, "dims": DIMS, "vocab": vocab_names, "keywords": kw_list,
        "alpha": alpha,
        "evaluation": [{"features": r["features"],
                        "popular": dict(zip(["r2", "tag_corr", "nbr_overlap"], map(float, r["popular"]))),
                        "recent": dict(zip(["r2", "tag_corr", "nbr_overlap"], map(float, r["recent"])))}
                       for r in results],
    }, indent=1))
    print(f"\nwrote model/ (predictor.npz {(MODEL / 'predictor.npz').stat().st_size / 1e6:.1f} MB, "
          f"genome.npz {(MODEL / 'genome.npz').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
