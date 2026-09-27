"""Shared feature code for training (train.py) and the weekly build (build.py)."""
import numpy as np

ENCODER = "sentence-transformers/all-mpnet-base-v2"
TMDB_GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
               "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
               "Romance", "Science Fiction", "Thriller", "War", "Western", "TV Movie"]
DECADES = list(range(1920, 2040, 10))
N_CAST = 3


def year_of(m):
    return int(m["release"][:4]) if m.get("release") else 0


def texts(m):
    """Two views of a film for the text encoder: its plot and its keyword tags."""
    genres = ", ".join(m["genres"])
    plot = f'{m["title"]}. {m.get("tagline", "")} {m["overview"]}'.strip()
    tags = f'{genres}. {", ".join(m["keywords"])}' if m["keywords"] else genres
    return plot, tags


def encode(films, model=None):
    from sentence_transformers import SentenceTransformer
    model = model or SentenceTransformer(ENCODER, device="cpu")
    plots, tags = zip(*(texts(m) for m in films))
    kw = dict(batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    return np.hstack([model.encode(list(plots), **kw), model.encode(list(tags), **kw)]).astype(np.float32)


def people_of(m):
    return [p[0] for p in m["directors"]], [p[0] for p in m["cast"][:N_CAST]]


def person_profiles(films, vecs):
    """Sum and count of taste vectors per person, separately for directing and acting."""
    d_sum, d_n, c_sum, c_n = {}, {}, {}, {}
    for m, v in zip(films, vecs):
        dirs, cast = people_of(m)
        for p in dirs:
            d_sum[p] = d_sum.get(p, 0) + v
            d_n[p] = d_n.get(p, 0) + 1
        for p in cast:
            c_sum[p] = c_sum.get(p, 0) + v
            c_n[p] = c_n.get(p, 0) + 1
    return d_sum, d_n, c_sum, c_n


def person_features(m, profiles, dim, own=None):
    """Mean taste vector of the director's and leads' other films.

    `own` is this film's own vector when it is part of the profiles (training),
    so it is left out and the feature never sees the answer.
    """
    d_sum, d_n, c_sum, c_n = profiles
    dirs, cast = people_of(m)
    out = []
    for ids, s, n in ((dirs, d_sum, d_n), (cast, c_sum, c_n)):
        tot, cnt = np.zeros(dim), 0
        for p in ids:
            if p in s:
                ps, pn = s[p], n[p]
                if own is not None:
                    ps, pn = ps - own, pn - 1
                if pn > 0:
                    tot += ps
                    cnt += pn
        mean = tot / cnt if cnt else np.zeros(dim)
        out.append(np.concatenate([mean, [np.log1p(cnt), float(cnt == 0)]]))
    return np.concatenate(out)


def add_imdb_scores(films, ratings):
    """Attach each film's IMDb rating (from imdb.ratings()) as m["imdb_score"]."""
    for m in films:
        m["imdb_score"] = ratings.get(m["imdb"], (None, 0))[0] if m.get("imdb") else None


def meta_features(m):
    """Genres, decade, rating, runtime, language, keyword count.

    The rating is IMDb's where there is one (add_imdb_scores): the model learns from
    older films whose ratings have settled, while TMDB ratings of new releases run
    0.6-0.9 stars high. There's no vote count for the same reason: every training
    film has had years to collect votes, so a new release's few votes would read as
    an obscure film. Dropping it cost nothing in held-out accuracy.
    """
    y = year_of(m)
    g = [float(x in m["genres"]) for x in TMDB_GENRES]
    dec = [float(y // 10 * 10 == d) for d in DECADES]
    rating = m.get("imdb_score") or m["score"]
    return np.array(g + dec + [rating / 10, min(m["runtime"], 240) / 120, float(m["lang"] == "en"),
                               min(len(m["keywords"]), 40) / 40], dtype=np.float64)


def keyword_features(m, kw_index):
    x = np.zeros(len(kw_index))
    for k in m["keywords"]:
        i = kw_index.get(k)
        if i is not None:
            x[i] = 1.0
    return x
