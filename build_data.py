"""Build docs/movies.js from the MovieLens 25M dataset.

Download https://files.grouplens.org/datasets/movielens/ml-25m.zip and unzip it
into raw/ first, then run:  python3 build_data.py

Output is a small JS file (window.MOVIE_DATA = {...}) so the static site works
from file:// as well as GitHub Pages.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "raw" / "ml-25m"
OUT = Path(__file__).parent / "docs" / "movies.js"

POOL_SIZE = 1500      # most-rated movies that have genome scores
DIMS = 24             # PCA dimensions used by the preference model
TOP_TAGS = 6          # descriptive tags shown per movie
VOCAB = 300           # tags that the taste profile can talk about

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
    r"based on book|adapted from:book|fun movie|funniest movies|love|stupid as hell|long|adapted from:comic|adapted from:game)$"
)


def main():
    movies = pd.read_csv(RAW / "movies.csv")
    links = pd.read_csv(RAW / "links.csv", dtype={"imdbId": str})
    tags = pd.read_csv(RAW / "genome-tags.csv")

    print("reading ratings...")
    ratings = pd.read_csv(RAW / "ratings.csv", usecols=["movieId", "rating"])
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    del ratings

    print("reading genome scores...")
    scores = pd.read_csv(RAW / "genome-scores.csv")
    genome = scores.pivot(index="movieId", columns="tagId", values="relevance")
    del scores

    pool = stats.loc[stats.index.intersection(genome.index)]
    pool = pool.sort_values("count", ascending=False).head(POOL_SIZE)
    ids = pool.index.to_numpy()

    G = genome.loc[ids].to_numpy(dtype=np.float64)          # (n, 1128)
    tag_names = tags.set_index("tagId").loc[genome.columns, "tag"].tolist()

    # PCA on the genome, scaled so each component has unit variance.
    mu = G.mean(axis=0)
    Gc = G - mu
    U, S, Vt = np.linalg.svd(Gc, full_matrices=False)
    Z = U[:, :DIMS] * np.sqrt(len(ids))                      # unit variance
    explained = (S[:DIMS] ** 2).sum() / (S ** 2).sum()
    print(f"PCA {DIMS} dims explain {explained:.1%} of genome variance")

    # Quality prior: Bayesian-shrunk mean rating plus log popularity.
    c, m = pool["count"].to_numpy(), pool["mean"].to_numpy()
    prior_n, prior_m = 500, m.mean()
    bayes = (c * m + prior_n * prior_m) / (c + prior_n)
    z = lambda a: (a - a.mean()) / a.std()
    quality = 0.6 * z(bayes) + 0.4 * z(np.log(c))

    # Taste vocabulary: descriptive tags with the most spread across the pool.
    usable = np.array([not BLOCK.match(t) for t in tag_names])
    spread = G.std(axis=0) * usable
    vocab = np.argsort(-spread)[:VOCAB]
    # Correlation of each vocab tag with each component: affinity = C @ theta.
    Gv = (G[:, vocab] - G[:, vocab].mean(0)) / G[:, vocab].std(0)
    C = (Gv.T @ Z) / len(ids)                                # (VOCAB, DIMS)

    # Per-movie display tags: most distinctive vocab tags for that movie.
    vocab_mean = G[:, vocab].mean(0)
    rename = {"mindfuck": "mind-bending", "dystopic future": "dystopian future",
              "distopia": "dystopia", "sci fi": "sci-fi"}
    vocab_names = [rename.get(tag_names[i], tag_names[i]) for i in vocab]

    meta = movies.set_index("movieId").loc[ids]
    link = links.set_index("movieId").loc[ids]
    out_movies = []
    for i, mid in enumerate(ids):
        raw_title = meta.at[mid, "title"].strip()
        mt = re.match(r"^(.*)\s+\((\d{4})\)\s*$", raw_title)
        title, year = (mt.group(1), int(mt.group(2))) if mt else (raw_title, 0)
        # "Matrix, The" -> "The Matrix"; also handles "(a.k.a. ...)" tails
        title = re.sub(r"\s*\(a\.k\.a\..*\)$", "", title)
        title = re.sub(r"^(.*), (The|A|An|Les|La|Le|El|Il|Das|Die)(\s*\(.*\))?$",
                       r"\2 \1\3", title)
        top, stems = [], set()
        for t in np.argsort(-(G[i, vocab] - vocab_mean)):
            stem = re.sub(r"[^a-z]", "", vocab_names[t])[:4]
            if stem not in stems:
                top.append(t)
                stems.add(stem)
            if len(top) == TOP_TAGS:
                break
        tmdb = link.at[mid, "tmdbId"]
        out_movies.append([
            title,
            year,
            meta.at[mid, "genres"].replace("(no genres listed)", ""),
            link.at[mid, "imdbId"],
            int(tmdb) if pd.notna(tmdb) else 0,
            int(c[i]),
            round(float(m[i]), 2),
            round(float(quality[i]), 2),
            [round(float(v), 2) for v in Z[i]],
            [int(t) for t in top],
        ])

    data = {
        "source": "MovieLens 25M (GroupLens Research), tag genome",
        "dims": DIMS,
        "vocab": vocab_names,
        "affinity": [[round(float(v), 3) for v in row] for row in C],
        "fields": ["title", "year", "genres", "imdb", "tmdb", "count",
                   "rating", "quality", "vec", "tags"],
        "movies": out_movies,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.MOVIE_DATA=" + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {OUT} ({OUT.stat().st_size/1e3:.0f} kB, {len(ids)} movies)")


if __name__ == "__main__":
    main()
