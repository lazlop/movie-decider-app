# Reel Duel

A phone-first "this or that" game that finds the movie you want to watch tonight.
You pick between two movies at a time. After about a dozen picks it names one film.

Static site, no backend: everything the browser needs lives in `docs/`, so it can be
hosted on GitHub Pages. A weekly GitHub Action refreshes the movie list from TMDB.

## Run locally

Open `docs/index.html` in a browser, or run `python3 -m http.server -d docs` and visit
http://localhost:8000.

## Host on GitHub Pages

1. Push the repo, then go to **Settings → Pages → Build and deployment**, choose
   "Deploy from a branch", branch `main`, folder `/docs`.
2. Add your TMDB key under **Settings → Secrets and variables → Actions** as
   `TMDB_API_KEY`. The workflow in `.github/workflows/refresh-movies.yml` then
   rebuilds `docs/movies.js` every Monday. You can also run it by hand from the
   Actions tab.

## Data pipeline

```
pipeline/tmdb.py training   # one-time: TMDB data for every MovieLens-scored film
pipeline/train.py           # one-time: learn MovieLens tag scores from TMDB data -> model/
pipeline/tmdb.py pool       # weekly: current game pool from TMDB
pipeline/build.py           # weekly: model + pool -> docs/movies.js
```

Setup: `python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install -r requirements.txt`.
Put `TMDB_API_KEY=...` in `.env` (git-ignored). Training also needs MovieLens 25M
unzipped into `raw/` (git-ignored):
`curl -LO https://files.grouplens.org/datasets/movielens/ml-25m.zip`.

**The pool.** The most-voted TMDB films for every release year (5 a year in the
1920s–30s rising to 45 a year from 1990 on), plus what's playing in US theaters.

**Taste profiles.** MovieLens's tag genome scores about 14,000 films on 1,128
descriptive tags. That's great mood and theme data, but it stops around 2019. So:

- Films MovieLens scored keep their real scores, reduced to a 24-dimension "taste
  space" (PCA) plus 300 readable tags.
- Every other film gets a predicted profile. A ridge regression, trained on the
  ~13,000 films in both datasets, maps TMDB data to the MovieLens scores. Its
  inputs are a sentence-transformer embedding (`all-mpnet-base-v2`) of the plot
  and of the keyword list, keyword and genre flags, decade, and the average
  profile of the director's and leads' other films.
- `model/meta.json` records the held-out accuracy of each feature set.

`model/` is committed so the weekly build doesn't need MovieLens.

**Quality prior.** A shrunk TMDB rating, plus the vote count compared with films
from the same few years (new releases haven't had time to collect votes).

## How the game decides

- **Model.** Your taste is a weight vector θ over the 24 dimensions plus quality,
  with a Gaussian posterior. Each pick "A over B" is a probit observation
  P(A beats B) = Φ(θ·(x_A − x_B) + people), folded in with one moment-matching update.
- **People.** Directors and lead actors with 3 or more films in the pool get a
  small learned bonus each, capped so one favorite can't take over. The **Match
  on actors** switch turns the actor part off; directors always count.
- **Warm-up, rounds 1–3 (explore).** Pairs of well-known films chosen for the
  most expected information: pairs where the model can't predict your pick.
- **Narrowing, rounds 4–12 (explore and decide).** Double Thompson sampling. The
  game draws two plausible versions of your taste and shows each one's favorite film.
- **Final four (decide).** The game moves on once one film wins at least 30% of
  posterior draws after 7 or more rounds, or at round 12. The four likeliest films
  then play two semifinals and a final.
- **Seen it** swaps a movie out. **Neither** counts as a mild vote against both.
- **Release years** and **Skip movies still in theaters** limit the pool.
  "In theaters" means on TMDB's now-playing list, or released in the last ~2 months.

## Credits

Movie data and posters from [TMDB](https://www.themoviedb.org/). This product uses
the TMDB API but is not endorsed or certified by TMDB. Taste profiles from the
MovieLens 25M tag genome: F. Maxwell Harper and Joseph A. Konstan, 2015,
*The MovieLens Datasets: History and Context*, ACM TiiS. Non-commercial use.
