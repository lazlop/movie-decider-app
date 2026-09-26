# Reel Duel

A phone-first "this or that" game that finds the movie you want to watch tonight.
You pick between two movies at a time. After about a dozen picks it names one film.

Static site, no backend: everything lives in `docs/`, so it can be hosted on GitHub Pages.

## Run locally

Open `docs/index.html` in a browser, or run `python3 -m http.server -d docs` and visit http://localhost:8000.

## Host on GitHub Pages

Push the repo, then go to **Settings → Pages → Build and deployment** and choose
"Deploy from a branch", branch `main`, folder `/docs`.

## Rebuild the data

```
mkdir -p raw && cd raw
curl -LO https://files.grouplens.org/datasets/movielens/ml-25m.zip && unzip ml-25m.zip
cd .. && python3 build_data.py      # needs numpy + pandas; writes docs/movies.js
```

`build_data.py` takes the 1,500 most-rated movies that have MovieLens tag-genome
scores. It reduces the 1,128 genome tags to 24 PCA dimensions (the "taste space")
and adds a quality score from the average rating and the number of ratings. It
also keeps a 300-tag vocabulary so the app can describe your taste in words.

## How the game decides

- **Model.** Your taste is a weight vector θ over the 24 dimensions plus quality,
  with a Gaussian posterior. Each pick "A over B" is a probit observation
  P(A beats B) = Φ(θ·(x_A − x_B)), folded in with one moment-matching update.
- **Warm-up, rounds 1–3 (explore).** Pairs of well-known films chosen for the
  most expected information: pairs where the model can't predict your pick.
- **Narrowing, rounds 4–12 (explore and decide).** Double Thompson sampling.
  The game draws two plausible versions of your taste and shows each one's
  favorite film.
- **Final four (decide).** The game moves on once one film wins at least 30% of
  posterior draws after 7 or more rounds, or at round 12. The four likeliest
  films then play two semifinals and a final.
- **Seen it** swaps a movie out without learning anything. **Neither** counts as
  a mild vote against both movies.
- The certainty bulbs track this progress.

## Data and license

MovieLens 25M, F. Maxwell Harper and Joseph A. Konstan, 2015,
*The MovieLens Datasets: History and Context*, ACM TiiS. GroupLens allows
non-commercial use with attribution. The data ends in late 2019, so newer
releases are not included.
