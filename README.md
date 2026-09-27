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

**The pool.** The most-voted TMDB films for every release year, plus what's
playing in US theaters. How many films make it in is set in `config.json`:

| Setting | Default | Meaning |
|---|---|---|
| `per_year` | 5 a year from 1920, 10 from 1940, 18 from 1960, 30 from 1980, 45 from 1990 | Films per release year, as `[from_year, count]` steps |
| `scale` | 1.0 | Multiplies every step. 2.0 gives ~5,100 films (1.8 MB) instead of ~2,600 (0.9 MB), still well-known titles |
| `min_votes` / `min_votes_recent` | 200 / 80 | TMDB votes a film needs (the lower bar applies to the last two years) |
| `now_playing_min_votes` | 50 | Votes an in-theaters film needs |
| `first_year` | 1920 | Earliest release year |

Edit it (GitHub's web editor works) and the next build, weekly or run by hand,
uses it. The game scores every film each round; at 5,000 films that's ~30 ms
on a desktop, so pools up to several thousand films stay responsive on phones.

**Taste profiles.** MovieLens's tag genome scores about 14,000 films on 1,128
descriptive tags. That's great mood and theme data, but it stops around 2019. So:

- Films MovieLens scored keep their real scores, reduced to a 24-dimension "taste
  space" (PCA) plus 300 readable tags. PCA also merges redundant tags: "biopic",
  "biography" and "based on a true story" rise and fall together across films, so
  they collapse into one direction instead of being counted three times.
- Every other film gets a predicted profile. A ridge regression, trained on the
  ~13,000 films in both datasets, maps TMDB data to the MovieLens scores. Its
  inputs are a sentence-transformer embedding (`all-mpnet-base-v2`) of the plot
  and of the keyword list, keyword and genre flags, decade, and the average
  profile of the director's and leads' other films. (Tested: including the title
  in the encoded text, or encoding it separately, makes no measurable difference.)
- `model/meta.json` records the held-out accuracy of each feature set.

`model/` is committed so the weekly build doesn't need MovieLens.

**Quality prior.** A shrunk TMDB rating, plus the vote count compared with films
from the same few years (new releases haven't had time to collect votes).

## How the matching works

All of this runs in the browser, in `docs/app.js`. There's no server and nothing
is sent anywhere.

### Movies are points on a taste map

Each movie has 25 numbers:

- **24 taste coordinates.** Its position in the "taste space" built from the
  MovieLens tag genome (see [Data pipeline](#data-pipeline)). Nearby films feel
  alike: *Blade Runner*'s nearest neighbors are *Metropolis* and *THX 1138*,
  while *Notting Hill* is farther from it than most films are.
  The axes don't have names, but each lines up with readable tags ("dark",
  "space", "feel-good"…), which is how the game can describe your taste in words.
- **1 quality score.** From the TMDB rating and vote count (see *Quality prior*).

### What the game learns about you

Your taste is a set of 25 weights, one per number above. A movie's appeal to you
is its numbers multiplied by your weights and added up, plus a small bonus for
directors and actors you've shown you like.

The game never knows your weights exactly. It keeps a best guess **and** how unsure
it is about each one (a Gaussian distribution). It starts out knowing nothing
about your taste, except for a mild assumption that you prefer well-loved films
(quality weight starts at 0.8).

### What one tap does

Picking A over B tells the game "A appeals to you more than B". It nudges your
weights toward whatever makes A different from B and shrinks its uncertainty in
that direction. A surprising pick moves the weights more than an expected one.

Technically it's a probit model, P(A beats B) = Φ(appeal(A) − appeal(B)), updated
with one moment-matching step (assumed density filtering). The loser leaves the
game; the winner can come back (each film appears at most 3 times).

### The phases

| Phase | Rounds | Goal | How the two movies are chosen |
|---|---|---|---|
| **Warm-up** | 1–3 | Explore | From the best-known ~30% of the pool (40–300 films), it tries 400 random pairs and shows the one it learns the most from: two films where your pick is hard to predict *and* would move the uncertain weights. In practice that means very different films. |
| **Narrowing** | 4 up to 12 | Explore and decide | **Double Thompson sampling.** It draws two plausible versions of "you" from its current uncertainty and shows each one's favorite film. While it's unsure, the two draws disagree and you see varied options; as it gets sure, they converge on the same corner of the map. If both draws pick the same film, it pairs that film with the most informative challenger. About 15% of the time it redraws the challenger anyway, for variety. |
| **Final four** | 3 more picks | Decide | Starts once one film is the favorite in at least 30% of 200 imagined versions of you (from round 7 on), or at round 12 regardless. The four films that win most often are seeded 1–4 and play 1 vs 4 and 2 vs 3, then the final. These picks still update the model. |
| **Now showing** | — | — | The winner of the final, with its poster. |
| **Extra rounds** (optional) | 3–5 more | More certainty | **Play 5 more rounds** on the winner screen goes back to narrowing. A new final four starts after 3 picks if one film clearly leads, otherwise after 5. The previous finalists are allowed back in, since they were close calls. |

A typical game is 10–15 taps. The **certainty bulbs** follow this: one lights per
warm-up round, narrowing fills up to 8 as the leading film pulls ahead (or as the
round limit approaches), the final four lights 9–11, and the winner lights all 12.

The "Leaning toward…" line during play and the "More of / Less of" chips at the
end come from comparing your current weights with how each tag lines up with the
taste axes. Near-duplicate tags ("biopic", "biography", "biographical") point the
same way in the taste space, so labels skip any tag within cosine 0.9 of one
already shown. A few tags are used for matching but never shown as labels because
they're explicit or judgemental ("pornography", "lame"…); see `HIDDEN_TAGS` in
`docs/app.js`.

### Directors and actors

Directors and lead actors (top 3 billed) with **3 or more films in the pool** each
get their own small learned bonus. Pick two Villeneuve films and his others get a
boost. These bonuses are tracked independently (it keeps phones fast, and a dozen
picks can't teach much about how people relate to each other). Each bonus starts
at zero and is capped at ±0.9 so one favorite can't take over.

The **Match on actors** switch turns off the actor bonuses; directors always count.
The winner screen's **People** row lists anyone you picked at least twice who
earned a clearly positive bonus.

### Buttons that change the game

- **Skip** swaps a movie out without learning anything from it. Use it for films
  you can't judge; films you've seen are worth voting on. In the warm-up the
  replacement is again the most informative film, in narrowing it's a posterior
  draw's favorite, and in the final four it's the next-best film overall.
- **Neither** counts as a mild vote against both films. The game treats it as an
  average film beating each one, at reduced strength, and removes both.
- **Undo** restores the game exactly as it was before your last tap.
- **Seen it? Next pick** (under the winner) rules the winner out and shows the
  film the model now likes best.
- **Play 5 more rounds** (next to it) runs the extra rounds described above.
- The runners-up are the other finalists, topped up with the model's next favorites.

### The pool and filters

- **Release years** and **Skip movies still in theaters** decide which films can
  appear at all. The game needs at least 20 films in range to start.
- "In theaters" means on TMDB's US now-playing list when the list was built, or
  released in the last ~2 months. Films more than ~100 days old never count as in
  theaters, even if the list is stale.

### Tuning

The main knobs are constants at the top of `docs/app.js`:

| Constant | Value | Effect |
|---|---|---|
| `WARMUP_ROUNDS` | 3 | Rounds of pure exploration |
| `MIN_ROUNDS` / `MAX_ROUNDS` | 7 / 12 | Earliest and latest start of the final four |
| `LEADER_SHARE` | 0.3 | How dominant the leader must be to end narrowing early |
| `MAX_SHOWS` | 3 | How often one film can reappear |
| `PERSON_MIN_FILMS` | 3 | Films needed before a person gets a bonus |
| `PERSON_PRIOR_VAR`, `PERSON_CAP` | 0.12, 0.9 | How fast and how far person bonuses can grow |

## Credits

Movie data and posters from [TMDB](https://www.themoviedb.org/). This product uses
the TMDB API but is not endorsed or certified by TMDB. Taste profiles from the
MovieLens 25M tag genome: F. Maxwell Harper and Joseph A. Konstan, 2015,
*The MovieLens Datasets: History and Context*, ACM TiiS. Non-commercial use.
