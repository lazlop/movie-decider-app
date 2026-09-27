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

**The pool.** The most-voted TMDB feature films (60 minutes or longer) for every
release year, plus what's playing in US theaters. Its size is set in `config.json`; see
[Configuration](#configuration).

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

**Quality and popularity.** Quality is the TMDB rating shrunk toward the mean
(so a 9.0 from 40 votes doesn't beat an 8.5 from 20,000). Popularity is the vote
count compared with films of the same age, since new releases haven't had time to
collect votes: films over a year old are compared with films released within two
years of them, and newer films with the 15 films closest to them in age (a
film's votes climb for its first few months, so a year is too coarse). Popularity
has a floor of −2.5 standard deviations. A few films get in with far fewer votes
than the rest of the pool (limited releases on the now-playing list, or this
year's trending films), and without the floor they would win nearly every game
for players who favor lesser-known films.

The game learns how much you care about quality and popularity separately. The
start screen's **Favor: Popular / Balanced / Lesser-known** sets where the
popularity weight starts; your picks can move it from there.

## Configuration

Everything you can adjust, from most to least likely to need it.

### Movie pool: `config.json`

Controls which films enter the game. It's read by `pipeline/tmdb.py pool`, so a
change takes effect on the next build: the Monday refresh, a manual run of the
**Refresh movie list** workflow (Actions tab → Run workflow), or locally with
`python3 pipeline/tmdb.py pool && .venv/bin/python pipeline/build.py`. You can edit
the file in GitHub's web editor.

```json
{
  "pool": {
    "first_year": 1920,
    "per_year": [[1920, 5], [1940, 10], [1960, 18], [1980, 30], [1990, 45]],
    "scale": 1.0,
    "min_votes": 200,
    "min_runtime": 60,
    "min_votes_recent": 80,
    "now_playing_min_votes": 50
  }
}
```

| Setting | Default | What it does |
|---|---|---|
| `per_year` | see above | How many films each release year contributes, as `[from_year, count]` steps. Each step holds until the next: `[1990, 45]` means 45 films a year from 1990 on. Within a year, films are ranked by TMDB vote count (the current year by popularity, since its films are still collecting votes). |
| `scale` | `1.0` | Multiplies every `per_year` count, then rounds. The quickest way to grow or shrink the whole pool. |
| `min_votes` | `200` | TMDB votes a film needs to be considered at all. Raise it for only well-known films; lower it for deeper cuts. A year with fewer qualifying films than its count just contributes fewer. |
| `min_runtime` | `60` | Minutes a film must run. Keeps out shorts, music videos and TV specials (*Steamboat Willie*, *Thriller*, *How the Grinch Stole Christmas!*), and the next most-voted feature takes each one's place. Films with no runtime on TMDB yet (some brand-new releases) are kept. `0` turns it off. |
| `min_votes_recent` | `80` | The same bar for the current and previous year, which haven't had time to collect votes. |
| `now_playing_min_votes` | `50` | Votes a film on TMDB's US now-playing list needs to be added even if it missed its year's cut. |
| `first_year` | `1920` | Earliest release year fetched. |

What `scale` does in practice (measured September 2026):

| `scale` | Films | `movies.js` | Films per year from 1990 | Example of the least-voted films added |
|---|---|---|---|---|
| `1.0` | ~2,600 | 0.9 MB | 45 | — |
| `2.0` | ~5,100 | 1.8 MB | 90 | *Mighty Aphrodite*, *Rob Roy*, *Colossus: The Forbin Project* |

To reshape rather than resize, edit the steps. For example, more recent films:
`[[1920, 5], [1960, 15], [1990, 45], [2010, 80]]`.

Things to keep in mind:

- **Speed.** Each round the game scores every film 200 times. At 5,000 films
  that's about 30 ms on a desktop and roughly 100–150 ms on a phone, so pools
  of several thousand films are fine.
- **Predicted profiles.** Films MovieLens never scored (mostly post-2019 and
  obscure titles) get predicted taste profiles, which are rougher. Bigger pools
  and newer-leaning steps raise that share; the build prints it ("predicting N").
- **The warm-up** always draws from the best-known ~30% of whatever is in range
  (by vote count, whatever the **Favor** setting), so a bigger pool doesn't make
  the first rounds more obscure. Films shown in
  recent warm-ups on the same device are held back, and the next most-voted
  films take their place.

### Game behavior: constants in `docs/app.js`

Change these, commit, and GitHub Pages serves the new behavior. No rebuild needed.

| Constant | Default | What it does |
|---|---|---|
| `WARMUP_ROUNDS` | `3` | Rounds of pure exploration before narrowing. |
| `POPULARITY` | `+0.3` / `0` / `−0.3` | The start screen's **Favor** options: the starting popularity weight (per standard deviation of popularity). Balanced is the default; Popular starts about where the game did before the option existed. |
| `POP_PRIOR_VAR` | `0.1` | How far picks can move the popularity weight from where Favor started it. `0` fixes it at the Favor value. |
| `WARM_CHOICES` | `20` | Each warm-up pair is picked at random from this many of the most informative candidates. `1` always shows the single best pair (the same few films every game); higher means more variety and slightly less informative openers. |
| `RECENT_MAX` / `RECENT_DAYS` | `36` / `14` | Warm-up films remembered in the browser (localStorage) and held out of the next games' warm-ups: at most this many films, each for at most this many days. |
| `MIN_ROUNDS` / `MAX_ROUNDS` | `7` / `12` | Earliest round the final four can start, and the round it starts regardless. Lower both for shorter games. |
| `LEADER_SHARE` | `0.3` | How often one film must come out on top across 200 imagined versions of you to end narrowing early. Higher means longer, surer games. |
| `MAX_SHOWS` | `3` | How many times one film can appear in a game. |
| `MIN_POOL` | `20` | Fewest films in the chosen year range that still allows starting. |
| `PERSON_MIN_FILMS` | `3` | Films a director or actor needs in the pool to get a learned bonus. |
| `PERSON_PRIOR_VAR` / `PERSON_CAP` | `0.12` / `0.9` | How quickly person bonuses grow, and their ceiling. Lower both to make directors and actors matter less. |
| `SAME_TAG` | `0.9` | How closely two tags must align to count as duplicates on labels. |
| `HIDDEN_TAGS` | list | Tags used for matching but never shown as labels. |
| `PRESETS` | 5 ranges | The year-range shortcut buttons on the start screen. |
| extra rounds | `+3` / `+5` | In `moreRounds()`: earliest and latest pick at which **Play 5 more rounds** starts a new final four. |

### Build and model: `pipeline/`

| Setting | Where | Default | Needs |
|---|---|---|---|
| `TOP_TAGS` | `build.py` | `6` | Rebuild. Tags stored per film (tickets show 3). |
| `SAME_TAG` | `build.py` | `0.9` | Rebuild. Duplicate threshold for those stored tags. |
| Quality / popularity | `build.py` | quality: rating shrunk toward the mean by 300 votes; popularity: log votes minus the median of films within 2 years. Both z-scored. | Rebuild. |
| `NEW_DAYS` / `NEW_PEERS` | `build.py` | `365` / `15` | Rebuild. Films younger than `NEW_DAYS` get popularity relative to the `NEW_PEERS` films closest to them in age instead of their 2-year window. |
| `POP_FLOOR` | `build.py` | `-2.5` | Rebuild. Lowest popularity score (in standard deviations). Lower it and the few films with very few votes start to dominate **Lesser-known** games. |
| `N_CAST` | `features.py` | `3` | Retrain. Lead actors per film used for matching and prediction. |
| `ENCODER` | `features.py` | `all-mpnet-base-v2` | Retrain. The sentence-transformer used for plots and keywords. |
| `DIMS` | `train.py` | `24` | Retrain. Size of the taste space. |
| `VOCAB` | `train.py` | `300` | Retrain. Readable tags kept for labels. |
| `KW_MIN` | `train.py` | `25` | Retrain. How often a TMDB keyword must appear to be used as a feature. |
| `PCA_MIN_VOTES` | `train.py` | `150` | Retrain. Films the taste space is fitted on. |
| `BLOCK`, `RENAME` | `train.py` | lists | Retrain. Tags excluded from or renamed in the vocabulary. |

"Rebuild" means `.venv/bin/python pipeline/build.py` (or the weekly workflow).
"Retrain" means running the one-time steps again (`pipeline/tmdb.py training`,
then `pipeline/train.py`, which prints held-out accuracy) and committing
`model/`. It needs MovieLens in `raw/`.

### Refresh schedule and secrets

- **Schedule:** `.github/workflows/refresh-movies.yml`, `cron: "0 9 * * 1"`
  (Mondays 09:00 UTC). It also has a manual **Run workflow** button.
- **`TMDB_API_KEY`:** a repository secret (Settings → Secrets and variables →
  Actions). Locally, put it in `.env`.
- **Workflow permissions:** Settings → Actions → General → "Read and write", so
  the workflow can commit the refreshed `docs/movies.js`.

## How the matching works

All of this runs in the browser, in `docs/app.js`. There's no server and nothing
is sent anywhere.

### Movies are points on a taste map

Each movie has 26 numbers:

- **24 taste coordinates.** Its position in the "taste space" built from the
  MovieLens tag genome (see [Data pipeline](#data-pipeline)). Nearby films feel
  alike: *Blade Runner*'s nearest neighbors are *Metropolis* and *THX 1138*,
  while *Notting Hill* is farther from it than most films are.
  The axes don't have names, but each lines up with readable tags ("dark",
  "space", "feel-good"…), which is how the game can describe your taste in words.
- **1 quality score.** From the TMDB rating (see *Quality and popularity*).
- **1 popularity score.** How widely seen it is compared with films of the same
  age (see *Quality and popularity*).

### What the game learns about you

Your taste is a set of 26 weights, one per number above. A movie's appeal to you
is its numbers multiplied by your weights and added up, plus a small bonus for
directors and actors you've shown you like.

The game never knows your weights exactly. It keeps a best guess **and** how unsure
it is about each one (a Gaussian distribution). It starts out knowing nothing
about your taste, except for a mild assumption that you prefer well-rated films
(quality weight starts at 0.5) and the popularity weight set by the start
screen's Favor option (+0.3, 0 or −0.3).

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
| **Warm-up** | 1–3 | Explore | From the best-known ~30% of the pool (40–300 films), it tries 400 random pairs and shows one of the 20 it would learn the most from (films from recent warm-ups on this device are held back): two films where your pick is hard to predict *and* would move the uncertain weights. In practice that means very different films. |
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

- **Info ↗** on each ticket opens the film's TMDB page (synopsis, cast,
  trailer) in a new tab. It doesn't count as a pick.
- **Next option** swaps a movie out without learning anything from it. Use it for films
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
- **Favor: Popular / Balanced / Lesser-known** (Balanced by default) doesn't
  remove any films. It sets the popularity weight the game starts with; your picks
  can still move it. It doesn't affect the warm-up, which always shows well-known
  films.

## Credits

Movie data and posters from [TMDB](https://www.themoviedb.org/). This product uses
the TMDB API but is not endorsed or certified by TMDB. Taste profiles from the
MovieLens 25M tag genome: F. Maxwell Harper and Joseph A. Konstan, 2015,
*The MovieLens Datasets: History and Context*, ACM TiiS. Non-commercial use.
