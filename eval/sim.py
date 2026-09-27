"""Replay the game in docs/app.js against real MovieLens users.

    .venv/bin/python eval/sim.py                                   # the game as it ships
    .venv/bin/python eval/sim.py '{"short":{"MAX_ROUNDS":8}}' 3    # variants, 3 games per user

Each player is a MovieLens user from eval/prepare.py. Their game pool is the current
pool films they rated, and they answer every duel (and the opening lineup) by their
own rating plus some noise. We score the film the game names.

Output columns:
  winner pct   where the winner falls among the user's own ratings (0.5 = typical film)
  rating       the winner's rating, next to the user's average
  finalists vs crowd   how much more the user liked the final four than all MovieLens
               users did: the part that's personal taste rather than "good film". The
               cleanest measure of personalisation (the winner's own number is inflated,
               since the user picks it from the four).
  top-rated film pct   baseline: everyone gets the pool's highest-quality film
  winner popularity    the winner's popularity (SDs, from movies.js), and where it falls
               among the popularity of the player's own pool films
  predicted winners    share of winners whose profile was swapped for a predicted one,
               next to their share of the pool (they should roughly match)

Variant keys (JSON object per variant; anything omitted is as shipped):
  any constant from app.js read below, e.g. MAX_ROUNDS, LEADER_SHARE, LINEUP_SCALE
  "lineup": false     skip the opening lineup
  "q_mu", "q_var"     quality weight prior (0.5, 0.06 in app.js)
  "pop_mu": -0.8      starting popularity weight, as set by the start screen's Favor
                      buttons (Balanced, 0, is the default)
  "people": false     no director/actor bonuses
  "actors": false     director bonuses only (the start screen's Match on actors switch)
  "noise": 1.0        rating noise per judgement (default 0.5 stars)
  "quality": "tmdb"   TMDB rating shrunk by 300 votes instead of movies.js's quality
  "pred": "stretch"   how swapped-in predictions are spread: "match" (build.py, default),
                      "stretch" (old per-axis stretch), "raw" (as the ridge predicts), "none"
                      (no swap: every rated film keeps its real profile)
  "pred_scale": 1.1   multiply matched predicted lengths (with "match")

The pool is the latest docs/movies.js, so rebuild first to test build.py changes.
Caveat: MovieLens users have seen every film in their pool; real players meet films
they don't know. Post-2019 films have no MovieLens ratings and can't be scored.
"""
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "eval" / "cache"
sys.path.insert(0, str(ROOT / "pipeline"))
from build import match_lengths  # noqa: E402

# ---- the game's data and constants ----
_s = (ROOT / "docs" / "movies.js").read_text()
DATA = json.loads(_s[len("window.MOVIE_DATA="):_s.rindex(";")])
F = {f: i for i, f in enumerate(DATA["fields"])}
M = DATA["movies"]
K = DATA["dims"]; Q = K; POP = K + 1; D = K + 2
X0 = np.array([r[F["vec"]] + [r[F["quality"]], r[F["popularity"]]] for r in M], float)
TMDB = np.array([r[F["tmdb"]] for r in M])
PREDICTED = np.array([r[F["predicted"]] for r in M], bool)

_app = (ROOT / "docs" / "app.js").read_text()
APP = {k: float(v) for k, v in re.findall(r"^\s*const ([A-Z_]+) = ([\d.]+);", _app, re.M)}
for need in ("WARMUP_ROUNDS", "LINEUP_SIZE", "LINEUP_SCALE", "WARM_CHOICES", "MIN_ROUNDS",
             "MAX_ROUNDS", "LEADER_SHARE", "MAX_SHOWS", "PERSON_MIN_FILMS", "PERSON_PRIOR_VAR",
             "PERSON_CAP", "POP_PRIOR_VAR"):
    assert need in APP, f"app.js constant {need} not found"

_people = [list(dict.fromkeys(r[F["directors"]] + r[F["cast"]])) for r in M]
_credits = Counter(p for ps in _people for p in ps)
_bidx = {p: i for i, p in enumerate(p for p, n in _credits.items() if n >= APP["PERSON_MIN_FILMS"])}
BP = [[_bidx[p] for p in ps if p in _bidx] for ps in _people]
BP_DIR = [[_bidx[p] for p in dict.fromkeys(r[F["directors"]]) if p in _bidx] for r in M]
NP = len(_bidx)

# Lineup vibes, as in app.js vibeOf(): mostly the film's first-listed genre.
_PRIMARY_VIBE = {
    "Action": "Action", "Adventure": "Adventure", "Fantasy": "Adventure", "Family": "Adventure",
    "Science Fiction": "Sci-fi", "Comedy": "Comedy", "Horror": "Horror",
    "Thriller": "Thriller", "Crime": "Thriller", "Mystery": "Thriller", "Romance": "Romance",
}


def _vibe(genres):
    gs = genres.split("|") if genres else []
    g = set(gs)
    if "Animation" in g:
        return "Animated"
    if {"Romance", "Comedy"} <= g and not g & {"Drama", "War"}:
        return "Rom-com"
    if "Science Fiction" in g and gs[0] in ("Action", "Adventure"):
        return "Sci-fi"
    return _PRIMARY_VIBE.get(gs[0], "Drama") if gs else "Drama"


VIBE = [_vibe(r[F["genres"]]) for r in M]

# ---- MovieLens players ----
_ur = pd.read_csv(CACHE / "ratings.csv")
_pos = {t: i for i, t in enumerate(TMDB)}
_ur = _ur[_ur.tmdb.isin(_pos)].copy()
_ur["idx"] = _ur.tmdb.map(_pos)
PLAYERS = {u: (g.idx.to_numpy(), g.rating.to_numpy()) for u, g in _ur.groupby("user")}
_fm = pd.read_csv(CACHE / "film_mean.csv").set_index("tmdb")["mean"]
CROWD = np.array([_fm.get(t, np.nan) for t in TMDB])

# ---- quality variants ----
_votes = np.array([r[F["votes"]] for r in M], float)
_score = np.array([r[F["score"]] for r in M], float)
_z = lambda a: (a - a.mean()) / a.std()
QUALITY = {"build": X0[:, Q].copy(),
           "tmdb": _z((_votes * _score + 300 * _score.mean()) / (_votes + 300))}

# ---- predicted-profile test: a fixed 30% of real-profile films get out-of-fold predictions ----
_oof = np.load(CACHE / "oof.npz")
_zp_of = dict(zip(_oof["tmdb"].tolist(), _oof["zpred"].astype(float)))
SWAP = np.array([not PREDICTED[i] and TMDB[i] in _zp_of for i in range(len(M))])
SWAP &= np.random.default_rng(7).random(len(M)) < 0.3


def taste_variant(mode, scale=1.0):
    X = X0.copy()
    if mode == "none":
        return X
    idx = np.where(SWAP)[0]
    zp = np.array([_zp_of[TMDB[i]] for i in idx])
    real = X0[~SWAP & ~PREDICTED, :K]
    if mode == "match":
        zp = match_lengths(zp, real) * scale
    elif mode == "stretch":
        st = np.clip(real.std(0) / (zp.std(0) + 1e-9), 1, 1.8)
        zp = zp.mean(0) + (zp - zp.mean(0)) * st
    X[idx, :K] = zp
    return X


_XV = {}


def taste(mode, scale=1.0):
    if (mode, scale) not in _XV:
        _XV[mode, scale] = taste_variant(mode, scale)
    return _XV[mode, scale]


class Game:
    def __init__(self, idx, rating, rng, cfg):
        self.c = {**APP, **cfg}
        self.rng, self.idx, self.rating = rng, idx, rating
        X = taste(self.c.get("pred", "match"), self.c.get("pred_scale", 1.0))[idx].copy()
        X[:, Q] = QUALITY[self.c.get("quality", "build")][idx]
        self.X = X
        self.bp = [(BP if self.c.get("actors", True) else BP_DIR)[i] for i in idx]
        self.people = self.c.get("people", True)
        self.mu = np.zeros(D)
        self.mu[Q] = self.c.get("q_mu", 0.5)
        self.mu[POP] = self.c.get("pop_mu", 0.0)
        self.S = np.diag([1.5 / K] * K + [self.c.get("q_var", 0.06), self.c["POP_PRIOR_VAR"]])
        self.pMu = np.zeros(NP)
        self.pVar = np.full(NP, self.c["PERSON_PRIOR_VAR"])
        self.out = np.zeros(len(idx), bool)
        self.shows = np.zeros(len(idx), int)
        self.round = 0

    def avail(self):
        return np.where(~self.out & (self.shows < self.c["MAX_SHOWS"]))[0]

    def familiar(self, av, share=0.3, cap=300):     # av is sorted by vote count
        return av[:max(40, min(cap, round(len(av) * share)))]

    def sample(self, n):
        L = np.linalg.cholesky(self.S + 1e-12 * np.eye(D))
        th = self.mu + self.rng.standard_normal((n, D)) @ L.T
        w = self.pMu + np.sqrt(self.pVar) * self.rng.standard_normal((n, NP))
        return th, w

    def util(self, th, w, av):
        U = th @ self.X[av].T
        if self.people:
            for j, i in enumerate(av):
                if self.bp[i]:
                    U[:, j] += w[:, self.bp[i]].sum(1)
        return U

    def pdiff(self, a, b, scale=1.0):
        d = {}
        if self.people:
            for p in self.bp[a]:
                d[p] = d.get(p, 0) + 1 / scale
            for p in self.bp[b]:
                d[p] = d.get(p, 0) - 1 / scale
        return {p: v for p, v in d.items() if v}

    def info(self, a, b):
        d = self.X[a] - self.X[b]
        v, m = d @ self.S @ d, self.mu @ d
        for p, dp in self.pdiff(a, b).items():
            v += self.pVar[p] * dp * dp
            m += self.pMu[p] * dp
        q = norm.cdf(m / np.sqrt(1 + v))
        return v * q * (1 - q)

    def observe(self, w, l, scale=1.0):
        d = (self.X[w] - self.X[l]) / scale
        pd_ = self.pdiff(w, l, scale)
        Sd = self.S @ d
        v, m = d @ Sd, self.mu @ d
        for p, dp in pd_.items():
            v += self.pVar[p] * dp * dp
            m += self.pMu[p] * dp
        s = np.sqrt(1 + v)
        z = m / s
        r = -z if z < -8 else norm.pdf(z) / norm.cdf(z)
        k = r / (s * s) * (r + z)
        self.mu += r / s * Sd
        self.S -= k * np.outer(Sd, Sd)
        cap = self.c["PERSON_CAP"]
        for p, dp in pd_.items():
            vp = self.pVar[p]
            self.pMu[p] = np.clip(self.pMu[p] + r / s * vp * dp, -cap, cap)
            self.pVar[p] = max(1e-4, vp - k * vp * vp * dp * dp)

    def judge(self, films):
        noise = self.c.get("noise", 0.5)
        return films[int(np.argmax([self.rating[f] + noise * self.rng.standard_normal() for f in films]))]

    def duel(self, a, b):
        self.shows[a] += 1
        self.shows[b] += 1
        w = self.judge([a, b])
        l = b if w == a else a
        self.observe(w, l)
        self.out[l] = True
        self.round += 1
        return w

    def lineup(self):
        fam = self.familiar(self.avail(), 0.4, 600)
        spread = lambda a, b: (self.X[a] - self.X[b]) @ self.S @ (self.X[a] - self.X[b])
        chosen = [fam[self.rng.integers(len(fam))]]
        while len(chosen) < self.c["LINEUP_SIZE"]:
            used = {VIBE[self.idx[c]] for c in chosen}
            cands = [m for m in fam if VIBE[self.idx[m]] not in used] or [m for m in fam if m not in chosen]
            scored = sorted(((min(spread(m, c) for c in chosen), m) for m in cands), key=lambda t: -t[0])
            chosen.append(scored[self.rng.integers(min(3, len(scored)))][1])
        w = self.judge(chosen)
        for l in chosen:
            if l != w:
                self.observe(w, l, self.c["LINEUP_SCALE"])
                self.out[l] = True
        self.round += 1

    def leaderboard(self, av, n=200):
        th, w = self.sample(n)
        ids, c = np.unique(av[self.util(th, w, av).argmax(1)], return_counts=True)
        o = np.argsort(-c)
        return ids[o], c[o] / n

    def mean_rank(self, av):
        return av[np.argsort(-self.util(self.mu[None], self.pMu[None], av)[0])]

    def play(self):
        c, rng = self.c, self.rng
        if c.get("lineup", True):
            self.lineup()
        while self.round < c["WARMUP_ROUNDS"]:
            fam = self.familiar(self.avail())
            scored = []
            for _ in range(400):
                a, b = fam[rng.integers(len(fam))], fam[rng.integers(len(fam))]
                if a != b:
                    scored.append((self.info(a, b), a, b))
            scored.sort(key=lambda t: -t[0])
            _, a, b = scored[rng.integers(min(int(c["WARM_CHOICES"]), len(scored)))]
            self.duel(a, b)
        while True:
            av = self.avail()
            ids, share = self.leaderboard(av)
            if (self.round >= c["MIN_ROUNDS"] and share[0] >= c["LEADER_SHARE"]) or self.round >= c["MAX_ROUNDS"]:
                break
            th, w = self.sample(9)
            U = self.util(th, w, av)
            a = av[U[0].argmax()]
            pair = None
            for t in range(1, 9):
                u = U[t].copy()
                u[av == a] = -np.inf
                if rng.random() < 0.85:
                    pair = (a, av[u.argmax()])
                    break
            if pair is None:
                others = [b for b in av[rng.integers(len(av), size=250)] if b != a]
                pair = (a, max(others, key=lambda b: self.info(a, b)))
            self.duel(*pair)
        seeds = list(ids[:4])
        for m in self.mean_rank(av):
            if len(seeds) >= 4:
                break
            if m not in seeds:
                seeds.append(m)
        win = self.duel(self.duel(seeds[0], seeds[3]), self.duel(seeds[1], seeds[2]))
        return win, seeds


def percentile(ratings, x):
    return ((ratings < x).sum() + 0.5 * (ratings == x).sum()) / len(ratings)


def play_one(job):
    user, cfg, seed = job
    films, ratings = PLAYERS[user]
    order = np.argsort(films)                       # pool order = vote count, as in the game
    idx, rl = films[order], ratings[order]
    game = Game(idx, rl, np.random.default_rng(seed), cfg)
    win, seeds = game.play()
    top_q = rl[np.argmax(game.X[:, Q])]
    return dict(pct=percentile(rl, rl[win]), rating=rl[win], user_mean=rl.mean(),
                vs_crowd=np.nanmean([rl[s] - CROWD[idx[s]] for s in seeds]),
                top_q_pct=percentile(rl, top_q), taps=game.round,
                win_pop=X0[idx[win], POP], win_pop_pct=percentile(X0[idx, POP], X0[idx[win], POP]),
                swapped_win=SWAP[idx[win]], swapped_share=SWAP[idx].mean())


def main():
    variants = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"as shipped": {}}
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    users = sorted(PLAYERS)
    print(f"{len(users)} players, {reps} games each; pool {len(M)} films built {DATA['built']}")
    with Pool(max(1, cpu_count() - 2)) as pool:
        for name, cfg in variants.items():
            jobs = [(u, cfg, 1000 * k + int(u)) for u in users for k in range(reps)]
            r = pd.DataFrame(pool.map(play_one, jobs, chunksize=4))
            se = r.pct.std() / np.sqrt(len(r))
            print(f"{name:24} winner pct {r.pct.mean():.3f}±{se:.3f} | rating {r.rating.mean():.2f} "
                  f"(user avg {r.user_mean.mean():.2f}) | finalists vs crowd {r.vs_crowd.mean():+.2f} "
                  f"| top-rated film pct {r.top_q_pct.mean():.3f} | taps {r.taps.mean():.1f} "
                  f"| winner popularity {r.win_pop.mean():+.2f} ({r.win_pop_pct.mean():.0%} of pool) "
                  f"| predicted winners {r.swapped_win.mean():.2f} (pool {r.swapped_share.mean():.2f})",
                  flush=True)


if __name__ == "__main__":
    main()
