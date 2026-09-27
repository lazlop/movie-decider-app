/* Reel Duel
 *
 * Each movie is a point in a 24-dim "taste space" (PCA of the MovieLens tag
 * genome, predicted from TMDB data for films MovieLens never scored) plus a
 * quality score. The viewer's taste is a weight vector theta with a Gaussian
 * posterior N(mu, S). A pick "A over B" is a probit observation
 *   P(A beats B) = Phi(theta . (xA - xB) + people bonus)
 * folded in with one moment-matching (assumed density filtering) step.
 *
 * Popularity (votes relative to films of the same era) is learned like quality,
 * but kept separate from it; the start screen's Favor setting picks its starting
 * weight, so players can ask for lesser-known films and picks can still move it.
 *
 * People bonus: directors and lead actors with 3+ films in the pool get their
 * own small weight (actors can be switched off on the start screen), kept as independent Gaussians (cheap on a phone, and too
 * few picks to learn correlations between people anyway).
 *
 * Pair selection moves from exploring to deciding:
 *   lineup    one pick from six well-known films of different genres, spread
 *             far apart in taste space (counted as the pick beating each other)
 *   warm-up   pairs of well-known films, drawn from the most informative ones
 *             (films opened with on recent visits are held back)
 *   narrowing double Thompson sampling: two posterior draws, each one's favourite
 *   final 4   a small bracket between the posterior's four likeliest winners
 */
(() => {
  "use strict";

  const DATA = window.MOVIE_DATA;
  const K = DATA.dims;
  const Q = K, POP = K + 1;        // quality and popularity weights follow the taste dims
  const D = K + 2;
  const WARMUP_ROUNDS = 3;         // counting the opening lineup, if played
  const LINEUP_SIZE = 6;
  const LINEUP_SCALE = 1.5;        // softens the lineup's five comparisons, which all hinge on one tap
  const WARM_CHOICES = 20;         // each warm-up pair is one of this many most informative
  const RECENT_MAX = 60;           // lineup and warm-up films remembered across visits (about six nights)
  const RECENT_DAYS = 14;          // ...and forgotten after this long
  const MIN_ROUNDS = 7;
  const MAX_ROUNDS = 12;
  const LEADER_SHARE = 0.3;        // Thompson share that triggers the final four
  const MAX_SHOWS = 3;             // a movie can return at most this often
  const BULBS = 12;
  const MIN_POOL = 20;
  const PERSON_MIN_FILMS = 3;
  const PERSON_PRIOR_VAR = 0.12;
  const PERSON_CAP = 0.9;
  const POP_PRIOR_VAR = 0.1;       // how far picks can move the popularity weight from its start
  // Start-screen popularity setting: starting weight per SD of era-relative votes.
  const POPULARITY = { popular: ["Popular", 0.3], balanced: ["Balanced", 0], obscure: ["Lesser-known", -0.3] };
  const POSTER = "https://image.tmdb.org/t/p/w185";
  const DAY = 864e5;

  const F = Object.fromEntries(DATA.fields.map((f, i) => [f, i]));
  const built = new Date(DATA.built + "T12:00:00");
  const today = new Date();
  const movies = DATA.movies.map((r, id) => {
    const release = new Date(r[F.release] + "T12:00:00");
    const age = (today - release) / DAY;
    return {
      id,
      title: r[F.title],
      year: release.getFullYear(),
      genres: r[F.genres] ? r[F.genres].split("|") : [],
      imdb: r[F.imdb],
      tmdb: r[F.tmdb],
      votes: r[F.votes],
      score: r[F.score],
      x: Float64Array.from([...r[F.vec], r[F.quality], r[F.popularity] ?? 0]),
      tags: r[F.tags],
      directors: r[F.directors],
      cast: r[F.cast],
      poster: r[F.poster],
      runtime: r[F.runtime],
      // Still in theaters: on TMDB's now-playing list at build time, or out for under ~2 months.
      theaters: age < 100 && (r[F.nowPlaying] === 1 || age < 60),
    };
  });
  const vocab = DATA.vocab;
  // Tags the model still uses for matching but that don't make good labels:
  // explicit, judgemental, or artifacts of MovieLens itself.
  const HIDDEN_TAGS = new Set([
    "pornography", "sexualized violence", "male nudity", "sex", "sexual", "sexy",
    "horrible", "idiotic", "lame", "shallow", "stupidity", "plot holes",
    "movielens top pick", "very interesting", "funny as hell", "islam", "women", "stereotypes",
  ]);
  const shownTag = (t) => !HIDDEN_TAGS.has(vocab[t]);
  const affinityRows = DATA.affinity;
  const peopleNames = DATA.people;

  // People who appear often enough for the game to learn something about them.
  const credits = new Map();
  for (const m of movies) for (const p of new Set([...m.directors, ...m.cast])) credits.set(p, (credits.get(p) || 0) + 1);
  const bonusIndex = new Map();
  for (const [p, n] of credits) if (n >= PERSON_MIN_FILMS) bonusIndex.set(p, bonusIndex.size);
  const bonusPerson = [...bonusIndex.keys()];
  const P = bonusIndex.size;
  for (const m of movies) {
    m.bp = [...new Set([...m.directors, ...m.cast])].filter((p) => bonusIndex.has(p)).map((p) => bonusIndex.get(p));
    m.bpDir = m.directors.filter((p) => bonusIndex.has(p)).map((p) => bonusIndex.get(p));
  }
  // People whose weight counts for a film, depending on the "Match on actors" switch.
  const bpOf = (m) => (!m.bpDir ? m.bp || [] : st && st.filters.actors ? m.bp : m.bpDir);

  const YEAR_MIN = Math.floor(Math.min(...movies.map((m) => m.year)) / 10) * 10;
  const YEAR_MAX = Math.max(today.getFullYear(), built.getFullYear());
  const PRESETS = [
    ["Any year", YEAR_MIN, YEAR_MAX],
    ["Before 1980", YEAR_MIN, 1979],
    ["80s & 90s", 1980, 1999],
    ["2000s & 2010s", 2000, 2019],
    ["Last 10 years", YEAR_MAX - 9, YEAR_MAX],
  ];

  const GENRE_LABEL = { "Science Fiction": "Sci-Fi" };
  const STOCK = [
    ["Animation", "--t-mint"], ["Family", "--t-mint"], ["Horror", "--t-salmon"],
    ["Science Fiction", "--t-blue"], ["Documentary", "--t-sand"], ["War", "--t-sand"],
    ["Western", "--t-sand"], ["History", "--t-sand"], ["Romance", "--t-pink"],
    ["Fantasy", "--t-lilac"], ["Music", "--t-pink"], ["Comedy", "--t-mustard"],
    ["Action", "--t-orange"], ["Crime", "--t-slate"], ["Thriller", "--t-salmon"],
    ["Mystery", "--t-slate"], ["Adventure", "--t-orange"], ["Drama", "--t-lilac"],
  ];
  // Opening-lineup vibe: mostly the film's primary (first-listed) TMDB genre, so
  // Pulp Fiction is a Thriller, not a Comedy. The lineup shows one film per vibe.
  const has = (m, ...gs) => gs.some((g) => m.genres.includes(g));
  const PRIMARY_VIBE = {
    Action: "Action", Adventure: "Adventure", Fantasy: "Adventure", Family: "Adventure",
    "Science Fiction": "Sci-fi", Comedy: "Comedy", Horror: "Horror",
    Thriller: "Thriller", Crime: "Thriller", Mystery: "Thriller", Romance: "Romance",
  };
  function vibeOf(m) {
    if (has(m, "Animation")) return "Animated";
    if (has(m, "Romance") && has(m, "Comedy") && !has(m, "Drama", "War")) return "Rom-com";
    if (has(m, "Science Fiction") && ["Action", "Adventure"].includes(m.genres[0])) return "Sci-fi";   // Star Wars, Dune
    return PRIMARY_VIBE[m.genres[0]] || "Drama";
  }
  for (const m of movies) m.vibe = vibeOf(m);
  const stockOf = (m) => `var(${(STOCK.find(([g]) => m.genres.includes(g)) || [0, "--t-sand"])[1]})`;

  // ---------- math ----------
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const matVec = (S, v) => S.map((row) => dot(row, v));
  const pdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  function cdf(z) {
    // Abramowitz-Stegun 7.1.26 via erf
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }
  function gauss() {
    let u = 0; while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }
  function cholesky(S) {
    const n = S.length, L = S.map(() => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = S[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-9)) : s / L[j][j];
      }
    }
    return L;
  }

  // ---------- state ----------
  let st = null;
  let history = [];
  let busy = false;
  const filters = { from: YEAR_MIN, to: YEAR_MAX, skipTheaters: false, actors: true, popularity: "balanced" };

  function freshState() {
    const mu = new Float64Array(D);
    mu[Q] = 0.5;                                   // people usually prefer well-rated films
    mu[POP] = (POPULARITY[filters.popularity] || POPULARITY.balanced)[1];
    const S = Array.from({ length: D }, (_, i) => {
      const row = new Float64Array(D);
      row[i] = i < K ? 1.5 / K : i === Q ? 0.06 : POP_PRIOR_VAR;
      return row;
    });
    return {
      filters: { ...filters },
      mu, S,
      pMu: new Float64Array(P),
      pVar: new Float64Array(P).fill(PERSON_PRIOR_VAR),
      round: 0,
      roundStart: 0,        // where the current stretch of narrowing began
      minRound: MIN_ROUNDS,
      maxRound: MAX_ROUNDS,
      overtime: false,
      phase: "lineup",
      lineup: null,
      out: new Set(),       // lost a duel, rejected, or skipped
      seen: new Set(),      // skipped or marked seen: never listed as a runner-up
      shows: {},
      pair: null,
      bracket: null,
      picks: [],
      certainty: 0,
      winner: null,
      recent: new Set(loadRecent().map(([id]) => id)),
    };
  }

  function snapshot() {
    return {
      ...st,
      mu: Float64Array.from(st.mu),
      S: st.S.map((r) => Float64Array.from(r)),
      pMu: Float64Array.from(st.pMu),
      pVar: Float64Array.from(st.pVar),
      out: new Set(st.out),
      seen: new Set(st.seen),
      shows: { ...st.shows },
      pair: st.pair && [...st.pair],
      lineup: st.lineup && [...st.lineup],
      bracket: st.bracket && JSON.parse(JSON.stringify(st.bracket)),
      picks: [...st.picks],
    };
  }

  // Sparse person difference between two films: +1 for the winner's people, -1 for the loser's.
  function personDiff(w, l, scale) {
    const d = new Map();
    for (const p of bpOf(w)) d.set(p, (d.get(p) || 0) + 1 / scale);
    for (const p of bpOf(l)) d.set(p, (d.get(p) || 0) - 1 / scale);
    for (const [p, v] of d) if (v === 0) d.delete(p);
    return d;
  }

  // One probit moment-matching update: winner beats loser.
  function observe(w, l, scale = 1) {
    const d = w.x.map((v, i) => (v - l.x[i]) / scale);
    const pd = personDiff(w, l, scale);
    const Sd = matVec(st.S, d);
    let v = dot(d, Sd), m = dot(st.mu, d);
    for (const [p, dp] of pd) { v += st.pVar[p] * dp * dp; m += st.pMu[p] * dp; }
    const s = Math.sqrt(1 + v), z = m / s;
    const r = z < -8 ? -z : pdf(z) / cdf(z);
    const k = (r / (s * s)) * (r + z);
    for (let i = 0; i < D; i++) {
      st.mu[i] += (r / s) * Sd[i];
      for (let j = 0; j < D; j++) st.S[i][j] -= k * Sd[i] * Sd[j];
    }
    for (const [p, dp] of pd) {
      const vp = st.pVar[p];
      st.pMu[p] = Math.max(-PERSON_CAP, Math.min(PERSON_CAP, st.pMu[p] + (r / s) * vp * dp));
      st.pVar[p] = Math.max(1e-4, vp - k * vp * vp * dp * dp);
    }
  }

  const inRange = (m, f) => m.year >= f.from && m.year <= f.to && !(f.skipTheaters && m.theaters);
  const available = () => movies.filter((m) =>
    inRange(m, st.filters) && !st.out.has(m.id) && (st.shows[m.id] || 0) < MAX_SHOWS);

  function sampleTaste() {
    const L = cholesky(st.S);
    const z = Array.from({ length: D }, gauss);
    const theta = st.mu.map((m, i) => { let s = m; for (let k = 0; k <= i; k++) s += L[i][k] * z[k]; return s; });
    const w = st.pMu.map((m, p) => m + Math.sqrt(st.pVar[p]) * gauss());
    return { theta, w };
  }
  const meanTaste = () => ({ theta: st.mu, w: st.pMu });
  const utility = (m, t) => { let u = dot(t.theta, m.x); for (const p of bpOf(m)) u += t.w[p]; return u; };

  const argmax = (pool, taste, exclude = new Set()) => {
    let best = null, bv = -Infinity;
    for (const m of pool) {
      if (exclude.has(m.id)) continue;
      const u = utility(m, taste);
      if (u > bv) { bv = u; best = m; }
    }
    return best;
  };

  // Expected information of a comparison: posterior variance of the utility gap
  // weighted by how uncertain the outcome is.
  function info(a, b) {
    const d = a.x.map((v, i) => v - b.x[i]);
    let v = dot(d, matVec(st.S, d)), m = dot(st.mu, d);
    for (const [p, dp] of personDiff(a, b, 1)) { v += st.pVar[p] * dp * dp; m += st.pMu[p] * dp; }
    const q = cdf(m / Math.sqrt(1 + v));
    return v * q * (1 - q);
  }

  // Warm-up films from recent visits on this device, as [tmdb id, time shown].
  const RECENT_KEY = "reelduel.recentWarm";
  function loadRecent() {
    try {
      const cutoff = Date.now() - RECENT_DAYS * DAY;
      return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]").filter(([, t]) => t > cutoff);
    } catch (e) { return []; }
  }
  function rememberWarm(shown) {
    try {
      const ids = new Set(shown.map((m) => m.tmdb)), now = Date.now();
      const kept = loadRecent().filter(([id]) => !ids.has(id));
      localStorage.setItem(RECENT_KEY, JSON.stringify([...kept, ...[...ids].map((id) => [id, now])].slice(-RECENT_MAX)));
    } catch (e) { /* storage unavailable */ }
  }

  function familiar(pool, share = 0.3, cap = 300) {
    const n = Math.max(40, Math.min(cap, Math.round(pool.length * share)));
    // Hold back recent warm-up films, unless the filters leave too few without them.
    const fresh = pool.filter((m) => !st.recent.has(m.tmdb));
    return (fresh.length >= n ? fresh : pool).slice(0, n);   // pool is sorted by vote count
  }

  function bestPartner(a, pool, tries = 250) {
    let best = null, bv = -1;
    for (let t = 0; t < tries; t++) {
      const b = pool[(Math.random() * pool.length) | 0];
      if (b.id === a.id) continue;
      const v = info(a, b);
      if (v > bv) { bv = v; best = b; }
    }
    return best;
  }

  // One of the most informative random pairs, not always the very best, so the
  // same few extreme films don't open every game.
  function warmPair(pool) {
    const fam = familiar(pool);
    const scored = [];
    for (let t = 0; t < 400; t++) {
      const a = fam[(Math.random() * fam.length) | 0], b = fam[(Math.random() * fam.length) | 0];
      if (a.id === b.id) continue;
      scored.push([info(a, b), a, b]);
    }
    scored.sort((p, q) => q[0] - p[0]);
    const [, a, b] = scored[(Math.random() * Math.min(WARM_CHOICES, scored.length)) | 0];
    return [a, b];
  }

  // Six well-known films with six different vibes, picked at random so rarer
  // ones like Rom-com get their turn. Within those vibes, films are chosen
  // greedily so each is as far as possible (under the prior) from those
  // already in, with a little randomness. A wider slice than the warm-up's, so
  // every vibe has candidates.
  function lineupFilms(pool) {
    const fam = familiar(pool, 0.4, 600);
    const vibes = new Set([...new Set(fam.map((m) => m.vibe))].sort(() => Math.random() - 0.5).slice(0, LINEUP_SIZE));
    const spread = (a, b) => { const d = a.x.map((v, i) => v - b.x[i]); return dot(d, matVec(st.S, d)); };
    const first = fam.filter((m) => vibes.has(m.vibe));
    const chosen = [first[(Math.random() * first.length) | 0]];
    while (chosen.length < LINEUP_SIZE) {
      const used = new Set(chosen.map((m) => m.vibe));
      let cands = fam.filter((m) => vibes.has(m.vibe) && !used.has(m.vibe));
      if (!cands.length) cands = fam.filter((m) => !chosen.includes(m));   // narrow years: vibes run out
      if (!cands.length) break;
      const scored = cands.map((m) => [Math.min(...chosen.map((c) => spread(m, c))), m]).sort((p, q) => q[0] - p[0]);
      chosen.push(scored[(Math.random() * Math.min(3, scored.length)) | 0][1]);
    }
    return chosen.sort(() => Math.random() - 0.5);
  }

  function narrowPair(pool) {
    const a = argmax(pool, sampleTaste());
    for (let t = 0; t < 8; t++) {
      const b = argmax(pool, sampleTaste(), new Set([a.id]));
      if (b && b.id !== a.id && Math.random() < 0.85) return [a, b];
    }
    return [a, bestPartner(a, pool)];
  }

  // Share of posterior draws in which each movie comes out on top.
  function leaderboard(pool, n = 200) {
    const counts = new Map();
    for (let i = 0; i < n; i++) {
      const m = argmax(pool, sampleTaste());
      counts.set(m.id, (counts.get(m.id) || 0) + 1);
    }
    return [...counts].map(([id, c]) => ({ m: movies[id], share: c / n })).sort((a, b) => b.share - a.share);
  }
  const meanRanking = (pool) => {
    const t = meanTaste();
    return [...pool].sort((a, b) => utility(b, t) - utility(a, t));
  };

  function affinities() {
    return affinityRows.map((row, t) => ({ t, tag: vocab[t], v: dot(row, st.mu.subarray(0, K)) }))
      .filter((a) => !HIDDEN_TAGS.has(a.tag));
  }
  // Keep one of each near-duplicate tag ("biopic" / "biography"): tags that point
  // the same way in taste space say the same thing.
  const SAME_TAG = 0.9;
  const tagDir = affinityRows.map((row) => { const n = Math.sqrt(dot(row, row)) || 1; return row.map((v) => v / n); });
  function distinct(list, n) {
    const out = [];
    for (const a of list) {
      if (out.some((b) => dot(tagDir[a.t], tagDir[b.t]) >= SAME_TAG)) continue;
      out.push(a);
      if (out.length === n) break;
    }
    return out;
  }

  // ---------- flow ----------
  function nextPair() {
    const pool = available();
    if (pool.length < 2) return finish(meanRanking(pool)[0] || movies[st.picks.at(-1)?.[0] ?? 0]);

    if (st.phase === "lineup") {
      const films = lineupFilms(pool);
      rememberWarm(films);
      st.lineup = films.map((m) => m.id);
      films.forEach((m) => { st.shows[m.id] = 1; });
      return renderLineup();
    }
    if (st.phase === "warm" && st.round >= WARMUP_ROUNDS) st.phase = "narrow";
    if (st.phase === "narrow") {
      const lb = leaderboard(pool);
      const share = lb[0].share;
      const progress = Math.max((st.round - st.roundStart) / (st.maxRound - st.roundStart), share / LEADER_SHARE);
      st.certainty = Math.min(8, Math.round(progress * 8));
      if ((st.round >= st.minRound && share >= LEADER_SHARE) || st.round >= st.maxRound) {
        return startFinal(lb, pool);
      }
      st.pair = narrowPair(pool).map((m) => m.id);
    } else if (st.phase === "warm") {
      st.certainty = st.round;
      const pair = warmPair(pool);
      rememberWarm(pair);
      st.pair = pair.map((m) => m.id);
    } else {
      st.pair = st.bracket.matches[st.bracket.stage];
    }
    st.pair.forEach((id) => { st.shows[id] = (st.shows[id] || 0) + 1; });
    render();
  }

  function startFinal(lb, pool) {
    const seeds = lb.slice(0, 4).map((e) => e.m.id);
    for (const m of meanRanking(pool)) {
      if (seeds.length >= 4) break;
      if (!seeds.includes(m.id)) seeds.push(m.id);
    }
    st.phase = "final";
    st.bracket = { seeds, matches: [[seeds[0], seeds[3]], [seeds[1], seeds[2]]], winners: [], stage: 0 };
    st.certainty = 9;
    st.pair = st.bracket.matches[0];
    render();
  }

  function pick(slot) {
    if (busy || !st.pair) return;
    history.push(snapshot());
    const w = movies[st.pair[slot]], l = movies[st.pair[1 - slot]];
    observe(w, l);
    st.out.add(l.id);
    st.picks.push([w.id, l.id]);
    st.round++;
    animateOut(slot, () => {
      if (st.phase === "final") return advanceBracket(w.id);
      nextPair();
    });
  }

  // The lineup pick beats each of the other five. They leave the game, like any loser.
  function pickLineup(i) {
    if (busy || !st.lineup) return;
    history.push(snapshot());
    const w = movies[st.lineup[i]], losers = st.lineup.filter((id) => id !== w.id).map((id) => movies[id]);
    for (const l of losers) { observe(w, l, LINEUP_SCALE); st.out.add(l.id); }
    st.picks.push([w.id, ...losers.map((l) => l.id)]);
    st.round++;
    st.lineup = null;
    st.phase = "warm";
    busy = true;
    [...$("lineup").children].forEach((el, j) => el.classList.add(j === i ? "chosen" : "dropped"));
    setTimeout(nextPair, reduceMotion() ? 0 : 480);
  }

  // Straight to the pairwise warm-up, which then runs its full length.
  function skipLineup() {
    if (busy || !st.lineup) return;
    history.push(snapshot());
    st.lineup = null;
    st.phase = "warm";
    nextPair();
  }

  function advanceBracket(winnerId) {
    const b = st.bracket;
    b.winners.push(winnerId);
    b.stage++;
    if (b.stage === 2) b.matches.push([b.winners[0], b.winners[1]]);
    if (b.stage === 3) return finish(movies[winnerId]);
    st.certainty = 9 + b.stage;
    st.pair = b.matches[b.stage];
    render();
  }

  function neither() {
    if (busy || !st.pair) return;
    history.push(snapshot());
    const pool = available();
    // A soft "the typical movie beats both of these".
    const avg = { x: new Float64Array(D), bp: [] };
    for (const m of pool) m.x.forEach((v, i) => { avg.x[i] += v / pool.length; });
    avg.x[Q] = avg.x[POP] = 0;
    for (const id of st.pair) { observe(avg, movies[id], 1.6); st.out.add(id); }
    st.round++;
    if (st.phase === "final") {
      // Drop this match and refill it from the next-best films.
      const b = st.bracket;
      const repl = meanRanking(available()).filter((m) => !b.seeds.includes(m.id)).slice(0, 2).map((m) => m.id);
      if (repl.length < 2) return finish(meanRanking(available())[0] || movies[st.pair[0]]);
      b.seeds.push(...repl);
      b.matches[b.stage] = repl;
      st.pair = repl;
      return animateOut(-1, render);
    }
    animateOut(-1, nextPair);
  }

  // Swap one movie out without learning anything from it.
  function skip(slot) {
    if (busy || !st.pair) return;
    history.push(snapshot());
    const goneId = st.pair[slot], keep = movies[st.pair[1 - slot]];
    st.out.add(goneId);
    st.seen.add(goneId);
    const pool = available().filter((m) => m.id !== keep.id);
    let repl;
    if (st.phase === "final") {
      const b = st.bracket;
      repl = meanRanking(pool).find((m) => !b.seeds.includes(m.id) && !b.winners.includes(m.id));
      if (repl) {
        b.seeds.push(repl.id);
        b.matches[b.stage][slot] = repl.id;
      }
    } else if (st.phase === "warm") {
      repl = bestPartner(keep, familiar(pool));
      if (repl) rememberWarm([repl]);
    } else {
      repl = argmax(pool, sampleTaste());
    }
    if (!repl) return finish(keep);
    st.pair = [...st.pair];
    st.pair[slot] = repl.id;
    st.shows[repl.id] = (st.shows[repl.id] || 0) + 1;
    renderSlot(slot, true);
    updateUndo();
  }

  function undo() {
    if (busy || !history.length) return;
    st = history.pop();
    if (st.winner != null) return renderWin();
    if (st.lineup) return renderLineup();
    render();
  }

  function finish(winner) {
    st.winner = winner.id;
    st.certainty = BULBS;
    st.pair = null;
    renderWin();
  }

  // Back to narrowing for a few more picks, then a fresh final four. The last
  // finalists were close calls, so they're allowed back in.
  function moreRounds() {
    history.push(snapshot());
    for (const id of st.bracket ? st.bracket.seeds : []) {
      if (!st.seen.has(id)) { st.out.delete(id); delete st.shows[id]; }
    }
    st.winner = null;
    st.bracket = null;
    st.phase = "narrow";
    st.overtime = true;
    st.roundStart = st.round;
    st.minRound = st.round + 3;
    st.maxRound = st.round + 5;
    nextPair();
    window.scrollTo(0, 0);
  }

  function seenWinner() {
    history.push(snapshot());
    st.out.add(st.winner);
    st.seen.add(st.winner);
    const next = meanRanking(available())[0];
    if (next) finish(next);
  }

  // ---------- rendering ----------
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtCount = (n) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
  const fmtRuntime = (min) => min ? `${Math.floor(min / 60)}h ${String(min % 60).padStart(2, "0")}m` : "";
  const serial = (id) => String((id * 7919 + 104729) % 1000000).padStart(6, "0");
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const genreText = (m) => m.genres.slice(0, 2).map((g) => GENRE_LABEL[g] || g).join(" · ");
  const names = (ids) => ids.map((p) => peopleNames[p]);

  function ticketHTML(m, { interactive = false } = {}) {
    const len = m.title.length;
    const cls = len > 34 ? "xlong" : len > 18 ? "long" : "";
    const tags = m.tags.filter(shownTag).slice(0, 3).map((t) => vocab[t]).join(" · ");
    const dir = names(m.directors.slice(0, 1))[0];
    const leads = names(m.cast.slice(0, 2)).join(", ");
    const creditsLine = [dir && `<span class="dir">Dir. ${esc(dir)}</span>`, leads && esc(leads)].filter(Boolean).join(" · ");
    const poster = m.poster
      ? `<img class="poster" src="${POSTER}${m.poster}" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
      : "";
    return `
      <div class="ticket" style="--stock:${stockOf(m)}" ${interactive ? `role="button" tabindex="0" aria-label="${esc("Pick " + m.title)}"` : ""}>
        <div class="stub"><span class="admit">Admit one</span><span class="serial">${serial(m.id)}</span></div>
        <div class="face">
          <div class="meta">${m.year} · ${esc(genreText(m))}${m.theaters ? ' <span class="now">In theaters</span>' : ""}</div>
          <h2 class="title ${cls}">${esc(m.title)}</h2>
          ${creditsLine ? `<div class="credits">${creditsLine}</div>` : ""}
          <div class="tags">${esc(tags)}</div>
          <div class="foot">
            <span>★ ${m.score.toFixed(1)} · ${fmtRuntime(m.runtime)}</span>
            ${interactive ? `<span class="acts"><a class="info" href="https://www.themoviedb.org/movie/${m.tmdb}" target="_blank" rel="noopener" aria-label="${esc("About " + m.title + " (opens TMDB)")}">Info ↗</a><button class="skip" type="button">Next option</button></span>` : `<span>${fmtCount(m.votes)} votes</span>`}
          </div>
        </div>
        ${poster}
        <div class="stamp" aria-hidden="true">Admit</div>
      </div>`;
  }

  function renderSlot(slot, enter) {
    const el = $(slot === 0 ? "slotA" : "slotB");
    el.innerHTML = ticketHTML(movies[st.pair[slot]], { interactive: true });
    const t = el.querySelector(".ticket");
    if (enter && !reduceMotion()) t.classList.add("enter");
    t.addEventListener("click", (e) => {
      if (e.target.closest(".info")) return;
      if (e.target.closest(".skip")) return skip(slot);
      pick(slot);
    });
    t.addEventListener("keydown", (e) => {
      if (e.target !== t) return;
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(slot); }
    });
  }

  function renderBulbs() {
    const b = $("bulbs");
    if (!b.children.length) b.innerHTML = "<i></i>".repeat(BULBS);
    [...b.children].forEach((el, i) => el.classList.toggle("on", i < st.certainty));
    b.setAttribute("aria-valuenow", st.certainty);
  }

  function phaseLabel() {
    if (st.winner != null) return "Now showing";
    if (st.phase === "lineup") return "Opening pick";
    if (st.phase === "final") return ["Final four · Semi 1", "Final four · Semi 2", "The final"][st.bracket.stage];
    if (st.phase === "warm") return `Warm-up · Round ${st.round + 1}`;
    if (st.overtime) return `Extra rounds · ${st.round - st.roundStart + 1} of ${st.maxRound - st.roundStart}`;
    return `Narrowing · Round ${st.round + 1}`;
  }

  function renderLeaning() {
    const el = $("leaning");
    if (st.phase === "final") {
      el.innerHTML = st.bracket.stage === 2
        ? "Last call. <b>Which one tonight?</b>"
        : "The four films most likely to be the one. <b>Two semifinals, then the final.</b>";
      return;
    }
    if (st.round < 2) {
      el.textContent = "Tap the one you'd rather watch tonight. Go with your gut; there are no wrong answers.";
      return;
    }
    const top = distinct(affinities().sort((a, b) => b.v - a.v), 3).map((a) => a.tag);
    el.innerHTML = `Leaning toward <b>${top.map(esc).join("</b>, <b>")}</b>`;
  }

  function updateUndo() { $("undo").disabled = !history.length; }

  function render() {
    busy = false;
    show("duel");
    $("phase").textContent = phaseLabel();
    renderBulbs();
    renderLeaning();
    renderSlot(0, true);
    renderSlot(1, true);
    updateUndo();
  }

  function renderLineup() {
    busy = false;
    show("lineup");
    $("phase").textContent = phaseLabel();
    renderBulbs();
    $("lineup").innerHTML = st.lineup.map((id, i) => {
      const m = movies[id];
      const poster = m.poster ? `<img src="${POSTER}${m.poster}" alt="" decoding="async" onerror="this.remove()">` : "";
      return `<button type="button" class="pick" style="--stock:${stockOf(m)};--i:${i}" aria-label="${esc(`${m.title} (${m.year}), ${m.vibe}`)}">
        ${poster}<span class="pick-text"><span class="vibe">${esc(m.vibe)}</span><span class="pick-title">${esc(m.title)}</span><span class="pick-year">${m.year}</span></span>
      </button>`;
    }).join("");
  }

  function animateOut(slot, then) {
    busy = true;
    const ts = [$("slotA"), $("slotB")].map((s) => s.querySelector(".ticket"));
    ts.forEach((t, i) => t && t.classList.add(i === slot ? "chosen" : "dropped"));
    setTimeout(then, reduceMotion() ? 0 : slot >= 0 ? 520 : 300);
  }

  // People the viewer kept choosing: learned a clear positive weight and won at least twice.
  function favouritePeople() {
    const wins = new Map();
    for (const [w] of st.picks) for (const p of bpOf(movies[w])) wins.set(p, (wins.get(p) || 0) + 1);
    return [...wins].filter(([p, n]) => n >= 2 && st.pMu[p] > 0.1)
      .sort((a, b) => st.pMu[b[0]] - st.pMu[a[0]]).slice(0, 3)
      .map(([p, n]) => ({ name: peopleNames[bonusPerson[p]], n }));
  }

  function renderWin() {
    busy = false;
    const m = movies[st.winner];
    show("win");
    $("phase").textContent = phaseLabel();
    renderBulbs();
    const t = $("win-title");
    t.textContent = m.title;
    t.classList.toggle("long", m.title.length > 22);
    t.style.animation = "none"; void t.offsetWidth; t.style.animation = "";
    const poster = $("win-poster");
    $("win-feature").classList.toggle("no-poster", !m.poster);
    if (m.poster) {
      poster.onerror = () => $("win-feature").classList.add("no-poster");
      poster.src = `https://image.tmdb.org/t/p/w342${m.poster}`;
      poster.alt = `Poster for ${m.title}`;
    } else {
      poster.removeAttribute("src");
    }
    const dir = names(m.directors.slice(0, 1))[0];
    $("win-meta").textContent = [m.year, genreText(m), fmtRuntime(m.runtime), dir && `Dir. ${dir}`, `★ ${m.score.toFixed(1)} on TMDB`]
      .filter(Boolean).join(" · ");

    const aff = new Map(affinities().map((a) => [a.tag, a.v]));
    const fits = m.tags.filter(shownTag).map((t) => vocab[t]).filter((t) => aff.get(t) > 0).sort((a, b) => aff.get(b) - aff.get(a)).slice(0, 3);
    const tagsText = (fits.length ? fits : m.tags.filter(shownTag).slice(0, 3).map((t) => vocab[t])).map((s) => `<b>${esc(s)}</b>`).join(", ");
    const lead = names(m.cast.slice(0, 2)).join(" and ");
    $("win-why").innerHTML = (fits.length
      ? `Picked from ${st.picks.length} choices. It hits what you kept choosing: ${tagsText}.`
      : `Picked from ${st.picks.length} choices. Expect ${tagsText}.`) + (lead ? ` Starring ${esc(lead)}.` : "");

    // The bracket picks the winner; this is the film the learned weights score highest.
    const top = meanRanking(movies.filter((r) => inRange(r, st.filters) && !st.seen.has(r.id)))[0];
    $("win-top").innerHTML = top.id === m.id
      ? "It's also the top rated based on your tags."
      : `Top rated based on your tags: <b>${esc(top.title)}</b> (${top.year}).`;

    const q = encodeURIComponent(m.title);
    $("win-links").innerHTML = [
      m.imdb && `<a href="https://www.imdb.com/title/${m.imdb}/" target="_blank" rel="noopener">IMDb</a>`,
      `<a href="https://www.themoviedb.org/movie/${m.tmdb}" target="_blank" rel="noopener">TMDB</a>`,
      `<a href="https://www.justwatch.com/us/search?q=${q}" target="_blank" rel="noopener">Where to stream</a>`,
      `<a href="https://www.youtube.com/results?search_query=${q}+${m.year}+trailer" target="_blank" rel="noopener">Trailer</a>`,
    ].filter(Boolean).join("");
    $("lookup-q").value = "";
    $("lookup-results").innerHTML = "";

    const sorted = affinities().sort((a, b) => b.v - a.v);
    $("taste-more").innerHTML = distinct(sorted, 5).map((a) => `<span>${esc(a.tag)}</span>`).join("");
    $("taste-less").innerHTML = distinct([...sorted].reverse(), 3).map((a) => `<span>${esc(a.tag)}</span>`).join("");
    const fav = favouritePeople();
    $("taste-people-row").hidden = !fav.length;
    $("taste-people").innerHTML = fav.map((f) => `<span>${esc(f.name)} ×${f.n}</span>`).join("");

    // Other finalists were close calls, even the ones that lost a head-to-head.
    const b = st.bracket;
    const list = b ? [...new Set([...b.winners, ...b.seeds])].filter((id) => id !== m.id && !st.seen.has(id)) : [];
    for (const r of meanRanking(available())) {
      if (list.length >= 3) break;
      if (r.id !== m.id && !list.includes(r.id)) list.push(r.id);
    }
    $("runners").innerHTML = list.slice(0, 3).map((id) => {
      const r = movies[id];
      return `<li style="--stock:${stockOf(r)}"><span class="rt">${esc(r.title)}</span><span class="ry">${r.year}</span></li>`;
    }).join("");
  }

  // ---------- winner screen: where would a given film rank tonight? ----------
  const fold = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  for (const m of movies) m.key = ` ${fold(m.title)} `;

  function lookup(query) {
    const q = fold(query);
    if (q.length < 2) return [];
    // Word-start matches first ("alien" finds Aliens before Paralien), then by vote count.
    const hits = movies.filter((m) => m.key.includes(q))
      .map((m) => [m.key.startsWith(` ${q}`) ? 0 : m.key.includes(` ${q}`) ? 1 : 2, m])
      .sort((a, b) => a[0] - b[0] || b[1].votes - a[1].votes);
    return hits.slice(0, 5).map(([, m]) => m);
  }

  function lookupStatus(m) {
    if (m.id === st.winner) return "Tonight's pick";
    const won = st.picks.find(([w]) => w === m.id);
    if (won) return "You picked it";
    const lost = st.picks.find((p) => p.indexOf(m.id) > 0);
    if (lost) return `You chose ${movies[lost[0]].title} over it`;
    if (st.seen.has(m.id)) return "You skipped it";
    return "";
  }

  function renderLookup() {
    const t = meanTaste();
    const pool = movies.filter((m) => inRange(m, st.filters)).map((m) => utility(m, t));
    const aff = new Map(affinities().map((a) => [a.t, a.v]));
    const tagList = (m, sign) => m.tags.filter((g) => shownTag(g) && sign * (aff.get(g) || 0) > 0.05)
      .sort((a, b) => sign * (aff.get(b) - aff.get(a))).slice(0, 2).map((g) => vocab[g]);
    $("lookup-results").innerHTML = lookup($("lookup-q").value).map((m) => {
      const u = utility(m, t), rank = 1 + pool.filter((v) => v > u).length;
      const inYears = inRange(m, st.filters);
      const pct = (100 * rank) / pool.length;
      const where = pct <= 50 ? `top ${Math.max(1, Math.ceil(pct))}%` : `bottom ${Math.max(1, Math.ceil(100 - pct))}%`;
      const fits = tagList(m, 1), against = tagList(m, -1);
      const notes = [
        inYears ? `${where} of ${pool.length.toLocaleString()}` : "outside your years",
        lookupStatus(m),
        fits.length && `fits: ${fits.join(", ")}`,
        against.length && `against: ${against.join(", ")}`,
      ].filter(Boolean).map(esc).join(" · ");
      return `<li style="--stock:${stockOf(m)}"><span class="lr">#${rank.toLocaleString()}</span>
        <span class="lt"><span class="rt">${esc(m.title)}</span> <span class="ry">${m.year}</span><span class="ln">${notes}</span></span></li>`;
    }).join("") || ($("lookup-q").value.trim().length >= 2 ? `<li class="none">Not in tonight's movie list.</li>` : "");
  }

  function show(name) {
    for (const s of ["start", "lineup", "duel", "win"]) $(`screen-${s}`).hidden = s !== name;
  }

  // ---------- start screen: year range ----------
  const yrFrom = $("yr-from"), yrTo = $("yr-to"), theatersBox = $("skip-theaters"), actors = $("use-actors");

  function renderFilters() {
    yrFrom.value = filters.from;
    yrTo.value = filters.to;
    $("yr-from-out").textContent = filters.from;
    $("yr-to-out").textContent = filters.to;
    const span = YEAR_MAX - YEAR_MIN;
    const fill = $("dual-fill");
    fill.style.left = `${((filters.from - YEAR_MIN) / span) * 100}%`;
    fill.style.right = `${((YEAR_MAX - filters.to) / span) * 100}%`;
    theatersBox.checked = filters.skipTheaters;
    actors.checked = filters.actors;
    for (const btn of $("popularity").children) btn.setAttribute("aria-pressed", String(btn.dataset.pop === filters.popularity));
    const n = movies.filter((m) => inRange(m, filters)).length;
    $("yr-count").textContent = `${n.toLocaleString()} movies`;
    $("start").disabled = n < MIN_POOL;
    $("start").textContent = n < MIN_POOL ? "Widen the years a little" : "Start the duel";
    for (const btn of $("presets").children) {
      btn.setAttribute("aria-pressed", String(+btn.dataset.from === filters.from && +btn.dataset.to === filters.to));
    }
    try { localStorage.setItem("reelduel.filters", JSON.stringify(filters)); } catch (e) { /* storage unavailable */ }
  }

  function setupFilters() {
    for (const el of [yrFrom, yrTo]) { el.min = YEAR_MIN; el.max = YEAR_MAX; }
    try {
      const saved = JSON.parse(localStorage.getItem("reelduel.filters") || "null");
      if (saved) Object.assign(filters, {
        from: Math.max(YEAR_MIN, Math.min(YEAR_MAX, +saved.from || YEAR_MIN)),
        to: Math.max(YEAR_MIN, Math.min(YEAR_MAX, +saved.to || YEAR_MAX)),
        skipTheaters: !!saved.skipTheaters,
        actors: saved.actors !== false,
        popularity: saved.popularity in POPULARITY ? saved.popularity : "balanced",
      });
    } catch (e) { /* storage unavailable */ }
    $("presets").innerHTML = PRESETS.map(([label, from, to]) =>
      `<button type="button" data-from="${from}" data-to="${to}">${label}</button>`).join("");
    $("presets").addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      filters.from = +btn.dataset.from;
      filters.to = +btn.dataset.to;
      renderFilters();
    });
    yrFrom.addEventListener("input", () => { filters.from = Math.min(+yrFrom.value, filters.to); renderFilters(); });
    yrTo.addEventListener("input", () => { filters.to = Math.max(+yrTo.value, filters.from); renderFilters(); });
    // When both thumbs meet, keep the one that can still move on top.
    yrFrom.addEventListener("pointerdown", () => { yrFrom.style.zIndex = 2; yrTo.style.zIndex = 1; });
    yrTo.addEventListener("pointerdown", () => { yrTo.style.zIndex = 2; yrFrom.style.zIndex = 1; });
    theatersBox.addEventListener("change", () => { filters.skipTheaters = theatersBox.checked; renderFilters(); });
    actors.addEventListener("change", () => { filters.actors = actors.checked; renderFilters(); });
    $("popularity").innerHTML = Object.entries(POPULARITY).map(([key, [label]]) =>
      `<button type="button" data-pop="${key}">${label}</button>`).join("");
    $("popularity").addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      filters.popularity = btn.dataset.pop;
      renderFilters();
    });
    renderFilters();
  }

  function start() {
    st = freshState();
    history = [];
    nextPair();
    window.scrollTo(0, 0);
  }

  function home() {
    st = freshState();
    history = [];
    show("start");
    $("phase").textContent = "Tonight's pick";
    renderBulbs();
  }

  // ---------- boot ----------
  const byTitle = (t, fallback) => movies.find((m) => m.title === t) || movies[fallback];
  $("sample").innerHTML = ticketHTML(byTitle("Toy Story", 1)) + ticketHTML(byTitle("Pulp Fiction", 0));
  $("pool-size").textContent = movies.length.toLocaleString();
  $("built").textContent = `Movie list updated ${built.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}.`;
  setupFilters();
  $("start").addEventListener("click", start);
  $("undo").addEventListener("click", undo);
  $("neither").addEventListener("click", neither);
  $("again").addEventListener("click", home);
  $("seen-winner").addEventListener("click", seenWinner);
  $("more-rounds").addEventListener("click", moreRounds);
  $("lineup").addEventListener("click", (e) => {
    const btn = e.target.closest(".pick");
    if (btn) pickLineup([...$("lineup").children].indexOf(btn));
  });
  $("lineup-skip").addEventListener("click", skipLineup);
  $("lookup-q").addEventListener("input", renderLookup);
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input")) return;
    if (!$("screen-lineup").hidden && e.key >= "1" && e.key <= String(LINEUP_SIZE)) return pickLineup(+e.key - 1);
    if ($("screen-duel").hidden) return;
    if (e.key === "ArrowUp" || e.key === "1") pick(0);
    else if (e.key === "ArrowDown" || e.key === "2") pick(1);
    else if (e.key === "u" || e.key === "Backspace") undo();
    else if (e.key === "n") neither();
  });
  home();
})();
