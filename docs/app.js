/* Reel Duel
 *
 * Each movie is a point in a 24-dim "taste space" (PCA of the MovieLens tag
 * genome) plus a quality score. The viewer's taste is a weight vector theta
 * with a Gaussian posterior N(mu, S). A pick "A over B" is a probit
 * observation  P(A beats B) = Phi(theta . (xA - xB)),  folded in with one
 * moment-matching (assumed density filtering) step.
 *
 * Pair selection moves from exploring to deciding:
 *   warm-up   pairs of well-known films that maximise expected information
 *   narrowing double Thompson sampling: two posterior draws, each one's favourite
 *   final 4   a small bracket between the posterior's four likeliest winners
 */
(() => {
  "use strict";

  const DATA = window.MOVIE_DATA;
  const K = DATA.dims;
  const D = K + 1;                 // taste dims + quality weight
  const WARMUP_ROUNDS = 3;
  const MIN_ROUNDS = 7;
  const MAX_ROUNDS = 12;
  const LEADER_SHARE = 0.3;        // Thompson share that triggers the final four
  const MAX_SHOWS = 3;             // a movie can return at most this often
  const BULBS = 12;

  const movies = DATA.movies.map((r, id) => ({
    id,
    title: r[0],
    year: r[1],
    genres: r[2] ? r[2].split("|") : [],
    imdb: r[3],
    tmdb: r[4],
    count: r[5],
    rating: r[6],
    x: Float64Array.from([...r[8], r[7]]),
    tags: r[9],
  }));
  const vocab = DATA.vocab;
  const affinityRows = DATA.affinity;

  const ERAS = {
    any: () => true,
    old: (m) => m.year < 1980,
    mid: (m) => m.year >= 1980 && m.year < 2000,
    new: (m) => m.year >= 2000,
  };

  const STOCK = [
    ["Animation", "--t-mint"], ["Children", "--t-mint"], ["Horror", "--t-salmon"],
    ["Sci-Fi", "--t-blue"], ["Documentary", "--t-sand"], ["War", "--t-sand"],
    ["Western", "--t-sand"], ["Romance", "--t-pink"], ["Fantasy", "--t-lilac"],
    ["Musical", "--t-pink"], ["Comedy", "--t-mustard"], ["Action", "--t-orange"],
    ["Crime", "--t-slate"], ["Thriller", "--t-salmon"], ["Mystery", "--t-slate"],
    ["Drama", "--t-lilac"],
  ];
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
  const diff = (a, b, scale = 1) => a.x.map((v, i) => (v - b.x[i]) / scale);

  // ---------- state ----------
  let st = null;
  let history = [];
  let busy = false;

  function freshState(era) {
    const mu = new Float64Array(D);
    mu[K] = 0.8;                                   // people usually prefer well-loved films
    const S = Array.from({ length: D }, (_, i) => {
      const row = new Float64Array(D);
      row[i] = i < K ? 1.5 / K : 0.15;
      return row;
    });
    return {
      era, mu, S,
      round: 0,
      phase: "warm",
      out: new Set(),       // lost a duel, rejected, or seen
      seen: new Set(),
      shows: {},
      pair: null,
      bracket: null,
      picks: [],
      certainty: 0,
      winner: null,
    };
  }

  function snapshot() {
    return {
      ...st,
      mu: Float64Array.from(st.mu),
      S: st.S.map((r) => Float64Array.from(r)),
      out: new Set(st.out),
      seen: new Set(st.seen),
      shows: { ...st.shows },
      pair: st.pair && [...st.pair],
      bracket: st.bracket && JSON.parse(JSON.stringify(st.bracket)),
      picks: [...st.picks],
    };
  }

  // One probit moment-matching update: winner beats loser.
  function observe(w, l, scale = 1) {
    const d = diff(w, l, scale);
    const Sd = matVec(st.S, d);
    const v = dot(d, Sd), m = dot(st.mu, d);
    const s = Math.sqrt(1 + v), z = m / s;
    const r = z < -8 ? -z : pdf(z) / cdf(z);
    const k = (r / (s * s)) * (r + z);
    for (let i = 0; i < D; i++) {
      st.mu[i] += (r / s) * Sd[i];
      for (let j = 0; j < D; j++) st.S[i][j] -= k * Sd[i] * Sd[j];
    }
  }

  const available = () => movies.filter((m) =>
    ERAS[st.era](m) && !st.out.has(m.id) && (st.shows[m.id] || 0) < MAX_SHOWS);

  function sampleTheta() {
    const L = cholesky(st.S);
    const z = Array.from({ length: D }, gauss);
    return st.mu.map((m, i) => { let s = m; for (let k = 0; k <= i; k++) s += L[i][k] * z[k]; return s; });
  }
  const argmax = (pool, theta, exclude = new Set()) => {
    let best = null, bv = -Infinity;
    for (const m of pool) {
      if (exclude.has(m.id)) continue;
      const u = dot(theta, m.x);
      if (u > bv) { bv = u; best = m; }
    }
    return best;
  };

  // Expected information of a comparison: posterior variance of the utility gap
  // weighted by how uncertain the outcome is.
  function info(a, b) {
    const d = diff(a, b);
    const v = dot(d, matVec(st.S, d));
    const p = cdf(dot(st.mu, d) / Math.sqrt(1 + v));
    return v * p * (1 - p);
  }

  function familiar(pool) {
    const n = Math.max(60, Math.min(300, Math.round(pool.length * 0.35)));
    return pool.slice(0, n);                        // pool is sorted by rating count
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

  function warmPair(pool) {
    const fam = familiar(pool);
    let best = null, bv = -1;
    for (let t = 0; t < 400; t++) {
      const a = fam[(Math.random() * fam.length) | 0], b = fam[(Math.random() * fam.length) | 0];
      if (a.id === b.id) continue;
      const v = info(a, b);
      if (v > bv) { bv = v; best = [a, b]; }
    }
    return best;
  }

  function narrowPair(pool) {
    const a = argmax(pool, sampleTheta());
    for (let t = 0; t < 8; t++) {
      const b = argmax(pool, sampleTheta(), new Set([a.id]));
      if (b && b.id !== a.id && Math.random() < 0.85) return [a, b];
    }
    return [a, bestPartner(a, pool)];
  }

  // Share of posterior draws in which each movie comes out on top.
  function leaderboard(pool, n = 200) {
    const counts = new Map();
    for (let i = 0; i < n; i++) {
      const m = argmax(pool, sampleTheta());
      counts.set(m.id, (counts.get(m.id) || 0) + 1);
    }
    return [...counts].map(([id, c]) => ({ m: movies[id], share: c / n })).sort((a, b) => b.share - a.share);
  }
  const meanRanking = (pool) => [...pool].sort((a, b) => dot(st.mu, b.x) - dot(st.mu, a.x));

  function affinities() {
    return affinityRows.map((row, t) => ({ tag: vocab[t], v: dot(row, st.mu.subarray(0, K)) }));
  }
  // Keep one of each near-duplicate tag ("dystopia" / "dystopian future").
  function distinct(list, n) {
    const out = [], stems = new Set();
    for (const a of list) {
      const stem = a.tag.replace(/[^a-z]/g, "").slice(0, 4);
      if (stems.has(stem)) continue;
      stems.add(stem);
      out.push(a);
      if (out.length === n) break;
    }
    return out;
  }

  // ---------- flow ----------
  function nextPair() {
    const pool = available();
    if (pool.length < 2) return finish(meanRanking(pool)[0] || movies[0]);

    if (st.phase === "warm" && st.round >= WARMUP_ROUNDS) st.phase = "narrow";
    if (st.phase === "narrow") {
      const lb = leaderboard(pool);
      const share = lb[0].share;
      const progress = Math.max(st.round / MAX_ROUNDS, share / LEADER_SHARE);
      st.certainty = Math.min(8, Math.round(progress * 8));
      if ((st.round >= MIN_ROUNDS && share >= LEADER_SHARE) || st.round >= MAX_ROUNDS) {
        return startFinal(lb, pool);
      }
      st.pair = narrowPair(pool).map((m) => m.id);
    } else if (st.phase === "warm") {
      st.certainty = st.round;
      st.pair = warmPair(pool).map((m) => m.id);
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
    const avg = { x: new Float64Array(D) };
    for (const m of pool) m.x.forEach((v, i) => { avg.x[i] += v / pool.length; });
    avg.x[K] = 0;
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

  function seen(slot) {
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
    } else {
      repl = argmax(pool, sampleTheta());
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
    show("duel");
    render();
  }

  function finish(winner) {
    st.winner = winner.id;
    st.certainty = BULBS;
    st.pair = null;
    renderWin();
  }

  function seenWinner() {
    history.push(snapshot());
    st.out.add(st.winner);
    st.seen.add(st.winner);
    const pool = available();
    const next = meanRanking(pool)[0];
    if (next) finish(next);
  }

  // ---------- rendering ----------
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtCount = (n) => n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
  const serial = (id) => String((id * 7919 + 104729) % 1000000).padStart(6, "0");
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function ticketHTML(m, { interactive = false, label = "" } = {}) {
    const len = m.title.length;
    const cls = len > 38 ? "xlong" : len > 22 ? "long" : "";
    const tags = m.tags.slice(0, 4).map((t) => vocab[t]).join(" · ");
    return `
      <div class="ticket" style="--stock:${stockOf(m)}" ${interactive ? `role="button" tabindex="0" aria-label="${esc(label + m.title)}"` : ""}>
        <div class="stub"><span class="admit">Admit one</span><span class="serial">${serial(m.id)}</span></div>
        <div class="face">
          <div class="meta">${m.year} · ${esc(m.genres.slice(0, 3).join(" · "))}</div>
          <h2 class="title ${cls}">${esc(m.title)}</h2>
          <div class="tags">${esc(tags)}</div>
          <div class="foot"><span>★ ${m.rating.toFixed(1)} / 5</span><span>${fmtCount(m.count)} ratings</span></div>
        </div>
        ${interactive ? `<button class="seen" type="button">Seen it</button>` : ""}
        <div class="stamp" aria-hidden="true">Admit</div>
      </div>`;
  }

  function renderSlot(slot, enter) {
    const el = $(slot === 0 ? "slotA" : "slotB");
    el.innerHTML = ticketHTML(movies[st.pair[slot]], { interactive: true, label: "Pick " });
    const t = el.querySelector(".ticket");
    if (enter && !reduceMotion()) t.classList.add("enter");
    t.addEventListener("click", (e) => {
      if (e.target.closest(".seen")) return seen(slot);
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
    if (st.phase === "final") return ["Final four · Semi 1", "Final four · Semi 2", "The final"][st.bracket.stage];
    if (st.phase === "warm") return `Warm-up · Round ${st.round + 1}`;
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

  function animateOut(slot, then) {
    busy = true;
    const ts = [$("slotA"), $("slotB")].map((s) => s.querySelector(".ticket"));
    ts.forEach((t, i) => t && t.classList.add(i === slot ? "chosen" : "dropped"));
    setTimeout(then, reduceMotion() ? 0 : slot >= 0 ? 520 : 300);
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
    $("win-meta").textContent = `${m.year} · ${m.genres.slice(0, 3).join(" · ")} · ★ ${m.rating.toFixed(1)} from ${fmtCount(m.count)} ratings`;

    const aff = new Map(affinities().map((a) => [a.tag, a.v]));
    const fits = m.tags.map((t) => vocab[t]).filter((t) => aff.get(t) > 0).sort((a, b) => aff.get(b) - aff.get(a)).slice(0, 3);
    const tagsText = (fits.length ? fits : m.tags.slice(0, 3).map((t) => vocab[t])).map((s) => `<b>${esc(s)}</b>`).join(", ");
    $("win-why").innerHTML = fits.length
      ? `Picked from ${st.picks.length} choices. It hits what you kept choosing: ${tagsText}.`
      : `Picked from ${st.picks.length} choices. Expect ${tagsText}.`;

    const q = encodeURIComponent(m.title);
    $("win-links").innerHTML = [
      m.imdb && `<a href="https://www.imdb.com/title/tt${m.imdb}/" target="_blank" rel="noopener">IMDb</a>`,
      m.tmdb && `<a href="https://www.themoviedb.org/movie/${m.tmdb}" target="_blank" rel="noopener">TMDB</a>`,
      `<a href="https://www.justwatch.com/us/search?q=${q}" target="_blank" rel="noopener">Where to stream</a>`,
      `<a href="https://www.youtube.com/results?search_query=${q}+${m.year}+trailer" target="_blank" rel="noopener">Trailer</a>`,
    ].filter(Boolean).join("");

    const sorted = affinities().sort((a, b) => b.v - a.v);
    $("taste-more").innerHTML = distinct(sorted, 5).map((a) => `<span>${esc(a.tag)}</span>`).join("");
    $("taste-less").innerHTML = distinct([...sorted].reverse(), 3).map((a) => `<span>${esc(a.tag)}</span>`).join("");

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

  function show(name) {
    for (const s of ["start", "duel", "win"]) $(`screen-${s}`).hidden = s !== name;
  }

  function start() {
    const era = document.querySelector('input[name="era"]:checked').value;
    st = freshState(era);
    history = [];
    nextPair();
    window.scrollTo(0, 0);
  }

  function home() {
    st = freshState("any");
    history = [];
    show("start");
    $("phase").textContent = "Tonight's pick";
    renderBulbs();
  }

  // ---------- boot ----------
  const byTitle = (t) => movies.find((m) => m.title === t) || movies[0];
  $("sample").innerHTML = ticketHTML(byTitle("Toy Story")) + ticketHTML(byTitle("Pulp Fiction"));
  $("start").addEventListener("click", start);
  $("undo").addEventListener("click", undo);
  $("neither").addEventListener("click", neither);
  $("again").addEventListener("click", home);
  $("seen-winner").addEventListener("click", seenWinner);
  document.addEventListener("keydown", (e) => {
    if ($("screen-duel").hidden || e.target.closest("input")) return;
    if (e.key === "ArrowUp" || e.key === "1") pick(0);
    else if (e.key === "ArrowDown" || e.key === "2") pick(1);
    else if (e.key === "u" || e.key === "Backspace") undo();
    else if (e.key === "n") neither();
  });
  home();
})();
