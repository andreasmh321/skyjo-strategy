// Skyjo rules engine, observation encoding and the v2 policy network, in plain JS.
// Line-by-line port of skyjo/fast.py and skyjo/rl/env.py (canonical encoding).
(function (root) {
  "use strict";
  const N_POS = 12, HIDDEN = 0, REVEALED = 1, REMOVED = 2;
  const SETUP = 0, CHOOSE = 1, PLACE = 2, DECIDE = 3, GAME_OVER = 5;
  const TAKE = 0, DRAW = 1, REP0 = 2, FLIP0 = 14, N_ACTIONS = 26;
  const FULL = [5, 10, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]; // index = value + 2
  const OBS_DIM_2P = 2 * 12 * 17 + 15 + 15 + 4 + 15 + 1 + 4 + 1; // 463

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffle(arr, rnd) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }

  // ------------------------------------------------------------------ engine
  function newGame(nPlayers, rnd, endScore) {
    const s = {
      n: nPlayers, endScore: endScore || 100, values: [], st: [], deck: [], disc: [],
      phase: SETUP, cur: 0, hand: null, closer: -1, finalLeft: 0, nTurns: 0, round: -1,
      nextStarter: -1, setupLeft: 2, totals: new Array(nPlayers).fill(0), nRounds: 0,
      nResh: 0, lastScores: new Array(nPlayers).fill(0), lastRaw: new Array(nPlayers).fill(0),
      lastDoubled: false, lastCloser: -1, history: [],
    };
    startRound(s, rnd);
    return s;
  }

  function startRound(s, rnd) {
    s.round += 1;
    const deck = [];
    for (let v = 0; v < 15; v++) for (let k = 0; k < FULL[v]; k++) deck.push(v - 2);
    shuffle(deck, rnd);
    s.values = []; s.st = [];
    for (let i = 0; i < s.n; i++) {
      const vals = [];
      for (let p = 0; p < N_POS; p++) vals.push(deck.pop());
      s.values.push(vals);
      s.st.push(new Array(N_POS).fill(HIDDEN));
    }
    s.disc = [deck.pop()];
    s.deck = deck;
    s.hand = null; s.closer = -1; s.finalLeft = 0; s.nTurns = 0; s.nResh = 0;
    s.phase = SETUP; s.cur = 0; s.setupLeft = 2;
  }

  function draw(s, rnd) {
    if (s.deck.length === 0) {
      const top = s.disc.pop();
      s.deck = shuffle(s.disc, rnd);
      s.disc = [top];
      s.nResh += 1;
    }
    return s.deck.pop();
  }

  // returns the columns removed
  function resolveTriples(s, i) {
    const out = [];
    for (let c = 0; c < 4; c++) {
      const b = 3 * c, st = s.st[i], v = s.values[i];
      if (st[b] === REMOVED) continue;
      if (st[b] === REVEALED && st[b + 1] === REVEALED && st[b + 2] === REVEALED &&
          v[b] === v[b + 1] && v[b] === v[b + 2]) {
        for (let q = b; q < b + 3; q++) { st[q] = REMOVED; s.disc.push(v[q]); }
        out.push(c);
      }
    }
    return out;
  }

  function legalMask(s) {
    const m = new Array(N_ACTIONS).fill(false), st = s.st[s.cur];
    if (s.phase === SETUP) {
      for (let p = 0; p < N_POS; p++) m[FLIP0 + p] = st[p] === HIDDEN;
    } else if (s.phase === CHOOSE) {
      m[TAKE] = true; m[DRAW] = true;
    } else if (s.phase === PLACE || s.phase === DECIDE) {
      for (let p = 0; p < N_POS; p++) {
        m[REP0 + p] = st[p] !== REMOVED;
        if (s.phase === DECIDE) m[FLIP0 + p] = st[p] === HIDDEN;
      }
    }
    return m;
  }

  function finishSetup(s, rnd) {
    let start;
    if (s.nextStarter >= 0) start = s.nextStarter;
    else {
      const sums = [];
      for (let i = 0; i < s.n; i++) {
        let t = 0;
        for (let p = 0; p < N_POS; p++) if (s.st[i][p] === REVEALED) t += s.values[i][p];
        sums.push(t);
      }
      const best = Math.max(...sums);
      const tied = sums.map((x, i) => (x === best ? i : -1)).filter((i) => i >= 0);
      start = tied[Math.floor(rnd() * tied.length)];
    }
    s.cur = start;
    s.phase = CHOOSE;
  }

  function endRound(s, rnd) {
    for (let i = 0; i < s.n; i++) {
      for (let p = 0; p < N_POS; p++) if (s.st[i][p] === HIDDEN) s.st[i][p] = REVEALED;
      resolveTriples(s, i);
    }
    const c = s.closer, raw = [];
    let othersMin = Infinity;
    for (let i = 0; i < s.n; i++) {
      let r = 0;
      for (let p = 0; p < N_POS; p++) if (s.st[i][p] !== REMOVED) r += s.values[i][p];
      raw.push(r);
      if (i !== c && r < othersMin) othersMin = r;
    }
    const doubled = raw[c] > 0 && othersMin <= raw[c];
    const scores = raw.map((r, i) => (doubled && i === c ? 2 * r : r));
    for (let i = 0; i < s.n; i++) s.totals[i] += scores[i];
    s.lastRaw = raw; s.lastScores = scores; s.lastDoubled = doubled; s.lastCloser = c;
    s.history.push({ raw, scores, closer: c, doubled,
                     boards: s.values.map((v) => v.slice()), st: s.st.map((x) => x.slice()) });
    s.nRounds += 1;
    s.nextStarter = c;
    if (Math.max(...s.totals) >= s.endScore) s.phase = GAME_OVER;
    else { s.phase = "round_over"; s.pendingRnd = true; }
  }

  // call after a round summary has been shown
  function nextRound(s, rnd) {
    if (s.phase === "round_over") startRound(s, rnd);
  }

  function endTurn(s, rnd) {
    s.nTurns += 1;
    const pl = s.cur;
    if (s.closer < 0) {
      if (s.st[pl].every((x) => x !== HIDDEN)) { s.closer = pl; s.finalLeft = s.n - 1; }
    } else s.finalLeft -= 1;
    if (s.closer >= 0 && s.finalLeft === 0) { endRound(s, rnd); return; }
    s.cur = (pl + 1) % s.n;
    s.phase = CHOOSE;
  }

  // Apply a real (non-canonical) action. Returns a small description of what happened.
  function step(s, a, rnd) {
    const cur = s.cur, info = { player: cur, action: a, phase: s.phase };
    if (!legalMask(s)[a]) throw new Error("illegal action " + a + " in phase " + s.phase);
    if (s.phase === SETUP) {
      s.st[cur][a - FLIP0] = REVEALED;
      info.revealed = s.values[cur][a - FLIP0];
      s.setupLeft -= 1;
      if (s.setupLeft === 0) {
        s.cur += 1; s.setupLeft = 2;
        if (s.cur === s.n) finishSetup(s, rnd);
      }
      return info;
    }
    if (s.phase === CHOOSE) {
      if (a === TAKE) { s.hand = s.disc.pop(); s.phase = PLACE; info.took = s.hand; }
      else { s.hand = draw(s, rnd); s.phase = DECIDE; info.drew = s.hand; }
      return info;
    }
    if (a < FLIP0) {
      const p = a - REP0, old = s.values[cur][p];
      info.pos = p; info.card = s.hand; info.replaced = old; info.wasHidden = s.st[cur][p] === HIDDEN;
      s.values[cur][p] = s.hand; s.st[cur][p] = REVEALED; s.disc.push(old);
    } else {
      const p = a - FLIP0;
      info.pos = p; info.discarded = s.hand; info.flipped = s.values[cur][p];
      s.disc.push(s.hand); s.st[cur][p] = REVEALED;
    }
    s.hand = null;
    info.triples = resolveTriples(s, cur);
    const closerBefore = s.closer;
    endTurn(s, rnd);
    info.closedRound = closerBefore < 0 && s.closer === cur;
    return info;
  }

  // ------------------------------------------------------------- encoding
  function code(s, pl, p) {
    const x = s.st[pl][p];
    if (x === HIDDEN) return 0;
    if (x === REMOVED) return 1;
    return s.values[pl][p] + 4;
  }

  function canonPerm(s, pl) {
    const cols = [], keys = [];
    for (let c = 0; c < 4; c++) {
      const ps = [3 * c, 3 * c + 1, 3 * c + 2];
      const cs = ps.map((p) => code(s, pl, p));
      for (let i = 1; i < 3; i++) {
        let j = i;
        while (j > 0 && cs[j - 1] > cs[j]) {
          [cs[j - 1], cs[j]] = [cs[j], cs[j - 1]];
          [ps[j - 1], ps[j]] = [ps[j], ps[j - 1]];
          j--;
        }
      }
      cols.push(ps);
      keys.push(cs[0] * 289 + cs[1] * 17 + cs[2]);
    }
    const order = [0, 1, 2, 3].sort((x, y) => keys[x] - keys[y] || x - y);
    const perm = [];
    for (const k of order) for (const p of cols[k]) perm.push(p);
    return perm;
  }

  function expectedHidden(s) {
    const counts = FULL.slice();
    for (let i = 0; i < s.n; i++)
      for (let p = 0; p < N_POS; p++) if (s.st[i][p] === REVEALED) counts[s.values[i][p] + 2] -= 1;
    for (const v of s.disc) counts[v + 2] -= 1;
    if (s.hand !== null) counts[s.hand + 2] -= 1;
    let num = 0, n = 0;
    for (let v = 0; v < 15; v++) { num += (v - 2) * counts[v]; n += counts[v]; }
    return { E: n > 0 ? num / n : 0, counts };
  }

  // Observation, canonical->real perm and canonical legal mask for the player to move.
  function encode(s) {
    const n = s.n, me = s.cur;
    const out = new Float32Array(n * 12 * 17 + 15 + 15 + 4 + 15 + 1 + 2 * n + 1);
    let o = 0, myPerm = null;
    for (let r = 0; r < n; r++) {
      const pl = (me + r) % n, bp = canonPerm(s, pl);
      if (r === 0) myPerm = bp;
      for (let k = 0; k < N_POS; k++) {
        const p = bp[k], x = s.st[pl][p];
        if (x === HIDDEN) out[o] = 1;
        else if (x === REMOVED) out[o + 1] = 1;
        else out[o + 2 + s.values[pl][p] + 2] = 1;
        o += 17;
      }
    }
    if (s.disc.length > 0) out[o + s.disc[s.disc.length - 1] + 2] = 1;
    o += 15;
    if (s.hand !== null) out[o + s.hand + 2] = 1;
    o += 15;
    out[o + s.phase] = 1;
    o += 4;
    const { E, counts } = expectedHidden(s);
    for (let v = 0; v < 15; v++) out[o + v] = counts[v] / 10;
    o += 15;
    out[o] = s.deck.length / 150; o += 1;
    for (let r = 0; r < n; r++) out[o + r] = s.totals[(me + r) % n] / 100;
    o += n;
    if (s.closer >= 0) out[o + ((s.closer - me) % n + n) % n] = 1;
    o += n;
    out[o] = E / 12;
    const lm = legalMask(s), mask = new Array(N_ACTIONS).fill(false);
    mask[TAKE] = lm[TAKE]; mask[DRAW] = lm[DRAW];
    for (let k = 0; k < N_POS; k++) {
      mask[REP0 + k] = lm[REP0 + myPerm[k]];
      mask[FLIP0 + k] = lm[FLIP0 + myPerm[k]];
    }
    return { obs: out, perm: myPerm, mask, E };
  }

  function toReal(a, perm) {
    if (a >= FLIP0) return FLIP0 + perm[a - FLIP0];
    if (a >= REP0) return REP0 + perm[a - REP0];
    return a;
  }

  // ---------------------------------------------------------------- network
  function loadNet(b64, meta) {
    let bytes;
    if (typeof atob === "function") {
      const bin = atob(b64);
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    } else bytes = new Uint8Array(Buffer.from(b64, "base64"));
    const all = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const w = {};
    for (const m of meta) w[m.name] = { data: all.subarray(m.offset, m.offset + m.size), shape: m.shape };
    return w;
  }

  function linear(W, b, x) {
    const [out, inp] = W.shape, y = new Float32Array(out), d = W.data;
    for (let i = 0; i < out; i++) {
      let acc = b.data[i];
      const row = i * inp;
      for (let j = 0; j < inp; j++) acc += d[row + j] * x[j];
      y[i] = acc;
    }
    return y;
  }

  function layerNormRelu(g, b, x) {
    const n = x.length;
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (x[i] - mean) * (x[i] - mean);
    v /= n;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let i = 0; i < n; i++) {
      const y = (x[i] - mean) * inv * g.data[i] + b.data[i];
      x[i] = y > 0 ? y : 0;
    }
    return x;
  }

  function forward(net, obs, mask) {
    let h = obs;
    for (const [lin, ln] of [["trunk.0", "trunk.1"], ["trunk.3", "trunk.4"], ["trunk.6", "trunk.7"]]) {
      h = linear(net[lin + ".weight"], net[lin + ".bias"], h);
      h = layerNormRelu(net[ln + ".weight"], net[ln + ".bias"], h);
    }
    const logits = linear(net["pi.weight"], net["pi.bias"], h);
    const value = linear(net["v.weight"], net["v.bias"], h)[0];
    for (let a = 0; a < N_ACTIONS; a++) if (!mask[a]) logits[a] = -1e9;
    let mx = -Infinity;
    for (let a = 0; a < N_ACTIONS; a++) mx = Math.max(mx, logits[a]);
    const probs = new Float32Array(N_ACTIONS);
    let z = 0;
    for (let a = 0; a < N_ACTIONS; a++) { probs[a] = mask[a] ? Math.exp(logits[a] - mx) : 0; z += probs[a]; }
    for (let a = 0; a < N_ACTIONS; a++) probs[a] /= z;
    return { logits, probs, value };
  }

  // Agent's choice for the player to move: real action, probabilities by real action, value.
  function agentMove(net, s) {
    const { obs, perm, mask } = encode(s);
    const { probs, value } = forward(net, obs, mask);
    let best = 0;
    for (let a = 1; a < N_ACTIONS; a++) if (probs[a] > probs[best]) best = a;
    const realProbs = new Array(N_ACTIONS).fill(0);
    for (let a = 0; a < N_ACTIONS; a++) realProbs[toReal(a, perm)] += probs[a];
    return { action: toReal(best, perm), probs: realProbs, value };
  }

  const api = {
    N_POS, HIDDEN, REVEALED, REMOVED, SETUP, CHOOSE, PLACE, DECIDE, GAME_OVER,
    TAKE, DRAW, REP0, FLIP0, N_ACTIONS, OBS_DIM_2P, FULL,
    mulberry32, shuffle, newGame, startRound, nextRound, draw, legalMask, step, canonPerm,
    expectedHidden, encode, toReal, loadNet, forward, agentMove, endRound, endTurn,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SkyjoCore = api;
})(typeof window !== "undefined" ? window : globalThis);
