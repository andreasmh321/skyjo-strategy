const C = require("./skyjo_core.js");
const fs = require("fs");
const W = JSON.parse(fs.readFileSync("weights.json"));
const net = C.loadNet(W.b64, W.meta);
const cases = JSON.parse(fs.readFileSync("cases.json"));
const toState = (x) => ({ n: 2, endScore: 100, values: x.values, st: x.st, deck: x.deck.slice(), disc: x.disc.slice(),
  phase: x.phase, cur: x.cur, hand: x.hand, closer: x.closer, finalLeft: x.finalLeft, nTurns: x.nTurns,
  nextStarter: x.nextStarter, setupLeft: x.setupLeft, totals: x.totals.slice(), nRounds: x.nRounds, history: [],
  nResh: 0, round: 0 });
let obsBad = 0, maskBad = 0, permBad = 0, argBad = 0, maxDiff = 0, vDiff = 0;
for (const c of cases) {
  const s = toState(c.state);
  const e = C.encode(s);
  for (let i = 0; i < e.obs.length; i++) if (Math.abs(e.obs[i] - c.obs[i]) > 1e-6) { obsBad++; break; }
  if (e.mask.some((m, i) => m !== c.mask[i])) maskBad++;
  if (e.perm.some((p, i) => p !== c.perm[i])) permBad++;
  const f = C.forward(net, e.obs, e.mask);
  for (let a = 0; a < 26; a++) if (c.mask[a]) maxDiff = Math.max(maxDiff, Math.abs(f.logits[a] - c.logits[a]));
  vDiff = Math.max(vDiff, Math.abs(f.value - c.value));
  const am = (arr, m) => { let b = -1; for (let a = 0; a < 26; a++) if (m[a] && (b < 0 || arr[a] > arr[b])) b = a; return b; };
  if (am(f.logits, e.mask) !== am(c.logits, c.mask)) argBad++;
}
console.log(`cases ${cases.length}: obs mismatches ${obsBad}, mask ${maskBad}, perm ${permBad}, argmax ${argBad}, max logit diff ${maxDiff.toExponential(2)}, max value diff ${vDiff.toExponential(2)}`);

// engine transitions
const trans = JSON.parse(fs.readFileSync("transitions.json"));
const rnd = C.mulberry32(1);
let bad = 0, checked = { plain: 0, round_end: 0, setup_done: 0 };
for (const t of trans) {
  const s = toState(t.before);
  s.history = []; s.lastScores = [0, 0];
  const lm = C.legalMask(s);
  const legal = lm.map((m, i) => (m ? i : -1)).filter((i) => i >= 0);
  if (JSON.stringify(legal) !== JSON.stringify(t.legal)) { bad++; console.log("legal mismatch", t.legal, legal); continue; }
  C.step(s, t.action, rnd);
  const A = t.after;
  if (t.kind === "plain") {
    const same = JSON.stringify([s.values, s.st, s.deck, s.disc, s.phase, s.cur, s.hand, s.closer, s.finalLeft, s.totals]) ===
                 JSON.stringify([A.values, A.st, A.deck, A.disc, A.phase, A.cur, A.hand, A.closer, A.finalLeft, A.totals]);
    if (!same) { bad++; if (bad < 4) console.log("plain mismatch", t.action, JSON.stringify(s.disc), JSON.stringify(A.disc)); }
  } else if (t.kind === "round_end") {
    const same = JSON.stringify([s.totals, s.lastScores, s.lastRaw, s.lastDoubled ? 1 : 0]) ===
                 JSON.stringify([A.totals, A.lastScores, A.lastRaw, A.lastDoubled]);
    if (!same) { bad++; console.log("round mismatch", s.totals, A.totals, s.lastScores, A.lastScores); }
  } else if (t.kind === "setup_done") {
    if (s.phase !== C.CHOOSE) bad++;
  }
  checked[t.kind]++;
}
console.log("transitions checked", checked, "mismatches", bad);

// reshuffle: empty deck, draw
const g = C.newGame(2, C.mulberry32(3));
g.disc = g.disc.concat(g.deck); g.deck = [];
const top = g.disc[g.disc.length - 1], before = g.disc.length;
g.phase = C.CHOOSE; g.cur = 0;
C.step(g, C.DRAW, C.mulberry32(4));
console.log("reshuffle ok:", g.disc.length === 1 && g.disc[0] === top && g.deck.length === before - 2 && g.nResh === 1);

// full games: agent vs random through the JS engine terminate and conserve cards
let games = 0, agentWins = 0;
const r2 = C.mulberry32(9);
for (let k = 0; k < 30; k++) {
  const s = C.newGame(2, r2);
  let guard = 0;
  while (s.phase !== C.GAME_OVER) {
    if (s.phase === "round_over") { C.nextRound(s, r2); continue; }
    let a;
    if (s.cur === 0) a = C.agentMove(net, s).action;
    else { const m = C.legalMask(s); const l = m.map((x, i) => (x ? i : -1)).filter((i) => i >= 0); a = l[Math.floor(r2() * l.length)]; }
    C.step(s, a, r2);
    let cnt = s.deck.length + s.disc.length + (s.hand !== null ? 1 : 0);
    for (let i = 0; i < 2; i++) for (let p = 0; p < 12; p++) if (s.st[i][p] !== C.REMOVED) cnt++;
    if (cnt !== 150) throw new Error("card count " + cnt);
    if (++guard > 100000) throw new Error("no end");
  }
  games++; if (s.totals[0] < s.totals[1]) agentWins++;
}
console.log(`agent vs random in JS: ${agentWins}/${games} wins`);
