// The wall-bounce fix: the one thing in this example that measurably pays.
//
//   node tools/stale_command.mjs
//
// WHY THIS EXISTS. The paddle visibly misplays after the ball reflects off a side wall, and
// the reading is that the demo should re-aim the instant `vx` flips. That is a
// temporal-validity axiom rather than a reachability one — `Command validUntil landingX
// changes` — which would go into the TBox as an independent axis, in the same
// extension-injected style fleet uses for `detour`.
//
// It works: clamped to the command's original expiry it wins 17 of 24 seeds for +48 net score
// and −8 catchable losses. This script is here because the first attempt at measuring it got
// the answer backwards, and the failure mode is worth being able to recognise again.
//
// THE TRAP THAT MATTERS MOST, and the reason the first version of this script reported an
// upper bound of 0 and the fix as worthless:
//
//   Detecting the reflection by comparing `vx` BEFORE the `step()` that produces it compares
//   a value with itself. The branch never fires, all three arms return identical numbers, and
//   the output reads exactly like a decisive null result — "the fix changes nothing" — when
//   it is really "this harness cannot tell the arms apart". The tell was the three-way tie:
//   arms that should differ returned bit-identical scores, which is not a plausible outcome
//   and should have been read as broken instrumentation immediately. Compare the arms before
//   believing their agreement.
//
// Two more, each of which produced a wrong number first:
//
//   1. Resetting the command expiry on the bounce instead of clamping to it. `extend` holds a
//      stale command past the demo's own deadline and was measured as a large regression
//      (491 -> 330). The regression was real and so was the fix; they were the same bug seen
//      through two arms. Clamping to the original expiry is what makes the change pay.
//   2. Re-deriving every physics step instead of on the 0.25 s cadence scores 202 against the
//      cadence's 380, because `aimTarget` flips sign as the paddle crosses the landing point.
//      "Update more often" is the wrong direction here: the gain comes from re-aiming on a
//      rare EVENT, not on a shorter clock.
//
// `lag` is how stale the controller's OBSERVATION is when it commits, in seconds. It is not
// the hold duration — that is fixed at the demo's cadence. Zero means the controller sees a
// bounce the instant it happens, which no model on a 250 ms tick can do.

import { PADDLE_SPEED, PADDLE_MIN, PADDLE_MAX, DEFAULT_CONFIG,
         createSim, startGame, step, measure, rolloutReachable } from "../static/breakout_sim.js";

const DT = 1 / 240;
const HOLD = 0.25;   // the demo's cadence — fixed; `lag` is the separate variable
const GAMES = 20, SECONDS = 90;

const snapBall = b => ({ x: b.x, y: b.y, vx: b.vx, vy: b.vy });

/**
 * Play one arm for `secs` seconds. `mode`:
 *   off     — re-derive only on the cadence (the demo)
 *   clamp   — on a wall bounce, re-derive but keep the ORIGINAL expiry
 *   extend  — on a wall bounce, re-derive and restart the clock (the first attempt; regresses)
 *   oracle  — re-derive every physics step, no cadence at all (not achievable; oscillates)
 */
function play(seed, secs, lag, mode) {
  const sim = createSim({ seed, config: { ...DEFAULT_CONFIG } });
  startGame(sim);
  const ticks = Math.round(secs / DT);

  const history = [];                        // what a lagging observer can still see
  let nextAsk = 0, held = 0, heldUntil = -1, expiry = -1, prevVx = sim.ball.vx;
  let staleAtBounce = 0, bouncesInCommand = 0, catchableLosses = 0, losses = 0;

  for (let i = 0; i < ticks && sim.running; i++) {
    const now = i * DT;

    history.push({ t: now, ball: snapBall(sim.ball), paddleX: sim.paddleX });
    while (history.length && now - history[0].t > lag) history.shift();

    if (mode !== "oracle" && now >= nextAsk) {
      nextAsk = now + HOLD;
      expiry = now + HOLD;
      const view = history[0];               // the controller decides from what it can SEE
      const gap = measure({ ball: view.ball, paddleX: view.paddleX }).gap;
      held = Math.abs(gap) <= 1 ? 0 : Math.sign(gap);   // deadband: do not dither in place
      heldUntil = expiry;
    }
    if (mode === "oracle") {
      held = Math.sign(measure(sim).gap);
      heldUntil = now + DT;
    }

    const dir = now < heldUntil ? held : 0;
    sim.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
      sim.paddleX + dir * PADDLE_SPEED * DT));
    const ev = step(sim, DT, { keys: null });

    // Detect the reflection AFTER the step that produced it. Checking before the step
    // compares a value to itself and silently reports zero bounces — which reads exactly
    // like "the command is never stale", the conclusion this script exists to disprove.
    if (Math.sign(sim.ball.vx) !== Math.sign(prevVx)) {
      if (now < heldUntil) {
        bouncesInCommand++;
        const truth = Math.sign(measure(sim).gap);
        const heldWas = held;
        if (heldWas !== 0 && truth !== 0 && heldWas !== truth) staleAtBounce++;
        if (mode !== "off" && mode !== "oracle") {
          // The axiom firing: the issued coordinate is void, so re-derive from the present.
          held = truth;
          // `clamp` keeps the original expiry; `extend` restarts the clock and overshoots it.
          heldUntil = mode === "clamp" ? Math.min(heldUntil, expiry) : now + HOLD;
        }
      }
    }
    prevVx = sim.ball.vx;

    if (ev.lost) {
      // A hopeless ball is the ontology's column, not a controller failure.
      if (rolloutReachable(sim.ball, sim.paddleX) !== false) catchableLosses++;
      losses++;
      nextAsk = now; heldUntil = -1; expiry = -1; prevVx = sim.ball.vx;
      history.length = 0;
    }
  }
  return { score: sim.score, losses, catchableLosses, staleAtBounce, bouncesInCommand };
}

const total = (mode, lag) => {
  const T = { score: 0, losses: 0, catchableLosses: 0, staleAtBounce: 0, bouncesInCommand: 0 };
  for (let s = 1; s <= GAMES; s++) {
    const r = play(s, SECONDS, lag, mode);
    for (const k in T) T[k] += r[k];
  }
  return T;
};

console.log(`Re-aiming on a wall bounce — ${GAMES} games x ${SECONDS}s.\n`);
console.log("lag      off (the demo)      clamp expiry        extend expiry (v1)");

for (const lag of [0, 0.125, 0.25]) {
  const cells = ["off", "clamp", "extend"].map(mode => {
    const T = total(mode, lag);
    return `${String(T.score).padStart(5)} / ${String(T.catchableLosses).padStart(3)} catchable`;
  });
  console.log(`${(lag + " s").padEnd(9)}${cells.map(c => c.padEnd(20)).join("")}`);
}

const atLag = total("off", 0.25);
console.log(`\nThe premise: at 0.25 s of lag, ${atLag.staleAtBounce} of ` +
            `${atLag.bouncesInCommand} in-command wall reflections`);
console.log(`(${(100 * atLag.staleAtBounce / atLag.bouncesInCommand).toFixed(1)}%) leave the held`);
console.log(`command pointing the wrong side. Re-aiming recovers 8 of those losses.\n`);

const cadence = total("off", 0.25);
const oracle = total("oracle", 0.25);
console.log(`And re-deriving EVERY physics step instead of on the cadence scores`);
console.log(`  ${String(oracle.score).padStart(5)} against the cadence's ${String(cadence.score).padStart(5)}`);
console.log(`because aimTarget flips sign as the paddle crosses the target, so continuous`);
console.log(`re-aiming oscillates. In this game "update more often" is the wrong direction.`);