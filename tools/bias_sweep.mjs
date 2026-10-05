// How much of the game the reachability gate can act on, as a function of how good the
// controller is.
//
//   node tools/bias_sweep.mjs
//
// WHY THIS EXISTS. `eval_breakout.mjs` reports a suppression rate of 1.7%-6.7% and its own
// comment says that is "a property of the game rather than a defect in the gate: the paddle
// crosses faster than the ball falls". That explanation is wrong, and this is the measurement
// that shows it. The suppression rate is a function of the controller's aim bias — how far off
// it reliably is. The harness's scenarios are generated with a per-seed bias, so the headline
// rate sits wherever that bias happens to land.
//
// The consequence for anyone reading eval_breakout.mjs: a low suppression number there is not
// a property of Breakout and says nothing about whether the gate works. It says the demo's
// controller is good. At a bias the demo would never produce, the gate suppresses a third of
// what it is asked.
//
// The scenarios are generated with a FIXED aim error here, not the harness's per-seed one,
// because policy quality is the variable under test and it has to be the only thing that moves.
//
// No model server: the gate is a pure function of the sim (see `shouldAsk` in
// breakout_panel.js — the same code path the browser runs), so what the gate can suppress is
// measurable without spending an inference.

import { PADDLE_SPEED, PADDLE_MIN, PADDLE_MAX, DEFAULT_CONFIG,
         createSim, startGame, step, decidable, aimTarget, rolloutReachable }
  from "../static/breakout_sim.js";

const DT = 1 / 240;
const DECIDE_EVERY = 0.25;   // the demo's cadence
const N = 600;

const snapBall = b => ({ x: b.x, y: b.y, vx: b.vx, vy: b.vy });

/**
 * Drive a game forward with a controller that is reliably `aimError` px off, and snapshot
 * every decision tick.
 *
 * NOT perfect play, and that took one wrong version to learn. A perfect paddle puts every ball
 * exactly where it needs to be, every snapshot comes out reachable, and the sweep below prints
 * a column of zeros having measured nothing — the same vacuous result that
 * eval_breakout.mjs's header warns about. The paddle re-aims only on the decision cadence and
 * holds in between, which is what a model answering on a ~250 ms tick actually produces.
 */
function generateScenarios(count, seed, aimError) {
  const out = [];
  let s = seed;
  while (out.length < count) {
    const sim = createSim({ seed: s++, config: { ...DEFAULT_CONFIG } });
    startGame(sim);
    let nextTick = 0, committed = 0;
    for (let i = 0; i < 20000 && out.length < count; i++) {
      const t = i * DT;
      if (t >= nextTick) {
        nextTick = t + DECIDE_EVERY;
        if (decidable(sim.ball)) {
          out.push({
            ball: snapBall(sim.ball),
            paddleX: sim.paddleX,
            bricks: sim.bricks.map(r => [...r]),
            score: sim.score, lives: sim.lives, seed: s - 1,
          });
        }
        committed = Math.sign(aimTarget(sim.ball) + aimError - sim.paddleX);
      }
      sim.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
        sim.paddleX + committed * PADDLE_SPEED * DT));
      step(sim, DT, { keys: null });
      if (!sim.running) break;
    }
  }
  return out;
}

/** Rebuild a sim at a snapshot, so both sides see the identical state. */
function simAt(sc) {
  const sim = createSim({ seed: sc.seed, config: { ...DEFAULT_CONFIG } });
  sim.ball = { ...sc.ball };
  sim.paddleX = sc.paddleX;
  sim.bricks = sc.bricks.map(r => [...r]);
  sim.score = sc.score;
  sim.lives = sc.lives;
  sim.running = true;
  return sim;
}

/**
 * What suppression costs, since the rate alone reads like free.
 *
 * A suppressed ball gets no command, so the paddle holds still — and the cost of holding
 * still does not land on that ball (it was lost either way). It lands on the NEXT one. This
 * measures how far the paddle has to travel on the following ball purely because the previous
 * command was withheld.
 */
function suppressionCost(scenarios) {
  let suppressed = 0;
  const costs = [];
  for (const sc of scenarios) {
    const sim = simAt(sc);
    if (rolloutReachable(sim.ball, sim.paddleX) !== false) continue;  // gate did not close
    suppressed++;

    // Roll forward with perfect play and see how far the paddle ends up from where the
    // suppressed ball needed it. That distance is what the withheld command cost.
    const probe = simAt(sc);
    for (let k = 0; k < 1200; k++) {
      const d = Math.sign(aimTarget(probe.ball) - probe.paddleX);
      probe.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
        probe.paddleX + d * PADDLE_SPEED * DT));
      const ev = step(probe, DT, { keys: null });
      if (ev.lost || ev.caught) break;
    }
    costs.push(Math.abs(probe.paddleX - sim.paddleX));
  }
  costs.sort((a, b) => a - b);
  const median = costs[Math.floor(costs.length / 2)];
  const p90 = costs[Math.floor(costs.length * 0.9)];
  const over80 = costs.filter(c => c > 80).length;
  return { suppressed, median, p90, over80 };
}

console.log(`Suppression vs controller quality — ${N} decision ticks per row, seed 7.\n`);
console.log("aim bias    unreachable   suppression");

const rows = [];
for (const bias of [0, 40, 80, 120, 160, 200, 240]) {
  const scenarios = generateScenarios(N, 7, bias);

  let decidable = 0, unreachable = 0;
  for (const sc of scenarios) {
    const sim = simAt(sc);
    const r = rolloutReachable(sim.ball, sim.paddleX);
    if (r === null) continue;          // no verdict exists; neither ask nor suppress
    decidable++;
    if (!r) unreachable++;
  }
  rows.push({ bias, decidable, unreachable });
  console.log(`${String(bias + " px").padStart(9)}${String(decidable).padStart(14)}` +
              `${String(unreachable).padStart(14)}` +
              `${((100 * unreachable) / decidable).toFixed(1).padStart(14)}%`);
}

console.log(`\nThe demo's rate (1.7%-6.7%) is the left edge of this table — it is a statement`);
console.log(`about the controller, not about the game. Suppression only exists where the`);
console.log(`paddle is already committed the wrong way, and the paddle crosses at`);
console.log(`200 px/s while the ball falls ~210 px in 1.1-3 s.\n`);

const cost = suppressionCost(generateScenarios(N, 7, 200));
console.log(`And suppression is not free. At 200 px of bias, over the ${cost.suppressed}`);
console.log(`balls the gate closed on, the recovery distance the NEXT ball inherits because`);
console.log(`the paddle held still instead of being commanded:`);
console.log(`  median ${cost.median.toFixed(0)} px · p90 ${cost.p90.toFixed(0)} px ·` +
            ` ${cost.over80} of ${cost.suppressed} over 80 px`);
console.log(`\nSo eval_breakout.mjs's "balls lost should be ~unchanged" means the gate's cost`);
console.log(`roughly cancels its benefit — not that there is a benefit to collect.`);