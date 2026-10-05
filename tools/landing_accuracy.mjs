// How accurate is the landing point the model is actually shown?
//
//   node tools/landing_accuracy.mjs
//
// WHY THIS EXISTS. Two claims in the README rest on this and are easy to get backwards.
//
// The claim that matters: a wall reflection does NOT degrade the prediction. `buildLandingX`
// expands not-yet-happened reflections through the `reflectX` triangle wave, so a bounce the
// ball has not reached yet is already priced into the number handed to the model. That is
// what makes re-aiming on a bounce worth +48 score in stale_command.mjs: the model is told the
// truth in advance and still has to wait out its 0.25 s cadence to act on it.
//
// THE TRAP IN THE OTHER DIRECTION, because the first version of this measurement printed
// "the shown gap points the wrong way 38.6% of the time", which is alarming and false:
//
//   It compared the sign of the shown gap against the ball's position relative to the paddle
//   AT THE MOMENT OF THE CATCH. But the paddle has been travelling toward the landing point
//   the whole descent, so by the catch it is frequently on the other side of the ball — the
//   controller did exactly what it was told and arrived. Comparing a prediction made at ask
//   time against a state observed later measures the paddle's motion, not the prediction's
//   error. The correct comparison is the sign shown at ask time against the direction that
//   was needed at ask time.
//
// The number worth reading is the absolute error, which is measured the same way either
// time: a median of 0.2 px against a catch tolerance of ±40 px is three orders of magnitude
// inside the window, which is the whole finding.

import { PADDLE_SPEED, PADDLE_MIN, PADDLE_MAX, DEFAULT_CONFIG,
         createSim, startGame, step, decidable, buildLandingX, measure, aimTarget }
  from "../static/breakout_sim.js";

const DT = 1 / 240;
const DECIDE_EVERY = 0.25;
const GAMES = 20;

const snapBall = b => ({ x: b.x, y: b.y, vx: b.vx, vy: b.vy });

/**
 * At every decision tick, compare the landing x the model is shown against where the ball is
 * actually caught under perfect play from that instant.
 *
 * Perfect play is used deliberately here: the question is what the OBSERVATION says, so the
 * controller must be correct in order to isolate it. This is a measurement of the predicted
 * value, not of any policy.
 */
const errors = [];
let compared = 0, signWrongAtAsk = 0, signWrongAtCatch = 0;
let reflN = 0, reflErr = 0, noReflN = 0, noReflErr = 0;

for (let game = 1; game <= GAMES; game++) {
  const sim = createSim({ seed: game, config: { ...DEFAULT_CONFIG } });
  startGame(sim);
  let nextTick = 0;

  for (let i = 0; i < 20000 && sim.running; i++) {
    const now = i * DT;
    if (now >= nextTick) {
      nextTick = now + DECIDE_EVERY;
      if (decidable(sim.ball)) {
        const shown = buildLandingX(sim.ball);
        const gapShown = shown - sim.paddleX;      // the sign the model is handed

        const probe = createSim({ seed: game, config: { ...DEFAULT_CONFIG } });
        probe.ball = snapBall(sim.ball);
        probe.paddleX = sim.paddleX;
        probe.bricks = sim.bricks.map(r => [...r]);
        probe.running = true;

        let caughtX = null, reflected = false, prevVx = probe.ball.vx;
        for (let k = 0; k < 1200; k++) {
          const d = Math.sign(aimTarget(probe.ball) - probe.paddleX);
          probe.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
            probe.paddleX + d * PADDLE_SPEED * DT));
          const ev = step(probe, DT, { keys: null });
          if (Math.sign(probe.ball.vx) !== Math.sign(prevVx)) reflected = true;
          prevVx = probe.ball.vx;
          if (ev.caught) { caughtX = probe.ball.x; break; }
          if (ev.lost) break;
        }

        if (caughtX !== null) {
          compared++;
          const err = Math.abs(caughtX - shown);
          errors.push(err);
          if (reflected) { reflN++; reflErr += err; } else { noReflN++; noReflErr += err; }

          if (Math.abs(gapShown) > 1) {
            // Correct comparison: what the model was told, against what it needed at ask
            // time. Identical by construction here — `shown` IS the landing point — so this
            // is a sanity check that the sign convention did not flip, not a measurement.
            if (Math.sign(gapShown) !== Math.sign(shown - sim.paddleX)) signWrongAtAsk++;
            // The artifact version, kept to document the 38.6% that was once reported.
            if (Math.sign(caughtX - probe.paddleX) !== Math.sign(gapShown)) signWrongAtCatch++;
          }
        }
      }
    }
    step(sim, DT, { keys: null });
  }
}

errors.sort((a, b) => a - b);
const quantile = q => errors[Math.floor(errors.length * q)];

console.log(`Landing-point accuracy — ${GAMES} games, ${compared} asked-and-catchable states.\n`);
console.log(`  |shown landingX - actual catch x|`);
console.log(`    median ${quantile(0.5).toFixed(1)} px   p90 ${quantile(0.9).toFixed(1)} px   max ${errors[errors.length - 1].toFixed(1)} px`);
console.log(`\n  split by whether a wall reflection intervened before the catch:`);
console.log(`    reflected: n=${String(reflN).padStart(3)}   mean err ${(reflErr / reflN).toFixed(2)} px`);
console.log(`    no bounce: n=${String(noReflN).padStart(3)}   mean err ${(noReflErr / noReflN).toFixed(2)} px`);
console.log(`\n  Reflections do NOT degrade the prediction — buildLandingX expands them in advance`);
console.log(`  through reflectX. The model is told the post-bounce truth before the bounce.`);
console.log(`\n  sign shown vs direction needed at ASK time : ${signWrongAtAsk} wrong (sanity check, should be 0)`);
console.log(`  sign shown vs paddle at CATCH time        : ${signWrongAtCatch} wrong (${(100 * signWrongAtCatch / compared).toFixed(1)}%)`);
console.log(`  -> the second number measures the paddle's motion during the descent, not the`);
console.log(`     prediction's error. It is reported only because it was once mistaken for one.`);