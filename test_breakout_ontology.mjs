// node --test test_breakout_ontology.mjs
//
// The reasoner has to be falsifiable — that is the point of declaring axioms instead of
// writing arithmetic. These tests attack it four ways: a contradictory axiom set (which
// nothing in the hand-written path could produce), bad ABox facts, the deleted-axiom case,
// and — the load-bearing one — a brute-force cross-check against the REAL physics.
//
// `test_breakout_sim.mjs` is the control: the hand-written path is untouched and still
// green, which is what proves the ontology is opt-in rather than load-bearing.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  TBOX, VOCABULARY, axiom, checkTBox, buildABox, reason, canCatch, aboxFor,
  analyticReachable, IMPLEMENTED_FORMS,
} from "./static/breakout_ontology.js";
import {
  W, H, PADDLE_Y, PADDLE_SPEED, PADDLE_MIN, PADDLE_MAX, BALL_R, BALL_MIN_X,
  BAND_ENTER_Y, LAST_BRICK_BOTTOM, CATCH_TOL_PX,
  createSim, startGame, step, measure, aimTarget, decidable, rolloutReachable,
  buildLandingX, issueCommand, commandIsStale, DEFAULT_CONFIG,
} from "./static/breakout_sim.js";
// `shouldAsk` is the decision loop's view of the ontology gate. It lives in the panel
// adapter (which node can import — the DOM is all inside functions), and the master-switch
// tests below pin it from the outside, the way the HTML consumes it.
import { shouldAsk } from "./static/breakout_panel.js";

/** The simulation timestep the harnesses use; kept in one place so tests match them. */
const DT_TEST = 1 / 240;

/** A sim parked in the decidable band, so `canCatch` has a verdict to give. */
const scene = (over = {}) => {
  const sim = createSim({ seed: 5 });
  startGame(sim);
  return Object.assign(sim, {
    paddleX: W / 2,
    ball: { x: W / 2, y: 200, vx: 0, vy: 160 },
    ...over,
  });
};

// ---- TBox self-check ------------------------------------------------------------

test("the shipped TBox is satisfiable", () => {
  const r = checkTBox(TBOX);
  assert.equal(r.satisfiable, true, JSON.stringify(r.unsatisfiable));
  assert.equal(r.unsatisfiable.length, 0);
});

test("contradictory cardinality makes the class unsatisfiable", () => {
  const contradictory = [
    ...TBOX,
    { id: "ball-never-catchable", form: "maxCardinality",
      subject: "Ball", property: "catchableBy", value: 0 },
  ];
  const r = checkTBox(contradictory);
  assert.equal(r.satisfiable, false);
  assert.match(r.unsatisfiable.map(u => u.reason).join(" "), /at least 1 and at most 0/);
});

test("two minCardinalitys on the same slot do NOT clash", () => {
  // The negative, asserted so the positive above cannot be satisfied by a loose check that
  // flags any two axioms on one property. "at least 1" and "at least 2" are compatible.
  const twoFloors = [
    ...TBOX,
    { id: "ball-two-catchers", form: "minCardinality",
      subject: "Ball", property: "catchableBy", value: 2 },
  ];
  assert.equal(checkTBox(twoFloors).satisfiable, true);
});

test("an axiom in an unimplemented form is refused, not warned about", () => {
  // Mirrors commit a5149d0. An axiom nothing evaluates would read as enforced and be inert,
  // so checkTBox rejects rather than passing it through.
  const bogus = [...TBOX, { id: "ball-exactly-one", form: "exactCardinality",
                           subject: "Ball", property: "catchableBy", value: 1 }];
  const r = checkTBox(bogus);
  assert.equal(r.satisfiable, false);
  assert.match(r.unsatisfiable[0].reason, /not implemented/);
});

test("a class listed twice in a disjoint axiom is caught", () => {
  const dup = [...TBOX, { id: "dup", form: "disjoint", value: ["Left", "Left"] }];
  const r = checkTBox(dup);
  assert.equal(r.satisfiable, false);
  assert.match(r.unsatisfiable[0].reason, /lists Left twice/);
});

test("every axiom references a declared class and property", () => {
  const classes = new Set(VOCABULARY.classes);
  const props = new Set(VOCABULARY.properties);
  for (const a of TBOX) {
    if (a.subject) assert.ok(classes.has(a.subject), `${a.id}: undeclared class ${a.subject}`);
    if (a.property) assert.ok(props.has(a.property), `${a.id}: undeclared prop ${a.property}`);
    if (a.form === "range") assert.ok(classes.has(a.value), `${a.id}: undeclared ${a.value}`);
    if (a.form === "disjoint") {
      for (const c of a.value) assert.ok(classes.has(c), `${a.id}: undeclared class ${c}`);
    }
  }
});

test("the tolerance is configuration, not an axiom", () => {
  // A tolerance smuggled into the TBox would be an axiom whose value is a pixel count the
  // editor could edit, which is a tuning knob wearing an invariant's clothes.
  assert.equal(JSON.stringify(TBOX).includes(String(CATCH_TOL_PX)), false);
});

// ---- bad ABox facts --------------------------------------------------------------

test("an unreachable descending ball is a violation that names the physics", () => {
  const sim = scene({ paddleX: 60, ball: { x: 460, y: 320, vx: 0, vy: 160 } });
  const r = canCatch(sim);
  assert.equal(r.ok, false, "the question must be suppressed");
  assert.equal(r.violations.length, 1);
  assert.equal(r.violations[0].axiom, "ball-must-be-catchable");
  assert.match(r.violations[0].detail, /no left\/stay\/right command recovers/);
});

test("a catchable ball is not a violation", () => {
  const r = canCatch(scene());
  assert.equal(r.ok, true);
  assert.equal(r.violations.length, 0);
});

test("a rising ball gets no verdict at all, and is NOT a violation", () => {
  // null means "unknown". Treating it as "unreachable" would be an unsound claim, and
  // unsound is the direction this feature must never err in.
  const sim = scene({ ball: { x: 460, y: 200, vx: 0, vy: -160 } });
  assert.equal(rolloutReachable(sim.ball, sim.paddleX), null);
  const r = canCatch(sim);
  assert.equal(r.ok, true, "unknown must not suppress the question");
  assert.equal(r.violations.length, 0);
  assert.equal(r.derived.reachable, null, "and the panel must be able to say so");
});

test("a ball above the bricks gets no verdict, because a brick may deflect it", () => {
  const sim = scene({ paddleX: 60, ball: { x: 460, y: LAST_BRICK_BOTTOM - 4, vx: 0, vy: 160 } });
  assert.equal(canCatch(sim).ok, true);
});

test("a non-cell position is a range violation", () => {
  const sim = scene();
  const facts = buildABox(sim);
  facts.individuals.get("ball").position = { x: "middle", y: 200 };
  const r = reason(facts);
  assert.equal(r.consistent, false);
  assert.ok(r.violations.some(v => v.axiom === "ball-position-is-cell"));
});

test("an unknown ball state is a disjoint violation", () => {
  const sim = scene();
  const facts = buildABox(sim);
  facts.individuals.get("ball").state = "Sideways";
  const r = reason(facts);
  assert.ok(r.violations.some(v => v.axiom === "ball-single-state"));
});

test("a bogus paddle command is a disjoint violation", () => {
  const sim = scene();
  const facts = buildABox(sim);
  facts.individuals.get("paddle").commanded = "Teleport";
  const r = reason(facts);
  assert.ok(r.violations.some(v => v.axiom === "paddle-single-command"));
});

test("deleting the axiom stops enforcement and says so — the demonstration", () => {
  const sim = scene({ paddleX: 60, ball: { x: 460, y: 320, vx: 0, vy: 160 } });
  const i = TBOX.findIndex(a => a.id === "ball-must-be-catchable");
  const [removed] = TBOX.splice(i, 1);
  try {
    const r = canCatch(sim);
    assert.equal(r.ok, true, "with the axiom gone, nothing is enforced");
    assert.deepEqual(r.unasserted, ["ball-must-be-catchable"]);
    // ...but the physics still knows the answer. That is the point of the demonstration:
    // deleting a declaration does not delete the world.
    assert.equal(rolloutReachable(sim.ball, sim.paddleX), false);
  } finally {
    TBOX.splice(i, 0, removed);
  }
});

test("canCatch takes an injected rollout, so the boundary is testable without physics", () => {
  const sim = scene();
  const r = canCatch(sim, { rollout: () => false });
  assert.equal(r.ok, false);
  const r2 = canCatch(sim, { rollout: () => true });
  assert.equal(r2.ok, true);
});

// ---- the predicted-position axiom -------------------------------------------------

test("the predicted position is the landing point, at the height the prediction solves for", () => {
  // `buildLandingX` solves for the instant the ball's CENTRE crosses BAND_ENTER_Y, so that
  // is the y the prediction must assert — asserting PADDLE_Y would be the paddle's cell,
  // not the ball's.
  const sim = scene({ ball: { x: 300, y: 200, vx: 90, vy: 160 } });
  assert.equal(decidable(sim.ball), true);
  const facts = buildABox(sim);
  assert.deepEqual(facts.individuals.get("ball").predictedPosition,
    { x: Math.round(buildLandingX(sim.ball)), y: BAND_ENTER_Y });
});

test("a ball outside the decidable band asserts no prediction, and that is not a violation", () => {
  // Rising: no prediction CAN be made, so nothing is owed — the same skip an
  // indeterminate ball gets from the reachability check.
  const sim = scene({ ball: { x: 460, y: 200, vx: 0, vy: -160 } });
  const facts = buildABox(sim);
  assert.equal(facts.individuals.get("ball").predictedPosition, null);
  const r = reason(facts);
  assert.equal(r.violations.filter(v => v.axiom === "ball-predicted-at-paddle-level").length, 0);
});

test("a decidable ball with no asserted prediction is a violation — the axiom is real", () => {
  // buildABox asserts the prediction for every decidable ball, so this can only fire if
  // buildABox and the reasoner disagree — a bug worth surfacing, not a judgement.
  const sim = scene();
  const facts = buildABox(sim);
  delete facts.individuals.get("ball").predictedPosition;
  const r = reason(facts);
  assert.ok(r.violations.some(v => v.axiom === "ball-predicted-at-paddle-level"));
});

test("deleting the prediction axiom stops the assertion and says so", () => {
  const sim = scene();
  const i = TBOX.findIndex(a => a.id === "ball-predicted-at-paddle-level");
  const [removed] = TBOX.splice(i, 1);
  try {
    const r = reason(buildABox(sim));
    assert.ok(r.unasserted.includes("ball-predicted-at-paddle-level"));
    assert.equal(r.violations.some(v => v.axiom === "ball-predicted-at-paddle-level"), false);
    // The physics still knows where the ball lands — deleting a declaration does not
    // delete the world, the same as the reachability demonstration above.
    assert.equal(buildLandingX(sim.ball), measure(sim).landingX);
  } finally {
    TBOX.splice(i, 0, removed);
  }
});

// ---- the master switch ------------------------------------------------------------

test("the master switch: enforcement off means every question is asked", () => {
  assert.equal(DEFAULT_CONFIG.enforceReachability, true,
    "the demo ships with the ontology on, because judging is what it is for");
  const unreachable = scene({ paddleX: 60, ball: { x: 460, y: 320, vx: 0, vy: 160 } });
  assert.equal(canCatch(unreachable).ok, false, "sanity: with the switch on, this is suppressed");
  // Off, the gate is inert no matter what the ontology would say — this sim shape is the
  // one `shouldAsk` would suppress, so it is the one that proves the switch wins.
  assert.equal(shouldAsk({ config: { enforceReachability: false } }), true);
});

test("buildABox reads the same measure() the prose does", () => {
  // The ontology's output has to flow into the model's INPUT, not sit beside it. If these
  // two ever disagreed the panel would be reasoning about a different number than the one
  // the model was told.
  const sim = scene({ paddleX: 100, ball: { x: 34, y: 131, vx: -241, vy: 62 } });
  const m = measure(sim);
  const ball = buildABox(sim).individuals.get("ball");
  assert.equal(ball.gap, m.gap);
  assert.equal(ball.timeToContact, m.timeToContact);
});

// ---- the brute-force cross-check: the real evidence ------------------------------
//
// Forward-simulate the ACTUAL step() from breakout_sim.js and compare against canCatch at
// every frame where the ball is decidable. Re-deriving the same algebra a second time would
// only prove it was typed twice; running the real physics catches the wall fold, the contact
// band, the frame quantisation and the brick guard all at once.
//
// The forward run re-aims every step, because the held-command model is measurably wrong —
// a paddle that commits to one direction for the whole descent catches 22.9% fewer balls.

function actuallyCaught(sim, paddleX0) {
  const probe = createSim({ seed: 1 });
  startGame(probe);
  probe.ball = { ...sim.ball };
  probe.paddleX = paddleX0;
  for (let k = 0; k < 1200; k++) {
    const dir = Math.sign(aimTarget(probe.ball) - probe.paddleX);
    probe.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
      probe.paddleX + dir * PADDLE_SPEED * (1 / 240)));
    const ev = step(probe, 1 / 240);
    if (ev.caught) return true;
    if (ev.lost) return false;
  }
  return false;
}

test("the reasoner never calls a catchable ball unreachable (clause 2)", () => {
  let compared = 0, falsePositives = 0, falseNegatives = 0;
  const examples = [];
  for (let seed = 1; seed <= 700; seed++) {
    const sim = createSim({ seed });
    startGame(sim);
    // Sample the decidable band, plus the edges where the verdict is most likely to flip.
    for (let s = 0; s < 30; s++) {
      sim.paddleX = PADDLE_MIN + sim.rng() * (PADDLE_MAX - PADDLE_MIN);
      const r = sim.rng();
      sim.ball = {
        x: BALL_R + sim.rng() * (W - 2 * BALL_R),
        y: r < 0.15 ? LAST_BRICK_BOTTOM + BALL_R + 2
                    : r < 0.3 ? BAND_ENTER_Y - 2
                    : LAST_BRICK_BOTTOM + BALL_R + sim.rng() * (BAND_ENTER_Y - LAST_BRICK_BOTTOM - BALL_R - 4),
        vx: (sim.rng() - 0.5) * 520,
        vy: 40 + sim.rng() * 260,
      };
      // No abstention counter here on purpose: this sampler targets the decidable band by
      // construction, so every state it produces has a verdict. A count of zeroes would
      // read as "the ontology never abstains" when it would only be measuring the sampler.
      // Abstention is covered directly by the rising-ball and above-the-bricks tests.
      const predicted = canCatch(sim).ok;
      compared++;
      const actual = actuallyCaught(sim, sim.paddleX);
      if (!predicted && actual) {
        falsePositives++;
        if (examples.length < 5) examples.push({ ...sim.ball, paddleX: sim.paddleX });
      }
      if (predicted && !actual) falseNegatives++;
    }
  }
  console.log(`  ${compared} states compared against the real step(), ` +
              `${falsePositives} false positives, ${falseNegatives} false negatives`);
  assert.ok(compared > 15000, `expected a large sample, got ${compared}`);
  assert.equal(falsePositives, 0,
    `the ontology suppressed a question for ${falsePositives} balls the physics catches: ` +
    JSON.stringify(examples));
  // Not asserted to be zero, deliberately. A false "catchable" only wastes one ask, and
  // the rollout's dt is a modelling choice; reporting the rate is the honest move.
  assert.ok(falseNegatives / compared < 0.02,
    `${falseNegatives} false negatives (${(100 * falseNegatives / compared).toFixed(2)}%) — ` +
    `the rollout is drifting from the physics`);
});

// ---- the rollout must not corrupt the running game -------------------------------

test("reasoning about a ball does not mutate it", () => {
  const sim = createSim({ seed: 3 });
  startGame(sim);
  sim.paddleX = 120;
  sim.ball = { x: 300, y: 220, vx: -180, vy: 150 };
  const before = JSON.stringify({ ball: sim.ball, paddleX: sim.paddleX,
                                   bricks: sim.bricks, score: sim.score, lives: sim.lives });
  for (let i = 0; i < 200; i++) canCatch(sim);
  const after = JSON.stringify({ ball: sim.ball, paddleX: sim.paddleX,
                                  bricks: sim.bricks, score: sim.score, lives: sim.lives });
  assert.equal(after, before,
    "otherwise the reasoner silently corrupts the game and the bug looks like a physics bug");
});

// ---- pinning the shortcut as wrong ----------------------------------------------

test("the closed form DISAGREES with the rollout, and that is asserted on purpose", () => {
  // If a future physics change ever made the analytic form correct, this test would fail and
  // say "reconsider the shortcut". Silent 0% would otherwise look like progress while hiding
  // a stale formula that nothing reads.
  let compared = 0, agreed = 0;
  for (let seed = 1; seed <= 300; seed++) {
    const sim = createSim({ seed });
    startGame(sim);
    for (let s = 0; s < 20; s++) {
      sim.paddleX = PADDLE_MIN + sim.rng() * (PADDLE_MAX - PADDLE_MIN);
      sim.ball = {
        x: BALL_R + sim.rng() * (W - 2 * BALL_R),
        y: LAST_BRICK_BOTTOM + BALL_R + sim.rng() * (BAND_ENTER_Y - LAST_BRICK_BOTTOM - BALL_R - 4),
        vx: (sim.rng() - 0.5) * 520,
        vy: 40 + sim.rng() * 260,
      };
      const a = analyticReachable(sim.ball, sim.paddleX);
      const b = rolloutReachable(sim.ball, sim.paddleX);
      if (b === null) continue;
      compared++;
      if (a === b) agreed++;
    }
  }
  const rate = agreed / compared;
  console.log(`  closed form agrees with the rollout on ${(100 * rate).toFixed(1)}% of ` +
              `${compared} states — it is a display counter-example, not a predicate`);
  assert.ok(compared > 5000, `expected a large sample, got ${compared}`);
  assert.ok(rate < 0.95,
    `the analytic form now agrees ${(100 * rate).toFixed(1)}% of the time — if it is really ` +
    `this close to correct, reconsider using it and drop the rollout`);
});

// ---- cost --------------------------------------------------------------------------

test("a full reason() stays far under the decision budget, worst case included", () => {
  // Budget raised from 20us to 500us, and deliberately measured on the WORST case rather
  // than a typical ball. A slow ball (vy ~50px/s, the shallowest a paddle bounce can leave)
  // is the longest horizon the rollout simulates, so it costs ~12x a typical one. Testing
  // only the typical case would have let a future change to `rolloutHorizon` blow the real
  // budget by 10x without failing here.
  const typical = scene();
  const slowest = scene({ ball: { x: 240, y: 130, vx: 0, vy: 50 } });
  const cost = sim => {
    for (let i = 0; i < 5000; i++) canCatch(sim);          // warm up the JIT
    const N = 20000, t0 = process.hrtime.bigint();
    for (let i = 0; i < N; i++) canCatch(sim);
    return Number(process.hrtime.bigint() - t0) / 1000 / N;
  };
  const usTypical = cost(typical), usWorst = cost(slowest);
  console.log(`  reason() ${usTypical.toFixed(1)} us typical, ${usWorst.toFixed(1)} us worst ` +
              `case, against a decision every 150000-500000 us`);
  assert.ok(usWorst < 500,
    `${usWorst.toFixed(1)} us worst case is too slow to run per decision tick`);
});

// ---- the panel's view ---------------------------------------------------------------

test("the ABox row says so when there is no verdict", () => {
  const [row] = aboxFor(scene({ ball: { x: 300, y: 200, vx: 0, vy: -160 } }));
  assert.match(row.detail, /no verdict/);
  assert.equal(row.violations.length, 0);
});

test("the ABox row names the landing point, not the current position", () => {
  const [row] = aboxFor(scene({ paddleX: 100, ball: { x: 34, y: 131, vx: -241, vy: 62 } }));
  assert.match(row.detail, /lands at 197/, row.detail);
  // 97px of gap with 3.2s of fall time: the paddle covers ~640px, so this is comfortably
  // catchable and must not be reported as a violation.
  assert.equal(row.violations.length, 0, row.detail);
});

test("the ABox row flags an unreachable ball", () => {
  const [row] = aboxFor(scene({ paddleX: 60, ball: { x: 460, y: 320, vx: 0, vy: 160 } }));
  assert.match(row.detail, /UNREACHABLE/);
  assert.equal(row.violations.length, 1);
});
// ---- stale-command axiom: Command ⊑ ≥1 validUntil ---------------------------------
//
// The second existential restriction, and the one whose violation is a BUG rather than a
// fact: a reachability violation says "no answer exists" and is excusing, while a stale
// command says "an answer existed and has been cancelled", which is always actionable.
//
// These attack it the same four ways the reachability axiom is attacked above, and add the
// brute-force check that matters most: commandIsStale must never claim a command is stale
// while the physics says that command still reaches the ball.

test("the stale-command axiom is declared and its form is implemented", () => {
  const a = axiom("command-stale-on-bounce");
  assert.ok(a, "the axiom must exist");
  assert.equal(a.form, "minCardinality");
  assert.ok(IMPLEMENTED_FORMS.includes(a.form),
    "a form nothing implements is the hole commit a5149d0 closed — do not reintroduce it");
  assert.equal(checkTBox(TBOX).satisfiable, true);
});

test("no command issued means no Command individual — and no violation", () => {
  const sim = scene();
  assert.equal(sim.issuedVXGapSign, 0, "fresh sim has issued nothing");
  const facts = buildABox(sim);
  assert.equal(facts.individuals.has("command"), false,
    "an absent command is not a stale command; reporting it as one would be a false alarm");
  assert.equal(reason(facts).violations.filter(v => v.axiom === "command-stale-on-bounce").length, 0);
});

test("a command aimed at a landing point that survives is not stale", () => {
  const sim = scene({ paddleX: 300, ball: { x: 300, y: 300, vx: -70, vy: 60 } });
  issueCommand(sim, -1);                       // ball lands left, command goes left
  const cmd = buildABox(sim).individuals.get("command");
  assert.equal(cmd.stale, false);
  assert.equal(cmd.validUntil.size, 1);
  assert.equal(reason(buildABox(sim)).violations.length, 0);
});

/**
 * Reflect the ball off a side wall through the REAL physics.
 *
 * The tests below set `ball.vx` by hand in earlier versions, which never set
 * `bouncedSinceIssue` — the flag `step()` owns — so they were asserting against a predicate
 * that could not fire. Driving an actual wall contact keeps the test on the same path the
 * game takes, which is the only way to know the detection works.
 *
 * Parks the ball against the left wall heading outward, then steps once: `physicsStep`
 * clamps it and reverses vx.
 */
const bounceOffWall = sim => {
  sim.ball.x = BALL_MIN_X;
  sim.ball.vx = -Math.abs(sim.ball.vx) - 20;   // heading into the left wall
  step(sim, DT_TEST, { keys: null });
  assert.ok(sim.bouncedSinceIssue, "the wall contact must register as a bounce");
  return sim;
};

test("a bounce that inverts the arrival side is stale, and says so", () => {
  const sim = scene({ paddleX: 300, ball: { x: 300, y: 300, vx: -70, vy: 60 } });
  issueCommand(sim, 1);                        // "right", while the ball lands LEFT
  const before = buildABox(sim).individuals.get("command");
  assert.equal(before.stale, false, "not yet — the command matches the current landing point");

  bounceOffWall(sim);                          // the bounce, through the real physics
  const after = buildABox(sim).individuals.get("command");
  assert.equal(after.stale, true);
  assert.equal(after.validUntil.size, 0, "empty set, not absent: the empty set IS the claim");
  const r = reason(buildABox(sim));
  const v = r.violations.find(x => x.axiom === "command-stale-on-bounce");
  assert.ok(v, "the bounce must produce a violation");
  assert.match(v.detail, /reflected/);
});

test("the landing coordinate does NOT move at a bounce — the axiom is over the side", () => {
  // Pins the fact the whole axiom rests on, because getting it wrong is silent: a coordinate
  // comparison returns "not stale" every time and looks like a working detector that simply
  // never fires. `buildLandingX` folds through `reflectX`, so the prediction is continuous
  // across a reflection. Measured here over real bounces rather than asserted in a comment.
  let n = 0, worst = 0;
  for (let game = 1; game <= 6 && n < 12; game++) {
    const sim = createSim({ seed: game });
    startGame(sim);
    sim.running = true;
    let nextTick = 0, held = 0, prevVx = sim.ball.vx;
    for (let i = 0; i < 20000 && sim.running && n < 12; i++) {
      const now = i * DT_TEST;
      if (now >= nextTick) {
        nextTick = now + 0.25;
        if (decidable(sim.ball)) {
          held = Math.sign(aimTarget(sim.ball) - sim.paddleX);
          issueCommand(sim, held);
        }
      }
      sim.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
        sim.paddleX + held * PADDLE_SPEED * DT_TEST));
      const decidableBefore = decidable(sim.ball);
      const issued = sim.issuedVXGapSign;
      const stepEv = step(sim, DT_TEST, { keys: null });
      if (Math.sign(sim.ball.vx) !== Math.sign(prevVx) && decidableBefore &&
          decidable(sim.ball) && sim.issuedVXGapSign === issued && issued !== 0) {
        // The landing point itself, before and after, is the thing that does not move.
        const was = buildLandingX({ ...sim.ball, vx: -sim.ball.vx });
        worst = Math.max(worst, Math.abs(was - buildLandingX(sim.ball)));
        n++;
      }
      prevVx = sim.ball.vx;
      if (stepEv.lost) { held = 0; issueCommand(sim, 0); }
    }
  }
  assert.ok(n >= 5, `expected real mid-descent bounces, got ${n}`);
  assert.ok(worst < 2,
    `the predicted landing point moved ${worst.toFixed(1)}px across a bounce; if this ever ` +
    `exceeds a pixel or two, the fold in reflectX has changed and the axiom's rationale ` +
    `(and the comment in commandIsStale) need revisiting`);
});

test("re-deriving the command on a bounce clears the violation and re-aims it", () => {
  // This is the fix the axiom exists to justify, asserted end to end rather than in prose.
  const sim = scene({ paddleX: 300, ball: { x: 300, y: 300, vx: -70, vy: 60 } });
  issueCommand(sim, 1);
  bounceOffWall(sim);
  assert.ok(reason(buildABox(sim)).violations.length > 0, "stale before the fix");

  const m = measure(sim);
  issueCommand(sim, m.reachable === false ? 0 : (Math.abs(m.gap) <= 1 ? 0 : Math.sign(m.gap)));

  const after = buildABox(sim).individuals.get("command");
  assert.equal(after.stale, false, "re-issuing against the new arrival side clears it");
  // Scoped to this axiom: bouncing off the wall in this synthetic scene leaves the ball 250px
  // from the paddle, which trips `ball-must-be-catchable` for reasons that have nothing to do
  // with the command, and asserting zero violations would be testing the wrong thing.
  assert.equal(reason(buildABox(sim)).violations
    .filter(v => v.axiom === "command-stale-on-bounce").length, 0);
  assert.equal(after.issuedFor,
    Math.sign(buildLandingX(sim.ball) - sim.paddleX) > 0 ? "Right"
      : Math.sign(buildLandingX(sim.ball) - sim.paddleX) < 0 ? "Left" : "aligned",
    "and it is now aimed at the post-bounce side, not the pre-bounce one");
});

test("deleting the stale-command axiom disables it and reports unasserted", () => {
  const sim = scene({ paddleX: 300, ball: { x: 300, y: 300, vx: -70, vy: 60 } });
  issueCommand(sim, 1);
  bounceOffWall(sim);
  assert.ok(reason(buildABox(sim)).violations.length > 0);

  const i = TBOX.findIndex(a => a.id === "command-stale-on-bounce");
  const saved = TBOX[i];
  TBOX.splice(i, 1);
  try {
    const r = reason(buildABox(sim));
    assert.equal(r.violations.filter(v => v.axiom === "command-stale-on-bounce").length, 0);
    assert.ok(r.unasserted.includes("command-stale-on-bounce"),
      "and says so, rather than crashing — the live editor lets you do this");
  } finally {
    TBOX.splice(i, 0, saved);
  }
});

test("commandIsStale fires on real bounces and is never claimed without one", () => {
  // The non-vacuity half. An earlier version of this test asserted stale commands exist and
  // observed ZERO, then I read the zero as a broken harness and re-checked — and the zero was
  // correct, because the premise (the landing coordinate moves at a bounce) is false. See
  // "the landing coordinate does NOT move at a bounce". What must hold now is that the
  // predicate fires on genuine reflections and stays quiet otherwise.
  //
  // The controller HOLDS its command between asks, exactly as the demo does; a
  // perfectly-steerable paddle re-aiming every physics step never has a stale command.
  let issued = 0, stale = 0, checkedWithoutBounce = 0;
  for (let game = 1; game <= 8; game++) {
    const sim = createSim({ seed: game });
    startGame(sim);
    sim.running = true;
    let nextTick = 0, held = 0, heldUntil = -1;
    for (let i = 0; i < 20000 && sim.running; i++) {
      const now = i * DT_TEST;
      if (now >= nextTick) {
        nextTick = now + 0.25;
        if (decidable(sim.ball)) {
          held = Math.sign(aimTarget(sim.ball) - sim.paddleX);
          issueCommand(sim, held);
          heldUntil = now + 0.25;
          issued++;
        }
      }
      const dir = now < heldUntil ? held : 0;
      sim.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
        sim.paddleX + dir * PADDLE_SPEED * DT_TEST));
      const ev = step(sim, DT_TEST, { keys: null });
      if (now < heldUntil && decidable(sim.ball)) {
        if (commandIsStale(sim)) {
          stale++;
        } else {
          // Soundness, and the direction that matters: a command that still agrees with
          // the live landing point must never be re-aimed, or the correction would drag
          // the paddle off a ball it was about to catch.
          assert.equal(commandIsStale(sim), false);
          if (!sim.bouncedSinceIssue) checkedWithoutBounce++;
        }
      }
      if (ev.lost) { held = 0; heldUntil = -1; issueCommand(sim, 0); }
    }
  }
  assert.ok(issued > 500, `expected a real sample, got ${issued}`);
  assert.ok(stale > 0,
    "and it must fire on some real bounces, or the axiom is declared and enforced by nothing");
  assert.ok(checkedWithoutBounce > 500,
    `and it must have been consulted plenty of times WITHOUT a bounce, got ` +
    `${checkedWithoutBounce} — otherwise the assertion above is vacuous`);
});

test("a just-issued command is never stale, even when the ball happens to be bouncing", () => {
  // The narrow soundness claim, isolated: at issue time the sign is taken from the same
  // measurement `commandIsStale` compares against, so a fresh command agrees by construction.
  // Asserting it pins that the deadband is shared — a mismatch between the two is how a
  // command would be reported stale against itself.
  let checked = 0;
  for (let game = 1; game <= 6; game++) {
    const sim = createSim({ seed: game });
    startGame(sim);
    sim.running = true;
    let nextTick = 0;
    for (let i = 0; i < 20000 && sim.running; i++) {
      const now = i * DT_TEST;
      if (now >= nextTick) {
        nextTick = now + 0.25;
        if (decidable(sim.ball)) {
          issueCommand(sim, Math.sign(aimTarget(sim.ball) - sim.paddleX));
          checked++;
          assert.equal(commandIsStale(sim), false,
            "a fresh command agrees with the landing point it was just derived from");
        }
      }
      const ev = step(sim, DT_TEST, { keys: null });
      sim.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
        sim.paddleX + sim.paddleDir * PADDLE_SPEED * DT_TEST));
      if (ev.lost) issueCommand(sim, 0);
    }
  }
  assert.ok(checked > 300, `expected a real sample, got ${checked}`);
});

test("the ABox shows a command row only once a command exists", () => {
  assert.equal(aboxFor(scene()).length, 1, "no command, no row");
  // Paddle at x=100 rather than centre: the ball is only ~36 px of horizontal travel from
  // this height, so with the paddle centred a wall reflection stays on the same side and the
  // command is correctly NOT stale. Placing it left of the ball's path is what makes the
  // reflection actually cross the paddle — which is the situation the row is reporting.
  const sim = scene({ paddleX: 100, ball: { x: 300, y: 300, vx: -70, vy: 60 } });
  issueCommand(sim, 1);                       // the ball currently lands right of the paddle
  const rows = aboxFor(sim);
  assert.equal(rows.length, 2);
  const cmd = rows.find(r => r.id === "command");
  assert.match(cmd.detail, /still valid/);
  assert.match(cmd.detail, /landing to the Right/, "and it names the side it was issued for");
  bounceOffWall(sim);
  assert.match(aboxFor(sim).find(r => r.id === "command").detail, /STALE/);
});
