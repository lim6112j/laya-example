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
  W, H, PADDLE_Y, PADDLE_SPEED, PADDLE_MIN, PADDLE_MAX, BALL_R,
  BAND_ENTER_Y, LAST_BRICK_BOTTOM, CATCH_TOL_PX,
  createSim, startGame, step, measure, aimTarget, decidable, rolloutReachable,
} from "./static/breakout_sim.js";

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