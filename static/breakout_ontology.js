// A TBox/ABox reasoner for the Breakout domain — the ontology behind the reachability rule.
//
// WHY A SEPARATE MODULE AND NOT A GENERALISED ontology.js. `ontology.js` is coherent,
// heavily tested, and its `reason()` has three hardcoded axiom ids that the README names as
// a deliberate coupling. Parameterising them is a rewrite of the thing those tests protect,
// and `buildABox` is fleet-shaped (`bus.onboard`, `facts.seats`) with no seam to generalise
// without a per-domain adapter — the same amount of code as a second module, with the
// rewrite's risk on top. So: two modules, one shared core for the domain-free editor.
//
// WHAT IS DECLARED HERE AND WHAT IS COMPUTED. The Tier 0 claim is `Ball ⊑ ≥1 catchableBy` —
// a descending ball below the bricks that no paddle command reaches is a violation, not a
// bad answer. The verdict itself comes from the rollout in breakout_sim.js, injected as an
// extension exactly the way fleet injects `detour`: cost is not logic, and a second
// implementation of reachability would be a second thing to get wrong.
//
// DEPENDENCY DIRECTION. This module imports breakout_sim.js; breakout_sim.js must not
// import this one. One-way, so there is no cycle — the same constraint fleet documents at
// fleet_sim.js:20-24.
//
// SCOPE, STATED UP FRONT, because it is the same subset ontology.js declares: an OWL-RL
// subset, closed-world. Absence of a fact is absence, not openness. That is right for an
// operational decision system and is NOT OWL's default.

import {
  PADDLE_Y, PADDLE_SPEED, BALL_R,
  LAST_BRICK_BOTTOM, BAND_ENTER_Y, CATCH_TOL_PX,
  decidable, rolloutReachable, buildLandingX, measure, commandIsStale,
} from "./breakout_sim.js";
import { checkTBox as coreCheckTBox } from "./ontology_core.js";

// ---- vocabulary ---------------------------------------------------------------
// Declared, not derived. These are the only class names the axioms may reference.

export const VOCABULARY = {
  // Left/Stay/Right are declared as CLASSES even though they are really command values: a
  // `disjoint` axiom lists class names by definition, and `paddle-single-command` asserts
  // over exactly those three. Declaring them as bare strings the vocabulary does not know
  // would let an axiom reference a name nothing else in the file can resolve.
  //
  // `Command` and `Stale` come from the stale-command axiom. `Command` is the individual a
  // held answer, and `Stale` is the state it falls into when the ball bounces and the
  // coordinate it was issued against no longer exists. See `command-stale-on-bounce`.
  classes: ["Ball", "Paddle", "GridCell", "Ascending", "Descending", "Caught", "Lost",
            "Left", "Stay", "Right", "Command", "Stale"],
  properties: ["position", "catchableBy", "commanded", "state", "issuedFor", "validUntil",
               "predictedPosition"],
  // A ball's lifecycle states are mutually exclusive. `Ball` itself is a thing, not a
  // state; the four below are the four mutually exclusive ones.
  states: ["Ascending", "Descending", "Caught", "Lost"],
  // The three things the paddle can be told to do. Exclusive, and always exactly one.
  commands: ["Left", "Stay", "Right"],
};

// ---- TBox: the axioms ---------------------------------------------------------
// Declarations, not control flow. Read by checkTBox and by reason(), and editable live.

export const TBOX = [
  {
    id: "ball-must-be-catchable",
    form: "minCardinality",
    subject: "Ball",
    property: "catchableBy",
    value: 1,
    comment: "Tier 0. A descending ball below the bricks that no paddle command reaches. " +
             "The verdict comes from the physics rollout in breakout_sim.js, injected — " +
             "this axiom declares that the answer exists, not how to compute it.",
  },
  {
    id: "ball-predicted-at-paddle-level",
    form: "minCardinality",
    subject: "Ball",
    property: "predictedPosition",
    value: 1,
    comment: "A ball that will reach paddle height has a predictable position when it " +
             "gets there. Computed from the CURRENT ball position by the physics " +
             "rollout (`buildLandingX`, with side-wall folding), injected exactly like " +
             "reachability — the axiom declares that the prediction exists and is part " +
             "of what the ontology says, not an ad-hoc read of the sim. Same injection " +
             "style as `ball-must-be-catchable`: cost is not logic, and a second " +
             "implementation of the landing point would be a second thing to get wrong.",
  },
  {
    id: "command-stale-on-bounce",
    form: "minCardinality",
    subject: "Command",
    property: "validUntil",
    value: 1,
    comment: "A command is issued against the SIDE of the paddle the ball was headed for when " +
             "the model answered. If the ball then reflects off a side wall, that side can " +
             "invert, so the command has no valid target and the paddle is steering toward " +
             "a trajectory that was cancelled. Stated over the side rather than the landing " +
             "coordinate because buildLandingX folds through reflectX and is continuous " +
             "across a reflection — the coordinate does not move, so comparing it detects " +
             "nothing. This is the one axiom in this file whose violation is actionable " +
             "rather than excusing: unlike a reachability violation, it is never right — " +
             "re-deriving the command from the live sim costs no model call.",
  },
  {
    id: "paddle-single-command",
    form: "disjoint",
    value: ["Left", "Stay", "Right"],
    comment: "Tier 0. The paddle obeys exactly one command; being in two at once is a " +
             "violation, which is what a stale in-flight answer would look like.",
  },
  {
    id: "ball-single-state",
    form: "disjoint",
    value: ["Ascending", "Descending", "Caught", "Lost"],
    comment: "Tier 0. A ball is rising or falling or caught or lost — never two at once.",
  },
  {
    id: "ball-position-is-cell",
    form: "range",
    subject: "Ball",
    property: "position",
    value: "GridCell",
  },
  {
    id: "paddle-position-is-cell",
    form: "range",
    subject: "Paddle",
    property: "position",
    value: "GridCell",
  },
];

export const axiom = id => TBOX.find(a => a.id === id);

/**
 * The axiom forms THIS reasoner evaluates.
 *
 * Per-domain on purpose, and not a shared list. Commit a5149d0 made `checkTBox` reject any
 * axiom whose form nothing implements, because the failure it prevents is an axiom that is
 * declared, displayed as enforced, and read by nothing. A global list would let a form
 * fleet's reasoner handles pass breakout's check and then be enforced by nothing —
 * reintroducing exactly that hole through the back door of a refactor.
 */
export const IMPLEMENTED_FORMS = ["maxCardinality", "minCardinality", "range", "disjoint"];

export const checkTBox = (tbox = TBOX) => coreCheckTBox(tbox, IMPLEMENTED_FORMS);

// ---- ABox: facts about one ball ------------------------------------------------

/**
 * Facts for the current ball and paddle.
 *
 * `catchableBy` is left EMPTY when the rollout says the ball cannot be caught, which under
 * closed-world reasoning is "no paddle command reaches it" — exactly what the
 * `ball-must-be-catchable` existential restriction checks against.
 *
 * `reachable === null` is the interesting case and it is NOT the same thing. null means
 * the ball is outside the decidable band: rising, above the bricks, or already past paddle
 * level. There is no sound verdict there, so the ABox says so explicitly
 * (`indeterminate: true`) instead of leaving the property empty, which under closed-world
 * would be read as a false "unreachable" and would violate clause 2 of the invariant — the
 * one error this whole feature must not make.
 *
 * @param sim     a live sim from breakout_sim.js
 * @param opts    {rollout} — injection seam, see `canCatch`
 */
export function buildABox(sim, opts = {}) {
  const rollout = opts.rollout ?? rolloutReachable;
  const m = measure(sim);
  const reachable = rollout(sim.ball, sim.paddleX);

  const facts = { individuals: new Map() };
  const add = (id, type, props = {}) => {
    facts.individuals.set(id, { id, type, ...props });
    return facts.individuals.get(id);
  };

  const state = sim.ball.vy > 0 ? "Descending" : "Ascending";

  add("ball", "Ball", {
    // Rounded to whole pixels because that is what the arena's grid cells are; the
    // reasoner and the renderer are looking at the same world, not a more precise one.
    position: { x: Math.round(sim.ball.x), y: Math.round(sim.ball.y) },
    // Empty set when unreachable — that IS the violation. Never inferred from x alone.
    catchableBy: reachable === true ? new Set(["Left", "Stay", "Right"]) : new Set(),
    indeterminate: reachable === null,
    state,
    // Kept on the fact so a violation message can quote the physics rather than just the
    // assertion that failed.
    gap: m.gap,
    timeToContact: m.timeToContact,
    // The `ball-predicted-at-paddle-level` output: where the ball WILL BE when it reaches
    // paddle height (centre y = BAND_ENTER_Y, the same instant `buildLandingX` solves
    // for), folded through the side walls. Asserted only where a prediction CAN be made —
    // outside the decidable band it is null, and the reasoner treats that the same way it
    // treats an indeterminate reachability verdict: no claim, not a violation.
    predictedPosition: decidable(sim.ball)
      ? { x: Math.round(buildLandingX(sim.ball)), y: BAND_ENTER_Y }
      : null,
  });

  add("paddle", "Paddle", {
    // y asserted as PADDLE_Y because the paddle IS at that cell — the paddle is a
    // horizontal bar and its vertical position is a constant of the geometry, not a
    // claim being made to satisfy the range axiom.
    position: { x: Math.round(sim.paddleX), y: PADDLE_Y },
    commanded: "Stay",
  });

  // The held command as its own individual, so `command-stale-on-bounce` has something to
  // assert over. `validUntil` is the set of conditions under which the command still has a
  // target; it is EMPTY when the command is stale, which is the violation. Empty-not-absent,
  // same convention as `catchableBy`: under closed-world reasoning the empty set is the
  // claim, not a missing fact.
  //
  // Absent entirely when no command has been issued, which is a different state and must
  // not be reported as a stale command — there is nothing to be stale. That is the same
  // reason `commandIsStale` requires a bounce to have occurred: it reports on commands, and
  // a command that is merely old has not lost its target.
  if (sim.paddleDir !== 0 || sim.issuedVXGapSign !== 0) {
    const stale = opts.isStale ? opts.isStale(sim) : commandIsStale(sim);
    const m2 = measure(sim);
    add("command", "Command", {
      // Which side of the paddle the ball was headed for when the model answered — the
      // target the command names. A side, not a coordinate: `buildLandingX` is continuous
      // across a reflection, so the coordinate cannot be what a bounce invalidates.
      issuedFor: sim.issuedVXGapSign === 0 ? "aligned"
        : sim.issuedVXGapSign > 0 ? "Right" : "Left",
      // Where the ball is headed now. Equal sides after a bounce is what makes the
      // command stale; unequal sides is a command that simply needs re-deriving.
      validUntil: stale ? new Set() : new Set(["LandingPoint"]),
      stale,
      commanded: sim.paddleDir === -1 ? "Left" : sim.paddleDir === 1 ? "Right" : "Stay",
      // Which side the ball is headed for now, for the violation message. Computed here
      // rather than in reason() so the reasoner stays a pure function of the ABox.
      nowSide: Math.abs(m2.gap) <= 1 ? "aligned" : m2.gap > 0 ? "Right" : "Left",
    });
  }

  return facts;
}

// ---- inference ----------------------------------------------------------------

/**
 * Check the ABox against the TBox.
 *
 * @param facts      from buildABox
 * @returns {consistent, violations, unasserted, derived}
 */
export function reason(facts, { extensions = {} } = {}) {
  const violations = [];
  const unasserted = [];

  // minCardinality: Ball ⊑ ≥1 catchableBy
  //
  // If the axiom has been deleted — which the live TBox editor lets you do — there is no
  // declared reachability constraint at all, so none is enforced, and that is REPORTED
  // rather than crashed on. The physics still knows the answer; the ontology has simply
  // stopped asserting it. That is the demonstration the editor exists for.
  const catchable = axiom("ball-must-be-catchable");
  if (!catchable) {
    unasserted.push("ball-must-be-catchable");
  } else {
    for (const [id, ind] of facts.individuals) {
      if (ind.type !== "Ball") continue;
      // An indeterminate ball is not a violating ball. Above the bricks a deflection may
      // still rescue it, so declaring it unreachable would be an unsound claim, and
      // unsound is the direction this feature must never err in.
      if (ind.indeterminate) continue;
      if (ind.catchableBy.size < catchable.value) {
        const m = extensions.measure?.(id);
        violations.push({
          axiom: catchable.id,
          subject: id,
          detail: m
            ? `unreachable: the ball falls ${m.gap > 0 ? "right" : "left"} of the paddle by ` +
              `${Math.abs(m.gap).toFixed(0)} px and arrives in ` +
              `${m.timeToContact.toFixed(2)} s, which no left/stay/right command recovers`
            : "catchable by no command — no left/stay/right recovers this ball",
        });
      }
    }
  }

  // minCardinality: Command ⊑ ≥1 validUntil
  //
  // The second existential restriction, and the one whose violation is a bug rather than a
  // fact. A reachability violation says "no answer exists"; a stale-command violation says
  // "an answer existed and has since been cancelled", which is always actionable —
  // re-derive it from the live sim, at no model cost.
  //
  // Like the reachability axiom, deleting it in the live editor disables the check rather
  // than crashing, and that is reported as unasserted.
  const validCmd = axiom("command-stale-on-bounce");
  if (!validCmd) {
    unasserted.push("command-stale-on-bounce");
  } else {
    for (const [id, ind] of facts.individuals) {
      if (ind.type !== "Command") continue;
      if (ind.validUntil.size < validCmd.value) {
        violations.push({
          axiom: validCmd.id,
          subject: id,
          detail: `the command was issued for a ball landing to the ${ind.issuedFor} of the paddle, ` +
            `but the ball has since reflected off a side wall and now lands to the ` +
            `${ind.nowSide} — the paddle is steering toward a trajectory that no longer exists`,
        });
      }
    }
  }

  // minCardinality: Ball ⊑ ≥1 predictedPosition, wherever a prediction can be made.
  //
  // The third existential restriction, and the one that GIVES something rather than
  // forbidding something: it is what makes the landing point the ontology's declared
  // output instead of an ad-hoc read of the sim. `buildABox` asserts the prediction for
  // every decidable ball, so a violation here is not a judgement about the ball — it is
  // buildABox and this check disagreeing, which is a real bug worth surfacing. Same
  // deletion contract as the other two existential restrictions: removing it in the live
  // editor stops the assertion and is reported, not crashed on.
  const predicted = axiom("ball-predicted-at-paddle-level");
  if (!predicted) {
    unasserted.push("ball-predicted-at-paddle-level");
  } else {
    for (const [id, ind] of facts.individuals) {
      if (ind.type !== "Ball") continue;
      // Outside the decidable band no prediction CAN be made, so nothing is owed — the
      // same skip an indeterminate ball gets from the reachability check.
      if (ind.indeterminate) continue;
      if (ind.predictedPosition == null) {
        violations.push({
          axiom: predicted.id,
          subject: id,
          detail: "the ball will reach paddle height, but no predicted position is asserted",
        });
      }
    }
  }

  // range: positions are field cells.
  for (const a of TBOX) {
    if (a.form !== "range") continue;
    for (const [id, ind] of facts.individuals) {
      if (ind.type !== a.subject) continue;
      const value = ind[a.property];
      if (value == null) continue;
      const isCell = value && typeof value === "object" &&
        Number.isFinite(value.x) && Number.isFinite(value.y);
      if (!isCell) {
        violations.push({ axiom: a.id, subject: id, detail: `${a.property} is not a ${a.value}` });
      }
    }
  }

  // disjoint: the ball is in exactly one state, the paddle obeys exactly one command.
  const ballStates = axiom("ball-single-state");
  for (const [id, ind] of facts.individuals) {
    if (ind.type !== "Ball" || !ballStates) continue;
    if (!ballStates.value.includes(ind.state)) {
      violations.push({
        axiom: ballStates.id, subject: id,
        detail: `state ${ind.state} is not one of ${ballStates.value.join(" / ")}`,
      });
    }
  }
  const paddleCmd = axiom("paddle-single-command");
  for (const [id, ind] of facts.individuals) {
    if (ind.type !== "Paddle" || !paddleCmd) continue;
    if (!paddleCmd.value.includes(ind.commanded)) {
      violations.push({
        axiom: paddleCmd.id, subject: id,
        detail: `command ${ind.commanded} is not one of ${paddleCmd.value.join(" / ")}`,
      });
    }
  }

  // Derived — Tier 1. Numerics come from an extension, by design.
  const ball = [...facts.individuals.values()].find(i => i.type === "Ball") || null;
  const derived = { landingX: null, reachable: null, gap: null, timeToContact: null };
  if (ball) {
    derived.gap = ball.gap;
    derived.timeToContact = ball.timeToContact;
    derived.landingX = extensions.buildLandingX?.() ?? null;
    derived.reachable = ball.indeterminate ? null : ball.catchableBy.size > 0;
  }

  return { consistent: violations.length === 0, violations, unasserted, derived };
}

/**
 * A feasibility predicate with the shape the decision loop expects, backed by the reasoner.
 * This is the Tier 0 boundary: the question is skipped by asking the ontology whether the
 * ball is consistent with the axioms, not by asking whether a number is big.
 *
 * `rollout` is injectable so a test can drive the boundary with a stub — the same seam
 * fleet uses for `detour`, and the reason `canCatch` is not where reachability is computed.
 *
 * Returns ok: true when the question should still be ASKED. A ball outside the decidable
 * band is ok: true even when it is hopeless, because the ontology has no verdict there and
 * suppressing the question would be an unsound claim dressed as a rule.
 */
export function canCatch(sim, opts = {}) {
  const rollout = opts.rollout ?? rolloutReachable;
  const facts = buildABox(sim, { rollout });
  const probe = reason(facts, {
    extensions: { measure: () => measure(sim), buildLandingX: () => buildLandingX(sim.ball) },
  });
  // Only the reachability axiom gates the question. The range and disjoint axioms describe
  // facts the simulator always satisfies; if one of them ever fires it is a real bug worth
  // surfacing, but it is not a reason to stop asking about a ball the model could still save.
  const blocking = probe.violations.filter(v => v.axiom === "ball-must-be-catchable");
  return {
    ok: blocking.length === 0,
    violations: blocking,
    allViolations: probe.violations,
    unasserted: probe.unasserted,
    derived: probe.derived,
  };
}

// ---- the panel's display helpers ----------------------------------------------

/**
 * The closed-form shortcut, kept ONLY so the panel can show what it computes next to what
 * the rollout computes, and so a test can pin that they disagree.
 *
 * Do not use this to decide anything. Four closed forms were written and measured against
 * the rollout over 40,000 states; the best agreed 83% of the time, and every one of the
 * errors ran the same way — declaring catchable balls unreachable, which is precisely the
 * failure this feature exists to fix. It is here as a counter-example, not a fallback.
 */
export function analyticReachable(ball, paddleX) {
  if (!decidable(ball)) return null;
  const t = (BAND_ENTER_Y - ball.y) / ball.vy;
  // One time point, no wall folding, no band search: the naive form, on purpose.
  const naiveX = ball.x + ball.vx * t;
  // One integrator frame of slack, sourced from the timestep rather than tuned to make a
  // test pass. It can only widen the catch, never narrow it: a false "unreachable" silences
  // a live question, a false "reachable" costs one ask.
  return Math.abs(naiveX - paddleX) <= PADDLE_SPEED * t + CATCH_TOL_PX;
}

/** One line describing what the ontology currently believes, for the panel. */
export function aboxFor(sim) {
  const m = measure(sim);
  const reachable = rolloutReachable(sim.ball, sim.paddleX);
  const facts = buildABox(sim);
  const probe = reason(facts);
  const undecidable = reachable === null;
  const detail = undecidable
    ? `at (${Math.round(sim.ball.x)}, ${Math.round(sim.ball.y)}) · no verdict: ` +
      (sim.ball.y - BALL_R <= LAST_BRICK_BOTTOM
        ? "still above the bricks, a deflection may yet reach it"
        : sim.ball.vy > 0 ? "already at paddle level" : "rising, away from the paddle")
    : reachable
      ? `at (${Math.round(sim.ball.x)}, ${Math.round(sim.ball.y)}) → lands at ` +
        `${Math.round(m.landingX)} · ${m.gap > 0 ? "right" : m.gap < 0 ? "left" : "level"} ` +
        `of the paddle by ${Math.abs(m.gap).toFixed(0)} px in ` +
        `${m.timeToContact.toFixed(2)} s · catchable`
      : `at (${Math.round(sim.ball.x)}, ${Math.round(sim.ball.y)}) → lands at ` +
        `${Math.round(m.landingX)} · ${Math.abs(m.gap).toFixed(0)} px away with ` +
        `${(PADDLE_SPEED * m.timeToContact).toFixed(0)} px of travel left · UNREACHABLE`;
  const rows = [{
    id: "ball",
    detail,
    violations: probe.violations,
    unasserted: probe.unasserted,
  }];

  // The held command, as its own row. Only shown once a command has been issued — before
  // that there is nothing to be stale, and an empty row would read as a finding.
  const cmd = facts.individuals.get("command");
  if (cmd) {
    rows.push({
      id: "command",
      detail: `issued for a ball landing to the ${cmd.issuedFor} of the paddle · ` +
              `${cmd.commanded}${cmd.stale ? " · STALE: the ball has since bounced and now " +
              `lands to the ${cmd.nowSide}, so this command points at a cancelled trajectory`
                : " · still valid"}`,
      violations: probe.violations.filter(v => v.subject === "command"),
      unasserted: probe.unasserted.filter(u => u === "command-stale-on-bounce"),
    });
  }

  return rows;
}