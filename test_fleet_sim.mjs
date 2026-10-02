// node --test test_fleet_sim.mjs
//
// Guards for the invariants that are cheap to break and expensive to discover:
// the insertion arithmetic, the seeded replay, and the token budget — the last of
// which is a hard raise inside Agent.system_one, not a soft warning.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  makeBus, createSim, step, newDemand, assignTo,
  costOf, insertionCost, plannedPath, isOnRoute,
  buildState, buildQuestion, buildBaselineQuestion, headEstimate,
  rankedByDeadhead, RULES, GRID,
} from "./static/fleet_sim.js";

const at = (b, x, y) => { b.x = x; b.y = y; return b; };
const pt = (x, y) => ({ x, y });

test("insertion cost is zero for a pickup already on the planned route", () => {
  const bus = at(makeBus("A"), 0, 0);
  bus.job = { phase: "to_pickup", passenger: { pickup: pt(10, 0), dest: pt(20, 0) } };
  assert.deepEqual(plannedPath(bus), [pt(10, 0), pt(20, 0)]);
  assert.equal(insertionCost(bus, pt(5, 0)), 0);
  assert.equal(isOnRoute(bus, pt(5, 0)), true);
});

test("insertion cost never exceeds the deadhead cost", () => {
  // Appending after the last waypoint is always an option, and it is always at most
  // the serial path in costOf, so the multi-pickup rule can never look worse than
  // the original nearest-bus objective.
  for (let s = 1; s <= 200; s++) {
    const sim = createSim({ seed: s, config: { demandEvery: 0.5 } });
    for (let i = 0; i < 400; i++) {
      for (const p of step(sim, 0.05)) assignTo(sim, rankedByDeadhead(sim, p)[0], p);
    }
    for (const bus of sim.buses) {
      for (let k = 0; k < 10; k++) {
        const pickup = pt(1 + Math.floor(sim.rng() * (GRID - 2)),
                          1 + Math.floor(sim.rng() * (GRID - 2)));
        assert.ok(insertionCost(bus, pickup) <= costOf(bus, pickup) + 1e-9,
          `seed ${s}: insertion ${insertionCost(bus, pickup)} > deadhead ${costOf(bus, pickup)}`);
      }
    }
  }
});

test("insertion cost is never negative", () => {
  for (let s = 1; s <= 50; s++) {
    const sim = createSim({ seed: s, config: { demandEvery: 0.5 } });
    for (let i = 0; i < 200; i++) {
      for (const p of step(sim, 0.05)) assignTo(sim, rankedByDeadhead(sim, p)[0], p);
    }
    for (const bus of sim.buses) {
      for (let k = 0; k < 20; k++) {
        assert.ok(insertionCost(bus, pt(1 + Math.floor(sim.rng() * 48),
                                         1 + Math.floor(sim.rng() * 48))) >= 0);
      }
    }
  }
});

test("the dispatch question fits the head budget on every checkpoint", () => {
  // Agent.system_one raises when options overflow head_max_len, and the english
  // checkpoint has the tightest budget of the three.
  const TIGHTEST_HEAD = 192;
  for (let s = 1; s <= 60; s++) {
    const sim = createSim({ seed: s, config: { demandEvery: 0.5 } });
    for (let i = 0; i < 300; i++) {
      for (const p of step(sim, 0.05)) assignTo(sim, rankedByDeadhead(sim, p)[0], p);
    }
    const q = buildQuestion(sim, newDemand(sim));
    assert.ok(headEstimate(q) <= TIGHTEST_HEAD,
      `seed ${s}: head estimate ${headEstimate(q)} exceeds ${TIGHTEST_HEAD}`);
    for (const text of Object.values(q.criteria)) {
      // 48 tokens per option, hard-capped in build_sequence
      assert.ok(Math.ceil(text.length / 4) <= 48, `option too long: ${text}`);
    }
  }
});

test("the rules line comes first in the state", () => {
  // build_sequence truncates the state from the right, so a rule at the tail of a
  // long state would be dropped silently.
  const sim = createSim({ seed: 3 });
  for (let i = 0; i < 200; i++) {
    for (const p of step(sim, 0.05)) assignTo(sim, rankedByDeadhead(sim, p)[0], p);
  }
  const keys = Object.keys(buildState(sim, newDemand(sim)));
  assert.equal(keys[0], "rules");
  assert.match(buildState(sim, newDemand(sim)).rules, /carry up to \d+ demands at once/);
});

test("the baseline arm reproduces the original demo's question", () => {
  const q = buildBaselineQuestion();
  assert.deepEqual(q.criteria, {
    bus_a: "assign to Bus A (red)",
    bus_b: "assign to Bus B (green)",
    bus_c: "assign to Bus C (blue)",
  });
});

test("scenarios replay identically from a seed", () => {
  const replay = () => {
    const sim = createSim({ seed: 42, config: { demandEvery: 0.5 } });
    for (let i = 0; i < 500; i++) {
      for (const p of step(sim, 0.05)) assignTo(sim, rankedByDeadhead(sim, p)[0], p);
    }
    return JSON.stringify({ buses: sim.buses, passengers: sim.passengers });
  };
  assert.equal(replay(), replay());
});

test("the rules and the seat count stay in step", () => {
  assert.ok(RULES.freeSeats > 0 && RULES.onRouteBlocks >= 0);
  const sim = createSim({ seed: 5 });
  const q = buildQuestion(sim, newDemand(sim));
  for (const text of Object.values(q.criteria)) {
    assert.match(text, new RegExp(`${RULES.freeSeats} seats free`));
  }
});
