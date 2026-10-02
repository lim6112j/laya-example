// node --test test_fleet_sim.mjs
//
// Guards for the invariants that are cheap to break and expensive to discover:
// the insertion arithmetic, the mode-aware cost model, the seat limit, and the token
// budget — the last of which is a hard raise inside Agent.system_one, not a warning.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  makeBus, createSim, step, newDemand, assignTo,
  insertStops, insertSeq, capacityOk, routeLength, busPos, plannedPath,
  detourCost, pickupDetour, isOnRoute, nearestCost, serialCost,
  feasible, seatsLeft, buildState, buildQuestion, buildBaselineQuestion,
  headEstimate, planHorizon, RULES, GRID,
} from "./static/fleet_sim.js";

const pt = (x, y) => ({ x, y });
const at = (b, x, y) => { b.x = x; b.y = y; return b; };
const pax = (id, pu, d) => ({ id, pickup: pu, dest: d, state: "waiting", bus: null });
const simWith = multi => ({ config: { multiPickup: multi } });

/** Drive a sim with a feasibility-aware greedy so runs are comparable across policies. */
function drive(sim, pick, ticks) {
  for (let i = 0; i < ticks; i++) {
    for (const p of step(sim, 0.05)) assignTo(sim, pick(sim, p), p);
    for (const p of sim.passengers) if (p.state === "waiting") assignTo(sim, pick(sim, p), p);
  }
  return sim;
}
const bestDetour = (sim, p) => {
  const ok = sim.buses.filter(b => feasible(b, p, sim));
  const pool = ok.length ? ok : sim.buses;
  return [...pool].sort((a, b) => detourCost(a, p, sim) - detourCost(b, p, sim))[0];
};

// ---- insertion ---------------------------------------------------------------

test("insertion always places a drop-off after its own pickup", () => {
  for (let seed = 1; seed <= 60; seed++) {
    const sim = createSim({ seed, config: { demandEvery: 0.4 } });
    for (let i = 0; i < 300; i++) {
      for (const p of step(sim, 0.05)) assignTo(sim, bestDetour(sim, p), p);
      for (const bus of sim.buses) {
        for (let k = 0; k < bus.stops.length; k++) {
          if (bus.stops[k].kind !== "dropoff") continue;
          // A passenger who is already aboard has had their pickup serviced, so only
          // the drop-off remains in the route and there is nothing to compare against.
          if (bus.onboard.has(bus.stops[k].p.id)) continue;
          const pickedUp = bus.stops.some(
            (s, m) => m < k && s.kind === "pickup" && s.p.id === bus.stops[k].p.id);
          assert.ok(pickedUp,
            `seed ${seed}: drop-off of #${bus.stops[k].p.id} precedes its pickup`);
        }
      }
    }
  }
});

test("the reported detour is exactly what the bus drives", () => {
  // The number quoted in the question text must be the number the physics executes.
  // This is the invariant the whole rewrite exists to protect.
  for (let seed = 1; seed <= 40; seed++) {
    const sim = createSim({ seed, config: { demandEvery: 0.4 } });
    for (let i = 0; i < 250; i++) {
      for (const p of step(sim, 0.05)) {
        for (const bus of sim.buses) {
          const ins = insertStops(bus, p);
          if (!ins) continue;
          const before = routeLength(busPos(bus), bus.stops);
          const after = routeLength(busPos(bus), ins.stops);
          assert.ok(Math.abs((after - before) - ins.delta) < 1e-9,
            `seed ${seed}: delta ${ins.delta} != ${after} - ${before}`);
        }
        assignTo(sim, bestDetour(sim, p), p);
      }
    }
  }
});

// ---- capacity ----------------------------------------------------------------

test("the seat limit rejects a route that would carry two at once", () => {
  // Test the predicate directly on hand-built routes. Note that insertStops with
  // seats=1 usually still succeeds — it simply puts the new pickup after the drop-off,
  // so the bus is never carrying two. Rejection happens when no such slot exists.
  const riding = pax(1, pt(10, 0), pt(20, 0));
  const fresh = pax(2, pt(5, 0), pt(25, 0));
  const aboard = new Set([riding.id]);

  const overlapping = [
    { kind: "pickup", point: fresh.pickup, p: fresh },   // picked up before #1 is dropped
    { kind: "dropoff", point: riding.dest, p: riding },
    { kind: "dropoff", point: fresh.dest, p: fresh },
  ];
  assert.equal(capacityOk(overlapping, aboard, 1), false, "seats=1 must reject two aboard");
  assert.equal(capacityOk(overlapping, aboard, 2), true, "seats=2 accepts two aboard");

  const sequential = [
    { kind: "dropoff", point: riding.dest, p: riding },
    { kind: "pickup", point: fresh.pickup, p: fresh },
    { kind: "dropoff", point: fresh.dest, p: fresh },
  ];
  assert.equal(capacityOk(sequential, aboard, 1), true, "dropping first frees the seat");
});

test("a bus at its planning horizon refuses new work", () => {
  const bus = at(makeBus("A"), 0, 0);
  while (bus.stops.length < planHorizon()) {
    const p = pax(bus.stops.length + 100, pt(bus.stops.length + 1, 0), pt(40, 40));
    if (!insertStops(bus, p)) break;
    bus.stops = insertStops(bus, p).stops;
  }
  assert.equal(bus.stops.length, planHorizon());
  assert.equal(insertStops(bus, pax(999, pt(1, 1), pt(2, 2))), null,
    "a bus at its horizon is committed and must refuse");
});

test("a bus never carries more passengers than it has seats", () => {
  for (const multi of [true, false]) {
    const sim = createSim({ seed: 5, config: { demandEvery: 0.3, multiPickup: multi } });
    drive(sim, bestDetour, 4000);
    for (const bus of sim.buses) {
      assert.ok(bus.onboard.size <= RULES.seats,
        `multi=${multi}: ${bus.onboard.size} aboard with ${RULES.seats} seats`);
    }
  }
});

test("a serial bus never plans more than one job", () => {
  const sim = createSim({ seed: 5, config: { demandEvery: 0.3, multiPickup: false } });
  drive(sim, bestDetour, 3000);
  for (const bus of sim.buses) assert.ok(bus.stops.length <= 2, `stops ${bus.stops.length}`);
});

test("the route stays inside the planning horizon", () => {
  const sim = createSim({ seed: 9, config: { demandEvery: 0.2, multiPickup: true } });
  drive(sim, bestDetour, 3000);
  for (const bus of sim.buses) {
    assert.ok(bus.stops.length <= planHorizon(),
      `route of ${bus.stops.length} exceeds horizon ${planHorizon()}`);
  }
});

test("delivered never exceeds demands", () => {
  // Catches a double-assignment: a passenger spliced into two routes has its drop-off
  // serviced twice. Found this way in the browser (delivered 50, demands 30).
  for (const multi of [true, false]) {
    const sim = createSim({ seed: 3, config: { demandEvery: 0.3, multiPickup: multi } });
    drive(sim, bestDetour, 4000);
    assert.ok(sim.deliveredCount <= sim.demandCount,
      `multi=${multi}: delivered ${sim.deliveredCount} > demands ${sim.demandCount}`);
    const done = sim.passengers.filter(p => p.state === "delivered").length;
    assert.equal(done, sim.deliveredCount, "deliveredCount must match delivered passengers");
  }
});

// ---- the mode-aware cost model ------------------------------------------------

test("a pickup on the route is free in multi-pickup mode and worthless in serial", () => {
  const bus = at(makeBus("B"), 0, 0);
  assignTo(simWith(true), bus, pax(10, pt(10, 0), pt(20, 0)));
  const onRoute = pax(11, pt(5, 0), pt(25, 0));

  assert.equal(pickupDetour(bus, onRoute, simWith(true)), 0, "multi: pickup is on the route");
  assert.equal(isOnRoute(bus, onRoute, simWith(true)), true);
  // The drop-off is past the end of the route, so the demand as a whole is not free.
  assert.ok(detourCost(bus, onRoute, simWith(true)) > 0);

  // A serial bus cannot stop until it is free, so the route buys it nothing.
  assert.equal(pickupDetour(bus, onRoute, simWith(false)), serialCost(bus, onRoute));
  assert.equal(isOnRoute(bus, onRoute, simWith(false)), false,
    "serial: a nearby pickup must not be reported as on-route");
});

test("a committed serial bus never reports a pickup as on-route", () => {
  // An *idle* serial bus with a nearby pickup genuinely is nearly free — it is free.
  // What must never happen is a bus with work to do claiming the route helps it.
  const sim = createSim({ seed: 11, config: { demandEvery: 0.5, multiPickup: false } });
  let onRoute = 0, total = 0;
  for (let i = 0; i < 1500; i++) {
    for (const p of step(sim, 0.05)) {
      for (const bus of sim.buses) {
        if (!bus.stops.length && !bus.queue.length) continue;   // idle: exempt
        total++;
        if (isOnRoute(bus, p, sim)) onRoute++;
      }
      assignTo(sim, bestDetour(sim, p), p);
    }
  }
  assert.equal(onRoute, 0, `${onRoute}/${total} committed serial buses wrongly on-route`);
});

test("seats are reported honestly: a busy serial bus offers none", () => {
  const bus = at(makeBus("A"), 0, 0);
  assert.equal(seatsLeft(bus, simWith(true)), RULES.seats);
  assignTo(simWith(true), bus, pax(1, pt(5, 5), pt(9, 9)));
  assert.equal(seatsLeft(bus, simWith(true)), RULES.seats);   // assigned, not yet aboard
  assert.equal(seatsLeft(at(makeBus("B"), 0, 0), simWith(false)), 1);
  const busy = at(makeBus("C"), 0, 0);
  assignTo(simWith(false), busy, pax(1, pt(5, 5), pt(9, 9)));
  assert.equal(seatsLeft(busy, simWith(false)), 0, "a busy serial bus has nothing to offer");
});

// ---- the question sent to laya ------------------------------------------------

test("the question fits the head budget on every checkpoint", () => {
  const TIGHTEST_HEAD = 192;         // english; typed-decisions and multilingual get 256
  for (const multi of [true, false]) {
    for (let seed = 1; seed <= 40; seed++) {
      const sim = createSim({ seed, config: { demandEvery: 0.4, multiPickup: multi } });
      drive(sim, bestDetour, 900);
      const q = buildQuestion(sim, newDemand(sim));
      assert.ok(headEstimate(q) <= TIGHTEST_HEAD,
        `seed ${seed}: head ${headEstimate(q)} > ${TIGHTEST_HEAD}`);
      for (const text of Object.values(q.criteria)) {
        assert.ok(Math.ceil(text.length / 4) <= 48, `option too long: ${text}`);
      }
    }
  }
});

test("a bus with no room is dropped from the options, but the options are never empty", () => {
  const sim = createSim({ seed: 4, config: { demandEvery: 0.2, multiPickup: true } });
  let sawOmitted = false;
  for (let i = 0; i < 3000; i++) {
    for (const p of step(sim, 0.05)) {
      const usable = sim.buses.filter(b => feasible(b, p, sim));
      const q = buildQuestion(sim, p);
      const keys = Object.keys(q.criteria);
      assert.ok(keys.length >= 1, "empty criteria is a hard failure in Agent.system_one");
      if (usable.length && usable.length < sim.buses.length) {
        sawOmitted = true;
        assert.equal(keys.length, usable.length, "infeasible buses must be omitted");
        for (const b of sim.buses) {
          const key = `bus_${b.id.toLowerCase()}`;
          if (usable.includes(b)) assert.ok(q.criteria[key], `${key} should be offered`);
          else assert.equal(q.criteria[key], undefined, `${key} must not be offered`);
        }
      } else if (!usable.length) {
        // Nothing can take it: the options must survive and say so.
        for (const b of sim.buses) {
          assert.match(q.criteria[`bus_${b.id.toLowerCase()}`], /full|wait/i,
            "a bus that cannot take the work must say why");
        }
      }
      assignTo(sim, bestDetour(sim, p), p);
    }
  }
  assert.ok(sawOmitted, "expected at least one capacity-filtered decision");
});

test("the rules line comes first in the state", () => {
  // build_sequence truncates the state from the right, so a rule at the tail of a long
  // state would be dropped silently.
  for (const multi of [true, false]) {
    const sim = createSim({ seed: 3, config: { demandEvery: 0.5, multiPickup: multi } });
    drive(sim, bestDetour, 600);
    const state = buildState(sim, newDemand(sim));
    assert.equal(Object.keys(state)[0], "rules");
    assert.match(state.rules, multi ? /carry up to \d+ demands/ : /one demand at a time/);
  }
});

test("the baseline arm reproduces the original demo's question", () => {
  assert.deepEqual(buildBaselineQuestion().criteria, {
    bus_a: "assign to Bus A (red)",
    bus_b: "assign to Bus B (green)",
    bus_c: "assign to Bus C (blue)",
  });
});

test("the rules and the seat count stay in step", () => {
  assert.ok(RULES.seats > 0 && RULES.onRouteBlocks >= 0);
  const sim = createSim({ seed: 5, config: { multiPickup: true } });
  const q = buildQuestion(sim, newDemand(sim));
  for (const text of Object.values(q.criteria)) {
    assert.match(text, new RegExp(`of ${RULES.seats} seats free`));
  }
});

test("scenarios replay identically from a seed", () => {
  const replay = () => {
    const sim = createSim({ seed: 42, config: { demandEvery: 0.5 } });
    drive(sim, bestDetour, 400);
    return JSON.stringify({ buses: sim.buses.map(b => b.stops), n: sim.deliveredCount });
  };
  assert.equal(replay(), replay());
});

test("multi-pickup carries more passengers over the same demand", () => {
  // The headline claim of the toggle: same rules, same cost function, different world.
  const run = multi => {
    const sim = createSim({ seed: 7, config: { demandEvery: 0.8, multiPickup: multi } });
    drive(sim, bestDetour, 9000);
    return sim;
  };
  const multi = run(true), serial = run(false);
  assert.ok(multi.deliveredCount > serial.deliveredCount,
    `multi ${multi.deliveredCount} should beat serial ${serial.deliveredCount}`);
});
