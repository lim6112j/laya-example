// node --test test_ontology.mjs
//
// The reasoner has to be falsifiable — that is the point of declaring axioms instead of
// writing arithmetic. These tests attack it three ways: check that it catches a
// contradictory axiom set (which nothing in the hand-written path could), check that it
// catches bad ABox facts, and check that it agrees with `capacityOk` on the thing it
// replaces.
//
// The existing 18 tests in test_fleet_sim.mjs are the control: the hand-written path is
// untouched and still green, which is what proves the injection is opt-in.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  TBOX, VOCABULARY, axiom, checkTBox, buildABox, reason, canServe,
} from "./static/ontology.js";

import {
  makeBus, createSim, step, newDemand, assignTo, insertStops, insertSeq,
  capacityOk, detourCost, nearestCost, feasible, planHorizon, RULES,
} from "./static/fleet_sim.js";

const pt = (x, y) => ({ x, y });
const pax = (id, pickup, dest) => ({ id, pickup, dest, state: "waiting", bus: null });
const mkStops = (p, kind = "pickup") => ({
  kind,
  point: kind === "pickup" ? p.pickup : p.dest,
  p,
});
const at = (b, x, y) => { b.x = x; b.y = y; return b; };

// ---- TBox self-check -----------------------------------------------------------

test("the shipped TBox is satisfiable", () => {
  const r = checkTBox(TBOX);
  assert.equal(r.satisfiable, true, JSON.stringify(r.unsatisfiable));
  assert.equal(r.unsatisfiable.length, 0);
});

test("contradictory cardinality makes the class unsatisfiable", () => {
  // A hand-written `seats: 4` cannot contradict itself. Two declared axioms can, and
  // saying which one is wrong would be a guess — the honest answer is that the class has
  // no members at all.
  const contradictory = [
    ...TBOX,
    { id: "bus-floor", form: "minCardinality", subject: "Bus",
      property: "onboardPassenger", value: 6 },
  ];
  const r = checkTBox(contradictory);
  assert.equal(r.satisfiable, false);
  assert.equal(r.unsatisfiable.length, 1);
  assert.match(r.unsatisfiable[0].reason, /at least 6 and at most 4/);
  assert.match(r.unsatisfiable[0].id, /bus-capacity/);
  assert.match(r.unsatisfiable[0].id, /bus-floor/);
});

test("a disjoint axiom listing a class twice is caught", () => {
  const r = checkTBox([...TBOX, { id: "dup", form: "disjoint", value: ["Waiting", "Waiting"] }]);
  assert.equal(r.satisfiable, false);
  assert.match(r.unsatisfiable[0].reason, /lists Waiting twice/);
});

test("the soft parameters are deliberately not axioms", () => {
  // onRouteBlocks and minRideBlocks are tuning values, not invariants. Putting them in
  // the TBox would assert the domain guarantees something it does not.
  const serialized = JSON.stringify(TBOX);
  assert.ok(!serialized.includes(`"value":${RULES.onRouteBlocks}`),
    "onRouteBlocks must stay configuration, not an axiom");
  assert.ok(!serialized.includes(`"value":${RULES.minRideBlocks}`),
    "minRideBlocks must stay configuration, not an axiom");
  assert.equal(axiom("bus-capacity").value, RULES.seats,
    "the seat limit, which IS an invariant, is asserted in the TBox");
});

// ---- ABox consistency ----------------------------------------------------------

test("a route that exceeds the seat limit is a violation, and names the axiom", () => {
  const seats = RULES.seats;
  const bus = at(makeBus("A"), 0, 0);
  // A full bus: every seat taken.
  const riders = [];
  for (let i = 1; i <= seats; i++) {
    const p = pax(i, pt(10 + i, 0), pt(20 + i, 0));
    riders.push(p);
    bus.onboard.add(p.id);
  }
  bus.stops = riders.map(p => ({ kind: "dropoff", point: p.dest, p }));
  const fresh = pax(99, pt(2, 2), pt(30, 30));
  // Pick the newcomer up before any drop-off: five aboard on a four-seat bus.
  const stops = [mkStops(fresh), ...bus.stops, mkStops(fresh, "dropoff")];
  const r = reason(buildABox(bus, stops, seats));
  assert.equal(r.consistent, false);
  const v = r.violations.find(x => x.axiom === "bus-capacity");
  assert.ok(v, `expected a bus-capacity violation, got ${JSON.stringify(r.violations)}`);
  assert.match(v.detail, /at most 4/);
});

test("a drop-off before a pickup frees the seat, so the route is legal", () => {
  const seats = RULES.seats;
  const bus = at(makeBus("A"), 0, 0);
  const riders = [];
  for (let i = 1; i <= seats; i++) {
    const p = pax(i, pt(10 + i, 0), pt(20 + i, 0));
    riders.push(p);
    bus.onboard.add(p.id);
  }
  bus.stops = riders.map(p => ({ kind: "dropoff", point: p.dest, p }));
  const fresh = pax(99, pt(25, 0), pt(30, 5));
  // Same full bus, but the newcomer is collected only after everyone is dropped.
  const stops = [...bus.stops, mkStops(fresh), mkStops(fresh, "dropoff")];
  const r = reason(buildABox(bus, stops, seats));
  assert.equal(r.consistent, true, JSON.stringify(r.violations));
});

test("a seat limit that disagrees with the TBox is reported, and the stricter wins", () => {
  // Found by this suite: the reasoner read the TBox's declared limit and ignored the
  // configured one, so a bus run with seats: 1 was still checked against 4. The check
  // now reports the disagreement rather than silently preferring either side.
  const bus = at(makeBus("A"), 0, 0);
  const a = pax(1, pt(10, 0), pt(20, 0));
  const fresh = pax(2, pt(5, 0), pt(25, 0));
  const stops = [mkStops(fresh), mkStops(a, "dropoff"), mkStops(fresh, "dropoff")];
  const r = reason(buildABox(bus, stops, 1));
  assert.equal(r.consistent, false);
  const v = r.violations.find(x => /disagrees with the TBox/.test(x.detail));
  assert.ok(v, `expected a configuration-mismatch violation, got ${JSON.stringify(r.violations)}`);
  assert.match(v.detail, /configured seat limit 1 disagrees with the TBox's 4/);
  assert.match(v.detail, /enforcing the stricter 1/);

  // And the same two-aboard route that is illegal at 1 seat is legal at 4.
  const ok = reason(buildABox(bus, stops, RULES.seats));
  assert.equal(ok.consistent, true, JSON.stringify(ok.violations));
});

test("a malformed cell is a range violation", () => {
  const bus = at(makeBus("A"), 0, 0);
  const facts = buildABox(bus, [], RULES.seats);
  facts.individuals.get(bus.id).occupies = "somewhere";
  const r = reason(facts);
  assert.equal(r.consistent, false);
  assert.ok(r.violations.some(v => v.axiom === "bus-occupies-cell"));
});

test("a passenger in a state outside the disjoint set is caught", () => {
  const bus = at(makeBus("A"), 0, 0);
  const facts = buildABox(bus, [], RULES.seats);
  facts.individuals.set("passenger:99", {
    id: "passenger:99", type: "Passenger", state: "Teleporting", pickup: pt(1, 1), destination: pt(2, 2),
  });
  const r = reason(facts);
  assert.equal(r.consistent, false);
  assert.ok(r.violations.some(v => v.axiom === "passenger-single-state"));
});

test("an unassigned demand violates the existential restriction", () => {
  // This is the assertion that says "a demand nobody has taken is a problem" — and it is
  // deliberately excluded from per-bus feasibility, because asking "can this bus take it"
  // is exactly the state where it is unsatisfied.
  const bus = at(makeBus("A"), 0, 0);
  const d = pax(9, pt(3, 3), pt(8, 8));
  const r = reason(buildABox(bus, [], RULES.seats, d));
  assert.ok(r.violations.some(v => v.axiom === "demand-must-be-served"));

  // Once somebody has taken it, the assertion holds.
  const served = buildABox(bus, [], RULES.seats, d);
  served.individuals.get(`demand:${d.id}`).servedBy = new Set(["bus_a"]);
  assert.ok(!reason(served).violations.some(v => v.axiom === "demand-must-be-served"));
});

test("per-bus feasibility ignores the existential restriction", () => {
  const bus = at(makeBus("A"), 0, 0);
  const d = pax(9, pt(3, 3), pt(8, 8));
  const r = canServe(bus, [], RULES.seats, d);
  assert.equal(r.ok, true, JSON.stringify(r.violations));
});

// ---- equivalence with the hand-written path -------------------------------------

test("the reasoner and capacityOk agree on candidate routes", () => {
  let compared = 0;
  for (let seed = 1; seed <= 40; seed++) {
    const sim = createSim({ seed, config: { demandEvery: 0.4, multiPickup: true } });
    for (let i = 0; i < 150; i++) {
      for (const p of step(sim, 0.05)) {
        for (const bus of sim.buses) {
          // Enumerate candidate routes the same way insertSeq does, and ask both.
          const base = bus.stops;
          const slots = base.length + 2;
          for (let a = 0; a < slots; a++) {
            for (let b = a + 1; b < slots; b++) {
              const cand = [];
              let bi = 0, k = 0;
              for (let idx = 0; idx < slots; idx++) {
                if (k === 0 && idx === a) { cand.push(mkStops(p, "pickup")); k = 1; }
                else if (k === 1 && idx === b) { cand.push(mkStops(p, "dropoff")); k = 2; }
                else cand.push(base[bi++]);
              }
              if (cand.length !== base.length + 2) continue;
              const hand = capacityOk(cand, bus.onboard, RULES.seats);
              const ont = canServe(bus, cand, RULES.seats).ok;
              compared++;
              assert.equal(ont, hand,
                `seed ${seed}: capacityOk=${hand} reasoner=${ont} for ${cand.length} stops`);
            }
          }
        }
        // Drive the sim forward with the hand-written path, as everywhere else.
        const ok = sim.buses.filter(bus => feasible(bus, p, sim));
        const pool = ok.length ? ok : sim.buses;
        assignTo(sim, [...pool].sort((x, y) => detourCost(x, p, sim) - detourCost(y, p, sim))[0], p);
      }
    }
  }
  assert.ok(compared > 5000, `expected a broad sweep, only compared ${compared}`);
});

test("insertStops returns the same route under either feasibility predicate", () => {
  const withOntology = (bus, p) =>
    insertStops(bus, p, RULES.seats, cand => canServe(bus, cand, RULES.seats).ok);

  for (let seed = 1; seed <= 30; seed++) {
    const sim = createSim({ seed, config: { demandEvery: 0.4, multiPickup: true } });
    for (let i = 0; i < 400; i++) {
      for (const p of step(sim, 0.05)) {
        for (const bus of sim.buses) {
          const hand = insertStops(bus, p);
          const ont = withOntology(bus, p);
          assert.equal(ont === null, hand === null,
            `seed ${seed} bus ${bus.id}: hand=${hand === null ? "null" : "route"} ` +
            `ont=${ont === null ? "null" : "route"}`);
          if (hand && ont) {
            assert.equal(ont.delta, hand.delta, `seed ${seed}: deltas differ`);
            assert.equal(ont.len, hand.len, `seed ${seed}: route lengths differ`);
          }
        }
        const ok = sim.buses.filter(bus => feasible(bus, p, sim));
        const pool = ok.length ? ok : sim.buses;
        assignTo(sim, [...pool].sort((x, y) => detourCost(x, p, sim) - detourCost(y, p, sim))[0], p);
      }
    }
  }
});

test("reasoning cost is negligible against the 136 ms decision baseline", () => {
  const bus = at(makeBus("A"), 10, 10);
  const p = pax(9, pt(12, 14), pt(20, 22));
  const stops = [mkStops(p)];
  const N = 20000;
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) canServe(bus, stops, RULES.seats, p);
  const perCallUs = Number(process.hrtime.bigint() - t0) / 1000 / N;
  // insertSeq calls the predicate per candidate route, ~45 per decision, so this budget
  // is per-call generous by design: the whole search must stay well under a millisecond.
  assert.ok(perCallUs < 100, `reasoning costs ${perCallUs.toFixed(1)} us per call`);
});

// ---- vocabulary ---------------------------------------------------------------

test("every axiom references a declared class and property", () => {
  const classes = new Set(VOCABULARY.classes);
  const props = new Set(VOCABULARY.properties);
  for (const a of TBOX) {
    if (a.subject) assert.ok(classes.has(a.subject), `${a.id}: undeclared class ${a.subject}`);
    if (a.property) assert.ok(props.has(a.property), `${a.id}: undeclared property ${a.property}`);
    if (a.form === "range") assert.ok(classes.has(a.value), `${a.id}: undeclared class ${a.value}`);
    if (a.form === "disjoint") {
      for (const c of a.value) assert.ok(classes.has(c), `${a.id}: undeclared class ${c}`);
    }
  }
});

test("states are exactly the three the TBox makes disjoint", () => {
  assert.deepEqual(axiom("passenger-single-state").value, VOCABULARY.states);
});
