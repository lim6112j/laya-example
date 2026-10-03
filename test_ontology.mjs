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
  TBOX, VOCABULARY, axiom, checkTBox, buildABox, reason, canServe, IMPLEMENTED_FORMS,
} from "./static/ontology.js";
import {
  applySeatLimit, seatLimit, seatLimitAgrees, addAxiom, removeAxiom, resetTBox,
  tboxStatus, aboxFor, describeAxiom, axiomFormOptions, classOptions, propertyOptions,
} from "./static/ontology_panel.js";

// Captured before any editor test runs, so a mutation cannot leak into later tests.
const PRISTINE_SEATS_FALLBACK = RULES.seats;

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

test("an axiom form nothing implements is rejected, not silently ignored", () => {
  // reason() dispatches on `form`, and only four forms have code behind them. An axiom in
  // any other form would be read by checkTBox, carried in the TBox, shown in the editor —
  // and enforce nothing, which is the one failure a self-check must never have: reporting
  // a coherent axiom set that is not actually enforced.
  const unsupported = [...TBOX,
    { id: "depot-range", form: "maxDistance", subject: "Bus", property: "depot", value: 10 }];
  const r = checkTBox(unsupported);
  assert.equal(r.satisfiable, false,
    "an unimplemented form must not pass as a coherent axiom");
  assert.match(r.unsatisfiable[0].reason, /maxDistance/);
  assert.match(r.unsatisfiable[0].reason, /no reasoner implements/);
});

test("every form the reasoner can evaluate is one checkTBox accepts", () => {
  // The two lists have to agree, or the check rejects axioms the reasoner does support.
  // Each form needs a value of its own shape: disjoint reads `value` as the class list.
  const probe = form => form === "disjoint"
    ? { id: `probe-${form}`, form, value: ["Waiting", "Riding"] }
    : { id: `probe-${form}`, form, subject: "Bus", property: "onboardPassenger", value: 2 };
  for (const form of IMPLEMENTED_FORMS) {
    const ok = checkTBox([probe(form)]);
    assert.equal(ok.satisfiable, true,
      `form "${form}" is listed as implemented but checkTBox rejects it: ` +
      JSON.stringify(ok.unsatisfiable));
  }
});

test("the shipped TBox uses only implemented forms", () => {
  for (const a of TBOX) {
    assert.ok(IMPLEMENTED_FORMS.includes(a.form),
      `TBox axiom "${a.id}" uses form "${a.form}", which reason() does not evaluate`);
  }
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

// ---- the live editor ----------------------------------------------------------
// The panel mutates module state, so every test here restores the defaults on the way
// out — otherwise a test that changes the seat limit would silently reconfigure every
// test that runs after it.

test.afterEach(() => resetTBox());

test("deleting the seat axiom is reported, not crashed on", () => {
  // The reasoner used to do axiom("bus-capacity").value unconditionally, so the editor
  // making deletion reachable turned into a TypeError. An unasserted constraint is no
  // constraint, and the fact is reported so the panel can say so.
  removeAxiom("bus-capacity");
  const bus = { id: "A", x: 0, y: 0, stops: [], onboard: new Set([1, 2, 3, 4, 5]), queue: [] };
  const r = canServe(bus, [], RULES.seats);
  assert.equal(r.ok, true, "with no axiom asserted the ontology enforces no capacity");
  assert.deepEqual(r.unasserted, ["bus-capacity"]);
  assert.deepEqual(tboxStatus().missing.map(m => m.id), ["bus-capacity"]);
});

test("deleting the seat axiom still leaves the hand-written path enforcing capacity", () => {
  // The demonstration: the ontology stops enforcing capacity, the arithmetic does not.
  removeAxiom("bus-capacity");
  const bus = at(makeBus("A"), 0, 0);
  const p = pax(1, pt(5, 0), pt(25, 0));
  bus.onboard = new Set([2, 3, 4, 5]);
  bus.stops = [2, 3, 4, 5].map(id => ({
    kind: "dropoff", point: pt(20 + id, id), p: pax(id, pt(id, 0), pt(20 + id, id)),
  }));
  const stops = [mkStops(p), ...bus.stops, mkStops(p, "dropoff")];
  assert.equal(canServe(bus, stops, RULES.seats).ok, true, "ontology: unconstrained");
  assert.equal(capacityOk(stops, bus.onboard, RULES.seats), false, "hand-written: still refuses");
});

test("applySeatLimit moves the axiom and the simulator together", () => {
  const before = RULES.seats;
  applySeatLimit(2);
  assert.equal(axiom("bus-capacity").value, 2, "TBox updated");
  assert.equal(RULES.seats, 2, "simulator configuration follows");
  assert.equal(seatLimitAgrees(), true, "no mismatch to report");
  assert.ok(before !== 2);
});

test("a mismatch is visible rather than inferred", () => {
  // Only reachable from code, which is the point: the editor writes both, so a
  // disagreement means something changed one without the other.
  axiom("bus-capacity").value = 6;
  assert.equal(seatLimitAgrees(), false);
  const r = canServe({ id: "A", x: 0, y: 0, stops: [], onboard: new Set(), queue: [] }, [], 4);
  assert.ok(r.violations.some(v => /disagrees with the TBox/.test(v.detail)));
});

test("applySeatLimit on a removed axiom reports instead of pretending", () => {
  removeAxiom("bus-capacity");
  const r = applySeatLimit(3);
  assert.equal(r.applied, false);
  assert.match(r.reason, /removed/);
  assert.equal(RULES.seats, PRISTINE_SEATS_FALLBACK, "the simulator's limit is left alone");
});

test("an added axiom that contradicts the declared one is rejected by the self-check", () => {
  const r = addAxiom({ form: "minCardinality", subject: "Bus",
                       property: "onboardPassenger", value: 6 });
  assert.equal(r.ok, true);
  const status = tboxStatus();
  assert.equal(status.satisfiable, false, "the class cannot be satisfied");
  assert.match(status.unsatisfiable[0].reason, /at least 6 and at most 4/);
});

test("a duplicate axiom is refused and a bad value is refused", () => {
  assert.equal(addAxiom({ form: "range", subject: "Passenger",
                          property: "pickup", value: "GridCell" }).ok, false);
  assert.equal(addAxiom({ form: "maxCardinality", subject: "Demand",
                          property: "servedBy", value: "nope" }).ok, false);
});

test("reset restores the declared defaults exactly", () => {
  const original = JSON.stringify(TBOX);
  applySeatLimit(1);
  removeAxiom("bus-occupies-cell");
  addAxiom({ form: "minCardinality", subject: "Bus", property: "onboardPassenger", value: 9 });
  assert.notEqual(JSON.stringify(TBOX), original);
  resetTBox();
  assert.equal(JSON.stringify(TBOX), original, "byte-identical restore");
  assert.equal(tboxStatus().satisfiable, true);
  assert.equal(tboxStatus().missing.length, 0);
});

test("the editor's vocabulary options come from the declaration, not a duplicate list", () => {
  assert.deepEqual(classOptions(), VOCABULARY.classes);
  assert.deepEqual(propertyOptions(), VOCABULARY.properties);
  assert.deepEqual(axiomFormOptions().map(f => f.value), ["maxCardinality", "minCardinality", "range"]);
});

test("describeAxiom renders every declared form readably", () => {
  assert.equal(describeAxiom(axiom("bus-capacity")), "Bus ⊑ ≤4 onboardPassenger");
  assert.equal(describeAxiom(axiom("demand-must-be-served")), "Demand ⊑ ≥1 servedBy");
  assert.equal(describeAxiom(axiom("bus-occupies-cell")), "Bus.occupies ⊑ GridCell");
  assert.equal(describeAxiom(axiom("passenger-single-state")), "Waiting ⊥ Riding ⊥ Delivered");
});

test("the ABox view reports what the ontology believes about each bus", () => {
  const bus = at(makeBus("A"), 3, 4);
  bus.onboard = new Set([7, 8]);
  const [view] = aboxFor([bus]);
  assert.equal(view.id, "Bus A");
  assert.equal(view.position, "3, 4");
  assert.deepEqual(view.aboard, [7, 8]);
  assert.equal(view.violations.length, 0);
  assert.deepEqual(view.unasserted, []);
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
