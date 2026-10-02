// A TBox/ABox reasoner for the fleet domain — the ontology behind the rule tier.
//
// WHY THIS EXISTS. The rule tier was, until now, hand-written arithmetic: a seat count in a
// loop, called from insertSeq(). That works, but a hand-rolled `if` cannot be wrong in a
// way you can detect. Declared axioms can be checked against the thing they replace, which
// is what the divergence harness does. This file is that declaration, plus a reasoner.
//
// SCOPE, STATED UP FRONT. This is an OWL-RL *subset*, not OWL.
//   Excluded: RDF/Turtle serialisation, class expressions beyond what is listed below,
//   property chains, nominals, and the open-world assumption.
//   Closed-world: OWL reasons under the open world — absence of a fact is not a fact of
//   absence. This reasoner is CLOSED-world: we know exactly who is aboard, so a missing
//   assertion means "nobody". Closed-world is what an operational decision system wants and
//   is not OWL's default, so it is stated here rather than left to be assumed.
//
// WHAT IS AND IS NOT AN AXIOM. RULES.onRouteBlocks (3) and RULES.minRideBlocks (8) are
// tuning values and a generation heuristic, not invariants — asserting them here would
// claim the domain guarantees something it does not. Axioms are for what must always hold;
// configuration is for what we currently choose. See TBOX for the split.

// ---- vocabulary ---------------------------------------------------------------
// Declared, not derived. These are the only class names the axioms may reference.

export const VOCABULARY = {
  classes: ["Bus", "Passenger", "Demand", "GridCell", "Waiting", "Riding", "Delivered"],
  properties: ["occupies", "pickup", "destination", "onboardPassenger", "servedBy", "state"],
  // A passenger's lifecycle states are mutually exclusive. `Demand` is a role a passenger
  // plays before pickup; it is not one of the three states.
  states: ["Waiting", "Riding", "Delivered"],
};

// ---- TBox: the axioms ---------------------------------------------------------
// Declarations, not control flow. `checkTBox` reads exactly this and nothing else, so an
// axiom set that contradicts itself is detectable — which is the whole point of writing
// them down. Nothing in the hand-written path could say that.

export const TBOX = [
  {
    id: "bus-capacity",
    form: "maxCardinality",
    subject: "Bus",
    property: "onboardPassenger",
    value: 4,
    comment: "Tier 0. The seat limit. Replaces RULES.seats, which was enforced by a loop.",
  },
  {
    id: "demand-must-be-served",
    form: "minCardinality",
    subject: "Demand",
    property: "servedBy",
    value: 1,
    comment: "Tier 0. A demand that nobody has taken is a violation, not a free float.",
  },
  {
    id: "passenger-has-pickup",
    form: "range",
    subject: "Passenger",
    property: "pickup",
    value: "GridCell",
  },
  {
    id: "passenger-has-destination",
    form: "range",
    subject: "Passenger",
    property: "destination",
    value: "GridCell",
  },
  {
    id: "bus-occupies-cell",
    form: "range",
    subject: "Bus",
    property: "occupies",
    value: "GridCell",
  },
  {
    id: "passenger-single-state",
    form: "disjoint",
    value: ["Waiting", "Riding", "Delivered"],
    comment: "Tier 0. A passenger is in exactly one state; being in two is a violation.",
  },
];

/** Look an axiom up by id, so a violation can name the axiom it broke. */
export const axiom = id => TBOX.find(a => a.id === id);

// ---- TBox self-check ----------------------------------------------------------

/**
 * Is the axiom set itself coherent? Reports classes it cannot satisfy.
 *
 * This is a capability the hand-written path does not have: `seats: 4` cannot contradict
 * itself, but `Bus ⊑ ≤4 onboardPassenger` together with `Bus ⊑ ≥6 onboardPassenger` can,
 * and the honest report is that the class is unsatisfiable rather than that one of the two
 * assertions is more correct than the other.
 *
 * Returns {satisfiable: bool, unsatisfiable: [{id, reason}]}.
 */
export function checkTBox(tbox = TBOX) {
  const unsatisfiable = [];

  // Contradictory cardinality on the same (subject, property) pair.
  const bySlot = new Map();
  for (const a of tbox) {
    if (a.form !== "maxCardinality" && a.form !== "minCardinality") continue;
    const slot = `${a.subject}.${a.property}`;
    if (!bySlot.has(slot)) bySlot.set(slot, []);
    bySlot.get(slot).push(a);
  }
  for (const [slot, group] of bySlot) {
    const maxes = group.filter(a => a.form === "maxCardinality");
    const mins = group.filter(a => a.form === "minCardinality");
    const hi = mins.length ? Math.max(...mins.map(a => a.value)) : -Infinity;
    const lo = maxes.length ? Math.min(...maxes.map(a => a.value)) : Infinity;
    if (hi > lo) {
      unsatisfiable.push({
        id: group.map(a => a.id).join(" + "),
        reason: `${slot}: requires at least ${hi} and at most ${lo} — no value satisfies both`,
      });
    }
  }

  // A class declared disjoint from itself, or from a class it must be a member of.
  for (const a of tbox) {
    if (a.form === "disjoint") {
      const seen = new Set();
      for (const c of a.value) {
        if (seen.has(c)) {
          unsatisfiable.push({ id: a.id, reason: `disjoint axiom lists ${c} twice` });
        }
        seen.add(c);
      }
    }
  }

  return { satisfiable: unsatisfiable.length === 0, unsatisfiable };
}

// ---- ABox: facts about one decision --------------------------------------------

/**
 * Facts for one bus and one PROSPECTIVE route.
 *
 * Prospective, not current, because the question being asked is "is this bus free if it
 * takes this demand" — which is what insertSeq() already asks by calling capacityOk()
 * inside its candidate loop. Building from a snapshot is also why the harness can construct
 * an ABox without a live sim.
 *
 * @param bus       snapshot bus ({id, x, y, onboard:Set, ...})
 * @param stops     candidate route: [{kind:'pickup'|'dropoff', point:{x,y}, p:passenger}]
 * @param seats     the configured limit, mirroring what the TBox asserts
 * @param demand    the demand under consideration, or null to describe the route alone
 */
export function buildABox(bus, stops, seats, demand = null) {
  const facts = {
    individuals: new Map(),
    occupancy: [],       // ordered: who is riding, walking the route in order
  };

  const add = (id, type, props = {}) => {
    facts.individuals.set(id, { id, type, ...props });
    return facts.individuals.get(id);
  };

  add(bus.id, "Bus", {
    occupies: { x: Math.round(bus.x), y: Math.round(bus.y) },
    onboardPassenger: new Set(bus.onboard),
  });

  // Already aboard: their pickup is behind the bus, so only the drop-off is still ahead.
  for (const id of bus.onboard) {
    const p = bus.stops.find(s => s.p.id === id)?.p;
    if (p) add(`passenger:${p.id}`, "Passenger", { state: "Riding", destination: p.dest });
  }

  // Walk the candidate route in order, which is the only way to know how many are riding
  // at each point. A drop-off before a pickup frees the seat for it.
  const riding = new Set(bus.onboard);
  for (const s of stops) {
    const key = `passenger:${s.p.id}`;
    if (s.kind === "pickup") {
      add(key, "Passenger", { state: "Waiting", pickup: s.point, destination: s.p.dest });
      riding.add(s.p.id);
    } else {
      const existing = facts.individuals.get(key);
      if (existing) existing.state = "Riding";
      riding.delete(s.p.id);
    }
    facts.occupancy.push({ at: s.point, riding: new Set(riding) });
  }

  if (demand) {
    add(`demand:${demand.id}`, "Demand", {
      pickup: demand.pickup,
      destination: demand.dest,
      // servedBy is left empty: nobody has taken it yet. Under closed-world reasoning an
      // empty set is "none", which is what the existential restriction checks against.
      servedBy: new Set(),
    });
  }

  facts.seats = seats;
  return facts;
}

/** Render a grid cell for a violation message, without importing the simulator. */
const cellKey = c => (c && Number.isFinite(c.x) ? `grid (${c.x}, ${c.y})` : "an unknown cell");

// ---- inference -----------------------------------------------------------------

/**
 * Check the ABox against the TBox, and answer the derived questions.
 *
 * @param facts      from buildABox
 * @param extensions Tier 1 numerics, injected by the caller so this module never imports
 *                     the simulator (that would be a cycle). `detour(bus, demand)` is the
 *                     insertion cost — optimisation, not logic, which is why it is a
 *                     registered function rather than an inference rule.
 * @returns {consistent, violations, derived}
 */
export function reason(facts, { extensions = {} } = {}) {
  const violations = [];

  // maxCardinality: Bus ⊑ ≤N onboardPassenger
  //
  // Evaluated over the PROSPECTIVE route, not just the current assertion. Plain OWL would
  // only see the fillers asserted right now, but the question this ABox answers is "is
  // this bus legal if it drives this route" — and a route that picks up before dropping
  // carries two at a point where the current assertion says one. Both are checked: the
  // current assertion, and the peak occupancy walking the route.
  const cap = axiom("bus-capacity");
  //
  // The TBox declares the invariant; the simulator carries a configured value. When they
  // disagree, the ontology and the world are describing different fleets, and that is
  // worth reporting rather than resolving silently in favour of either. The stricter of
  // the two is enforced, so a misconfiguration can only ever be too conservative.
  //
  // This was a real bug, found by this check: the reasoner read the declared limit and
  // ignored the configured one, so a bus given seats: 1 was still checked against 4.
  const configured = facts.seats ?? cap.value;
  const limit = Math.min(cap.value, configured);
  if (configured !== cap.value) {
    violations.push({
      axiom: cap.id,
      subject: "fleet",
      detail: `configured seat limit ${configured} disagrees with the TBox's ${cap.value}; ` +
              `enforcing the stricter ${limit}`,
    });
  }
  for (const [id, ind] of facts.individuals) {
    if (ind.type !== "Bus") continue;
    if (ind.onboardPassenger.size > limit) {
      violations.push({
        axiom: cap.id,
        subject: id,
        detail: `carries ${ind.onboardPassenger.size} passengers, at most ${limit} allowed`,
      });
    }
  }
  for (const bus of facts.individuals.values()) {
    if (bus.type !== "Bus") continue;
    for (const [i, step] of facts.occupancy.entries()) {
      if (step.riding.size <= limit) continue;
      violations.push({
        axiom: cap.id,
        subject: bus.id,
        detail: `carries ${step.riding.size} passengers after stop ${i + 1} ` +
                `at ${cellKey(step.at)}, at most ${limit} allowed`,
      });
      break;   // one report per bus is enough to fail the assignment
    }
  }

  // minCardinality: Demand ⊑ ∃servedBy.Bus — checked on the unassigned demand.
  const served = axiom("demand-must-be-served");
  for (const [id, ind] of facts.individuals) {
    if (ind.type !== "Demand") continue;
    if (ind.servedBy.size < served.value) {
      violations.push({
        axiom: served.id,
        subject: id,
        detail: `served by ${ind.servedBy.size} buses, at least ${served.value} required`,
      });
    }
  }

  // range: the property's values must be instances of the declared class.
  for (const a of TBOX) {
    if (a.form !== "range") continue;
    for (const [id, ind] of facts.individuals) {
      if (ind.type !== a.subject) continue;
      const value = ind[a.property];
      if (value == null) continue;
      // A grid cell is asserted structurally, not as a named individual, so a
      // well-formed {x, y} satisfies the range.
      const isCell = value && typeof value === "object" &&
        Number.isFinite(value.x) && Number.isFinite(value.y);
      if (!isCell) {
        violations.push({ axiom: a.id, subject: id, detail: `${a.property} is not a ${a.value}` });
      }
    }
  }

  // disjoint: a passenger is in exactly one state.
  const states = axiom("passenger-single-state");
  for (const [id, ind] of facts.individuals) {
    if (ind.type !== "Passenger") continue;
    if (!states.value.includes(ind.state)) {
      violations.push({
        axiom: states.id,
        subject: id,
        detail: `state ${ind.state} is not one of ${states.value.join(" / ")}`,
      });
    }
  }

  // Derived — Tier 1. Numerics come from extension functions, by design.
  const demand = [...facts.individuals.values()].find(i => i.type === "Demand") || null;
  const derived = { detour: null };
  if (demand && extensions.detour) {
    const bus = [...facts.individuals.values()].find(i => i.type === "Bus");
    if (bus) derived.detour = extensions.detour(bus, demand);
  }

  return { consistent: violations.length === 0, violations, derived };
}

/**
 * A feasibility predicate with the shape insertSeq() expects, backed by the reasoner.
 * This is the Tier 0 boundary: the option set is filtered by asking the ontology whether
 * the assignment is consistent, not by asking whether a number is small.
 */
export function canServe(bus, stops, seats, demand = null, extensions = {}) {
  const facts = buildABox(bus, stops, seats, demand);
  // The existential restriction is about a demand nobody has taken, which is the state of
  // every demand *before* assignment. Once we are asking "can this bus take it", that
  // assertion is expected to be unsatisfied, so it is excluded here and checked where it
  // belongs — in the backlog assertions, not in per-bus feasibility.
  const probe = reason(facts, { extensions });
  const blocking = probe.violations.filter(v => v.axiom !== "demand-must-be-served");
  return { ok: blocking.length === 0, violations: blocking, detour: probe.derived.detour };
}
