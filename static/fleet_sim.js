// Pure fleet simulation + the laya dispatch question.
//
// No DOM: the browser (fleet.html) and the A/B harness (eval_rules.mjs) both import
// this module, so the two can never drift. Everything here is a pure function of the
// sim state passed in, and the RNG is seeded, so a scenario replays exactly.
//
// A bus holds an ordered list of `stops` and the physics walks it. The cost model reads
// the same list, so the number quoted in the question text is the number the bus
// actually drives. That shared representation is the whole point: the previous version
// priced buses with an insertion heuristic while the physics served jobs one at a time,
// and the gap between those two models is exactly what the closed-loop harness caught.
//
// `multiPickup` switches between two worlds on the same representation:
//   true  — a demand is inserted into the route; a bus carries up to RULES.seats
//   false — a demand waits in the queue; a bus carries exactly one, ever
// The cost model branches with it, because a pickup on your route is only cheap if you
// can actually stop for it.

export const GRID = 50;

export const BUSES = {
  A: { label: "Bus A", color: "#f87171", css: "bus-a", colorName: "red", start: { x: 10, y: 40 } },
  B: { label: "Bus B", color: "#4ade80", css: "bus-b", colorName: "green", start: { x: 40, y: 10 } },
  C: { label: "Bus C", color: "#60a5fa", css: "bus-c", colorName: "blue", start: { x: 25, y: 25 } },
};

export const ASSIGNMENTS = ["bus_a", "bus_b", "bus_c"];

/**
 * The domain rules. Single source of truth: the state text and the question text are
 * both generated from this, so they cannot disagree.
 */
export const RULES = {
  // A bus may serve several demands at once. This is what makes the insertion
  // heuristic below meaningful: a bus already committed to a route can absorb a
  // nearby pickup for far less than a deadhead run would cost.
  multiPickup: true,
  // Hard capacity. Enforced in code, not described in text: a bus that cannot take a
  // demand is dropped from the question's options entirely.
  seats: 4,
  // A pickup within this many blocks of the planned route counts as "on route".
  onRouteBlocks: 3,
  // A demand shorter than this is not worth a detour at all.
  minRideBlocks: 8,
};

// ---- geometry ----------------------------------------------------------------

export const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
export const cellText = c => `grid (${c.x}, ${c.y})`;
// buses carry x/y as numbers; wrap them as a point for distance helpers
export const busPos = bus => ({ x: bus.x, y: bus.y });

// ---- seeded rng --------------------------------------------------------------

/** Deterministic RNG, so a scenario replays identically across runs and arms. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- simulation state --------------------------------------------------------

export const DEFAULT_CONFIG = {
  speed: 14,          // cells per second
  demandEvery: 5,     // seconds between demands
  multiPickup: RULES.multiPickup,
};

export function makeBus(id) {
  const spec = BUSES[id];
  return {
    id, label: spec.label, color: spec.color, css: spec.css,
    x: spec.start.x, y: spec.start.y,
    stops: [],           // remaining stops, in order: {kind, point, p}
    onboard: new Set(),  // ids of passengers currently riding
    queue: [],           // demands waiting their turn (serial mode)
  };
}

export function createSim({ seed = 1, busIds = ["A", "B", "C"], config = {} } = {}) {
  return {
    buses: busIds.map(makeBus),
    passengers: [],
    demandCount: 0,
    deliveredCount: 0,
    blocksDriven: 0,
    demandTimer: 0,
    config: { ...DEFAULT_CONFIG, ...config },
    rng: mulberry32(seed),
  };
}

function randomCell(sim) {
  return { x: 1 + Math.floor(sim.rng() * (GRID - 2)),
           y: 1 + Math.floor(sim.rng() * (GRID - 2)) };
}

/** Create one demand. Does not dispatch it — the caller decides how. */
export function newDemand(sim) {
  sim.demandCount++;
  const pickup = randomCell(sim);
  let dest = randomCell(sim);
  while (manhattan(pickup, dest) < RULES.minRideBlocks) dest = randomCell(sim);
  const passenger = { id: sim.demandCount, pickup, dest, state: "waiting", bus: null };
  sim.passengers.push(passenger);
  return passenger;
}

// ---- the route ----------------------------------------------------------------

const stop = (kind, p) => ({ kind, point: kind === "pickup" ? p.pickup : p.dest, p });

/** Blocks to walk the given stops from `from`, in order. */
export function routeLength(from, stops) {
  let d = 0, prev = from;
  for (const s of stops) { d += manhattan(prev, s.point); prev = s.point; }
  return d;
}

/** What this bus still has to drive, in order. This is the "current route". */
export const plannedPath = bus => bus.stops.map(s => s.point);

/**
 * Would walking these stops ever exceed the seat limit? Seeded with the passengers
 * already riding, whose pickups are behind the bus and whose dropoffs are still ahead.
 *
 * This is deliberately part of the same scan that picks the route, not a separate
 * check, so a cost quoted in the question text can never describe a route the bus
 * would refuse to drive.
 */
export function capacityOk(stops, onboard, seats) {
  const riding = new Set(onboard);
  for (const s of stops) {
    if (s.kind === "pickup") riding.add(s.p.id); else riding.delete(s.p.id);
    if (riding.size > seats) return false;
  }
  return true;
}

/**
 * Cheapest place to splice `newStops` into this bus's route, keeping their relative
 * order (a dropoff may never precede its own pickup). Routes are ~10 stops and there
 * are 3 buses, so the O(n^2) slot scan is free. Returns null when no placement fits
 * the seat limit.
 */
export function insertSeq(bus, newStops, seats = RULES.seats) {
  const base = bus.stops, here = busPos(bus);
  const before = routeLength(here, base);
  const slots = base.length + newStops.length;
  let best = null;

  const place = (chosen) => {
    if (chosen.length === newStops.length) {
      const cand = [];
      let bi = 0, k = 0;
      for (let idx = 0; idx < slots; idx++) {
        if (k < chosen.length && chosen[k] === idx) { cand.push(newStops[k]); k++; }
        else cand.push(base[bi++]);
      }
      if (!capacityOk(cand, bus.onboard, seats)) return;
      const len = routeLength(here, cand);
      if (best === null || len < best.len) best = { len, stops: cand, delta: len - before };
    }
    for (let i = (chosen.length ? chosen[chosen.length - 1] + 1 : 0); i < slots; i++) {
      place([...chosen, i]);
    }
  };
  place([]);
  return best;
}

/**
 * How many stops a bus will commit to planning ahead. A bus holding `seats` passengers
 * has at most that many outstanding pickup/dropoff pairs, so anything beyond this is a
 * commitment the bus cannot service soon. It is also what keeps the O(n^2) insertion
 * scan cheap: without a horizon the route grows by two stops per demand while demands
 * arrive faster than three buses can serve them, and the scan blows up.
 *
 * A bus at its horizon reports "full" even if a seat happens to be free, which is the
 * honest answer — it is committed.
 */
export const planHorizon = (seats = RULES.seats) => 2 * seats;

/** Splice a whole demand — pickup and dropoff — into the route. */
export function insertStops(bus, passenger, seats = RULES.seats) {
  if (bus.stops.length + 2 > planHorizon(seats)) return null;
  return insertSeq(bus, [stop("pickup", passenger), stop("dropoff", passenger)], seats);
}

/** Can this bus take this demand at all, in the world the sim is currently in? */
export function feasible(bus, passenger, sim) {
  if (!sim.config.multiPickup) return true;   // it can queue, however long that takes
  return insertStops(bus, passenger) !== null;
}

/** Seats not currently occupied. In serial mode a busy bus reports 0, honestly. */
export function seatsLeft(bus, sim) {
  if (!sim.config.multiPickup) {
    return (bus.stops.length === 0 && bus.queue.length === 0) ? 1 : 0;
  }
  return Math.max(0, RULES.seats - bus.onboard.size);
}

/**
 * Hand a demand to a bus. Returns false when the bus cannot take it (no seat, or it is
 * already at its planning horizon), in which case the passenger stays `waiting` for
 * another bus or a later tick.
 *
 * Only the serial mode uses `queue`: there a committed bus can always take more work
 * eventually. In multi-pickup mode an infeasible bus is filtered out of the question
 * instead, so a demand it could not take is never assigned to it in the first place.
 */
export function assignTo(sim, bus, passenger) {
  if (!bus) return false;
  if (sim.config.multiPickup) {
    const ins = insertStops(bus, passenger);
    if (!ins) return false;
    bus.stops = ins.stops;
  } else {
    // A serial bus can start one job; anything else waits its turn.
    if (bus.stops.length === 0 && bus.queue.length === 0) {
      bus.stops = [stop("pickup", passenger), stop("dropoff", passenger)];
    } else {
      bus.queue.push(passenger);
    }
  }
  passenger.bus = bus.id;
  passenger.state = "assigned";
  return true;
}

// ---- physics -----------------------------------------------------------------

/**
 * Advance physics by `dt` seconds. Returns the demands spawned this tick (empty
 * usually) so the caller can dispatch them; the spawner fires at most one per tick.
 */
export function step(sim, dt) {
  const spawned = [];
  sim.demandTimer += dt * 1000;
  if (sim.demandTimer >= sim.config.demandEvery * 1000) {
    sim.demandTimer = 0;
    spawned.push(newDemand(sim));
  }

  for (const bus of sim.buses) {
    // Serial only: pull the next queued job once the current one has run out. In
    // multi-pickup mode a demand is only ever assigned if it could be spliced into the
    // route, so the queue stays empty.
    if (!sim.config.multiPickup && !bus.stops.length && bus.queue.length) {
      const p = bus.queue.shift();
      bus.stops = [stop("pickup", p), stop("dropoff", p)];
    }
    const next = bus.stops[0];
    if (!next) continue;

    // axis-by-axis Manhattan stepping, as in the original loop
    const tx = next.point.x, ty = next.point.y;
    const prevX = bus.x, prevY = bus.y;
    if (bus.x !== tx) {
      bus.x += Math.sign(tx - bus.x) * Math.min(sim.config.speed * dt, Math.abs(tx - bus.x));
    } else if (bus.y !== ty) {
      bus.y += Math.sign(ty - bus.y) * Math.min(sim.config.speed * dt, Math.abs(ty - bus.y));
    }
    sim.blocksDriven += Math.abs(bus.x - prevX) + Math.abs(bus.y - prevY);

    if (bus.x === tx && bus.y === ty) {
      if (next.kind === "pickup") {
        next.p.state = "riding";
        bus.onboard.add(next.p.id);
      } else {
        next.p.state = "delivered";
        sim.deliveredCount++;
        bus.onboard.delete(next.p.id);
      }
      bus.stops.shift();
    }
  }
  return spawned;
}

// ---- costs -------------------------------------------------------------------

/** The naive dispatcher: straight-line distance to the pickup, ignoring all work. */
export const nearestCost = (bus, p) => manhattan(busPos(bus), p.pickup);

/**
 * Honest serial cost: blocks before this bus could begin the new demand, finishing
 * its current route and its whole queue first. This is what the world actually costs
 * when a bus carries one passenger at a time — which is why, in serial mode, a pickup
 * lying on the route buys the passenger nothing.
 */
export function serialCost(bus, p) {
  let from = busPos(bus), d = 0;
  for (const s of bus.stops) { d += manhattan(from, s.point); from = s.point; }
  for (const q of bus.queue) {
    d += manhattan(from, q.pickup); from = q.pickup;
    d += manhattan(from, q.dest); from = q.dest;
  }
  return d + manhattan(from, p.pickup);
}

/**
 * The multi-pickup rule, as arithmetic: the fewest extra blocks this bus would add to
 * the route it is already driving by also serving this demand — pickup *and* dropoff,
 * because serving the demand means driving to both ends.
 *
 * Mode-aware on purpose. In multi-pickup mode this is an insertion delta, so a pickup
 * already on the route costs almost nothing. In serial mode the bus physically cannot
 * stop until it is free, so the delta is the full serial cost and the route buys
 * nothing. Reporting "+0 blocks, on its route" for a bus that is not allowed to stop
 * is the exact dishonesty this model was rewritten to eliminate.
 */
export function detourCost(bus, p, sim) {
  if (!sim.config.multiPickup) return serialCost(bus, p);
  const ins = insertStops(bus, p);
  return ins ? ins.delta : serialCost(bus, p);
}

/**
 * How far the *pickup* alone sits from the route this bus already drives. Kept separate
 * from `detourCost` because they answer different questions: a demand can have its
 * pickup right on the route and its destination well past the end, which makes the
 * pickup free and the demand as a whole not.
 */
export function pickupDetour(bus, p, sim) {
  if (!sim.config.multiPickup) return serialCost(bus, p);
  const ins = insertSeq(bus, [stop("pickup", p)]);
  return ins ? ins.delta : serialCost(bus, p);
}

/** True when the pickup is close enough to the route to count as nearly free. */
export function isOnRoute(bus, p, sim) {
  return pickupDetour(bus, p, sim) <= RULES.onRouteBlocks;
}

function rankBy(sim, passenger, costFn) {
  return [...sim.buses].sort((a, b) => costFn(a, passenger, sim) - costFn(b, passenger, sim));
}

/** The naive ranking: nearest bus, straight-line. Also the greedy baseline. */
export const rankedByNearest = (sim, p) => rankBy(sim, p, nearestCost);
/** The rules-aware ranking: least added blocks to the route each bus already has. */
export const rankedByDetour = (sim, p) => rankBy(sim, p, detourCost);

export const greedyAssign = (sim, p) => rankedByNearest(sim, p)[0];

// ---- the text sent to laya ----------------------------------------------------

export function busText(bus, sim) {
  const at = `at ${cellText({ x: Math.round(bus.x), y: Math.round(bus.y) })}`;
  const free = seatsLeft(bus, sim);

  if (!sim.config.multiPickup) {
    if (!bus.stops.length) {
      // A bus with a queue but no active job is a real state: it has just dropped its
      // last passenger and has not been handed the next one yet.
      if (!bus.queue.length) {
        return `${bus.label} is idle and empty, ${at}, serves one demand at a time.`;
      }
      return `${bus.label} is between jobs, ${at}, with ${bus.queue.length} demand` +
        `${bus.queue.length === 1 ? "" : "s"} queued. It carries one at a time, and will ` +
        `start the next from its queue.`;
    }
    const first = bus.stops[0];
    const d = Math.round(manhattan(busPos(bus), first.point));
    const what = first.kind === "pickup" ? "picking up" : "dropping off";
    const q = bus.queue.length ? `, then ${bus.queue.length} more queued` : "";
    return `${bus.label} is ${what} at ${cellText(first.point)} in ${d} blocks, ${at}${q}. ` +
      `It carries one demand at a time, so it cannot take anything else until it is free.`;
  }

  if (!bus.stops.length) {
    return `${bus.label} is idle, ${at}, empty, ${free} of ${RULES.seats} seats free.`;
  }
  const first = bus.stops[0];
  const d = Math.round(manhattan(busPos(bus), first.point));
  const what = first.kind === "pickup" ? "next pickup" : "next drop-off";
  const aboard = bus.onboard.size;
  const rest = bus.stops.length - 1;
  return `${bus.label} carries ${aboard} of ${RULES.seats}, ${what} at ${cellText(first.point)} ` +
    `in ${d} blocks, ${rest} more stop${rest === 1 ? "" : "s"} after that, ${at}. ` +
    `${free} seat${free === 1 ? "" : "s"} free.`;
}

/**
 * The one-line form of a rule, regenerated from RULES. This sits FIRST in the state
 * dict because laya's build_sequence() truncates the state from the right
 * (`st = st[:room]`, truncate_left is never set by Agent.system_one) — anything at the
 * tail of a long state is dropped silently.
 */
export function rulesText(sim) {
  if (!sim.config.multiPickup) {
    return `Rules: a bus carries one demand at a time, so a pickup lying near its route ` +
      `buys nothing until the bus is free. Choose the bus that will reach the pickup soonest.`;
  }
  return `Rules: a bus may carry up to ${RULES.seats} demands at once, and a demand ` +
    `whose pickup lies within ${RULES.onRouteBlocks} blocks of a bus's current route is ` +
    `nearly free for that bus to add. Prefer the bus with the fewest added blocks, ` +
    `not the nearest one.`;
}

/** The state. `rules` first, for the truncation reason above. */
export function buildState(sim, passenger) {
  const ranked = rankedByNearest(sim, passenger);
  const [best, second] = ranked;
  const detourBest = rankedByDetour(sim, passenger)[0];
  const anyFeasible = sim.buses.some(bus => feasible(bus, passenger, sim));
  return {
    rules: rulesText(sim),
    demand: `A new passenger request arrived at ${cellText(passenger.pickup)} and wants to go to ` +
      `${cellText(passenger.dest)}, a ${manhattan(passenger.pickup, passenger.dest)}-block ride.`,
    fleet: ranked.map(bus => busText(bus, sim)).join(" "),
    recommendation: anyFeasible
      ? `${best.label} is nearest: ${Math.round(nearestCost(best, passenger))} blocks away ` +
        `(straight line); ${second.label} would need ${Math.round(nearestCost(second, passenger))}. ` +
        `But ${detourBest.label} adds only ` +
        `${Math.round(detourCost(detourBest, passenger, sim))} blocks to what it is already driving` +
        `${isOnRoute(detourBest, passenger, sim) ? ", because the pickup is on its way" : ""}. ` +
        `You may follow the nearest bus or override it if the situation warrants.`
      : `No bus has room for this demand right now, so it has to wait for a seat to free up.`,
    backlog: `${sim.passengers.filter(p => p.state === "waiting").length} passengers waiting, ` +
      `${sim.deliveredCount} delivered so far`,
  };
}

/** Per-option criteria: the derived numbers the rule produces, one per bus.
 *  Kept short on purpose — each option is capped at 48 tokens and the whole head at
 *  head_max_len (192 on the english checkpoint). */
function criteriaFor(bus, passenger, sim) {
  if (!feasible(bus, passenger, sim)) {
    return `${bus.label}: full, must wait for a seat to free up`;
  }
  if (!sim.config.multiPickup) {
    const cost = Math.round(detourCost(bus, passenger, sim));
    return `${bus.label}: +${cost} blocks before it can start this one; carries one at a time`;
  }
  // Two numbers, because they answer different questions: how far the pickup sits from
  // the route being driven, and what the whole demand costs to absorb.
  const off = Math.round(pickupDetour(bus, passenger, sim));
  const total = Math.round(detourCost(bus, passenger, sim));
  const where = isOnRoute(bus, passenger, sim)
    ? `pickup on its route, +${off} blocks`
    : `pickup ${off} blocks off route, +${off} blocks`;
  return `${bus.label}: ${where}, +${total} total; ${seatsLeft(bus, sim)} of ${RULES.seats} seats free`;
}

/**
 * The dispatch question. Buses that cannot take the demand are omitted from the
 * options entirely — a hard constraint enforced in code, so laya cannot pick an option
 * it never sees. If every bus is full the options are kept and say so, because an
 * empty option set is a hard failure inside Agent.system_one.
 */
export function buildQuestion(sim, passenger) {
  const usable = sim.buses.filter(bus => feasible(bus, passenger, sim));
  const shown = usable.length ? usable : sim.buses;
  const criteria = {};
  for (const bus of shown) {
    criteria[`bus_${bus.id.toLowerCase()}`] = criteriaFor(bus, passenger, sim);
  }
  return {
    type: "choice",
    instructions: sim.config.multiPickup ? INSTRUCTIONS_MULTI : INSTRUCTIONS_SERIAL,
    criteria,
  };
}

const INSTRUCTIONS_MULTI =
  "You are the dispatcher of an autonomous bus fleet. A new passenger demand just arrived. " +
  "Each bus can carry several demands at once, so a bus already driving past a pickup adds " +
  "it for very few extra blocks. Choose the bus that adds the fewest blocks to what it is " +
  "already driving, not simply the nearest one.";

const INSTRUCTIONS_SERIAL =
  "You are the dispatcher of an autonomous bus fleet. A new passenger demand just arrived. " +
  "Each bus carries one demand at a time, so a pickup near a busy bus buys nothing until that " +
  "bus is free. Choose the bus that will be free soonest and closest to the pickup.";

/** The original static question, kept as the A/B baseline arm. */
export function buildBaselineQuestion() {
  const criteria = {};
  for (const [id, spec] of Object.entries(BUSES)) {
    criteria[`bus_${id.toLowerCase()}`] = `assign to ${spec.label} (${spec.colorName})`;
  }
  return {
    type: "choice",
    instructions: "You are the dispatcher of an autonomous bus fleet. " +
      "A new passenger demand just arrived. Which bus should serve it?",
    criteria,
  };
}

/**
 * Guard against the hard failure in Agent.system_one: options that overflow
 * head_max_len raise at inference time. Rough chars/4 estimate — deliberately
 * pessimistic, and the server raises loudly if it is ever wrong.
 */
export function headEstimate(question) {
  let n = question.instructions.length / 4;
  for (const text of Object.values(question.criteria || {})) n += 1 + text.length / 4;
  return Math.ceil(n);
}
