// Pure fleet simulation + the laya dispatch question.
//
// No DOM: the browser (fleet.html) and the A/B harness (eval_rules.mjs) both import
// this module, so the two can never drift. Everything here is a pure function of the
// sim state passed in, and the RNG is seeded, so a scenario replays exactly.
//
// The domain rules live in RULES. They are not decoration: `insertionCost` below is
// the "a bus can pick up more than one demand" rule made computable, and the numbers
// it returns are what the question text is built from. laya reads text; it does not
// execute rules, so the arithmetic happens here and the model arbitrates over the
// result.

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
  freeSeats: 4,
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
  minRideBlocks: 8,
};

export function makeBus(id) {
  const spec = BUSES[id];
  return {
    id, label: spec.label, color: spec.color, css: spec.css,
    x: spec.start.x, y: spec.start.y,
    job: null,            // {phase: "to_pickup"|"to_dest", passenger}
    queue: [],            // passengers waiting for this bus to become free
  };
}

export function createSim({ seed = 1, busIds = ["A", "B", "C"], config = {} } = {}) {
  return {
    buses: busIds.map(makeBus),
    passengers: [],
    demandCount: 0,
    deliveredCount: 0,
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
  while (manhattan(pickup, dest) < sim.config.minRideBlocks) dest = randomCell(sim);
  const passenger = { id: sim.demandCount, pickup, dest, state: "waiting", bus: null };
  sim.passengers.push(passenger);
  return passenger;
}

export function assignTo(sim, bus, passenger) {
  if (!bus) return;
  passenger.bus = bus.id;
  passenger.state = "assigned";
  bus.queue.push(passenger);
}

/**
 * Advance physics by `dt` seconds. Returns the demands spawned this tick (empty
 * usually) so the caller can dispatch them; the spawner fires at most one per tick,
 * as in the original loop.
 */
export function step(sim, dt) {
  const spawned = [];
  sim.demandTimer += dt * 1000;
  if (sim.demandTimer >= sim.config.demandEvery * 1000) {
    sim.demandTimer = 0;
    spawned.push(newDemand(sim));
  }

  for (const bus of sim.buses) {
    // take the next queued job when free
    if (!bus.job && bus.queue.length) {
      bus.job = { phase: "to_pickup", passenger: bus.queue.shift() };
    }
    const target = bus.job
      ? (bus.job.phase === "to_pickup" ? bus.job.passenger.pickup : bus.job.passenger.dest)
      : null;
    if (target) {
      // axis-by-axis Manhattan stepping
      if (bus.x !== target.x) {
        bus.x += Math.sign(target.x - bus.x) *
                Math.min(sim.config.speed * dt, Math.abs(target.x - bus.x));
      } else if (bus.y !== target.y) {
        bus.y += Math.sign(target.y - bus.y) *
                Math.min(sim.config.speed * dt, Math.abs(target.y - bus.y));
      }
      if (bus.x === target.x && bus.y === target.y) {
        if (bus.job.phase === "to_pickup") {
          bus.job.phase = "to_dest";
          bus.job.passenger.state = "riding";
        } else {
          bus.job.passenger.state = "delivered";
          sim.deliveredCount++;
          bus.job = null;
        }
      }
    }
  }
  return spawned;
}

// ---- costs -------------------------------------------------------------------

/**
 * The waypoints this bus still has to visit, in order, after its current position.
 * This is the "current route" the multi-pickup rule refers to.
 */
export function plannedPath(bus) {
  if (bus.job && bus.job.phase === "to_dest") return [bus.job.passenger.dest];
  if (bus.job && bus.job.phase === "to_pickup") {
    return [bus.job.passenger.pickup, bus.job.passenger.dest];
  }
  if (bus.queue.length) {
    const first = bus.queue[0];
    return [first.pickup, first.dest];
  }
  return [];
}

/**
 * Deadhead cost: how many blocks this bus travels before it can start the new
 * demand, serving everything already committed to first. This is the yardstick the
 * original demo ranked by, and it is what "nearest bus" means when a bus can only
 * carry one passenger at a time.
 */
export function costOf(bus, pickup) {
  if (bus.job && bus.job.phase === "to_dest") {
    const p = bus.job.passenger;
    return manhattan(busPos(bus), p.dest) + manhattan(p.dest, pickup);
  }
  if (bus.job && bus.job.phase === "to_pickup") {
    const p = bus.job.passenger;
    return manhattan(busPos(bus), p.pickup) + manhattan(p.pickup, p.dest) +
      manhattan(p.dest, pickup);
  }
  if (bus.queue.length) {
    const first = bus.queue[0];
    return manhattan(busPos(bus), first.pickup) + manhattan(first.pickup, first.dest) +
      manhattan(first.dest, pickup);
  }
  return manhattan(busPos(bus), pickup);
}

/**
 * The multi-pickup rule, as arithmetic: the fewest extra blocks this bus would add
 * to its planned route by also serving `pickup`. Classic VRP insertion — try
 * slotting the pickup between each consecutive pair of waypoints, or append it after
 * the last one, and keep the cheapest.
 *
 * A pickup that lies on the current route therefore costs ~0 here, while
 * `costOf` would charge a full deadhead for the same bus. That gap is exactly the
 * domain knowledge the model is missing, and it is the number the question text
 * reports per bus.
 */
export function insertionCost(bus, pickup) {
  const path = plannedPath(bus);
  const here = busPos(bus);
  if (!path.length) return manhattan(here, pickup);

  // Every candidate is a delta against the path the bus already drives, so they are
  // all comparable: slot the pickup between two waypoints (replacing one leg with
  // two), or append it as a new final leg.
  let best = Infinity;
  let prev = here;
  for (const node of path) {
    const through = manhattan(prev, pickup) + manhattan(pickup, node);
    const direct = manhattan(prev, node);
    best = Math.min(best, through - direct);
    prev = node;
  }
  // append after the final waypoint
  best = Math.min(best, manhattan(prev, pickup));
  return Math.max(0, best);
}

/** True when this pickup is close enough to the route to count as nearly free. */
export function isOnRoute(bus, pickup) {
  return insertionCost(bus, pickup) <= RULES.onRouteBlocks;
}

function rankBy(sim, passenger, costFn) {
  return [...sim.buses].sort((a, b) => costFn(a, passenger.pickup) - costFn(b, passenger.pickup));
}

/** The original ranking: nearest bus by deadhead. Also the greedy baseline. */
export const rankedByDeadhead = (sim, p) => rankBy(sim, p, costOf);
/** The rules-aware ranking: least added blocks to the route each bus already has. */
export const rankedByDetour = (sim, p) => rankBy(sim, p, insertionCost);

export const greedyAssign = (sim, p) => rankedByDeadhead(sim, p)[0];

// ---- the text sent to laya ----------------------------------------------------

export function busText(bus, pickup) {
  const at = `at ${cellText({ x: Math.round(bus.x), y: Math.round(bus.y) })}`;
  if (bus.job === null && bus.queue.length === 0) {
    return `${bus.label} (idle, empty) is ${at}, ${Math.round(manhattan(busPos(bus), pickup))} blocks from the pickup, no jobs queued.`;
  }
  if (bus.job && bus.job.phase === "to_dest") {
    const p = bus.job.passenger;
    const remain = Math.round(manhattan(busPos(bus), p.dest));
    const afterDrop = remain + Math.round(manhattan(p.dest, pickup));
    return `${bus.label} (carrying a passenger) will drop off at ${cellText(p.dest)} in ${remain} blocks, ` +
      `then is ${afterDrop} blocks from the new pickup, ${bus.queue.length} jobs already queued.`;
  }
  if (bus.job && bus.job.phase === "to_pickup") {
    const p = bus.job.passenger;
    const remain = Math.round(manhattan(busPos(bus), p.pickup));
    const afterTrip = remain + Math.round(manhattan(p.pickup, p.dest)) + Math.round(manhattan(p.dest, pickup));
    return `${bus.label} (empty, driving to a pickup at ${cellText(p.pickup)}) is ${remain} blocks from it, ` +
      `then ${afterTrip} blocks from the new pickup, ${bus.queue.length} jobs already queued.`;
  }
  const first = bus.queue[0];
  const remain = Math.round(manhattan(busPos(bus), first.pickup));
  return `${bus.label} (empty, ${bus.queue.length} jobs queued) will reach its first pickup at ${cellText(first.pickup)} in ${remain} blocks, ` +
    `then is ${remain + Math.round(manhattan(first.pickup, first.dest)) + Math.round(manhattan(first.dest, pickup))} blocks from the new pickup.`;
}

/**
 * The one-line form of a rule, regenerated from RULES. This sits FIRST in the state
 * dict because laya's build_sequence() truncates the state from the right
 * (`st = st[:room]`, truncate_left is never set by Agent.system_one) — anything at the
 * tail of a long state is dropped silently.
 */
export function rulesText() {
  return `Rules: a bus may carry up to ${RULES.freeSeats} demands at once, and a demand ` +
    `whose pickup lies within ${RULES.onRouteBlocks} blocks of a bus's current route is ` +
    `nearly free for that bus to add. Prefer the bus with the fewest added blocks, ` +
    `not the nearest one.`;
}

/**
 * The state. `rules` first, for the truncation reason above.
 */
export function buildState(sim, passenger) {
  const ranked = rankedByDeadhead(sim, passenger);
  const [best, second] = ranked;
  const detourBest = rankedByDetour(sim, passenger)[0];
  return {
    rules: rulesText(),
    demand: `A new passenger request arrived at ${cellText(passenger.pickup)} and wants to go to ` +
      `${cellText(passenger.dest)}, a ${manhattan(passenger.pickup, passenger.dest)}-block ride.`,
    fleet: ranked.map(bus => busText(bus, passenger.pickup)).join(" "),
    recommendation: `${best.label} is nearest: ${Math.round(costOf(best, passenger.pickup))} blocks ` +
      `away (serving its current work first); ${second.label} would need ` +
      `${Math.round(costOf(second, passenger.pickup))} blocks. But ${detourBest.label} adds only ` +
      `${Math.round(insertionCost(detourBest, passenger.pickup))} blocks to its route` +
      `${isOnRoute(detourBest, passenger.pickup) ? ", because the pickup is on its way" : ""}. ` +
      `You may follow the nearest bus or override it if the situation warrants.`,
    backlog: `${sim.passengers.filter(p => p.state === "waiting").length} passengers waiting, ` +
      `${sim.deliveredCount} delivered so far`,
  };
}

/** Per-option criteria: the derived numbers the rule produces, one per bus.
 *  Kept short on purpose — each option is capped at 48 tokens and the whole head at
 *  head_max_len (192 on the english checkpoint). The deadhead distance is already in
 *  the state's `fleet` field via busText, so it is not repeated here. */
function criteriaFor(bus, pickup) {
  const detour = Math.round(insertionCost(bus, pickup));
  const where = isOnRoute(bus, pickup)
    ? `pickup on its route, +${detour} blocks`
    : `${detour} blocks off route, +${detour} blocks`;
  return `${bus.label}: ${where}; ${RULES.freeSeats} seats free`;
}

export const INSTRUCTIONS =
  "You are the dispatcher of an autonomous bus fleet. A new passenger demand just arrived. " +
  "Each bus can carry several demands at once, so a bus already driving past a pickup adds " +
  "it for very few extra blocks. Choose the bus that adds the fewest blocks to what it is " +
  "already driving, not simply the nearest one.";

/** The dispatch question, with criteria computed per bus. */
export function buildQuestion(sim, passenger) {
  const criteria = {};
  for (const bus of sim.buses) {
    criteria[`bus_${bus.id.toLowerCase()}`] = criteriaFor(bus, passenger.pickup);
  }
  return { type: "choice", instructions: INSTRUCTIONS, criteria };
}

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
