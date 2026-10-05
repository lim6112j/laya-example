// A seeded RNG, shared by the two simulators.
//
// WHY THIS IS ITS OWN FILE. `fleet_sim.js` grew one and `breakout_sim.js` needed the
// same thing: `resetBall()` used `Math.random()`, which makes a scenario replay
// differently on every run, and the whole point of these demos is that a browser and a
// headless harness must be comparing the same thing. Duplicating eight lines would be
// cheaper than a module — but two copies of a PRNG is two copies to keep in sync, and
// both sims re-export it (see below) so the split costs nothing at the call site.

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
