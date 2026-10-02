// A/B harness for the fleet dispatch rules.
//
//   uv run uvicorn server:app        # terminal 1
//   node eval_rules.mjs --n 200 --seed 7
//
// Imports the same static/fleet_sim.js the browser does, so there is no second
// implementation to drift out of sync.
//
// Scenarios are generated counterfactually: a sim is driven forward by the greedy
// policy and every decision tick is snapshotted. Both arms then answer the *same*
// snapshots, differing only in the question text. That isolates what we actually
// want to measure — did injecting the rule change the model's choice — instead of
// letting one arm's assignment steer the other arm's future inputs.

import {
  createSim, step, newDemand, assignTo,
  buildState, buildQuestion, buildBaselineQuestion,
  costOf, insertionCost, rankedByDeadhead, manhattan, busPos, headEstimate,
} from "./static/fleet_sim.js";

// Accepts both `--n 200` and `--n=200`.
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const m = argv[i].match(/^--([^=]+)(?:=(.*))?$/);
    if (!m) continue;
    if (m[2] !== undefined) out[m[1]] = m[2];
    else if (argv[i + 1] && !argv[i + 1].startsWith("--")) out[m[1]] = argv[++i];
    else out[m[1]] = true;
  }
  return out;
}
const args = parseArgs(process.argv.slice(2));
const N = Number(args.n ?? 200);
const SEED = Number(args.seed ?? 7);
const ENDPOINT = args.url ?? "http://localhost:8000/api/predict";
const MODEL = args.model ?? "auto";
const WARMUP = Number(args.warmup ?? 5);

// ---- scenario generation -----------------------------------------------------

/** Deep-copy the parts of a bus the cost model reads, so arms see identical inputs. */
const snapBus = b => ({
  id: b.id, label: b.label, x: b.x, y: b.y,
  job: b.job ? { phase: b.job.phase, passenger: b.job.passenger } : null,
  queue: [...b.queue],
});

function generateScenarios(count, seed) {
  const out = [];
  let s = seed;
  while (out.length < count) {
    const sim = createSim({ seed: s++, config: { demandEvery: 0.6 } });
    for (let i = 0; i < 3000 && out.length < count; i++) {
      for (const p of step(sim, 0.05)) {
        if (sim.demandCount < 3) continue;              // let the fleet get going
        out.push({
          buses: sim.buses.map(snapBus),
          passenger: p,
          backlog: sim.passengers.filter(q => q.state === "waiting").length,
          delivered: sim.deliveredCount,
        });
        assignTo(sim, rankedByDeadhead(sim, p)[0], p);  // fixed policy drives the sim
      }
    }
  }
  return out;
}

// ---- arms --------------------------------------------------------------------

/**
 * Rehydrate a snapshot into the shape buildState/buildQuestion expect, so the
 * text sent to the model is byte-identical to what the browser would send.
 */
const simOf = sc => ({ buses: sc.buses, passengers: [], deliveredCount: sc.delivered });

const ARMS = {
  "no-rules": sc => ({
    state: buildBaselineState(sc),
    question: buildBaselineQuestion(),
  }),
  "rules": sc => ({
    state: buildState(simOf(sc), sc.passenger),
    question: buildQuestion(simOf(sc), sc.passenger),
  }),
};

/** The no-rules arm keeps the original state, which had no `rules` key. */
function buildBaselineState(sc) {
  const full = buildState(simOf(sc), sc.passenger);
  const { rules, ...rest } = full;
  return rest;
}

// ---- scoring -----------------------------------------------------------------

/** The best achievable detour and deadhead for this scenario. */
function bests(sc) {
  let detour = Infinity, dead = Infinity;
  for (const b of sc.buses) {
    detour = Math.min(detour, insertionCost(b, sc.passenger.pickup));
    dead = Math.min(dead, costOf(b, sc.passenger.pickup));
  }
  return { detour, dead };
}

async function ask(arm, sc) {
  const { state, question } = ARMS[arm](sc);
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      state,
      questions: { assign: question },
      ...(MODEL !== "auto" ? { model: MODEL } : {}),
    }),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  const body = await res.json();
  const a = body.result.answers.assign;
  return {
    choice: a.choice,
    confidence: a.confidence ?? 0,
    act: (a.action || {}).act_probability ?? 1,
    latency: body.latency_ms,
    head: headEstimate(question),
  };
}

// ---- run ---------------------------------------------------------------------

const scenarios = generateScenarios(N, SEED);
console.error(`generated ${scenarios.length} scenarios from seed ${SEED}`);
console.error(`warming up (${WARMUP} calls, the first one pays the checkpoint load)…`);
for (let i = 0; i < WARMUP; i++) await ask("rules", scenarios[0]);

const results = {};
for (const arm of Object.keys(ARMS)) {
  const rows = [];
  for (const sc of scenarios) {
    let r;
    try {
      r = await ask(arm, sc);
    } catch (err) {
      console.error(`\nFAILED on arm ${arm}: ${err.message}`);
      process.exit(1);
    }
    const bus = sc.buses.find(b => `bus_${b.id.toLowerCase()}` === r.choice);
    if (!bus) { console.error(`unknown choice ${r.choice}`); process.exit(1); }
    const { detour, dead } = bests(sc);
    rows.push({
      detourRegret: insertionCost(bus, sc.passenger.pickup) - detour,
      deadRegret: costOf(bus, sc.passenger.pickup) - dead,
      // did it pick the low-detour bus, and separately the nearest one?
      pickedMinDetour: Math.abs(insertionCost(bus, sc.passenger.pickup) - detour) < 1e-6,
      pickedNearest: Math.abs(costOf(bus, sc.passenger.pickup) - dead) < 1e-6,
      confidence: r.confidence,
      act: r.act,
      latency: r.latency,
      head: r.head,
      // is there actually a conflict to resolve? if the nearest bus is also the
      // low-detour bus, the rule changes nothing and the scenario is uninformative
      tensionful: Math.abs(detour - dead) > 0.5,
    });
  }
  results[arm] = rows;
  process.stderr.write(`${arm}: done\n`);
}

results["greedy"] = scenarios.map(sc => {
  const { detour, dead } = bests(sc);
  const b = rankedByDeadhead(simOf(sc), sc.passenger)[0];
  return {
    detourRegret: insertionCost(b, sc.passenger.pickup) - detour,
    deadRegret: costOf(b, sc.passenger.pickup) - dead,
    pickedMinDetour: Math.abs(insertionCost(b, sc.passenger.pickup) - detour) < 1e-6,
    pickedNearest: true,
    tensionful: Math.abs(detour - dead) > 0.5,
  };
});

// ---- report ------------------------------------------------------------------

const mean = xs => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
const pct = x => `${(100 * x).toFixed(1)}%`;
const pick = (rows, k) => mean(rows.map(r => r[k]));

const arms = ["no-rules", "rules", "greedy"];
const tension = results["rules"].filter(r => r.tensionful);

console.log(`\n${scenarios.length} scenarios (seed ${SEED}), ${tension.length} with a real ` +
            `nearest-vs-least-detour conflict\n`);
const cols = ["mean detour regret", "mean deadhead regret", "picks least-detour",
              "picks nearest", "mean confidence", "act<0.5", "conf<0.45", "ms"];
const w = 22;
console.log("".padEnd(w) + arms.map(a => a.padStart(16)).join(""));
console.log("-".repeat(w + 16 * arms.length));
const rows = [
  ["mean detour regret", a => pick(results[a], "detourRegret").toFixed(2) + " blk"],
  ["mean deadhead regret", a => pick(results[a], "deadRegret").toFixed(2) + " blk"],
  ["picks least-detour", a => pct(mean(results[a].map(r => r.pickedMinDetour)))],
  ["picks nearest", a => pct(mean(results[a].map(r => r.pickedNearest)))],
  ["mean confidence", a => mean(results[a].map(r => r.confidence ?? 0)).toFixed(3)],
  ["act<0.5", a => results[a].some(r => r.act !== undefined) ? pct(mean(results[a].map(r => r.act < 0.5))) : "—"],
  ["conf<0.45", a => results[a].some(r => r.confidence !== undefined) ? pct(mean(results[a].map(r => r.confidence < 0.45))) : "—"],
  ["ms", a => results[a].some(r => r.latency) ? mean(results[a].map(r => r.latency)).toFixed(0) : "—"],
  ["head tokens (est)", a => results[a].some(r => r.head) ? String(Math.max(...results[a].map(r => r.head))) : "—"],
];
for (const [label, fn] of rows) {
  console.log(label.padEnd(w) + arms.map(a => String(fn(a)).padStart(16)).join(""));
}

// The same table restricted to the scenarios where the rule actually changes the
// answer — that is the subset the rules were written for.
if (tension.length) {
  console.log(`\nRestricted to the ${tension.length} conflicting scenarios:\n`);
  console.log("".padEnd(w) + arms.map(a => a.padStart(16)).join(""));
  console.log("-".repeat(w + 16 * arms.length));
  for (const [label, fn] of [
    ["mean detour regret", a => pick(results[a].filter(r => r.tensionful), "detourRegret").toFixed(2) + " blk"],
    ["picks least-detour", a => pct(mean(results[a].filter(r => r.tensionful).map(r => r.pickedMinDetour)))],
    ["mean deadhead regret", a => pick(results[a].filter(r => r.tensionful), "deadRegret").toFixed(2) + " blk"],
  ]) {
    console.log(label.padEnd(w) + arms.map(a => String(fn(a)).padStart(16)).join(""));
  }
}

// Confidence / act distributions, so the floors in fleet.html are set from data
// rather than guessed. Both are compressed near zero: temperature_by_options
// ships choice:3-5 at 1.76, which softens the logits, and a near-uniform 3-way
// distribution has low entropy confidence.
function percentiles(rows, key) {
  const xs = rows.map(r => r[key]).sort((a, b) => a - b);
  const q = p => xs[Math.floor(p * (xs.length - 1))];
  return [0.05, 0.10, 0.25, 0.50, 0.95].map(p => `${(p * 100).toFixed(0)}% ${q(p).toFixed(3)}`).join("  ");
}
console.log(`\nconfidence (rules arm): ${percentiles(results["rules"], "confidence")}`);
console.log(`act_probability      : ${percentiles(results["rules"], "act")}`);
console.log(`A floor set near the 10th percentile defers roughly the bottom 10%.`);

// ---- closed loop -------------------------------------------------------------

// The counterfactual comparison above isolates the decision. This one closes the
// loop: each policy drives its own sim to completion, so it also pays for the
// assignments it makes downstream.
//
// Caveat worth reading before trusting it: the simulator is still SERIAL — a bus
// serves its queue one job at a time — while insertionCost prices a bus as if it
// could multi-pickup. Until the simulator grows a manifest (phase 5), optimising
// for detour optimises for a capability the sim does not have, and this number can
// legitimately come out worse. That gap is the point of measuring it.
async function closedLoop(policy, seed, seconds = 120) {
  const sim = createSim({ seed, config: { demandEvery: 0.6 } });
  const ticks = Math.round(seconds / 0.05);
  let calls = 0;
  for (let i = 0; i < ticks; i++) {
    for (const p of step(sim, 0.05)) {
      if (sim.demandCount < 3) { assignTo(sim, rankedByDeadhead(sim, p)[0], p); continue; }
      let bus;
      if (policy === "greedy") {
        bus = rankedByDeadhead(sim, p)[0];
      } else {
        const { state, question } = ARMS[policy]({ ...sim, passenger: p });
        const res = await fetch(ENDPOINT, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ state, questions: { assign: question },
            ...(MODEL !== "auto" ? { model: MODEL } : {}) }),
        });
        if (!res.ok) { bus = rankedByDeadhead(sim, p)[0]; }
        else {
          const ch = (await res.json()).result.answers.assign.choice;
          bus = sim.buses.find(b => `bus_${b.id.toLowerCase()}` === ch) || rankedByDeadhead(sim, p)[0];
        }
        calls++;
      }
      assignTo(sim, bus, p);
    }
  }
  const riding = sim.passengers.filter(p => p.state === "riding").length;
  return { delivered: sim.deliveredCount, demands: sim.demandCount, riding, calls };
}

if (args.closed !== false) {
  console.log(`\nClosed loop, ${args.loopsecs ?? 120}s per policy:\n`);
  const w2 = 22;
  console.log("".padEnd(w2) + ["no-rules", "rules", "greedy"].map(a => a.padStart(14)).join(""));
  console.log("-".repeat(w2 + 14 * 3));
  const runs = {};
  for (const p of ["no-rules", "rules", "greedy"]) {
    const rows = [];
    for (const sd of [11, 12, 13]) rows.push(await closedLoop(p, sd));
    runs[p] = rows;
  }
  const line = (label, fn) =>
    console.log(label.padEnd(w2) + ["no-rules", "rules", "greedy"].map(p => String(fn(runs[p])).padStart(14)).join(""));
  line("delivered", r => mean(r.map(x => x.delivered)).toFixed(1));
  line("demands", r => mean(r.map(x => x.demands)).toFixed(1));
  line("still riding", r => mean(r.map(x => x.riding)).toFixed(1));
  line("calls", r => mean(r.map(x => x.calls)).toFixed(0));
  console.log(`\nSerial simulator: the detour objective is only correct once buses can actually`);
  console.log(`multi-pickup (phase 5). Treat "delivered" here as a canary, not a verdict.`);
}
