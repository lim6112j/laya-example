# laya-example

Demo of [laya](https://pypi.org/project/laya/): fast, non-autoregressive routing of
typed questions (choice / score / yes-no) over support-ticket-style state, using
pre-trained ModernBERT decision checkpoints. English input routes to the
`english` checkpoint, Hindi (and other languages) to `multilingual`, with an
explicit override for `typed-decisions`.

## Quickstart

```sh
uv sync
uv run python main.py
```

The three checkpoints (`english`, `multilingual`, `typed-decisions`) are
downloaded from Hugging Face on first use and cached in `~/.cache/huggingface`.

### Intel Macs (x86_64)

`uv sync` handles both architectures from one lockfile — the Intel-only caps in
`pyproject.toml` are marked on `platform_machine`, and both branches are declared
per package so `uv` forks rather than resolving a single version that satisfies
everything:

| | Intel (x86_64) | Apple Silicon (arm64) |
|---|---|---|
| torch | 2.2.2 | 2.11.0 |
| numpy | 1.26.4 | 2.4.6 |
| transformers | 4.57.6 | 5.18.0 |

Intel is capped because PyTorch stopped publishing macOS x86_64 wheels after
2.2.2, and that version forces two caps downstream: it is built against the
numpy 1.x C ABI (numpy 2.x fails with `_ARRAY_API not found`), and transformers 5.x
requires torch>=2.5, so on 2.2.2 it disables PyTorch and leaves `nn` undefined —
the `NameError: name 'nn' is not defined` that greets a fresh Intel checkout.

Two consequences worth knowing:

- **Python is 3.11 on both Macs** (`requires-python = ">=3.11,<3.12"`).
  transformers' ModernBERT module uses `@torch.compile`, and torch 2.2.2's Dynamo
  refuses Python 3.12+. The upper bound is global, not per-arch, so the Silicon
  Mac is held to 3.11 too even though torch 2.11 supports 3.12+. Per-arch lockfiles
  would recover that.
- **Intel runs on CPU.** MPS autocast needs torch>=2.4, so `laya_startup.py`'s
  `_default_device()` probes `torch.autocast` rather than asking
  `torch.backends.mps.is_available()` — on Intel those two disagree, the first
  says yes while the second raises. Expect ~440 ms per prediction rather than
  the ~35 ms an MPS machine gets.

Verified on Intel: `main.py` runs both checkpoints, 19 Python and 45 Node tests
pass. The arm64 column is from resolution, not execution.

## Web UI

A browser UI for the same state / questions → result flow, backed by a
long-lived FastAPI server (the ~7s model preload happens once at startup).
The UI is branded **CJet**; the underlying decision engine is the laya library:

```sh
uv run uvicorn server:app
```

Then open http://localhost:8000. Pick one of twenty built-in examples from the
domain-grouped dropdown (support tickets in English/Korean, plus laya's
pre-tuned preset workflows — email triage, prompt guardrails, content
moderation, model routing — with one benign and one flagged example each) to
autofill both the state and the question cards, or enter the state as JSON and
build questions with the form (type `choice` / `score` / `noul`, instructions,
and per-option criteria rows). Optionally override the model checkpoint and
run: results render per question with probability bars and the routing
metadata (model, repo, reason, latency).

- `server.py` — FastAPI app; `POST /api/predict` validates the payload against
  laya's question schema with pydantic (choice → dict criteria, score → list,
  noul → none) and serves `static/index.html`
- `laya_startup.py` — shared startup used by both `main.py` and the server
  (offline hub + fast-build monkeypatch + preload; see below)
- `test_server.py` — stubbed-API tests, no model load:
  `uv run pytest test_server.py`
- `test_fleet_sim.mjs` — 18 simulation invariants, no model load:
  `node --test test_fleet_sim.mjs`

### laya plays Breakout

`static/breakout.html` (linked from the UI header) is an Atari-Breakout-style
canvas game where laya controls the paddle. laya reads text, not pixels: every
decision tick serializes the game state into a short text observation
(ball position relative to the paddle, direction, estimated time to paddle
level, bricks remaining) and asks one typed question —
`move: choice(left/stay/right)`. The gameplay screen shows the canvas, the
exact state text laya receives ("Laya's view"), and the decision with
probability bars, confidence and latency ("Laya's decision"). Toggle autoplay
off to play by arrow keys and compare; sliders tune the decision interval and
ball speed, and the model override selects the checkpoint.

It also carries an **Ontology** panel — the same TBox editor and ABox view as the
fleet demo — declaring one rule: `Ball ⊑ ≥1 catchableBy`, so a ball the physics
cannot recover is a fact rather than a mistake. Off by default. The two counters
under the canvas split the loss into *provably lost* and *lost after a catchable
ask*, which is the honest form of "the demo stops blaming the model for
physics-impossible catches". See
[The same idea in Breakout](#the-same-idea-in-breakout--and-where-the-fleet-invariant-does-not-carry-over).

### laya runs the warehouse

`static/warehouse.html` (linked from the UI header) simulates a cargo house
and three delivery robots on crossing routes, each with a forward lidar. When
a robot's lidar sees another robot blocking its path, it serializes the
situation to prose (who is ahead and how far, other robots' relative positions
and motions, deliveries completed, seconds blocked) and asks laya
`move: choice(left/stay/right)` — left/right sidestep around the obstacle,
stay waits — so laya resolves the multi-agent blocking/deadlock problem. The screen
shows the canvas with lidar cones and route lines, per-robot lidar chips,
"Laya's view" (the state text the blocked robot sent) and "Laya's decisions"
(action bars, confidence, latency, history). Hard collision prevention is
independent of laya — physics never lets robots overlap; laya decides
strategy. Toggle autoplay off to watch blocked robots stop forever: the
deadlock laya prevents.

### laya dispatches the bus fleet

`static/fleet.html` (linked from the UI header) simulates an autonomous bus
fleet on a 50×50 grid: passengers appear with a pickup and a destination, and
every new demand is dispatched by laya —
`assign: choice(bus_a/bus_b/bus_c)`. Buses drive Manhattan-style through an
ordered list of stops, so a bus can be part-way through several demands at once.
The screen shows the grid with each bus's remaining route drawn as a dashed
polyline and an onboard-count badge, per-bus status chips (`Bus C · 3/4 · drop
grid (39, 39) (26) · +4 stops`), "Laya's view" (the dispatch state sent) and
"Laya's decisions" (assignment bars, confidence, latency, history). Toggle laya
dispatch off to compare against a greedy nearest-bus heuristic; sliders tune
demand frequency and bus speed.

The simulation, the domain rules and the question text live in
`static/fleet_sim.js` — a pure, seeded, DOM-free module. `fleet.html` is only
the DOM layer over it, and `eval_rules.mjs` imports the same file, so the
browser and the A/B harness cannot drift apart.

#### Domain rules: where they go, and why

The interesting question this demo answers is **where a domain rule belongs in
a System-1 model**. The rules here — a bus may carry several demands at once,
and a demand near a bus's current route is nearly free to add — are not in the
weights. laya does not execute rules, it reads text, so a bare assertion
("a bus may carry 2 demands") changes nothing. The rules are injected in four
places, each doing a different job:

| Tier | Where | What it carries |
| --- | --- | --- |
| 0 | code, not text | hard constraints — filter the option set so a rule is *unviolable* rather than *likely* |
| 1 | per-option `criteria` | the rule's derived numbers, one per bus |
| 2 | question `instructions` | the policy, constant across decisions |
| 3 | first key of the state | scenario-level rules |
| 4 | fine-tuning | last resort, gated on measurement |

Tier 1 is the one that matters, because of how the checkpoint is wired:
`build_sequence()` puts each option's criteria immediately before its `[MASK]`,
and the decision is the hidden state *at that mask*. The criteria are the
scorer's literal input.

The arithmetic lives in JS, not the model. `detourCost()` is the multi-pickup
rule made computable — the classic VRP insertion heuristic: fewest extra blocks
for a bus to also serve this demand, by splicing its pickup and drop-off into the
route it is already driving. `pickupDetour()` prices the pickup alone, because
the two answer different questions: a demand can have its pickup right on the
route and its destination well past the end, which makes the pickup free and the
demand as a whole not. Each option's criteria reports both:

```
bus_b: "Bus B: pickup on its route, +0 blocks, +14 total; 2 of 4 seats free"
```

A bus that has no room is **dropped from the options entirely** rather than
described as unavailable. That is Tier 0 doing its job: a rule the model cannot
see an option for is unviolable, not merely discouraged.

The state also carries a `rules:` line, and it is the **first** key: laya's
`build_sequence()` truncates the state from the right (`st = st[:room]`,
`truncate_left` is never set by `Agent.system_one`), so anything at the tail of
a long state is dropped silently. At the current fleet size that is not yet
binding — a mid-episode state is ~272 tokens against a ~459-token budget on the
`english` checkpoint — but it is free insurance as the state grows.

#### Measuring it

```sh
uv run uvicorn server:app            # terminal 1
node eval_rules.mjs --n 200 --seed 7 # terminal 2
```

The harness generates scenarios *counterfactually*: a sim is driven forward by
a fixed policy and every decision tick is snapshotted, then every arm answers
the *same* snapshots differing only in question text. That isolates the decision
instead of letting one arm's assignments steer the other arm's future inputs.

> **The no-rules baseline was re-measured and the old number was wrong.** The arm
> reported below used to strip only the `rules:` key, which left the rule in two other
> places: `recommendation` ended with a sentence built from `detourCost()` and
> `isOnRoute()` — that sentence *is* the rule — and `fleet` was `busText()`, which
> reports seats and multi-stop plans and so presupposes the world the rule describes.
> "No rules" was quietly a rules baseline. `buildBaselineState()` is a separate builder
> with none of that in it, and the clean baseline is roughly **twice as bad** (19.25 →
> 38.33 blocks). The rules arm is unchanged, so the apparent improvement grew — and that
> growth is not a new result, it is an old baseline being corrected.

Over 200 scenarios, 195 of which had a real nearest-vs-least-detour conflict:

```
                              no-rules           rules  rules-nofilter          greedy
mean detour regret           38.33 blk        6.43 blk        6.23 blk       34.60 blk
picks least-detour               39.0%           79.0%           83.6%           48.7%
picks nearest                    65.5%           55.0%           60.0%           76.5%
capacity violations                 51               0               4              48
mean confidence                  0.057           0.103           0.051           0.000
head tokens (est)                   48             137             137               —
ms                                 81             136             141               —
```

Two things to read here. The **injection works** — regret against the
least-detour bus drops ~6× and the model follows the criteria. And the
**capacity filter (Tier 0) is doing real work**: `rules` never assigns to a bus
that has no room, while the unfiltered arm does so four times and the naive
nearest-bus dispatcher forty-eight. Omitting an infeasible option makes the rule
unviolable rather than merely likely.

Note also how close `no-rules` sits to `greedy` (38.33 vs 34.60). Out of the box,
laya is barely better than "send it to the closest bus" on a domain it was never
trained on — which is the whole reason the injection is doing something.

`no-rules` and `greedy` show *negative* nearest-regret, which is not a bug: the
optimum is restricted to buses that can take the work, and the naive dispatcher
picks the closest bus whether or not it has room.

#### Does it transfer? laya vs Jev

Everything above rests on one model, and the obvious objection is that reading
criteria off an option is specific to how laya's scorer reads a `[MASK]`. That is
testable, because **Jev** (TypeSafe, `typesafe/jev-1.13`) has the same interface:
the same typed questions over the same state, answering with the same
`{type, choice, probabilities, confidence}`. `server.py` takes `model: "jev"` and
swaps the backend; `jev.py` normalises the response into laya's envelope so no
client has a branch on who answered.

```sh
node eval_rules.mjs --compare --n 200 --seed 7     # both models, both arms
node eval_rules.mjs --model jev --n 200            # one model
```

200 scenarios, 195 conflicting. Same scenarios, same scoring, both models:

| | arm | detour regret | picks least-detour | capacity viol. | $ / decision | ms |
|---|---|---|---|---|---|---|
| laya | no-rules | 38.33 blk | 39.0% | 51 | $0 local | 81 |
| laya | rules | 6.43 blk | 79.0% | **0** | $0 local | 136 |
| Jev | no-rules | 35.18 blk | 45.5% | 47 | $0.000022 | 276 |
| Jev | rules | **1.54 blk** | **91.0%** | **0** | $0.000032 | 269 |

**It transfers, and transfers better.** Both models are close to useless on this
domain out of the box — laya's no-rules arm (38.33) is *worse than the naive
nearest-bus dispatcher* (34.60), and Jev's (35.18) is no better. Neither has
priors about a bus fleet. With the criteria injected, laya drops 6× and Jev drops
**23×**, landing at 1.54 blocks — within ~2% of the cost-optimal choice on a
scenario set neither model was trained for. So the technique is a property of
System-1 models, not of laya, which is the load-bearing claim in the IR section
below.

Three things worth separating out:

- **The capacity filter behaves differently by model.** With the option removed,
  both models record 0 violations. Without the filter, laya commits 4 and Jev
  commits **0** — Jev reads "full, no room" in the criteria and declines on its
  own. Tier 0 is what makes the rule unviolable for a weaker reader; a stronger
  one can be told. `rules-nofilter` on Jev (0.98 blk) is in fact slightly *better*
  than filtered (1.54) — with all three options visible it picks better among the
  feasible ones, so on Jev the filter costs a little and buys nothing.
- **Confidence is not comparable across models.** laya's rules arm sits at
  p50 0.033 because `temperature_by_options["choice:3-5"] = 1.76` flattens its
  logits; Jev reports p50 0.880 on the same question. The `CONF_FLOOR` tuned on
  laya (0.002) is effectively a no-op for Jev — every Jev decision clears it. A
  gate is a per-model constant, and moving models means re-deriving it.
- **Cost and latency.** Jev is $0.000032 a decision at this question size
  ($0.042/1M input tokens) — about six-tenths of a cent for 200 decisions. The
  real difference is that it is a network call: 269 ms against laya's 136 ms, and
  laya's marginal cost is zero because it runs locally.

Note that this comparison measures *how well each model uses what it is given*,
not domain knowledge: the no-rules arm still contains a nearest-bus recommendation
in prose, by design. Both models get the same hint.

#### The two worlds, and which one the rule is for

`fleet.html` has a **Multi-pickup physics** toggle, and the harness runs the
closed loop in both. Same rules, same cost function, different world.

Closed loop, 200 s per policy, one demand every 1 s, 3 seeds. Cells are
"delivered / blocks per demand":

```
                                     rules                greedy
----------------------------------------------------------------
multi-pickup on               189.7 / 36.0          189.3 / 38.0
multi-pickup off              133.0 / 40.2          127.3 / 39.9
```

**Multi-pickup is worth +43% throughput** to the rules dispatcher and +49% to
greedy. That is the headline, and it is the capability the rule was written for.

The detour rule itself is a smaller, different win: **~5% less distance driven**
(36.0 vs 38.0 blocks per demand) with multi-pickup, and no throughput gain at all
— at one demand per second both policies are capacity-saturated, so they deliver
the same 189. Without multi-pickup it is worth +4% throughput and nothing on
distance. Reported as measured, because the earlier version of this README
claimed the rule "did not pay off" and the real reason was that the simulator
could not multi-pickup at all.

#### Multi-pickup physics

The toggle switches the simulator between two worlds on the *same* route
representation, and the cost model switches with it.

**On** — a bus holds an ordered `stops` list and carries up to `RULES.seats` (4)
at once. A new demand is spliced in at the cheapest feasible position, with the
drop-off always after its own pickup. Buses carry a real seat limit, so a bus
with no room is **removed from the question's options** and the demand waits for
one to free up (counted in the "No seat" HUD field). The bus draws its remaining
route and an onboard-count badge so the plan is visible.

**Off** — a bus carries exactly one demand and queues the rest, which is the
original serial behaviour, preserved exactly.

`detourCost()` and `isOnRoute()` branch on the mode, and that branch is the
point rather than a detail: **a pickup on your route is only cheap if you can
actually stop for it.** In serial mode `detourCost` is the full serial cost and
`isOnRoute` is false for any committed bus, so the same rule honestly reports no
advantage in a world that cannot deliver on it. A test asserts that no committed
serial bus ever reports a pickup as on-route.

The planning horizon reuses `seats` as "how many more passengers a bus commits
to". It is not a second capacity — it is what stops a bus planning forty
orders — and it also keeps the O(n²) insertion scan cheap.

#### What the rewrite actually fixed

The earlier version priced buses with an insertion heuristic while the physics
served one job at a time — two different models of the same bus, and the gap
between them is what the closed loop kept catching. A bus now holds an ordered
`stops` list; the physics walks it and every cost function reads it, so the
number quoted in the question text is the number the bus drives. A test asserts
that identity directly (`routeLength(after) - routeLength(before) === delta`).

Three real bugs the work surfaced, all now covered by tests:

- **A passenger could be assigned twice.** `dispatch()` awaits the prediction, so
  a passenger still in `waiting` was picked up a second time by the retry loop,
  and its drop-off was counted twice — visible in the browser as *delivered 50,
  demands 30*. Fixed with a `dispatching` guard; the `delivered ≤ demands`
  invariant now pins it.
- **The route grew without bound.** Two stops per demand, against demands
  arriving faster than three buses can serve them, and the insertion scan blew
  up — a real hang, not a slow path. Fixed by `planHorizon`.
- **`busText` crashed in serial mode** on a bus with a queue but no active job,
  which is a real state: it has just dropped its last passenger and has not been
  handed the next one yet.

One measurement bug is worth recording too, because it nearly produced a false
negative: an early version of the throughput comparison drove the new demands
through one policy and the retry queue through another, so at low load — where
nothing ever waits — the detour policy was never actually exercised and the two
arms came out identical. The browser's default demand rate hid this as well: at
3 s or more every bus is idle on arrival and any policy collapses to
nearest-bus, so the slider now runs 0.5–4 s.

#### Confidence gating

`fleet.html` falls back to the deterministic cost model when laya's confidence
is below `CONF_FLOOR`, and shows a **decided / deferred** counter in the HUD —
because a gate silently caps how often the model actually decides, and a gate
tuned until laya "always agrees" has removed laya from the loop.

The floor is set from the measured distribution, not guessed. Confidence on this
workload is heavily compressed near zero (p10 0.004, p50 0.041, p95 1.000),
because the checkpoint ships `temperature_by_options["choice:3-5"] = 1.76`,
which softens the logits, and a near-uniform 3-way distribution has low entropy
confidence. An earlier guess of 0.45 deferred **100%** of decisions — the rules
were in the prompt and laya never got to use them.

`CONF_FLOOR` is **0.002**, the measured 10th percentile of the rules arm's confidence
over 200 scenarios (p10 0.002, p50 0.033, p95 1.000). It defers roughly a tenth of
decisions. An intermediate value of 0.013 sat near the 20th percentile and deferred twice
what the comment beside it claimed, which is how it went unnoticed — the number and the
claim now agree. Re-derive it with `node eval_rules.mjs --n 200`; the distribution moves
with the criteria text.

The p95 of 1.000 is the capacity filter showing through — when only one bus has room,
the model is not choosing, and the HUD counter is what makes that visible rather than an
accident.

**The floor is per-model, and this is the trap worth knowing about.** It is
calibrated to laya's distribution (p50 0.033), which is compressed near zero by the
checkpoint's own temperature. Jev reports p50 0.880 for the same question, so the
same 0.002 defers nothing at all — selecting `jev` in the browser defers 0 of 27
decisions, while laya defers about a fifth. Moving models means re-deriving this
constant from that model's percentiles, and reusing a floor across models is not
neutral: it silently disables the gate rather than mis-setting it.

`act_probability` is reported but deliberately *not* gated on: measured at
1.000 across every percentile here, so a floor on it would be dead code. Jev does
not return the field at all, which the `?? 1` in `askCJet` absorbs.

#### The ontology, and the enforce toggle

`static/fleet.html` has an **Ontology** panel (left side) with a live TBox
editor, an ABox view of what the reasoner currently believes, and one checkbox:
**enforce Tier 0 with the ontology**.

**Tier 0** is the hard-constraint layer — here, the seat limit. Everything else in
this demo (detour distance, nearest-bus) is a *preference*. Tier 0 can be
answered two ways:

- the hand-written `capacityOk` — an integer comparison, `onboard.size < seats`.
  This is the default and what the browser runs.
- the TBox/ABox reasoner in `static/ontology.js` — axioms are *declared*
  ("a bus may not exceed the seat limit") and a reasoner decides whether an
  assignment is consistent with them.

The toggle swaps the predicate. Nothing else changes: `ontology.js` never imports
`fleet_sim.js`, so the dependency is one-way and the two cannot form a cycle.

**Turning it on should not change any decision. If it does, that is a bug.** That
is the whole point. The ontology is a few hundred lines of reasoner replacing a
one-line integer check, so "do the two reach the same answer?" is the evidence
that the declared axioms faithfully reimplement the code they replace — a
hand-rolled `if` cannot be wrong in a way you can detect, but a declared axiom set
can be compared against the thing it displaces.

`node eval_rules.mjs` runs exactly that comparison, deterministically:

```
600 bus/demand pairs compared, 0 disagreements
```

**That 600 is weaker than it looks, and the panel says so.** `insertStops` returns
`null` at the planning horizon *before* consulting the seat predicate — either
implementation's. Re-measured: of the 600 pairs, **221 were refused by both**
layers without the predicate ever being called, so the seat check actually fired
on **379**. The equivalence is real but rests on fewer comparisons than the
headline suggests.

The browser demo is a *worse* place to check this than the harness, and the toggle's
own note explains why: the demo runs on wall-clock, so two runs see different
demand counts. Counting deliveries on screen is not that check. What is visible
there is that the two behave the same — the harness is where it is settled.

The stronger evidence is `test_ontology.mjs`, which compares the two predicates
directly over 5,000+ candidate routes with the horizon bypassed, and passes.

Editing the TBox (e.g. dropping the seat limit 4 → 2) restarts the simulator on
the same seed, because routes planned under the old limit would otherwise linger.
`checkTBox()` reports satisfiability live, so a contradictory axiom set is visible
before it silently changes dispatch.

##### What this ontology is not

The TBox is data, but the reasoner is **not** data-driven, and the difference is
worth being precise about because the demo's own framing invites the wrong reading.

`reason()` dispatches on an axiom's `form`, and only four forms have code behind
them — `IMPLEMENTED_FORMS` is that list. Everything else about an axiom *is* data:
`subject`, `property`, and `value` are read from the TBox, which is why dropping the
seat limit 4 → 2 changes dispatch with no code change at all. But the **kind** of
constraint is code. Adding a genuinely new kind — "a bus may not drift more than
10 cells from its depot" — is not a TBox edit.

Before this was enforced, such an axiom passed `checkTBox()` as coherent, appeared
in the editor, and was silently never evaluated: the self-check reported a rule as
enforced while it was inert. `checkTBox()` now rejects any axiom whose `form` is not
in `IMPLEMENTED_FORMS`, and the editor's form dropdown is derived from that same
list rather than restating it. So the failure mode is gone, but the **cost** is
still there — a new form requires code in `reason()` *and* an entry in the list, and
the axiom is refused until both exist. This is fail-fast, not extensibility.

Two related couplings remain, both deliberate:

- **Three axiom ids are hardcoded** — `axiom("bus-capacity")`,
  `axiom("demand-must-be-served")`, `axiom("passenger-single-state")`. Only
  `range` is evaluated by a generic loop over the TBox. So the *value* of a
  capacity axiom is data; the *connection* between an id and its check is not.
- **Tier 1 is not in the ontology at all.** `reason()` takes `detour` as an
  injected extension function, because insertion cost is optimisation rather than
  logic. A violation the reasoner reports is always about Tier 0.

A real OWL 2 RL engine (`owlrl`) was evaluated and rejected. Two findings, one of
which is decisive and one of which is not.

**It cannot run here at all.** `owlrl` is Python-only with no JS build, and
`canServe` is an ES module shared by three consumers — `fleet.html`, `test_ontology.mjs`
and `eval_rules.mjs`. Moving it server-side would delete the TBox editor, the ABox
panel and the enforce toggle from the browser demo, and push both test suites over
HTTP. That settles it on its own.

**It is also slower, though not fatally so.** Measured against this repo's actual
predicate (`owlrl` 7.6.2 / `rdflib` 7.6.0, in-process, minimal ABox of one bus and
four passengers):

| | per check | vs. now |
|---|---|---|
| this reasoner | 3.4–4.2 µs | — |
| `owlrl`, ABox rebuilt each call | 23–25 ms | ~6,000× |
| `owlrl`, graph reused, closure re-run | 11–12 ms | ~3,000× |
| `owlrl`, closure on an already-settled graph | 10–12 ms | ~2,500× |

The predicate runs 3.00× per demand (measured over 37 demand ticks, 111 calls), so
`owlrl` would add ~70 ms per demand — 2% CPU at the slowest demand rate, 28% at the
fastest. Slow, but not disqualifying on its own. The decisive number is the last row:
`expand()` is forward-chaining and re-runs the whole rule set on every call, so even
with the graph kept alive and nothing new to infer it still costs ~10 ms. About half
the cost is avoidable ABox reconstruction; the other half is structural.

That second half is the part worth noting, because it is what this module is already
built to avoid. The TBox is a parsed module constant rather than re-parsed turtle; the
ABox is a hand-built `Map` rather than an RDF graph; and `reason()` is a single pass
over the axioms rather than a fixpoint. All three exist to keep those two costs at
zero. An OWL-RL adoption would reintroduce both.

Note also that the rebuild figure is a *floor*: it was measured on a minimal ABox, and
inference cost scales with graph size, so the real demo's graph would be larger.

**What the ontology actually buys here**, stated without overclaiming: for a seat
limit, one line of `if` is sufficient and the reasoner is 300 lines of overkill.
What it adds is that the rule is *declared*, so it can be contradicted
(`checkTBox` catches `≥6` against `≤4`), it can name what it caught
(`bus-capacity :: carries 5 passengers, at most 4 allowed`), and it can be
compared against the code it replaces. A hand-written `if` cannot be wrong in any
way you can detect. The tradeoff only starts paying if rules multiply far enough
to interfere with each other — and this demo is not there yet.

#### The same idea in Breakout — and where the fleet invariant does *not* carry over

`static/breakout.html` now carries an Ontology panel too, with the same TBox editor, ABox
view and enforce checkbox. The **TBox is data, but the reasoner is code** and the axiom
**ids are hardcoded** — so is the "Tier 1 arrives as an injected extension" split. What is
new is the constraint: reachability.

`Ball ⊑ ≥1 catchableBy`. A descending ball below the brick field that no `left/stay/right`
reaches is a violation, not a wrong answer. The verdict is computed by forward-simulating
the physics (`rolloutReachable` in `breakout_sim.js`) and injected as an extension, exactly
as fleet injects `detour` — cost is not logic.

**Why not the obvious arithmetic.** "time to paddle × paddle speed ≥ gap" is what you would
write, and it is wrong. Four closed forms were measured against the rollout over 40,000
states; the best agreed **83%** of the time and every residual error ran the same way —
declaring balls uncatchable that the physics catches. That is precisely the failure this
feature exists to fix, because it would exculpate the model on balls it genuinely lost. The
rollout costs **~9 µs typical, ~81 µs worst case** against a decision every 150–500 ms.

**A bug the tests caught, worth recording.** The rollout first ran for a fixed 2.5 s. The
brute-force sweep found **1136 of 21000** balls declared unreachable that the physics
caught — every one a slow ball. A paddle bounce can leave the ball descending at ~50 px/s,
and the ~210 px to the paddle then takes over 4 seconds. The horizon is now derived from
`ball.vy` rather than fixed. That number is the reason the sweep forward-simulates the *real*
`step()` rather than re-deriving the same algebra twice.

```
21000 states compared against the real step(), 0 false positives, 22 false negatives
```

**The invariant had to be restated, because the fleet one is false here.** Fleet's is
*"turning it on should not change any decision — if it does, that is a bug."* That holds
there by construction: the ontology filters an option set a caller already assembled. In
breakout it **cannot** hold. Once the ball is provably gone, `left/stay/right` has no correct
answer, so the option set *is* the question — gating it necessarily changes what is asked.
Claiming otherwise would be false. What replaces it:

> **Clause 1 — equivalence (inherited).** The reasoner's feasibility answer and the
> hand-written arithmetic agree wherever both apply.
>
> **Clause 2 — soundness of the gate (new).** Enforcement must never suppress a question
> whose answer could have changed the outcome. A false "unreachable" is the costly error; a
> false "catchable" merely wastes one ask. `CATCH_TOL_PX` is therefore non-negative by
> construction and asserted as such, so it cannot be "tidied" negative later.

Clause 2 is the one the 21,000-state sweep enforces, and it is one-directional on purpose.
Enforce **defaults to off**, so the demo ships byte-identical to before.

**Above the bricks the ontology is silent.** A descending ball above `y = 121` can be
deflected before it reaches the paddle, so no sound verdict exists. The decidable band is
`ball.y ∈ (121, 331)`; outside it the panel renders "no verdict", never a violation. A
verdict is about the geometry *at that tick*, but below the brick field no new information
can change the trajectory — a ball declared unreachable stays unreachable.

**What it does not do.** It does not improve the score, and the HUD says so with two
counters that move independently of the toggle:

```
provably lost 0 · lost after a catchable ask 1
```

The second is the model's real failure rate. Splitting the first out is the honest
presentation of "the demo stops blaming the model for physics-impossible catches" — it is a
*different quantity*, not a better one.

**`node eval_breakout.mjs` is the on/off A/B.** `ontology-off` is the demo exactly as it was;
`ontology-on` is the same game with the gate closed. It reports what actually changes, and
deliberately does not report score as the headline:

```
600 decision ticks sampled (seed 11, 0.25s cadence)
  ontology-off: 600 asked · ontology-on: 560 asked, 40 suppressed (6.7%)
  suppressed but actually catchable : 0   <- must be 0; this is clause 2
  asked but hopeless                : 0
  closed form disagrees with the rollout on 12.8% of states
```

The suppression rate is genuinely low, and that is a fact about the game rather than a
defect in the gate: the paddle crosses at 200 px/s while the ball falls the ~210 px in
1.1–3 s, so it recovers from most bad positions before the ball escapes. Hopeless balls need
the paddle already committed the wrong way. Measured across seeds 3/7/11 the rate runs
1.7%–6.7%.

Two measurement traps this harness hit and now guards against in its own comments, because
both produced a clean sheet of zeros that meant nothing: generating scenarios with a
**perfect** paddle (every ball trivially catchable, nothing to suppress), and an aim bias too
small to matter (unreachable rate 0.0% below ~60 px of bias).

**The observation changed too, and it was wrong before.** `buildState()` compared the
paddle to the ball's *current* x. The ball drifts and reflects on the way down, so over
80,000 decidable states that named the wrong side **12.4%** of the time. It now reports the
landing point, which takes that to **0** by construction, and both the prose and the reasoner
read the same `measure()` so the number the model sees and the number reasoned over cannot
diverge. Worth being straight about what that did and did not buy: it did **not** change the
score — 25 seeded games with a perfect controller lost 1025 balls aiming at current x and
1025 aiming at the landing point. A paddle driving toward the landing point sweeps through
the current x on the way. The gain is that the model is no longer *told* the wrong thing,
which starts to matter the moment a controller commits to one side for a whole descent.

**Two modules, one shared core.** `breakout_ontology.js` is separate from `ontology.js`
rather than a generalisation of it: the fleet reasoner has three hardcoded axiom ids the
README names as a deliberate coupling, and parameterising them is a rewrite of the thing
`test_ontology.mjs` protects. What was genuinely shared — `checkTBox`, `describeAxiom`,
`addAxiom`, `removeAxiom`, `resetTBox`, and the editor itself — moved to
`static/ontology_core.js` as `mountTBoxEditor({ root, adapter })`. `ontology.js` re-exports
the moved names, so `test_ontology.mjs` and `eval_rules.mjs` are untouched and still pass.

**`IMPLEMENTED_FORMS` deliberately did not move.** Both domains happen to implement the same
four forms, so a shared list is the obvious tidy-up and it would re-open the hole commit
`a5149d0` closed: a form fleet's `reason()` handles would pass breakout's `checkTBox` and then
be enforced by nothing. Each domain exports its own, and `checkTBox` takes it as an argument.

**Browser verification.** The panel was checked in headless Chromium, not just in node: panel
mounts with 5 axioms and a satisfiable TBox, the toggle flips without error, the ABox
populates with a live verdict, duplicate axioms are refused *by name*
("ball-must-be-catchable already asserts that"), and add/delete/reset all repaint. Two bugs
were caught only there — a stray `</content>` left in the spliced script, and a stale axiom
list after an edit — neither of which any node test could see.

#### Fine-tuning

Not done, and the measurement says it should stay that way. `laya` ships no
training code, though `common.py` does expose `build_model`, `collate_items`,
`proper_reward` and `td_lambda_targets`, and the config is named
`rl_agent_config.json` with a `training` block — so a finetune means writing a
loop against `DecisionModel` ourselves, with the simulator as both reward oracle
and expert. Two costs to name up front: 421M parameters on MPS iterates slowly,
and narrow fine-tuning of a third-party checkpoint tends to wreck the
calibration that `confidence` currently provides — which would also break the
gate above.

The gate was: finetune only if the rules arm left regret materially above the
floor on scenarios where the rule is clearly correct. It does not — regret falls
38.33 → 6.43 blocks on laya and 35.18 → 1.54 on Jev, without touching a weight.
The remaining gap is not something the model fails to learn; it is the ~5% of
distance the routing objective can still give back, and a finetune would be
paying 421M parameters of training to chase it. The cross-model result is what
settles this: if the technique were a property of the checkpoint rather than of
System-1 models in general, Jev would not have improved 23× without training.

## IR 자료

> 이 섹션은 위 기술 작업에서 **실제로 측정된 값만** 재구성한 요약입니다.
> 시장 규모, 고객, 매출, 도입 사례는 이 저장소에 근거가 없어 기재하지 않았습니다.
> 본 저장소는 시뮬레이션 데모이며, 프로덕션 검증 전입니다.

### 한 줄 요약

**업무 도메인 규칙을 AI 의사결정 모델에 재학습 없이 반영할 수 있고, 그것이 실제로
판단을 바꾸며 우회로 비용을 줄였음을 측정했습니다.** 같은 규칙·같은 비용 함수로,
운영 방식이 그 규칙을 수용하는지에 따라 가치가 사라지거나 나타납니다.

### 무엇을 만들었는가

좌석·픽업 순서·다중 승객 등 **업무에 고유한 제약과 지식**을 가진 의사결정 문제에
범용 사전학습 체크포인트(421M 파라미터)를 그대로 연결했습니다. 모델은 1회
비-autoregressive 추론으로 판단하며, 추론 속도는 판단당 약 150ms입니다.

핵심은 규칙을 "가르쳐서" 넣지 않는다는 점입니다. 규칙에서 파생되는 **수치**(이
버스에 추가되는 블록 수, 남은 좌석)를 계산해 판단 옵션에 붙이면, 모델이 그 수치를
따릅니다. 모델을 고쳐-tune할 필요가 없습니다.

### 측정된 결과

**규칙이 모델에 도달하는가** (200개 시나리오 중 195개가 실제 충돌 케이스)

| 지표 | 규칙 없음 | 규칙 반영 |
|---|---|---|
| 최소우회 regret | 38.33 블록 | **6.43 블록** |
| 최소우회 버스 선택률 | 39.0% | **79.0%** |

**모델이 바뀌어도 같은 결과인가** — laya(CJet)와 Jev(TypeSafe) 동일 인터페이스 대조

| | 규칙 없음 | 규칙 반영 | 판단당 비용 | 지연 |
|---|---|---|---|---|
| **laya (CJet)** | 38.33 블록 | 6.43 블록 | $0 (로컬) | 136 ms |
| **Jev** | 35.18 블록 | **1.54 블록** | $0.000032 | 269 ms |

두 모델 모두 **규칙 없이는 쓸모가 없습니다** — laya의 no-rules(38.33)는 단순 최근접
휴리스틱(34.60)보다 오히려 나쁩니다. 규칙을 넣으면 laya는 6배, Jev는 **23배**
개선되고 1.54 블록까지 내려갑니다. 즉 이 기법은 laya의 `[MASK]` 구조 특유한
것이 아니라 **System-1 모델 일반의 성질**입니다. IR 주장에서 가장 무게가 실리는
부분이 이겁니다.

**하드 제약(좌석)이 실제로 지켜지는가**

| | 규칙 없음 | 규칙 반영 | 규칙 반영(필터 미적용) | 단순 최근접 |
|---|---|---|---|---|
| laya 좌석 초과 배정 | 51건 | **0건** | 4건 | 48건 |
| Jev 좌석 초과 배정 | 47건 | **0건** | **0건** | 48건 |

좌석이 없는 버스를 **선택지에서 삭제**했기 때문에 위반이 0입니다. 규칙을
"설명"한 것이 아니라 "제거"해서, 모델이 보지 못한 선택지를 고를 수 없게 한
결과입니다.

다만 **모델에 따라 필터의 가치가 다릅니다.** 필터 없이 criteria 문장만으로
알려주면 laya는 4건 위반하지만 Jev는 **0건**입니다 — Jev는 "full, no room"을
읽고 스스로 거절합니다. Tier 0는 읽는 능력이 약한 모델을 위해 규칙을
"불가능하게" 만드는 장치이며, 잘 읽는 모델에게는 필터가 아무것도 보장하지
않습니다(사실 Jev에서는 필터가 오히려 regret을 0.98 → 1.54로 조금 악화시킵니다).

> **이전 수치는 정정되었습니다.** 위 "규칙 없음" 수치는 당초 `rules:` 키만 지우고
> 다시 측정한 값이었고, 그 상태에는 우리 규칙이 다른 두 경로로 남아 있었습니다
> (`recommendation` 문장에 `detourCost()`로 만든 문장이 포함됨). 깨끗하게 다시 만든
> 기준선은 약 2배 나쁩니다(19.25 → 38.33). 규칙 arm는 그대로이므로 개선 폭이
> 커져 보이지만, **새 결과가 아니라 이전 기준선의 오류가 수정된 것**입니다.

**운영 방식이 바뀔 때** (시뮬레이션, 정책당 200초, 3시드)

```
                              rules              greedy
multi-pickup on        189.7 / 36.0        189.3 / 38.0
multi-pickup off       133.0 / 40.2        127.3 / 39.9
```

- 다중 탑승이 가능한 운행일 때: 처리량 **+43%**, 주행거리 **−5%**
- 직렬(1명씩)일 때: 규칙의 이득이 **0으로 사라짐**

### 해석: 이 실험에서 가장 값진 발견

**1. AI가 못 한 것이 아니라, 운영 방식이 규칙을 받쳐주지 않았던 것입니다.**

규칙 주입 직후 측정한 결과는 이렇았습니다 — 규칙은 모델에 잘 들어갔고
우회로 regret이 크게 줄었는데, 실제 처리량은 오히려 **적었습니다**(76.0 vs 78.3).
원인은 AI가 아니라 **버스가 구조상 한 명씩만 태울 수 있었던 것**이었습니다. 곧,
"도메인 규칙을 넣어라"가 아니라 **"그 규칙이 성립할 수 있는 운영 모델부터 갖춰라"**
는 순서를 말합니다.

**2. 기법은 특정 모델의 구조가 아니라 System-1 모델 일반의 성질입니다.**

동일 인터페이스를 가진 다른 System-1 모델(Jev)로 반사실 A/B를 돌렸을 때,
Jev는 규칙 없이는 laya와 마찬가지로 쓸모없었고(35.18 vs laya 38.33, 둘 다 단순
최근접 34.60 수준), 규칙 주입 시 **23배** 개선되며 1.54 블록까지 내려갔습니다.
laya의 `[MASK]` 스코어러 구조에 특화된 트릭이었다면 Jev에서는 작동하지 않아야
했습니다. 즉 특정 벤더 선택이 아니라 **방법론**을 사는 것이고, 모델 교체 시
기존 주입 코드가 그대로 유지됩니다.

### 조건과 비용

- 모델 크기·지연: 421M 파라미터, 판단당 ~150ms (기준 입력 대비 +40ms).
  비교 대상 Jev는 269ms, 판단당 $0.000032 (판단 200회 ≈ 0.6센트)
- 학습 비용 0: 파인튜닝 미수행. 규칙 반영만으로 판단이 바뀌며, 파인튜닝 시
  calibration이 손상되어 신뢰도 게이트가 무력화될 위험이 있음
- 재현성: 시뮬레이션이 시드 고정·결정론적이며, 같은 시나리오에서 동일 재현
- **모델 교체 시 신뢰도 게이트는 재파생해야 합니다.** laya의 신뢰도는
  p50 0.033, Jev는 같은 질문에 p50 0.880입니다. laya에 맞춰 잡은 임계값
  (0.002)을 그대로 쓰면 Jev에서는 **무효**가 되어 27건 중 0건만 게이트됩니다.
  임계값은 모델별 상수이고, 모델을 바꾸면 재측정 없이 그대로 두면 게이트가
  조용히 꺼집니다.

### 한계 — 주장 범위를 넘어서는 부분

- **시뮬레이션 결과이며 운영 데이터가 아닙니다.** 실운영에서의 효과는 별도
  검증 대상입니다.
- 규칙 주입의 효과(−5% 주행거리)는 처리량 개선이 아니라 **효율 개선**입니다.
  다중 탑승의 +43%는 AI 기여가 아니라 **운영 방식 변경의 효과**입니다. 두 값을
  혼동하지 않는 것이 중요합니다.
- 강제 제약(좌석)이 있는 문제는 코드로 처리했습니다. 프롬프트에 "이러면 안 된다"고
  쓰는 것으로는 보장되지 않기 때문입니다. 다만 **모델이criteria를 제대로 읽으면
  필터 없이도 0건**이었습니다(Jev). 즉 코드로 제거하는 것은 "안전한 기본값"이지
  유일한 방법은 아닙니다. 자연어로 표현되는 **선호**(soft preference)만 모델에
  맡기는 경계가 명확하지 않은 영역은 남아 있습니다.
- **Jev 비교는 단일 스냅샷(1.13-20260917) 1개입니다.** 버전 간 차이가 이번
  격차에 기여했는지 알 수 없습니다. 시나리오·판정 코드는 동일하나 200회 1회
  실행이므로 seed 재현성은 있으나 통계적 표본은 아닙니다.
- 규모 검증 없음: 버스 3대, 격자 50×50, 421M 모델 기준. 파라미터 스케일업과
  동시 에이전트 수 증가에 대한 검증은 없습니다.

### 다음 검증 과제

1. 실제 운영 데이터로 동일 프로토콜(반사실 A/B + closed loop) 재실행
2. 파인튜닝 게이트: 규칙 반영 후 regret이 기준선을 크게 넘을 때만 재고려 (현재
   38.33 → 6.43으로 통과하지 않음 → **파인튜닝하지 않음**)
3. 언어 분기: 현재 영어 체크포인트 기준이며, 다국어 입력에서의 규칙 유효성 미검증
4. 모델 3종 이상으로 확대: 현 결론은 2개 모델(421M 로컬 / 원격 API)이며,
   Jev 스냅샷 변경 후 재측정

## Sensor anomaly detection

`sensor_anomaly_detection.py` is a standalone, stdlib-only script (no laya
dependency, nothing new to install) that demonstrates statistical anomaly
detection on synthetic sensor data:

```sh
uv run python sensor_anomaly_detection.py
```

### What it does

1. Generates a deterministic temperature + humidity feed (500 samples each,
   5-minute interval, periodic signal + Gaussian noise, `SEED=42`) and injects
   known faults: spike, out-of-range, flatline (stuck sensor), and transient
   drift (ramp up + ramp down).
2. Runs six detectors and merges their flags per reading:
   - `range` — raw value outside the sensor's physical limits
   - `global_z` / `rolling_z` — z-score vs whole series / trailing window, on
     detrended residuals (reading minus the expected periodic signal)
   - `rate` — jump between consecutive readings over the rate limit
   - `flatline` — N consecutive identical readings
   - `level_shift` — median of a trailing window vs the preceding window
     (catches drift; medians keep a single spike from masquerading as a shift)
3. Prints flagged readings (severity = number of agreeing detectors) and
   precision/recall against the injected ground truth.

### Results (SEED=42)

| Sensor       | Precision | Recall | Notes                                            |
| ------------ | --------- | ------ | ------------------------------------------------ |
| temperature  | 80%       | 100%   | spike and out-of-range fully caught              |
| humidity     | 89%       | 70%    | flatline 100%; drift caught ~2/3 through its ramp |

Slow ramps are the hardest shape for windowed statistics: a trailing-window
z-score mathematically caps at √12/2 ≈ 1.73σ on any linear ramp (the slope
cancels), which is why drift needs the dedicated level-shift detector — and
even then it only fires once ~14 of its last 20 samples are inside the ramp.
The residual false positives are statistical noise tails plus documented
"settling" flags while window detectors clear an anomaly's aftermath.

### Raising precision

Three levers, most effective first:

1. **Require detector agreement** — keep only readings flagged by ≥2 detectors
   (`if len(methods) >= MIN_AGREEING_DETECTORS`). On this data all false
   positives are single-detector flags.
2. **Raise thresholds** (`Z_THRESHOLD`, `LEVEL_SHIFT_THRESHOLD`) — fewer false
   alarms, more missed events.
3. **Confirm-before-alert** — require 2–3 consecutive flags before alerting
   (the usual pattern for live streams).

### Note on laya / Jev

This script deliberately does **not** use laya: its checkpoints are trained on
support-ticket text, so numeric sensor data is out-of-domain. The sensible
architecture is statistics for detection and a System One model (laya, or
TypeSafe AI's frontier [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev))
as a downstream judgment layer — feeding detector evidence as state and asking
typed questions (fault type = choice, urgency = score, needs-maintenance = yes-no).

## Startup performance

Stock `laya` + `Router(preload=True)` spent ~100s before the first prediction on
an Apple-silicon (MPS) machine. Two independent causes, both worked around in
`main.py` without touching the installed package (so the patch survives
`uv sync` / reinstalls):

| Phase                          | Before  | After   |
| ------------------------------ | ------- | ------- |
| Hub file checks (per run)      | ~1s     | 0s      |
| Model build (per checkpoint)   | ~29s    | ~0.1s   |
| Weight load + MPS transfer     | ~2s     | ~2s     |
| **Total (`uv run python main.py`)** | **~100s** | **~7s** |

Predictions are bit-identical before and after the patch (verified on both the
English and Hindi routing paths).

### Cause 1: per-run hub checks

Every run, `Agent.__init__` calls `snapshot_download()`, which performs a
network round-trip to Hugging Face to validate each cached file — even when
everything is already in the local cache. The `Fetching 5 files` progress bars
appear on every run while transferring `0.00B`.

**Fix:** set `HF_HUB_OFFLINE=1` before importing `laya` (`main.py:5`). Skipped
entirely when checkpoints are cached; unset it (or don't set it) when a newer
checkpoint must be pulled.

### Cause 2: wasted random initialization

`laya.common.build_model()` constructs the encoder with
`AutoModel.from_config()`, which *randomly initializes all ~400M parameters* —
about 29s per checkpoint. `Agent.__init__` then immediately calls
`load_state_dict(strict=True)` and overwrites every one of those tensors with
the checkpoint weights. That's ~90s of pure waste across the three checkpoints.

**Fix:** `main.py` replaces `laya.agent.build_model` with `_fast_build_model()`
(`main.py:14-29`), which wraps the same construction in transformers'
`no_init_weights()` context manager. Parameter tensors are still allocated (so
shapes and the strict state load are unchanged), but the expensive random fills
are skipped: ~29s → ~0.1s per checkpoint.

#### Why not a meta-device build?

A `torch.device("meta")` build is even faster, but `to_empty()` leaves
**non-persistent buffers** uninitialized — ModernBERT registers its RoPE
`inv_freq` buffers with `persistent=False`, so they are not part of the
checkpoint `state_dict` and would contain garbage. This silently corrupts
position embeddings and changes predictions (the English routing flipped from
`billing` to `sales` when tested). `no_init_weights()` skips only the parameter
initializers, so those buffers are computed normally.

#### What the monkeypatch does and doesn't change

- Replaces only the encoder/`DecisionModel` constructor. Everything else in the
  load path (checkpoint download, `_verify_compatibility()` architecture checks,
  strict `load_state_dict`, temperature clamping, device placement) is untouched.
- `no_init_weights()` is a private-ish API (`transformers.initialization`); if a
  future transformers release moves it, the patch fails loudly at import time
  rather than producing wrong results.
- The patch is applied before `from laya import Router` so every checkpoint
  `Router` loads uses it.

### Remaining startup cost

The residual ~7s is mostly unavoidable: importing `torch`/`transformers`
(~1s) and moving ~1.16B parameters onto the MPS device (~2s per checkpoint
round of allocation + transfer). If startup ever matters more than first-token
latency, run the router as a long-lived server process and preload once.
