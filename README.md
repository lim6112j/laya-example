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
- `test_fleet_sim.mjs` — simulation invariants, no model load:
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
`assign: choice(bus_a/bus_b/bus_c)`. Buses drive Manhattan-style to the
pickup, carry the passenger to the destination and loop for queued jobs. The
screen shows the grid, per-bus status chips, "Laya's view" (the dispatch
state sent) and "Laya's decisions" (assignment bars, confidence, latency,
history). Toggle laya dispatch off to compare against a greedy nearest-bus
heuristic; sliders tune demand frequency and bus speed.

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

The arithmetic lives in JS, not the model. `insertionCost()` is the multi-pickup
rule made computable — the classic VRP insertion heuristic: fewest extra blocks
for a bus to also serve this pickup, by slotting it between waypoints on the
route it is already driving. A pickup already on that route therefore costs ~0,
where `costOf()` (the serial nearest-bus measure) would charge a full deadhead
for the same bus. Each option's criteria reports the result:

```
bus_b: "Bus B: pickup on its route, +0 blocks; 4 seats free"
```

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
the greedy policy and every decision tick is snapshotted, then both arms answer
the *same* snapshots differing only in question text. That isolates the decision
instead of letting one arm's assignments steer the other arm's future inputs.

Over 200 scenarios, 197 of which had a real nearest-vs-least-detour conflict:

```
                              no-rules           rules          greedy
mean detour regret            13.38 blk        1.29 blk       11.66 blk
picks least-detour               31.0%           93.9%           30.5%
picks nearest                    76.5%           37.5%          100.0%
mean deadhead regret            5.54 blk       25.13 blk        0.00 blk
head tokens (est)                   48             119               —
ms                                110             149               —
```

So the injection works — mean regret against the least-detour bus drops ~10× and
the model follows the criteria. It is not free: deadhead regret rises, the head
grows 48 → 119 tokens, and latency goes 110 → 149 ms.

**And it does not yet pay off end to end.** The closed loop in the same run
delivers 76.0 (rules) vs 78.3 (greedy) out of 200 demands. That is not noise to
be tuned away — it is the honest consequence of a gap: the simulator is still
**serial** (a bus serves its queue one job at a time) while `insertionCost`
prices a bus as if it could multi-pickup. Optimising for detour optimises for a
capability the world does not have yet. Multi-pickup physics (seats, a
manifest, en-route insertion) is the next piece of work, and the closed-loop
number is the canary that will show whether it landed.

#### Confidence gating

`fleet.html` falls back to the deterministic cost model when laya's confidence
is below `CONF_FLOOR`, and shows a **decided / deferred** counter in the HUD —
because a gate silently caps how often the model actually decides, and a gate
tuned until laya "always agrees" has removed laya from the loop.

The floor is set from the measured distribution, not guessed. Confidence on this
workload is heavily compressed near zero (p10 0.011, p50 0.057, p95 0.194),
because the checkpoint ships `temperature_by_options["choice:3-5"] = 1.76`,
which softens the logits, and a near-uniform 3-way distribution has low entropy
confidence. An earlier guess of 0.45 deferred **100%** of decisions — the rules
were in the prompt and laya never got to use them.

`act_probability` is reported but deliberately *not* gated on: measured at
1.000 across every percentile here, so a floor on it would be dead code.

#### Fine-tuning

Not done, and gated on the measurement above. `laya` ships no training code,
though `common.py` does expose `build_model`, `collate_items`, `proper_reward`
and `td_lambda_targets`, and the config is named `rl_agent_config.json` with a
`training` block — so a finetune means writing a loop against `DecisionModel`
ourselves, with the simulator as both reward oracle and expert. Two costs to
name up front: 421M parameters on MPS iterates slowly, and narrow fine-tuning
of a third-party checkpoint tends to wreck the calibration that `confidence`
currently provides — which would also break the gate above. If the rules arm
closes the gap on the conflicting scenarios, do not finetune.

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
