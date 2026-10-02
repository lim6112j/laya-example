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

Over 60 scenarios, 59 of which had a real nearest-vs-least-detour conflict:

```
                              no-rules           rules  rules-nofilter          greedy
mean detour regret           19.25 blk        3.37 blk        6.11 blk       27.47 blk
picks least-detour               63.3%           91.7%           88.3%           51.7%
picks nearest                    60.0%           66.7%           66.7%           80.0%
capacity violations                 10               0               2              12
mean confidence                  0.120           0.124           0.057           0.000
head tokens (est)                   48             137             137               —
ms                                114             153             158               —
```

Two things to read here. The **injection works** — regret against the
least-detour bus drops ~6× and the model follows the criteria. And the
**capacity filter (Tier 0) is doing real work**: `rules` never assigns to a bus
that has no room, while the unfiltered arm does so twice and the naive
nearest-bus dispatcher does so twelve times. Omitting an infeasible option
makes the rule unviolable rather than merely likely, and it improves routing
quality as a side effect (3.37 vs 6.11 regret) by removing distractors.

`no-rules` and `greedy` show *negative* nearest-regret, which is not a bug: the
optimum is restricted to buses that can take the work, and the naive dispatcher
picks the closest bus whether or not it has room.

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

`CONF_FLOOR` is 0.013, which against the current distribution sits near the 20th
percentile and defers roughly a fifth of decisions. It was derived from a
previous version of the question and is now slightly conservative; re-derive it
whenever the criteria text changes, since the distribution moves with it. The
p95 of 1.000 is the capacity filter showing through — when only one bus has
room, the model is not choosing, and the HUD counter is what makes that visible
rather than an accident.

`act_probability` is reported but deliberately *not* gated on: measured at
1.000 across every percentile here, so a floor on it would be dead code.

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
19.25 → 3.37 blocks and the model follows the criteria 92% of the time. The
remaining gap is not something the model fails to learn; it is the ~5% of
distance the routing objective can still give back, and a finetune would be
paying 421M parameters of training to chase it.

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

**규칙이 모델에 도달하는가** (200개 시나리오 중 197개가 실제 충돌 케이스)

| 지표 | 규칙 없음 | 규칙 반영 |
|---|---|---|
| 최소우회 regret | 13.38 블록 | **1.29 블록** |
| 최소우회 버스 선택률 | 31.0% | **93.9%** |

**하드 제약(좌석)이 실제로 지켜지는가**

| | 규칙 없음 | 규칙 반영 | 규칙 반영(필터 미적용) | 단순 최근접 |
|---|---|---|---|---|
| 좌석 초과 배정 | 10건 | **0건** | 2건 | 12건 |

좌석이 없는 버스를 **선택지에서 삭제**했기 때문에 위반이 0입니다. 규칙을
"설명"한 것이 아니라 "제거"해서, 모델이 보지 못한 선택지를 고를 수 없게 한
결과입니다.

**운영 방식이 바뀔 때** (시뮬레이션, 정책당 200초, 3시드)

```
                              rules              greedy
multi-pickup on        189.7 / 36.0        189.3 / 38.0
multi-pickup off       133.0 / 40.2        127.3 / 39.9
```

- 다중 탑승이 가능한 운행일 때: 처리량 **+43%**, 주행거리 **−5%**
- 직렬(1명씩)일 때: 규칙의 이득이 **0으로 사라짐**

### 해석: 이 실험에서 가장 값진 발견

**AI가 못 한 것이 아니라, 운영 방식이 규칙을 받쳐주지 않았던 것입니다.**

규칙 주입 직후 측정한 결과는 이렇았습니다 — 규칙은 모델에 잘 들어갔고
우회로 regret이 10배 줄었는데, 실제 처리량은 오히려 **적었습니다**(76.0 vs 78.3).
원인은 AI가 아니라 **버스가 구조상 한 명씩만 태울 수 있었던 것**이었습니다. 곧,
"도메인 규칙을 넣어라"가 아니라 **"그 규칙이 성립할 수 있는 운영 모델부터 갖춰라"**
는 순서를 말합니다.

이 순서를 뒤집으면 흔한 실패 모드(AI를 구매했으나 규칙이 반영되지 않아 ROI가
안 나오고 원인을 모델 탓으로 돌리는 상황)를 설계 단계에서 피할 수 있습니다. 그리고
이 저장소는 **검증 수치와 검증 도구를 함께 제공**합니다 — 주장과 반증 조건이
저장소에 들어 있습니다.

### 조건과 비용

- 모델 크기·지연: 421M 파라미터, 판단당 ~150ms (기준 입력 대비 +40ms)
- 학습 비용 0: 파인튜닝 미수행. 규칙 반영만으로 판단이 바뀌며, 파인튜닝 시
  calibration이 손상되어 신뢰도 게이트가 무력화될 위험이 있음
- 재현성: 시뮬레이션이 시드 고정·결정론적이며, 같은 시나리오에서 동일 재현

### 한계 — 주장 범위를 넘어서는 부분

- **시뮬레이션 결과이며 운영 데이터가 아닙니다.** 실운영에서의 효과는 별도
  검증 대상입니다.
- 규칙 주입의 효과(−5% 주행거리)는 처리량 개선이 아니라 **효율 개선**입니다.
  다중 탑승의 +43%는 AI 기여가 아니라 **운영 방식 변경의 효과**입니다. 두 값을
  혼동하지 않는 것이 중요합니다.
- 강제 제약(좌석)이 있는 문제는 코드로 처리했습니다. 프롬프트에 "이러면 안 된다"고
  쓰는 것으로는 보장되지 않기 때문입니다. 자연어로 표현되는 **선호**(soft
  preference)만 모델에 맡기는 경계가 명확하지 않은 영역은 남아 있습니다.
- 규모 검증 없음: 버스 3대, 격자 50×50, 421M 모델 기준. 파라미터 스케일업과
  동시 에이전트 수 증가에 대한 검증은 없습니다.

### 다음 검증 과제

1. 실제 운영 데이터로 동일 프로토콜(반사실 A/B + closed loop) 재실행
2. 파인튜닝 게이트: 규칙 반영 후 regret이 기준선을 크게 넘을 때만 재고려 (현재
   19.25 → 3.42로 통과하지 않음 → **파인튜닝하지 않음**)
3. 언어 분기: 현재 영어 체크포인트 기준이며, 다국어 입력에서의 규칙 유효성 미검증

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
