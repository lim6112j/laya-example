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
