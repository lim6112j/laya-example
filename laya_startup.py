"""Shared laya router startup: offline hub + fast model build + preload.

Extracted from main.py so both the CLI demo and the web server share one
implementation. See README "Startup performance" for the two workarounds.
"""

import os
import transformers
import transformers.initialization
import laya
import laya.common
import laya.agent as _laya_agent

# The laya checkpoints are already in the local HF cache (~/.cache/huggingface), so skip
# the per-run hub file checks (the "Fetching 5 files" bars + network HEAD requests).
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _fast_build_model(cfg, encoder_dir=None):
    # laya's build_model() constructs the encoder with AutoModel.from_config(), which
    # randomly initialises ~400M params (~29s per checkpoint) that load_state_dict()
    # then immediately overwrites with the checkpoint weights. Wrapping the build in
    # no_init_weights() skips that wasted init (~29s -> ~0.1s per checkpoint). Unlike a
    # meta-device build it also initialises non-persistent buffers (e.g. ModernBERT's
    # RoPE inv_freq), which the checkpoint state_dict does not contain.
    src = encoder_dir if encoder_dir and os.path.exists(
        encoder_dir) else cfg["encoder"]
    ecfg = transformers.AutoConfig.from_pretrained(src)
    with transformers.initialization.no_init_weights():
        enc = transformers.AutoModel.from_config(
            ecfg, attn_implementation="sdpa")
        model = laya.common.DecisionModel(enc, cfg.get("head_layers", 2),
                                          len(cfg.get("act_costs", {})) + 1)
    return model


def build_router():
    """Apply the fast-build monkeypatch and return a preloaded Router."""
    _laya_agent.build_model = _fast_build_model
    return laya.Router(preload=True)