"""Shared laya router startup: offline hub + fast model build + preload.

Extracted from main.py so both the CLI demo and the web server share one
implementation. See README "Startup performance" for the two workarounds.
"""

import os
import torch
import transformers
import laya
import laya.common
import laya.agent as _laya_agent

try:
    # transformers 5.x exposes no_init_weights from its own `initialization` module.
    from transformers.initialization import no_init_weights
except ImportError:
    # transformers 4.x (the only line installable on Intel Macs, where torch stops at
    # 2.2.2) keeps it in modeling_utils instead.
    from transformers.modeling_utils import no_init_weights

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
    with no_init_weights():
        enc = transformers.AutoModel.from_config(
            ecfg, attn_implementation="sdpa")
        model = laya.common.DecisionModel(enc, cfg.get("head_layers", 2),
                                          len(cfg.get("act_costs", {})) + 1)
    return model


def _default_device():
    """Pick a device laya can actually run autocast on.

    laya picks MPS whenever torch reports it available, but MPS autocast only landed
    in torch 2.4 — and 2.2.2 is the newest torch with an Intel Mac wheel. On such a
    setup, torch.backends.mps.is_available() is True while
    torch.autocast(device_type="mps") raises, so predictions die with
    "User specified an unsupported autocast device_type 'mps'". Fall back to CPU there.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            with torch.autocast(device_type="mps", dtype=torch.float16):
                pass
            return "mps"
        except RuntimeError:
            return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_router():
    """Apply the fast-build monkeypatch and return a preloaded Router."""
    _laya_agent.build_model = _fast_build_model
    return laya.Router(preload=True, device=_default_device())
