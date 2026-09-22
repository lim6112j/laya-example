import os

# The laya checkpoints are already in the local HF cache (~/.cache/huggingface), so skip
# the per-run hub file checks (the "Fetching 5 files" bars + network HEAD requests).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import transformers
import transformers.initialization
import laya.agent as _laya_agent
from laya.common import DecisionModel


def _fast_build_model(cfg, encoder_dir=None):
    # laya's build_model() constructs the encoder with AutoModel.from_config(), which
    # randomly initialises ~400M params (~29s per checkpoint) that load_state_dict()
    # then immediately overwrites with the checkpoint weights. Wrapping the build in
    # no_init_weights() skips that wasted init (~29s -> ~0.1s per checkpoint). Unlike a
    # meta-device build it also initialises non-persistent buffers (e.g. ModernBERT's
    # RoPE inv_freq), which the checkpoint state_dict does not contain.
    src = encoder_dir if encoder_dir and os.path.exists(encoder_dir) else cfg["encoder"]
    ecfg = transformers.AutoConfig.from_pretrained(src)
    with transformers.initialization.no_init_weights():
        enc = transformers.AutoModel.from_config(ecfg, attn_implementation="sdpa")
        model = DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
    return model


_laya_agent.build_model = _fast_build_model

import laya
from laya import Router

# Preload checkpoints into memory for instant sub-35ms routing
router = Router(preload=True)

# 1. State in any language or schema
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

# 2. Define your typed questions
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?"
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
    }
}

# 3. English state -> automatically routed to laya (ModernBERT-large, 39.5 ms)
res_en = router.predict(state, questions)
# -> billing (confidence: 0.94)
print("Department :", res_en["answers"]["department"]["choice"])
print("Routing    :", res_en["routing"]["model"])                 # -> english

# 4. Hindi state -> automatically routed to laya-multilingual (mmBERT-base, 32.8 ms)
res_hi = router.predict(
    {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions)
# -> billing (confidence: 0.86)
print("Department :", res_hi["answers"]["department"]["choice"])
print("Routing    :", res_hi["routing"]["model"]
      )                 # -> multilingual

# 5. Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")
