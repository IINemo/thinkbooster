#!/bin/bash
# =============================================================================
# Install the `step_reasoning` UncertaintyHead into the active `luh` package.
# =============================================================================
# The new K2-Think-V2 UHead checkpoint
#   rediska0123/uhead_hs_K2-Think-V2_mixed_code10K_steps_vllm_10epochs
# declares `head_type: step_reasoning`, a head class that exists ONLY in the
# author's research repo (not in any IINemo/llm-uncertainty-head branch):
#   https://github.com/cant-access-rediska0123/uncertainty4reasoning
#     -> luh/heads/uncertainty_head_steps_reasoning.py
#
# That head is a self-contained `UncertaintyHeadBase` subclass and its inference
# entry (`_compute_tensors(llm_inputs, X, X_attn_mask)`, reading
# `llm_inputs["claims"]`) is identical to the `claim` head that thinkbooster's
# vLLM UHead path already runs. So we drop the file into the installed `luh` and
# register `step_reasoning` in AutoUncertaintyHead.MODEL_MAPPING. Idempotent.
#
# Run (inside the uenv + venv):
#   uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc \
#     "source \$SCRATCH/venvs/tb/bin/activate; bash scripts/slurm/install_step_reasoning_head.sh"
# =============================================================================
set -euo pipefail

SRC_URL="https://raw.githubusercontent.com/cant-access-rediska0123/uncertainty4reasoning/main/luh/heads/uncertainty_head_steps_reasoning.py"

LUH=$(python -c 'import luh, os; print(os.path.dirname(luh.__file__))')
echo "luh package: $LUH"

echo "==> fetching step_reasoning head class"
curl -fsSL "$SRC_URL" -o "$LUH/heads/uncertainty_head_steps_reasoning.py"
echo "    wrote $(wc -l < "$LUH/heads/uncertainty_head_steps_reasoning.py") lines"

echo "==> registering step_reasoning in AutoUncertaintyHead.MODEL_MAPPING"
python - "$LUH/auto_uncertainty_head.py" <<'PY'
import sys
path = sys.argv[1]
s = open(path).read()
imp = "from .heads.uncertainty_head_steps_reasoning import UncertaintyHeadStepReasoning"
anchor_imp = "from .heads.uncertainty_head_claim import UncertaintyHeadClaim"
if imp not in s:
    assert anchor_imp in s, "import anchor not found in auto_uncertainty_head.py"
    s = s.replace(anchor_imp, anchor_imp + "\n" + imp, 1)
anchor_map = '"claim": UncertaintyHeadClaim,'
entry = '"step_reasoning": UncertaintyHeadStepReasoning,'
if '"step_reasoning"' not in s:
    assert anchor_map in s, "MODEL_MAPPING anchor not found"
    s = s.replace(anchor_map, anchor_map + "\n        " + entry, 1)
open(path, "w").write(s)
print("    patched auto_uncertainty_head.py")
PY

echo "==> verifying"
python - <<'PY'
from luh.auto_uncertainty_head import AutoUncertaintyHead as A
assert "step_reasoning" in A.MODEL_MAPPING, "step_reasoning not registered"
print("    registered head types:", sorted(A.MODEL_MAPPING))
print("    step_reasoning ->", A.MODEL_MAPPING["step_reasoning"].__name__)
PY
echo "OK: step_reasoning head installed."
