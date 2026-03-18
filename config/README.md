# Configuration

Hydra-based configuration for ThinkBooster evaluation framework.

## Directory Structure

```
config/
├── experiments/              # Full experiment configs (entry points)
│   ├── adaptive_scaling/     # Adaptive scaling experiments
│   ├── baseline/             # Baseline (single-shot)
│   ├── beam_search/          # Beam search experiments
│   ├── chain_of_thought/     # Chain-of-thought
│   ├── deepconf/             # DeepConf (offline/online)
│   ├── extended_thinking/    # Extended thinking
│   ├── offline_best_of_n/    # Offline best-of-N
│   ├── online_best_of_n/     # Online best-of-N
│   ├── self_consistency/     # Self-consistency
│   └── uncertainty_cot/      # Uncertainty-guided CoT
│
├── dataset/                  # Dataset configs
│   ├── math_500.yaml         # MATH-500
│   ├── olympiadbench.yaml    # OlympiadBench
│   ├── gaokao2023en.yaml     # GaoKao 2023 English
│   ├── minerva_math.yaml     # Minerva Math
│   ├── aime_2025.yaml        # AIME 2025
│   ├── human_eval_plus.yaml  # HumanEval+ (code)
│   ├── mbpp_plus.yaml        # MBPP+ (code)
│   └── ...
│
├── model/                    # Model/backend configs
│   ├── openrouter.yaml       # OpenRouter API (gpt-4o-mini, Claude, etc.)
│   ├── vllm_qwen3.yaml       # vLLM local (Qwen3)
│   ├── openai.yaml           # OpenAI API
│   ├── deepseek.yaml         # DeepSeek API
│   └── hf_qwen3.yaml         # HuggingFace local
│
├── strategy/                 # Strategy hyperparameters
│   ├── beam_search.yaml
│   ├── offline_bon.yaml
│   ├── online_bon.yaml
│   ├── self_consistency.yaml
│   ├── adaptive.yaml
│   ├── deepconf.yaml
│   └── ...
│
├── scorer/                   # Scorer configs
│   ├── entropy.yaml          # Entropy scorer
│   ├── prm.yaml              # Process Reward Model
│   ├── perplexity.yaml       # Perplexity scorer
│   ├── sequence_prob.yaml    # Sequence probability
│   ├── llm_critic.yaml       # LLM-as-a-critic
│   └── ...
│
├── evaluation/               # Evaluation method configs
├── generation/               # Generation parameters (temperature, top_p, etc.)
├── prompts/                  # Prompt templates
└── system/                   # System settings (device, seed)
```

## Usage

### Run an experiment config

```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/offline_best_of_n/math500/offline_bon_openrouter_gpt4o_mini_math500_entropy
```

### Override specific values

```bash
python scripts/run_tts_eval.py \
  --config-path ../config \
  --config-name experiments/offline_best_of_n/math500/offline_bon_openrouter_gpt4o_mini_math500_entropy \
  dataset.subset=10 \
  report_to=''
```

## Experiment Config Structure

Experiment configs compose from modular components via Hydra defaults:

```yaml
# @package _global_
defaults:
  - /config
  - /dataset/math_500
  - /model/openrouter
  - /strategy/offline_bon
  - /scorer/entropy
  - /evaluation/default
  - _self_

run_name: "offline_bon_openrouter_gpt4o_mini_math500_entropy_seed${system.seed}_${now:%H-%M-%S}"
report_to: wandb
wandb_project: llm-tts-eval-math500

model:
  model_path: "openai/gpt-4o-mini"

dataset:
  data_name: math
  prompt_file: "${hydra:runtime.cwd}/config/prompts/qwen25_math_official.txt"
```

## Naming Convention

Experiment config filenames follow:
```
{strategy_prefix}_{backend}_{model}_{dataset}_{scorer}.yaml
```

Examples:
- `offline_bon_openrouter_gpt4o_mini_math500_entropy.yaml`
- `beam_search_vllm_thinking_qwen3_8b_olympiadbench_prm.yaml`
- `baseline_vllm_qwen25_math_7b_instruct_aime2025.yaml`

See `tests/test_config_naming.py` for the enforced naming rules.

## Creating New Experiments

1. Pick the strategy directory under `experiments/`
2. Copy an existing config for the same strategy
3. Modify model, dataset, scorer as needed
4. Run with `dataset.subset=1` to verify
