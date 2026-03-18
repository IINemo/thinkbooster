# Project Structure

Overview of the ThinkBooster codebase organization.

---

## Directory Tree

```
thinkbooster/
├── llm_tts/                              # Main library package
│   ├── strategies/                       # TTS strategy implementations
│   │   ├── strategy_base.py             # Abstract base class (generate_trajectories_batch)
│   │   ├── deepconf/                    # DeepConf (offline/online confidence-based)
│   │   │   ├── strategy.py
│   │   │   └── utils.py
│   │   ├── strategy_beam_search.py      # Beam search with step-level scoring
│   │   ├── strategy_online_best_of_n.py # Online best-of-N (step-by-step)
│   │   ├── strategy_offline_best_of_n.py# Offline best-of-N (generate-then-score)
│   │   ├── strategy_self_consistency.py # Majority voting across N paths
│   │   ├── adaptive_scaling_best_of_n.py# Adaptive compute scaling
│   │   ├── strategy_uncertainty_cot.py  # Uncertainty-guided chain-of-thought
│   │   ├── phi.py                       # Phi decoding
│   │   ├── strategy_extended_thinking.py# Extended thinking wrapper
│   │   ├── strategy_baseline.py         # Single-shot baseline
│   │   ├── strategy_chain_of_thought.py # Basic chain-of-thought
│   │   └── metadata_builder.py          # Strategy metadata helpers
│   │
│   ├── generators/                       # Step candidate generators (backends)
│   │   ├── base.py                      # Base interface + StepCandidate dataclass
│   │   ├── api.py                       # OpenAI-compatible API backend
│   │   ├── vllm.py                      # vLLM backend
│   │   └── huggingface.py               # HuggingFace transformers backend
│   │
│   ├── scorers/                          # Step scoring implementations
│   │   ├── step_scorer_base.py          # Base scorer interface
│   │   ├── step_scorer_prm.py           # Process Reward Model scorer
│   │   ├── step_scorer_uncertainty.py   # Uncertainty-based scorer (entropy, perplexity)
│   │   ├── step_scorer_confidence.py    # Confidence scorer
│   │   ├── step_scorer_llm_critic.py    # LLM-as-a-critic scorer
│   │   ├── step_scorer_reward_base.py   # Base for reward model scorers
│   │   ├── multi_scorer.py             # Composite scorer (multiple scorers)
│   │   ├── majority_voting.py           # Majority voting scorer
│   │   └── estimator_uncertainty_pd.py  # Predictive distribution uncertainty
│   │
│   ├── step_boundary_detectors/          # Step/answer boundary detection
│   │   ├── base.py                      # Base detector interface
│   │   ├── non_thinking/                # Structured step markers (- Step, <Answer>:)
│   │   └── thinking/                    # Thinking model markers (<think>, </think>)
│   │
│   ├── models/                           # Model wrappers
│   │   ├── blackboxmodel_with_streaming.py  # OpenAI-compatible with streaming + logprobs
│   │   └── base.py                      # Base model interface
│   │
│   ├── evaluation/                       # Correctness evaluation
│   │   ├── exact_match.py              # Qwen2.5-Math exact match (numeric, boolean, char, string)
│   │   ├── llm_as_a_judge.py           # LLM-based correctness verification
│   │   ├── human_eval_plus_evaluator.py# HumanEval+ code evaluation
│   │   ├── mbpp_plus_evaluator.py      # MBPP+ code evaluation
│   │   ├── grader.py                   # Math grading utilities
│   │   ├── parser.py                   # Answer extraction parser
│   │   ├── math_normalize.py           # Math expression normalization
│   │   ├── alignscore.py              # Semantic similarity
│   │   └── latex2sympy/               # LaTeX to SymPy conversion
│   │
│   ├── datasets/                         # Dataset loaders
│   │   ├── gsm8k.py
│   │   ├── human_eval_plus.py
│   │   └── mbpp_plus.py
│   │
│   ├── integrations/                     # Third-party integrations
│   │   └── langchain_chat_model.py      # LangChain/LangGraph chat model
│   │
│   ├── utils/                            # Shared utilities
│   │   ├── answer_extraction.py         # Answer extraction from model output
│   │   ├── flops.py                     # FLOP computation
│   │   ├── parallel.py                  # Parallel execution helpers
│   │   ├── telegram.py                  # Telegram notifications
│   │   └── torch_dtype.py              # Torch dtype utilities
│   │
│   └── early_stopping.py                # Early stopping conditions for streaming
│
├── service_app/                          # OpenAI-compatible REST API service
│   ├── main.py                          # FastAPI app entrypoint
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py                  # /v1/chat/completions endpoint
│   │   │   ├── debugger.py              # /debugger SSE endpoint
│   │   │   └── models.py               # /v1/models endpoint
│   │   └── models/
│   │       └── openai_compat.py         # OpenAI-compatible request/response models
│   ├── core/
│   │   ├── strategy_manager.py          # Strategy factory and lifecycle
│   │   ├── prm_scorer_factory.py        # PRM model loading
│   │   ├── debugger_events.py           # Visual debugger event processing
│   │   ├── config.py                    # Service configuration
│   │   └── logging_config.py            # Logging setup
│   └── static/debugger/                 # Visual debugger web UI (HTML/JS/CSS)
│
├── scripts/
│   ├── run_tts_eval.py                  # Main evaluation pipeline
│   ├── local/submit.sh                  # Local experiment submission helper
│   └── slurm/submit.sh                 # SLURM cluster submission
│
├── config/                               # Hydra configuration
│   ├── experiments/                     # Complete experiment configs
│   │   ├── adaptive_scaling/
│   │   ├── baseline/
│   │   ├── beam_search/
│   │   ├── chain_of_thought/
│   │   ├── deepconf/
│   │   ├── extended_thinking/
│   │   ├── offline_best_of_n/
│   │   ├── online_best_of_n/
│   │   ├── self_consistency/
│   │   └── uncertainty_cot/
│   ├── dataset/                         # Dataset configs (math_500, olympiadbench, etc.)
│   ├── model/                           # Model configs (openrouter, vllm, hf, etc.)
│   ├── strategy/                        # Strategy hyperparameters
│   ├── scorer/                          # Scorer configs (entropy, prm, pd, etc.)
│   ├── generation/                      # Generation parameters
│   ├── evaluation/                      # Evaluation configs
│   ├── prompts/                         # Prompt templates
│   └── system/                          # System settings (device, seed)
│
├── tests/                                # Test suite
│   ├── strategy_registry.py             # Strategy registry and validation
│   ├── test_config_naming.py            # Config naming convention tests
│   ├── deepconf/                        # DeepConf tests
│   ├── evaluation/                      # Evaluator tests
│   ├── service_app/                     # API integration tests
│   └── run_tts_eval/                    # Eval pipeline integration test
│
├── lm-polygraph/                         # Submodule: uncertainty estimation library
├── setup.sh                              # Installation script (all dependencies)
├── pyproject.toml                        # Package configuration
├── docker-compose.yml                    # Docker deployment
├── Makefile                              # Dev commands (format, lint, test)
└── .github/workflows/test.yml            # CI pipeline
```

---

## Core Components

### 1. Strategies (`llm_tts/strategies/`)

All strategies inherit from `StrategyBase` and implement `generate_trajectories_batch()`:

| Strategy | File | Description |
|----------|------|-------------|
| Beam Search | `strategy_beam_search.py` | Step-level beam search with scorer-guided expansion |
| Online Best-of-N | `strategy_online_best_of_n.py` | Generate K candidates per step, pick best |
| Offline Best-of-N | `strategy_offline_best_of_n.py` | Generate N full trajectories, score and select |
| Self-Consistency | `strategy_self_consistency.py` | Majority voting across N independent paths |
| Adaptive Scaling | `adaptive_scaling_best_of_n.py` | Dynamic compute allocation based on confidence |
| Uncertainty CoT | `strategy_uncertainty_cot.py` | Uncertainty-guided chain-of-thought branching |
| DeepConf | `deepconf/strategy.py` | Confidence-based scaling (offline/online modes) |
| Phi Decoding | `phi.py` | Phi decoding with clustering |
| Extended Thinking | `strategy_extended_thinking.py` | Wrapper for thinking-mode models |
| Baseline | `strategy_baseline.py` | Single-shot generation (no scaling) |

### 2. Generators (`llm_tts/generators/`)

Backend-agnostic step candidate generation. Strategies call generators to produce step candidates.

| Generator | Backend | Use case |
|-----------|---------|----------|
| `api.py` | OpenAI-compatible API | OpenRouter, DeepSeek, any OpenAI SDK endpoint |
| `vllm.py` | vLLM | Fast local GPU inference with batching |
| `huggingface.py` | HuggingFace transformers | Local inference without vLLM |

### 3. Scorers (`llm_tts/scorers/`)

Score reasoning steps to guide strategy decisions.

| Scorer | Description |
|--------|-------------|
| PRM (`step_scorer_prm.py`) | Process Reward Model (e.g., Qwen2.5-Math-PRM-7B) |
| Uncertainty (`step_scorer_uncertainty.py`) | Entropy, perplexity, sequence probability |
| LLM Critic (`step_scorer_llm_critic.py`) | LLM-as-a-critic evaluation |
| Confidence (`step_scorer_confidence.py`) | Token-level confidence scoring |
| Multi-scorer (`multi_scorer.py`) | Composite of multiple scorers |

### 4. Service (`service_app/`)

OpenAI-compatible REST API gateway. Strategy and scorer are selected via URL path:
```
/v1/beam_search/prm/chat/completions
/v1/self_consistency/chat/completions
```

Includes a visual debugger web UI at `/debugger` for inspecting strategy behavior.

### 5. Configuration (`config/`)

Hierarchical Hydra configuration. Experiment configs compose from dataset, model, strategy, scorer, and evaluation components:

```yaml
# config/experiments/offline_best_of_n/math500/offline_bon_openrouter_gpt4o_mini_math500_entropy.yaml
defaults:
  - /config
  - /dataset/math_500
  - /model/openrouter
  - /strategy/offline_bon
  - /scorer/entropy
  - /evaluation/default
  - _self_
```

### 6. Evaluation (`llm_tts/evaluation/`)

- **Exact match** — Qwen2.5-Math official grading logic (numeric, LaTeX, boolean, char, string)
- **LLM judge** — Multi-vote LLM-based correctness verification
- **Code evaluation** — HumanEval+ and MBPP+ via EvalPlus sandbox
- **AlignScore** — Semantic similarity

---

## Key Design Patterns

1. **Strategy pattern** — all strategies implement `generate_trajectories_batch()` from `StrategyBase`
2. **Generator abstraction** — strategies are backend-agnostic; same strategy works with API, vLLM, or HuggingFace
3. **Configuration composition** — Hydra configs compose from smaller config files via `defaults:`
4. **Registry pattern** — strategy registry in `tests/strategy_registry.py` enforces test coverage
5. **Plugin architecture** — early stopping, boundary detectors, and scorers are pluggable

---

## References

- [Contributing Guide](CONTRIBUTING.md) — setup, workflow, PR guidelines
- [Strategy Registration](STRATEGY_REGISTRATION.md) — how to add new strategies
- [Service API Guide](service/SERVICE_API_GUIDE.md) — REST API reference
- [DeepConf Guide](deepconf/DeepConf.md) — example strategy deep-dive
- [Architecture](architecture.md) — offline vs online strategy paradigms
