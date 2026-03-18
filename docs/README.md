# ThinkBooster Documentation

## Getting Started

- [Contributing Guide](contributing.md) — setup, development workflow, PR guidelines
- [Project Structure](project_structure.md) — codebase organization and design patterns

## Core Concepts

- [Architecture](architecture.md) — offline vs online strategy paradigms
- [Step Boundary Detectors](step_boundary_detectors.md) — how step/answer boundaries are detected
- [vLLM vs HuggingFace](vllm_vs_hf.md) — backend comparison and trade-offs
- [Strategy Registration](strategy_registration.md) — adding new strategies with test coverage

## Strategies

- [DeepConf](strategies/deepconf.md) — confidence-based test-time scaling (offline/online)
- [Online Best-of-N](strategies/online_bon.md) — step-by-step generation with PRM scoring

## Service & Integrations

- [Service API Guide](service/api_guide.md) — OpenAI-compatible REST API reference
- [LangGraph Integration](integrations/langgraph.md) — LangChain/LangGraph with uncertainty
- [KernelAct Integration](integrations/kernelact.md) — CUDA kernel optimization benchmark
- [Telegram Notifications](integrations/telegram.md) — experiment notifications via Telegram bot

## Evaluation

- [Evaluation Protocol](evaluation/README.md) — overview and quick links
- [Datasets](evaluation/datasets.md) — supported benchmarks (MATH-500, OlympiadBench, AIME, HumanEval+, etc.)
- [Metrics](evaluation/metrics.md) — accuracy, tokens, FLOPs calculation
- [Models](evaluation/models.md) — model configurations
- [WandB](evaluation/wandb.md) — experiment logging and reporting
- [FLOP Calculator](evaluation/flop_calculator.md) — compute cost estimation

## Infrastructure

- [SLURM Guide](slurm.md) — running experiments on HPC clusters
