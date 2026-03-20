# Task: Tree of Thoughts — Implementation Verification

## Goal

Verify that our beam search implementation (used as ToT) correctly reproduces results from the original paper, then run experiments with Qwen2.5-Math-7B-Instruct.

## Background

- **Paper**: [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) (Yao et al., 2023)
- **Our implementation**: Beam search strategy (`llm_tts/strategies/strategy_beam_search.py`) with LLM-as-a-critic scorer (`llm_tts/scorers/step_scorer_llm_critic.py`), introduced in PR #161
- **Original code**: https://github.com/princeton-nlp/tree-of-thought-llm
- **Original prompts**: https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/src/tot/prompts
- **Original trajectories (for comparison)**: https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/logs

## Phase 1: Reproduce original paper results (Game of 24 + GPT-4)

The paper reports **74% success rate** on Game of 24 (indices 900–999, 100 puzzles) using GPT-4 with ToT (b=5).

### Steps

1. **Compare prompts with original**
   - Our prompts: `config/prompts/tree-of-thought/game24/` (propose_fewshot.txt, value_intermediate.txt, value_final.txt)
   - Original prompts: https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/src/tot/prompts/game24.py
   - Ensure propose prompt, value prompt, and step format match exactly

2. **Create experiment config**
   - Dataset: `config/dataset/game24.yaml` (already exists, indices 900–1000)
   - Model: GPT-4 via OpenRouter (`openai/gpt-4` or `openai/gpt-4-turbo`)
   - Strategy: beam search with `beam_width=5` (paper uses b=5)
   - Scorer: LLM-as-a-critic (`config/scorer/llm_critic.yaml`)
   - Create config at `config/experiments/beam_search/game24/beam_search_openrouter_gpt4_game24_llm_critic.yaml`

3. **Implement Game of 24 evaluator**
   - The task is NOT exact match — need to verify the expression equals 24
   - Check if expression uses exactly the 4 given numbers
   - Parse and evaluate arithmetic expression
   - May need a custom evaluator in `llm_tts/evaluation/`

4. **Run experiment**
   ```bash
   CUDA_VISIBLE_DEVICES="" python scripts/run_tts_eval.py \
     --config-path ../config \
     --config-name experiments/beam_search/game24/beam_search_openrouter_gpt4_game24_llm_critic \
     dataset.subset=10  # start with 10, then full 100
   ```

5. **Compare results**
   - Target: ~74% success rate (paper result with GPT-4)
   - Compare trajectories with original logs: https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/logs
   - If significantly off, debug: check beam expansion, scoring, pruning behavior

### Key differences to watch for
- Our beam search does step-level scoring; original ToT does value-based voting
- Prompt format for "propose" and "value" steps must match paper exactly
- Temperature and sampling parameters must match (paper uses temperature=0.7 for propose, temperature=1.0 for value)

## Phase 2: Run experiments with Qwen2.5-Math-7B-Instruct (4 math datasets)

After Phase 1 confirms correctness, run beam search with LLM-as-a-critic on:

1. **MATH-500** — `config/experiments/beam_search/math500/`
2. **OlympiadBench** — `config/experiments/beam_search/olympiadbench/`
3. **GaoKao 2023 En** — `config/experiments/beam_search/gaokao2023en/`
4. **Minerva Math** — `config/experiments/beam_search/minerva_math/`

### For each dataset
- Model: Qwen2.5-Math-7B-Instruct (vLLM backend, 2 GPUs)
- Scorer: LLM-as-a-critic
- Beam width: 4 (our standard)
- Seeds: 42, 43, 44 (3 seeds per dataset)
- Configs already exist in `config/experiments/beam_search/*/window_all/mean/` with `llm_critic` suffix

### Submission
```bash
./scripts/local/submit.sh --strategy beam_search --dataset math500 --scorers llm_critic --seeds 3
./scripts/local/submit.sh --strategy beam_search --dataset olympiadbench --scorers llm_critic --seeds 3
./scripts/local/submit.sh --strategy beam_search --dataset gaokao2023en --scorers llm_critic --seeds 3
./scripts/local/submit.sh --strategy beam_search --dataset minerva_math --scorers llm_critic --seeds 3
```

## References

- **Paper**: Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", NeurIPS 2023. https://arxiv.org/abs/2305.10601
- **Original code**: https://github.com/princeton-nlp/tree-of-thought-llm
- **Original prompts**: https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/src/tot/prompts
- **Original trajectories**: https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/logs
- **Our LLM-as-a-critic PR**: https://github.com/IINemo/thinkbooster/pull/161
