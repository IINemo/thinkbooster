#!/usr/bin/env python3
"""Convert experiment results (from run_tts_eval.py) to the Visual Debugger format.

Usage:
    python scripts/convert_results_to_debugger.py <output_dir> [--out debugger_payload.json]

Example:
    python scripts/convert_results_to_debugger.py outputs/2026-02-04/beam_search_math500_09-09-47
    # Opens: service_app/static/debugger/index.html with the generated cached_examples.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import yaml

from service_app.core.debugger_events import convert_strategy_result_to_debugger_run

# Add the project root to sys.path so we can import service_app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy type -> metadata mapping (mirrors visual_debugger_demo.py)
# ---------------------------------------------------------------------------

STRATEGY_META = {
    "baseline": {
        "id": "baseline",
        "name": "Baseline (Raw CoT)",
        "family": "single_pass",
        "summary": "Single-pass raw chain-of-thought without search or reranking.",
    },
    "beam_search": {
        "id": "beam_search",
        "name": "Beam Search (ToT)",
        "family": "tree_search",
        "summary": "Tree-of-thought expansion with beam pruning.",
    },
    "adaptive": {
        "id": "adaptive",
        "name": "Adaptive Best-of-N",
        "family": "reranking",
        "summary": "Online best-of-n with adaptive scaling across steps.",
    },
    "online_best_of_n": {
        "id": "online_best_of_n",
        "name": "Online Best-of-N",
        "family": "reranking",
        "summary": "Iterative candidate generation with stepwise reranking.",
    },
    "offline_best_of_n": {
        "id": "offline_best_of_n",
        "name": "Offline Best-of-N",
        "family": "reranking",
        "summary": "Generate full trajectories first, then rerank at the end.",
    },
    "self_consistency": {
        "id": "self_consistency",
        "name": "Self-Consistency",
        "family": "sample_and_vote",
        "summary": "Sample diverse trajectories and select by answer consensus.",
    },
}

SCORER_META = {
    "prm": {
        "id": "prm",
        "name": "PRM",
        "direction": "higher_better",
        "summary": "Process Reward Model trajectory quality score.",
    },
    "self_verification": {
        "id": "self_verification",
        "name": "Self-Verification (LLM Critic)",
        "direction": "higher_better",
        "summary": "LLM-as-a-judge verification scoring.",
    },
    "sequence_prob": {
        "id": "sequence_prob",
        "name": "Sequence Prob",
        "direction": "higher_better",
        "summary": "Cumulative sequence probability from token logprobs.",
    },
    "perplexity": {
        "id": "perplexity",
        "name": "Perplexity",
        "direction": "lower_better",
        "summary": "Per-token perplexity estimated from generation logprobs.",
    },
    "entropy": {
        "id": "entropy",
        "name": "Entropy",
        "direction": "lower_better",
        "summary": "Mean token entropy of decoded reasoning steps.",
    },
}


def load_config(output_dir: Path) -> dict:
    """Load the Hydra config from the experiment output directory."""
    config_path = output_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_results(output_dir: Path) -> list:
    """Load results.json from the experiment output directory."""
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    with open(results_path) as f:
        return json.load(f)


def get_strategy_info(config: dict) -> dict:
    """Get strategy metadata from the config."""
    strategy_type = config.get("strategy", {}).get("type", "baseline")
    meta = STRATEGY_META.get(strategy_type)
    if meta:
        return deepcopy(meta)
    return {"id": strategy_type, "name": strategy_type, "family": "unknown"}


def get_scorer_info(config: dict) -> dict | None:
    """Get scorer metadata from the config."""
    scorer_cfg = config.get("scorer")
    if not scorer_cfg:
        return None
    scorer_type = scorer_cfg.get("type", "")
    # self_consistency uses consensus scoring, no external scorer
    strategy_type = config.get("strategy", {}).get("type", "")
    if strategy_type == "self_consistency":
        return None
    meta = SCORER_META.get(scorer_type)
    if meta:
        return deepcopy(meta)
    return {
        "id": scorer_type,
        "name": scorer_type,
        "direction": "higher_better",
        "summary": "",
    }


def convert_experiment(
    output_dir: Path,
    filter_incorrect: bool = False,
    max_samples: int | None = None,
) -> dict:
    """Convert an experiment output directory to the debugger cached_examples format.

    Returns a dict matching the cached_examples.json schema.
    """
    config = load_config(output_dir)
    results = load_results(output_dir)

    strategy_info = get_strategy_info(config)
    scorer_info = get_scorer_info(config)

    strategy_cfg = config.get("strategy", {})
    scorer_cfg = config.get("scorer", {}) or {}
    generation_cfg = config.get("generation", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    budget = strategy_cfg.get("max_steps", 10)

    model_config = {
        "provider": model_cfg.get("provider", "unknown"),
        "model_id": model_cfg.get("model_name", model_cfg.get("model_path", "")),
        "api_key_masked": "sk-...eval",
    }

    data_name = dataset_cfg.get("data_name", "unknown")
    strategy_type = strategy_cfg.get("type", "unknown")

    # Count correct/incorrect
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = correct / total if total > 0 else 0

    if filter_incorrect:
        results = [r for r in results if not r.get("is_correct")]

    if max_samples is not None:
        results = results[:max_samples]

    examples = []
    for sample in results:
        sample_idx = sample.get("index", 0)
        question = sample.get("question", "")
        gold_answer = str(sample.get("gold_answer", ""))
        extracted_answer = str(sample.get("extracted_answer", ""))
        is_correct = sample.get("is_correct", False)

        # Build the strategy result dict that the converter expects
        strategy_result = {
            "trajectory": sample.get("generated_trajectory", ""),
            "extracted_answer": extracted_answer,
            "steps": sample.get("steps", []),
            "validity_scores": sample.get("validity_scores", []),
            "token_stats": sample.get("token_stats"),
            "completed": sample.get("completed", True),
        }

        # Propagate tree visualization fields
        for key in (
            "step_candidates",
            "all_trajectories",
            "all_trajectory_steps",
            "all_scores",
            "all_step_scores",
            "all_traces",
            "best_idx",
        ):
            if key in sample:
                strategy_result[key] = sample[key]

        try:
            run_payload = convert_strategy_result_to_debugger_run(
                strategy=strategy_info,
                scorer=scorer_info,
                strategy_result=strategy_result,
                budget=budget,
                latency_ms=0,
                model_config=model_config,
                generation_config=generation_cfg,
                strategy_config=dict(strategy_cfg),
                scorer_config=dict(scorer_cfg),
                has_gold_answer=bool(gold_answer),
                gold_answer=gold_answer,
            )
        except Exception:
            log.exception(f"Failed to convert sample {sample_idx}, skipping")
            continue

        correctness_mark = "correct" if is_correct else "INCORRECT"
        title = f"[{sample_idx}] [{correctness_mark}] {question[:70]}"
        description = (
            f"Gold: {gold_answer} | Predicted: {extracted_answer} | "
            f"{'Correct' if is_correct else 'Incorrect'}"
        )

        scorer_catalog = [scorer_info] if scorer_info else []

        payload = {
            "scenario": {
                "id": f"sample_{sample_idx}",
                "title": title,
                "description": description,
                "prompt": question,
                "ground_truth": gold_answer,
                "input_source": "experiment_results",
                "model_config": model_config,
                "strategy_count": 1,
                "scorer_count": len(scorer_catalog),
                "run_count": 1,
            },
            "available_budgets": [budget],
            "selected_budget": budget,
            "strategy_catalog": [],
            "scorer_catalog": scorer_catalog,
            "strategies": [
                {
                    "id": strategy_info["id"],
                    "strategy_id": strategy_info["id"],
                    "scorer_id": scorer_info["id"] if scorer_info else None,
                    "name": strategy_info["name"],
                    "family": strategy_info["family"],
                    "summary": strategy_info.get("summary", ""),
                    "run": run_payload,
                    "comparison_rank": 1,
                }
            ],
        }

        examples.append(
            {
                "id": f"sample_{sample_idx}",
                "title": title,
                "description": description,
                "available_budgets": [budget],
                "default_budget": budget,
                "payloads": {str(budget): payload},
            }
        )

    log.info(
        f"Converted {len(examples)} samples from {output_dir.name} "
        f"({data_name}, {strategy_type}, accuracy={accuracy:.1%})"
    )
    return {"examples": examples}


def main():
    parser = argparse.ArgumentParser(
        description="Convert experiment results to Visual Debugger format."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Path to experiment output directory (containing results.json and .hydra/config.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the debugger JSON. Default: <output_dir>/debugger_payload.json",
    )
    parser.add_argument(
        "--incorrect-only",
        action="store_true",
        help="Only include incorrect samples (useful for debugging failures)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Also copy the output to cached_examples.json for direct use with the debugger",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        log.error(f"Not a directory: {output_dir}")
        sys.exit(1)

    result = convert_experiment(
        output_dir=output_dir,
        filter_incorrect=args.incorrect_only,
        max_samples=args.max_samples,
    )

    out_path = args.out or (output_dir / "debugger_payload.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Wrote {out_path} ({len(result['examples'])} examples)")

    if args.install:
        cached_path = (
            PROJECT_ROOT
            / "service_app"
            / "static"
            / "debugger"
            / "cached_examples.json"
        )
        with open(cached_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"Installed to {cached_path}")
        log.info(
            f"Open in browser: file://{PROJECT_ROOT}/service_app/static/debugger/index.html"
        )


if __name__ == "__main__":
    main()
