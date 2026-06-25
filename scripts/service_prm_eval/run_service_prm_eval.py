#!/usr/bin/env python
"""
Service PRM evaluation harness.

Implements the test plan in ``thinkbooster_service_prm_test_exps.md`` /
``docs/service/running_with_prm.md``:

  1. Compare plain CoT reasoning vs PRM scoring in *offline best-of-N*
     (generation model: ``openai/gpt-oss-20b`` on OpenRouter).
  2. Run on two datasets: Gaokao2023en (math) + HumanEval+ (code).
  3. Time how long each strategy takes.

Two strategies per dataset:

  * ``cot``  — plain Chain-of-Thought, a single direct generation against
               OpenRouter (``https://openrouter.ai/api/v1``). This is the baseline.
  * ``prm``  — offline best-of-N: the request is sent to the *ThinkBooster service*
               (``<service-url>/v1/offline_bon/prm``). The service samples N
               trajectories on OpenRouter and re-ranks them with a local PRM
               (Qwen2.5-Math-PRM-7B on the GPU).

Generation always runs on OpenRouter; only the PRM runs locally on the GPU. The
OpenRouter key is read from the ``OPENROUTER_API_KEY`` environment variable (or
``.env`` at the repo root) and is forwarded to the service via
``extra_body["tts_api_key"]`` — never hard-coded here.

Correctness is graded with the same evaluators the offline pipeline uses:
  * Gaokao   -> ``thinkbooster.evaluation.EvaluatorExactMatch`` (math_equal)
  * HumanEval -> ``thinkbooster.evaluation.EvaluatorHumanEvalPlus`` (EvalPlus full suite)

Outputs (under ``--out``):
  * ``<dataset>__<strategy>/samples.jsonl`` — one record per problem (incremental,
    resumable)
  * ``summary.json`` — accuracy + timing per (dataset, strategy) cell
  * ``report.md``    — human-readable comparison table

Example (run on the GPU box after launching the service):

    python scripts/service_prm_eval/run_service_prm_eval.py \
        --datasets gaokao humaneval \
        --strategies cot prm \
        --service-url http://localhost:8001 \
        --n 8 --agg min --concurrency 8

See scripts/service_prm_eval/README.md for the full workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Repo paths / imports
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("service_prm_eval")

MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
CODE_SYSTEM_PROMPT = (
    "Do not output any other code except for asked self-contained Python script. "
    "Do not provide any guides on how to run it. Code will be parsed from codeblock "
    "to check it's correct."
)

GAOKAO_DATA_NAME = "gaokao2023en"
GAOKAO_HF_PATH = "test-time-compute/test_gaokao2023en"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    index: int
    question: str  # user-message content sent to the model
    gold: str  # gold answer (math: parsed answer; code: canonical solution)
    system_prompt: str
    task_id: Optional[str] = None  # HumanEval task id, used by EvalPlus


@dataclass
class SampleResult:
    index: int
    dataset: str
    strategy: str
    question: str
    gold: str
    task_id: Optional[str]
    content: str  # full model response (best trajectory for PRM)
    selected_answer: Optional[str]  # PRM-selected extracted answer (PRM only)
    aggregated_score: Optional[float]  # winning PRM score (PRM only)
    n_trajectories: Optional[int]  # how many candidates PRM ranked
    client_latency_s: float  # wall-clock measured by this client
    server_elapsed_s: Optional[float]  # server-side elapsed_time (PRM only)
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    error: Optional[str] = None
    is_correct: Optional[bool] = None  # filled in during evaluation

    def to_json(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "dataset": self.dataset,
            "strategy": self.strategy,
            "task_id": self.task_id,
            "gold": self.gold,
            "content": self.content,
            "selected_answer": self.selected_answer,
            "aggregated_score": self.aggregated_score,
            "n_trajectories": self.n_trajectories,
            "client_latency_s": round(self.client_latency_s, 3),
            "server_elapsed_s": self.server_elapsed_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "is_correct": self.is_correct,
            # question kept last (long); useful for debugging
            "question": self.question,
        }


@dataclass
class CellSummary:
    dataset: str
    strategy: str
    config: Dict[str, Any]
    n_total: int = 0
    n_errors: int = 0
    n_correct: int = 0
    latencies: List[float] = field(default_factory=list)
    server_elapsed: List[float] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    wall_clock_s: float = 0.0

    @property
    def n_scored(self) -> int:
        return self.n_total - self.n_errors

    def to_json(self) -> Dict[str, Any]:
        def _stats(xs: List[float]) -> Dict[str, Optional[float]]:
            xs = [x for x in xs if x is not None]
            if not xs:
                return {
                    "mean": None,
                    "median": None,
                    "min": None,
                    "max": None,
                    "sum": None,
                }
            return {
                "mean": round(statistics.mean(xs), 3),
                "median": round(statistics.median(xs), 3),
                "min": round(min(xs), 3),
                "max": round(max(xs), 3),
                "sum": round(sum(xs), 3),
            }

        acc = (self.n_correct / self.n_scored) if self.n_scored else None
        return {
            "dataset": self.dataset,
            "strategy": self.strategy,
            "config": self.config,
            "n_total": self.n_total,
            "n_errors": self.n_errors,
            "n_scored": self.n_scored,
            "n_correct": self.n_correct,
            "accuracy": round(acc, 4) if acc is not None else None,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "throughput_problems_per_min": (
                round(self.n_total / self.wall_clock_s * 60, 2)
                if self.wall_clock_s > 0
                else None
            ),
            "client_latency_s": _stats(self.latencies),
            "server_elapsed_s": _stats(self.server_elapsed),
            "output_tokens": _stats([float(x) for x in self.output_tokens]),
        }


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_gaokao(subset: Optional[int]) -> List[Problem]:
    from datasets import load_dataset

    from thinkbooster.evaluation.parser import parse_ground_truth, parse_question

    log.info("Loading Gaokao2023en from HF: %s", GAOKAO_HF_PATH)
    ds = load_dataset(GAOKAO_HF_PATH, split="test")
    problems: List[Problem] = []
    for i, ex in enumerate(ds):
        if subset is not None and i >= subset:
            break
        question = parse_question(ex, GAOKAO_DATA_NAME)
        _, gold = parse_ground_truth(ex, GAOKAO_DATA_NAME)
        problems.append(
            Problem(
                index=i,
                question=question,
                gold=str(gold),
                system_prompt=MATH_SYSTEM_PROMPT,
            )
        )
    log.info("Loaded %d Gaokao problems", len(problems))
    return problems


def load_humaneval(subset: Optional[int]) -> List[Problem]:
    from thinkbooster.datasets.human_eval_plus import load_human_eval_plus

    log.info("Loading HumanEval+ via evalplus API")
    data = load_human_eval_plus(subset_size=subset)
    problems: List[Problem] = []
    for i, ex in enumerate(data):
        problems.append(
            Problem(
                index=i,
                question=ex["question"],
                gold=ex["answer"],
                system_prompt=CODE_SYSTEM_PROMPT,
                task_id=ex["task_id"],
            )
        )
    log.info("Loaded %d HumanEval+ problems", len(problems))
    return problems


DATASET_LOADERS = {
    "gaokao": load_gaokao,
    "humaneval": load_humaneval,
}


# --------------------------------------------------------------------------- #
# Generation backends
# --------------------------------------------------------------------------- #
def _with_retries(fn, *, attempts: int, base_delay: float = 2.0):
    """Call fn() with simple exponential backoff. Returns fn() or raises last error."""
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = base_delay * (2**attempt)
            log.warning(
                "API call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _extract_content(resp) -> str:
    """Pull the assistant text; fall back to a `reasoning` field if content is empty."""
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        # Some reasoning models (gpt-oss) may surface text under `reasoning`.
        reasoning = getattr(msg, "reasoning", None)
        if isinstance(reasoning, str):
            content = reasoning.strip()
    return content


def run_cot(client, problem: Problem, args) -> SampleResult:
    """Plain CoT baseline: one direct OpenRouter generation."""

    def _call():
        return client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": problem.system_prompt},
                {"role": "user", "content": problem.question},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    t0 = time.perf_counter()
    try:
        resp = _with_retries(_call, attempts=args.retries)
    except Exception as exc:  # noqa: BLE001
        return SampleResult(
            index=problem.index,
            dataset=args._dataset,
            strategy="cot",
            question=problem.question,
            gold=problem.gold,
            task_id=problem.task_id,
            content="",
            selected_answer=None,
            aggregated_score=None,
            n_trajectories=None,
            client_latency_s=time.perf_counter() - t0,
            server_elapsed_s=None,
            input_tokens=None,
            output_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency = time.perf_counter() - t0

    usage = getattr(resp, "usage", None)
    return SampleResult(
        index=problem.index,
        dataset=args._dataset,
        strategy="cot",
        question=problem.question,
        gold=problem.gold,
        task_id=problem.task_id,
        content=_extract_content(resp),
        selected_answer=None,
        aggregated_score=None,
        n_trajectories=1,
        client_latency_s=latency,
        server_elapsed_s=None,
        input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )


def run_prm(client, problem: Problem, args, api_key: str) -> SampleResult:
    """Offline best-of-N with PRM re-ranking, via the ThinkBooster service."""

    def _call():
        return client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": problem.system_prompt},
                {"role": "user", "content": problem.question},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body={
                "tts_api_key": api_key,  # authenticates + routes to OpenRouter
                "tts_num_trajectories": args.n,
                "tts_score_aggregation": args.agg,
            },
        )

    t0 = time.perf_counter()
    try:
        resp = _with_retries(_call, attempts=args.retries)
    except Exception as exc:  # noqa: BLE001
        return SampleResult(
            index=problem.index,
            dataset=args._dataset,
            strategy="prm",
            question=problem.question,
            gold=problem.gold,
            task_id=problem.task_id,
            content="",
            selected_answer=None,
            aggregated_score=None,
            n_trajectories=None,
            client_latency_s=time.perf_counter() - t0,
            server_elapsed_s=None,
            input_tokens=None,
            output_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency = time.perf_counter() - t0

    dump = resp.model_dump()
    meta = dump["choices"][0].get("tts_metadata", {}) or {}
    token_stats = meta.get("token_stats", {}) or {}
    all_traj = meta.get("all_trajectories", []) or []
    return SampleResult(
        index=problem.index,
        dataset=args._dataset,
        strategy="prm",
        question=problem.question,
        gold=problem.gold,
        task_id=problem.task_id,
        content=_extract_content(resp),
        selected_answer=meta.get("selected_answer"),
        aggregated_score=meta.get("aggregated_score"),
        n_trajectories=len(all_traj) or args.n,
        client_latency_s=latency,
        server_elapsed_s=meta.get("elapsed_time"),
        input_tokens=token_stats.get("input_tokens"),
        output_tokens=token_stats.get("output_tokens"),
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def grade_results(dataset: str, results: List[SampleResult]) -> None:
    """Fill in ``is_correct`` for each non-errored result, in place."""
    scored = [r for r in results if r.error is None]
    if not scored:
        return

    if dataset == "gaokao":
        from thinkbooster.evaluation import EvaluatorExactMatch

        evaluator = EvaluatorExactMatch(
            dataset_answer_format="numeric", data_name=GAOKAO_DATA_NAME
        )
        scores = evaluator(
            [r.question for r in scored],
            [r.content for r in scored],
            [r.gold for r in scored],
        )
    elif dataset == "humaneval":
        from thinkbooster.evaluation import EvaluatorHumanEvalPlus

        evaluator = EvaluatorHumanEvalPlus(mode="full")
        scores = evaluator(
            [r.question for r in scored],
            [r.content for r in scored],
            [r.gold for r in scored],
            task_ids=[r.task_id for r in scored],
        )
    else:
        raise ValueError(f"Unknown dataset for grading: {dataset}")

    for r, s in zip(scored, scores):
        r.is_correct = bool(s >= 1.0)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _load_done_indices(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read an existing samples.jsonl so we can resume without re-querying."""
    done: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # only treat successful records as "done" so errors get retried
            if rec.get("error") is None:
                done[rec["index"]] = rec
    return done


def _record_from_json(rec: Dict[str, Any], dataset: str, strategy: str) -> SampleResult:
    return SampleResult(
        index=rec["index"],
        dataset=dataset,
        strategy=strategy,
        question=rec.get("question", ""),
        gold=rec.get("gold", ""),
        task_id=rec.get("task_id"),
        content=rec.get("content", ""),
        selected_answer=rec.get("selected_answer"),
        aggregated_score=rec.get("aggregated_score"),
        n_trajectories=rec.get("n_trajectories"),
        client_latency_s=rec.get("client_latency_s", 0.0),
        server_elapsed_s=rec.get("server_elapsed_s"),
        input_tokens=rec.get("input_tokens"),
        output_tokens=rec.get("output_tokens"),
        error=rec.get("error"),
        is_correct=rec.get("is_correct"),
    )


def run_cell(
    dataset: str,
    strategy: str,
    problems: List[Problem],
    args,
    out_dir: Path,
    api_key: str,
) -> CellSummary:
    """Run one (dataset, strategy) cell: generate, grade, persist."""
    from openai import OpenAI

    cell_dir = out_dir / f"{dataset}__{strategy}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    samples_path = cell_dir / "samples.jsonl"

    if strategy == "cot":
        client = OpenAI(
            base_url=args.openrouter_base, api_key=api_key, timeout=args.timeout
        )
        runner = lambda p: run_cot(client, p, args)  # noqa: E731
        config = {
            "strategy": "cot_baseline",
            "n": 1,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    elif strategy == "prm":
        client = OpenAI(
            base_url=f"{args.service_url.rstrip('/')}/v1/offline_bon/prm",
            api_key="unused",  # service reads the key from extra_body["tts_api_key"]
            timeout=args.timeout,
        )
        runner = lambda p: run_prm(client, p, args, api_key)  # noqa: E731
        config = {
            "strategy": "offline_bon/prm",
            "n": args.n,
            "agg": args.agg,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "service_url": args.service_url,
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    args._dataset = dataset  # passed through to runners for labeling

    done = _load_done_indices(samples_path) if args.resume else {}
    todo = [p for p in problems if p.index not in done]
    log.info(
        "[%s/%s] %d problems (%d already done, %d to run), concurrency=%d",
        dataset,
        strategy,
        len(problems),
        len(done),
        len(todo),
        args.concurrency,
    )

    results: List[SampleResult] = [
        _record_from_json(done[p.index], dataset, strategy)
        for p in problems
        if p.index in done
    ]

    write_lock = threading.Lock()
    file_mode = "a" if (args.resume and samples_path.exists()) else "w"
    t_cell0 = time.perf_counter()
    completed = 0
    with (
        samples_path.open(file_mode) as fout,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futures = {pool.submit(runner, p): p for p in todo}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            with write_lock:
                fout.write(json.dumps(res.to_json(), ensure_ascii=False) + "\n")
                fout.flush()
            completed += 1
            if completed % max(1, args.log_every) == 0 or completed == len(todo):
                errs = sum(1 for r in results if r.error)
                log.info(
                    "[%s/%s] %d/%d done (%d errors so far)",
                    dataset,
                    strategy,
                    completed,
                    len(todo),
                    errs,
                )
    wall = time.perf_counter() - t_cell0

    # Grade everything (cheap re-grade of resumed records keeps results consistent).
    log.info("[%s/%s] grading %d results...", dataset, strategy, len(results))
    grade_results(dataset, results)

    # Rewrite samples.jsonl with is_correct populated, sorted by index.
    results.sort(key=lambda r: r.index)
    with samples_path.open("w") as fout:
        for r in results:
            fout.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")

    summary = CellSummary(dataset=dataset, strategy=strategy, config=config)
    summary.n_total = len(results)
    summary.wall_clock_s = wall
    for r in results:
        if r.error:
            summary.n_errors += 1
            continue
        if r.is_correct:
            summary.n_correct += 1
        summary.latencies.append(r.client_latency_s)
        if r.server_elapsed_s is not None:
            summary.server_elapsed.append(r.server_elapsed_s)
        if r.output_tokens is not None:
            summary.output_tokens.append(r.output_tokens)

    sj = summary.to_json()
    (cell_dir / "summary.json").write_text(json.dumps(sj, indent=2))
    log.info(
        "[%s/%s] DONE acc=%s n=%d errors=%d wall=%.1fs",
        dataset,
        strategy,
        sj["accuracy"],
        summary.n_total,
        summary.n_errors,
        wall,
    )
    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_report(summaries: List[CellSummary], out_dir: Path, args) -> None:
    rows = [s.to_json() for s in summaries]
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "args": {
                    "model": args.model,
                    "n": args.n,
                    "agg": args.agg,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "service_url": args.service_url,
                    "concurrency": args.concurrency,
                    "timeout": args.timeout,
                },
                "cells": rows,
            },
            indent=2,
        )
    )

    lines: List[str] = []
    lines.append("# Service PRM evaluation — CoT vs offline best-of-N (PRM)\n")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Generation model: `{args.model}` (OpenRouter)")
    lines.append(
        f"- PRM best-of-N: N={args.n}, aggregation=`{args.agg}`, "
        f"service=`{args.service_url}`"
    )
    lines.append(
        f"- Sampling: temperature={args.temperature}, "
        f"max_tokens={args.max_tokens}\n"
    )

    lines.append("## Accuracy & timing\n")
    lines.append(
        "| Dataset | Strategy | N | Acc | Correct/Scored | Errors | "
        "Wall (s) | Latency/prob med (s) | Server elapsed med (s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cfg = r["config"]
        n = cfg.get("n", 1)
        acc = f"{r['accuracy']*100:.1f}%" if r["accuracy"] is not None else "—"
        lat = r["client_latency_s"]["median"]
        srv = r["server_elapsed_s"]["median"]
        lines.append(
            f"| {r['dataset']} | {cfg.get('strategy', r['strategy'])} | {n} | {acc} | "
            f"{r['n_correct']}/{r['n_scored']} | {r['n_errors']} | "
            f"{r['wall_clock_s']:.0f} | {lat if lat is not None else '—'} | "
            f"{srv if srv is not None else '—'} |"
        )

    # Per-dataset CoT vs PRM delta
    lines.append("\n## CoT vs PRM (per dataset)\n")
    by_ds: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["strategy"]] = r
    lines.append(
        "| Dataset | CoT acc | PRM acc | Δ (pp) | CoT wall (s) | PRM wall (s) | "
        "PRM/CoT wall ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for ds, cells in by_ds.items():
        cot = cells.get("cot")
        prm = cells.get("prm")
        cot_acc = cot["accuracy"] if cot else None
        prm_acc = prm["accuracy"] if prm else None
        delta = (
            f"{(prm_acc - cot_acc) * 100:+.1f}"
            if (cot_acc is not None and prm_acc is not None)
            else "—"
        )
        cot_wall = cot["wall_clock_s"] if cot else None
        prm_wall = prm["wall_clock_s"] if prm else None
        ratio = (
            f"{prm_wall / cot_wall:.1f}×"
            if (cot_wall and prm_wall and cot_wall > 0)
            else "—"
        )
        lines.append(
            f"| {ds} | {cot_acc*100:.1f}% | {prm_acc*100:.1f}% | {delta} | "
            f"{cot_wall:.0f} | {prm_wall:.0f} | {ratio} |"
            if (cot_acc is not None and prm_acc is not None)
            else (
                f"| {ds} | "
                f"{cot_acc*100:.1f}% | — | — | {cot_wall if cot_wall else '—'} | — | — |"
                if cot_acc is not None
                else f"| {ds} | — | "
                f"{prm_acc*100:.1f}% | — | — | {prm_wall if prm_wall else '—'} | — |"
            )
        )

    lines.append(
        "\n_See `summary.json` and each `*/samples.jsonl` for per-problem detail._\n"
    )
    (out_dir / "report.md").write_text("\n".join(lines))
    log.info("Report written to %s", out_dir / "report.md")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoT vs PRM offline best-of-N on Gaokao + HumanEval via the service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["gaokao", "humaneval"],
        choices=list(DATASET_LOADERS.keys()),
    )
    p.add_argument(
        "--strategies", nargs="+", default=["cot", "prm"], choices=["cot", "prm"]
    )
    p.add_argument(
        "--subset-gaokao",
        type=int,
        default=385,
        help="Number of Gaokao problems (full set = 385).",
    )
    p.add_argument(
        "--subset-humaneval",
        type=int,
        default=164,
        help="Number of HumanEval+ problems (full set = 164).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Global cap on problems per dataset (overrides --subset-* if smaller).",
    )
    p.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="OpenRouter model slug for generation.",
    )
    p.add_argument("--n", type=int, default=8, help="N trajectories for PRM best-of-N.")
    p.add_argument(
        "--agg",
        default="min",
        choices=["min", "mean", "max", "product", "last"],
        help="PRM per-step -> trajectory score aggregation.",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument(
        "--service-url",
        default="http://localhost:8001",
        help="ThinkBooster service base URL (PRM path is appended).",
    )
    p.add_argument(
        "--openrouter-base",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter base URL for the CoT baseline.",
    )
    p.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Env var holding the OpenRouter key.",
    )
    p.add_argument(
        "--concurrency", type=int, default=8, help="Parallel in-flight requests."
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts per request (exponential backoff).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request client timeout in seconds (OpenAI client). Raise for long "
        "best-of-N on long-output datasets (e.g. 1800) to avoid timeouts.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip problems already present (non-errored) in samples.jsonl.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output dir (default: outputs/service_prm_eval/<timestamp>).",
    )
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load datasets and print counts, then exit (no API calls).",
    )
    return p.parse_args(argv)


def load_env_key(args) -> str:
    """Resolve the OpenRouter key from env / repo .env without printing it."""
    # Best-effort: load .env at repo root if python-dotenv is available.
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass
    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise SystemExit(
            f"OpenRouter key not found in env var '{args.api_key_env}'. "
            f"Set it (e.g. in {REPO_ROOT/'.env'}) before running."
        )
    return key


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(
        args.out
        or REPO_ROOT
        / "outputs"
        / "service_prm_eval"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_dir)

    # Load datasets up front.
    subset_map = {"gaokao": args.subset_gaokao, "humaneval": args.subset_humaneval}
    problems_by_ds: Dict[str, List[Problem]] = {}
    for ds in args.datasets:
        subset = subset_map[ds]
        if args.limit is not None:
            subset = min(subset, args.limit) if subset is not None else args.limit
        problems_by_ds[ds] = DATASET_LOADERS[ds](subset)

    if args.dry_run:
        for ds, probs in problems_by_ds.items():
            log.info(
                "[dry-run] %s: %d problems; sample question:\n%s",
                ds,
                len(probs),
                probs[0].question[:300] if probs else "(none)",
            )
        return 0

    api_key = load_env_key(args)

    summaries: List[CellSummary] = []
    for ds in args.datasets:
        for strat in args.strategies:
            log.info("=== Running cell: dataset=%s strategy=%s ===", ds, strat)
            summary = run_cell(ds, strat, problems_by_ds[ds], args, out_dir, api_key)
            summaries.append(summary)
            # checkpoint the global report after each cell
            write_report(summaries, out_dir, args)

    log.info("All cells complete. Results in %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
