"""Force an answer at every BELT decision point of a never-stop run.

For each logged decision, the reasoning prefix up to that block is closed with
``</think>\\n\\n\\boxed{`` and decoded for at most 48 tokens, as in the BELT
paper's forced-answer protocol (appendix A.3). The paper's answers are integers
and it stops at the first ``}``; LaTeX answers nest braces (``\\frac{\\pi}{2}``),
so the answer here is the text up to the brace that closes ``\\boxed{``.
The answers are graded with ThinkBooster's exact-match evaluator, as the
evaluation runner grades extracted answers, in worker processes with a time
limit. This gives, per trajectory, when a correct answer first became available, and lets any stopping threshold be read off the
same trajectories.

    python scripts/belt_tutorial/forced_answers.py \
        --run <root>/runs/math500/natural/seed1 --out forced.jsonl
"""

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from thinkbooster.evaluation.exact_match import EvaluatorExactMatch

FORCE_SUFFIX = "</think>\n\n\\boxed{"


_EVALUATOR = None


def _init_worker(answer_format, data_name):
    global _EVALUATOR
    _EVALUATOR = EvaluatorExactMatch(
        dataset_answer_format=answer_format, data_name=data_name
    )


def _grade_one(pair):
    answer, gold = pair
    # Same call as the evaluation runner, which grades the extracted answer.
    return bool(_EVALUATOR._score_single(("", answer, gold), pre_extracted=True))


def grade_all(pairs, answer_format, data_name, seconds=10, workers=16):
    """Grade (answer, gold) pairs in worker processes with a hard time limit.

    Symbolic comparison can hang without returning, so a pair that exceeds the
    limit is graded by string equality and flagged; its worker pool is replaced.
    Returns {pair: (correct, timed_out)}.
    """
    import multiprocessing as mp

    todo = list(dict.fromkeys(pairs))
    results = {}
    ctx = mp.get_context("fork")
    while todo:
        pool = ctx.Pool(
            workers, initializer=_init_worker, initargs=(answer_format, data_name)
        )
        pending = [(pair, pool.apply_async(_grade_one, (pair,))) for pair in todo]
        todo = []
        broken = False
        for pair, job in pending:
            if broken:
                todo.append(pair)
                continue
            try:
                results[pair] = (job.get(timeout=seconds), False)
            except mp.TimeoutError:
                results[pair] = (pair[0].strip() == pair[1].strip(), True)
                broken = True  # a worker may be stuck; restart for the rest
        pool.terminate()
        pool.join()
        # Jobs that finished before the restart need not run again.
        todo = [p for p in todo if p not in results]
    return results


def boxed_content(text):
    """Text up to the brace closing an already opened \\boxed{, or None."""
    depth = 0
    for i, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return text[:i]
            depth -= 1
    return None


def prompt_text(config, question):
    template = config.dataset.get("prompt_template") or "{question}"
    return template.format(question=question) if "{question}" in template else question


def regrade(run, out):
    config = OmegaConf.load(run / ".hydra" / "config.yaml")
    by_index = {s["index"]: s for s in json.loads((run / "results.json").read_text())}
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    write_graded(rows, by_index, config, out)
    print(f"regraded {len(rows)} forced answers -> {out}")


def write_graded(rows, by_index, config, out):
    pairs = [
        (row["forced_answer"], str(by_index[row["index"]]["gold_answer"]))
        for row in rows
        if row["closed"]
    ]
    graded = grade_all(pairs, config.dataset.answer_format, config.dataset.data_name)
    for row in rows:
        pair = (row["forced_answer"], str(by_index[row["index"]]["gold_answer"]))
        row["correct"], row["grading_timeout"] = (
            graded[pair] if row["closed"] else (False, False)
        )
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="Regrade the answers already in --out instead of generating them",
    )
    args = parser.parse_args()

    if args.regrade:
        regrade(args.run, args.out)
        return

    # The evaluator imports initialize CUDA; vLLM must then spawn its engine.
    # vLLM reads this when it starts the engine, so setting it here is in time.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    config = OmegaConf.load(args.run / ".hydra" / "config.yaml")
    samples = json.loads((args.run / "results.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_path, revision=config.model.get("revision")
    )
    jobs = []
    for sample in samples:
        chat = [{"role": "user", "content": prompt_text(config, sample["question"])}]
        head = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        # raw_text keeps each block's trailing blank line; text strips it.
        steps = [
            s if isinstance(s, str) else (s.get("raw_text") or s.get("text") or "")
            for s in sample["steps"]
        ]
        for k, decision in enumerate(
            (sample.get("metadata") or {}).get("decisions") or []
        ):
            prefix = "".join(steps[: decision["step"]])
            jobs.append((sample["index"], k, head + prefix + FORCE_SUFFIX))

    llm = LLM(
        model=config.model.model_path,
        revision=config.model.get("revision"),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=int(config.model.max_context_budget) + 64,
        enable_prefix_caching=True,
        seed=int(config.system.seed),
    )
    params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=48,
        seed=int(config.system.seed),
    )
    outputs = llm.generate([j[2] for j in jobs], params)
    # Free the GPU before grading, which runs in forked CPU workers.
    del llm
    import gc

    gc.collect()

    by_index = {s["index"]: s for s in samples}
    rows = []
    for (index, k, _), output in zip(jobs, outputs):
        content = boxed_content(output.outputs[0].text)
        closed = content is not None
        answer = (content or output.outputs[0].text).strip()
        rows.append(
            {
                "index": index,
                "decision": k,
                "forced_answer": answer,
                "closed": closed,
                # Charged tokens end at the closing brace, as if decoding had
                # stopped there.
                "forced_tokens": (
                    len(tokenizer.encode(content + "}", add_special_tokens=False))
                    if closed
                    else len(output.outputs[0].token_ids)
                ),
            }
        )
    write_graded(rows, by_index, config, args.out)
    print(f"{len(jobs)} forced answers -> {args.out}")


if __name__ == "__main__":
    main()
