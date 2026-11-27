"""Force an answer at every BELT decision point of a never-stop run.

For each logged decision, the reasoning prefix up to that block is closed with
``</think>\\n\\n\\boxed{`` and decoded for at most 48 tokens, as in the BELT
paper's forced-answer protocol (appendix A.3). The paper's answers are integers
and it stops at the first ``}``; LaTeX answers nest braces (``\\frac{\\pi}{2}``),
so the answer here is the text up to the brace that closes ``\\boxed{``.
The answers are graded with ThinkBooster's exact-match evaluator. This gives, per trajectory, when a correct
answer first became available, and lets any stopping threshold be read off the
same trajectories.

    python scripts/belt_tutorial/forced_answers.py \
        --run <root>/runs/math500/natural/seed1 --out forced.jsonl
"""

import argparse
import json
import os
import signal
from pathlib import Path

# The evaluator imports initialize CUDA; vLLM must then spawn its engine.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from thinkbooster.evaluation.exact_match import EvaluatorExactMatch

FORCE_SUFFIX = "</think>\n\n\\boxed{"


def grade(evaluator, question, answer, gold, seconds=5):
    """Exact-match grade with a time limit; symbolic comparison can hang.

    Returns (correct, timed_out). On timeout, falls back to string equality.
    """

    def expire(*_):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(seconds)
    try:
        return bool(evaluator._score_single((question, "\\boxed{" + answer + "}", gold))), False
    except TimeoutError:
        return answer.strip() == gold.strip(), True
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    config = OmegaConf.load(args.run / ".hydra" / "config.yaml")
    samples = json.loads((args.run / "results.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_path, revision=config.model.get("revision")
    )
    evaluator = EvaluatorExactMatch(
        dataset_answer_format=config.dataset.answer_format,
        data_name=config.dataset.data_name,
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
        for k, decision in enumerate((sample.get("metadata") or {}).get("decisions") or []):
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
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=48,
        seed=int(config.system.seed),
    )
    outputs = llm.generate([j[2] for j in jobs], params)

    by_index = {s["index"]: s for s in samples}
    with args.out.open("w") as stream:
        for (index, k, _), output in zip(jobs, outputs):
            sample = by_index[index]
            content = boxed_content(output.outputs[0].text)
            closed = content is not None
            answer = (content or output.outputs[0].text).strip()
            correct, timed_out = (
                grade(evaluator, sample["question"], answer, str(sample["gold_answer"]))
                if closed
                else (False, False)
            )
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "decision": k,
                        "forced_answer": answer,
                        "closed": closed,
                        # Charged tokens end at the closing brace, as if decoding
                        # had stopped there.
                        "forced_tokens": (
                            len(tokenizer.encode(content + "}", add_special_tokens=False))
                            if closed
                            else len(output.outputs[0].token_ids)
                        ),
                        "correct": correct,
                        "grading_timeout": timed_out,
                    }
                )
                + "\n"
            )
    print(f"{len(jobs)} forced answers -> {args.out}")


if __name__ == "__main__":
    main()
