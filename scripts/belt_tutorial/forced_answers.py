"""Force an answer at every BELT decision point of a never-stop run.

For each logged decision, the reasoning prefix up to that block is closed with
``</think>\\n\\n\\boxed{`` and decoded until ``}`` (at most 32 tokens), as in the
BELT paper's forced-answer protocol (appendix A.3). The answers are graded with
ThinkBooster's exact-match evaluator. This gives, per trajectory, when a correct
answer first became available, and lets any stopping threshold be read off the
same trajectories.

    python scripts/belt_tutorial/forced_answers.py \
        --run <root>/runs/math500/natural/seed1 --out forced.jsonl
"""

import argparse
import json
import os
from pathlib import Path

# The evaluator imports initialize CUDA; vLLM must then spawn its engine.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from thinkbooster.evaluation.exact_match import EvaluatorExactMatch

FORCE_SUFFIX = "</think>\n\n\\boxed{"


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
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=32, stop=["}"],
        include_stop_str_in_output=False, seed=int(config.system.seed),
    )
    outputs = llm.generate([j[2] for j in jobs], params)

    by_index = {s["index"]: s for s in samples}
    with args.out.open("w") as stream:
        for (index, k, _), output in zip(jobs, outputs):
            sample = by_index[index]
            answer = output.outputs[0].text.strip()
            closed = output.outputs[0].finish_reason == "stop"
            correct = bool(
                closed
                and evaluator._score_single(
                    (sample["question"], "\\boxed{" + answer + "}", str(sample["gold_answer"]))
                )
            )
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "decision": k,
                        "forced_answer": answer,
                        "closed": closed,
                        "forced_tokens": len(output.outputs[0].token_ids),
                        "correct": correct,
                    }
                )
                + "\n"
            )
    print(f"{len(jobs)} forced answers -> {args.out}")


if __name__ == "__main__":
    main()
