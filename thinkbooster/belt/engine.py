"""Serve Qwen3-1.7B with BELT early stopping on vLLM.

``BeltEngine`` is the deployment path: it loads the model into vLLM once,
attaches the BELT probe and a fitted stopping policy, and answers questions
with BELT deciding, every few reasoning blocks, whether the model has thought
enough. A stop closes ``</think>`` and lets the model write its final answer,
so the saved tokens are never generated.

    from thinkbooster.belt.engine import BeltEngine

    engine = BeltEngine("belt-policy.json")          # backend="hf" by default
    for out in engine.generate(["What is 17 + 25?"]):
        print(out.answer, out.thinking_tokens, out.stopped_early)

The scorer can read the probe's hidden states two ways:

* ``backend="hf"`` (default): a second, Transformers copy of the backbone
  re-reads the prefix at each decision. Any batch size.
* ``backend="vllm"``: the states are captured by the generating engine, so a
  decision costs a few milliseconds. vLLM then runs in eager mode and takes at
  most ``min(16, 131072 // max_context)`` sequences per batch (9 at 14,336).

The engine reproduces the configuration of ``scripts/run_tts_eval.py`` with
``config/experiments/belt/tutorial/*_qwen3_1p7b.yaml``: BELT's protocol of a
decision at every fourth blank-line block from block 4 to block 128.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from thinkbooster.belt.policy import BeltStoppingPolicy
from thinkbooster.belt.runtime import BACKBONE, BACKBONE_REVISION, DEFAULT_CHECKPOINT

# Qwen3 end-of-turn and end-of-text.
QWEN3_STOP_TOKEN_IDS = (151645, 151643)
PROMPT = (
    "Answer the following question. Reply with the answer only. "
    "Answer enclosed with \\boxed{{}}.\n<Question>: {question}"
)

Chat = List[Dict[str, str]]


@dataclass
class BeltResult:
    """One answered request."""

    answer: Optional[str]  # extracted \\boxed{} content, if any
    text: str  # the final answer the model wrote after thinking
    thinking_tokens: int  # reasoning tokens actually generated
    output_tokens: int  # all generated tokens, answer included
    stopped_early: bool  # True when BELT closed thinking
    stop_decision: Optional[int]  # 1-based decision index where BELT stopped
    completion_reason: Optional[str]
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: str = ""  # full thinking plus answer


class BeltEngine:
    """vLLM engine plus BELT probe and stopping policy, loaded once."""

    def __init__(
        self,
        policy: Union[str, BeltStoppingPolicy],
        *,
        backend: str = "hf",
        checkpoint_repo: str = DEFAULT_CHECKPOINT,
        probe_seed: int = 0,
        threshold: Optional[float] = None,
        max_context: int = 14336,
        gpu_memory_utilization: float = 0.6,
        scorer_device: str = "cuda:0",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        max_step_tokens: int = 1024,
        max_new_tokens: int = 2048,
        min_steps: int = 4,
        query_stride: int = 4,
        max_blocks: int = 128,
        seed: int = 0,
    ):
        if backend not in ("hf", "vllm"):
            raise ValueError("backend must be 'hf' or 'vllm'")
        from vllm import LLM

        from thinkbooster.belt.vllm_wire import MAX_CAPTURE_TOKENS
        from thinkbooster.generators.vllm import VLLMStepGenerator
        from thinkbooster.scorers.step_scorer_belt import StepScorerBELT
        from thinkbooster.step_boundary_detectors import ThinkingMarkerDetector
        from thinkbooster.strategies.strategy_belt import StrategyBELT

        self.backend = backend
        self.max_context = max_context
        if not isinstance(policy, BeltStoppingPolicy):
            policy = BeltStoppingPolicy.from_file(policy)
        if threshold is not None:
            policy = dataclasses.replace(policy, threshold=float(threshold))

        # The scorer is created first so a policy that does not match the probe
        # fails before vLLM takes the GPU.
        scorer = StepScorerBELT(
            checkpoint_repo=checkpoint_repo, seed=probe_seed, device=scorer_device
        )
        policy.validate_checkpoint(scorer.checkpoint_sha256)

        llm_kwargs = dict(
            model=BACKBONE,
            revision=BACKBONE_REVISION,
            tokenizer_revision=BACKBONE_REVISION,
            dtype="bfloat16",
            max_model_len=max_context,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=seed,
        )
        if backend == "vllm":
            if max_context > MAX_CAPTURE_TOKENS:
                raise ValueError(
                    f"backend='vllm' needs max_context <= {MAX_CAPTURE_TOKENS}"
                )
            llm_kwargs.update(
                enforce_eager=True,
                worker_extension_cls=(
                    "thinkbooster.belt.vllm_worker.BeltHiddenStatesExtension"
                ),
            )
        self.llm = LLM(**llm_kwargs)
        model = self.llm
        if backend == "vllm":
            from thinkbooster.belt.vllm_backend import (
                DEFAULT_CACHE_ENTRIES,
                BeltVLLMBackend,
            )

            model = BeltVLLMBackend(self.llm, max_cache_entries=DEFAULT_CACHE_ENTRIES)
            scorer.bind_feature_provider(
                model,
                model.get_tokenizer(),
                getattr(getattr(model, "model_config", None), "hf_config", None),
            )
        self.batch_limit = (
            min(16, MAX_CAPTURE_TOKENS // max_context) if backend == "vllm" else None
        )

        # BELT overrides the stop strings per call (blank line, </think>);
        # min_step_tokens must be 0 so short blocks are not merged.
        detector = ThinkingMarkerDetector(
            min_step_tokens=0, max_step_tokens=max_step_tokens
        )
        self.step_generator = VLLMStepGenerator(
            model=model,
            thinking_mode=True,
            detector=detector,
            stop_token_ids=list(QWEN3_STOP_TOKEN_IDS),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_context_budget=max_context,
            disable_thinking_mode=False,
            seed=seed,
        )
        self.strategy = StrategyBELT(
            step_generator=self.step_generator,
            scorer=scorer,
            policy=policy,
            max_steps=1024,
            min_steps=min_steps,
            query_stride=query_stride,
            max_blocks=max_blocks,
        )
        self.policy = policy

    @staticmethod
    def chat(question: str, prompt: str = PROMPT) -> Chat:
        """The user turn BELT's probe was trained on."""
        return [{"role": "user", "content": prompt.format(question=question)}]

    def strategy_at(self, threshold: Optional[float]):
        """The BELT strategy at another stop threshold, on the same engine.

        A very high threshold (``1e9``) never stops early: the model thinks
        until it closes ``</think>`` itself, the baseline BELT is measured
        against. Policy thresholds must be finite.
        """
        from thinkbooster.strategies.strategy_belt import StrategyBELT

        if threshold is None:
            return self.strategy
        s = self.strategy
        return StrategyBELT(
            step_generator=s.step_generator,
            scorer=s.scorer,
            policy=dataclasses.replace(self.policy, threshold=float(threshold)),
            max_steps=s.max_steps,
            min_steps=s.min_steps,
            query_stride=s.query_stride,
            max_blocks=s.max_blocks,
        )

    def generate(
        self,
        requests: Sequence[Union[str, Chat]],
        threshold: Optional[float] = None,
    ) -> List[BeltResult]:
        """Answer questions (plain strings) or chats, with BELT early stopping."""
        strategy = self.strategy_at(threshold)
        chats = [self.chat(r) if isinstance(r, str) else list(r) for r in requests]
        step = self.batch_limit or len(chats) or 1
        out: List[BeltResult] = []
        for start in range(0, len(chats), step):
            batch = chats[start : start + step]
            results = strategy.generate_trajectories_batch(
                batch, sample_indices=list(range(start, start + len(batch)))
            )
            out.extend(self._result(r) for r in results)
        return out

    @staticmethod
    def _result(r: Dict[str, Any]) -> BeltResult:
        meta = r.get("metadata") or {}
        decisions = meta.get("decisions") or []
        stop = next((k for k, d in enumerate(decisions) if d.get("stop")), None)
        stats = r.get("token_stats") or {}
        # The first reasoning_steps steps are the thinking blocks; the closing
        # control text and the answer follow them.
        steps = (r.get("steps") or [])[: r.get("reasoning_steps") or 0]
        thinking = sum(len(getattr(s, "token_ids", None) or []) for s in steps)
        return BeltResult(
            answer=r.get("extracted_answer"),
            text=r.get("answer_step") or "",
            thinking_tokens=thinking,
            output_tokens=int(stats.get("output_tokens", 0) or 0),
            stopped_early=r.get("completion_reason") == "belt_stop",
            stop_decision=None if stop is None else stop + 1,
            completion_reason=r.get("completion_reason"),
            decisions=decisions,
            trajectory=r.get("trajectory", ""),
        )
