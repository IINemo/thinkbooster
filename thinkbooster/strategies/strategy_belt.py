"""End thinking using a fitted policy on frozen BELT signals."""

from thinkbooster.belt import BeltStoppingPolicy
from thinkbooster.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from thinkbooster.utils.answer_extraction import extract_answer

from .strategy_base import StrategyBase


def _completion_reason(candidate):
    reason = (candidate.other_data or {}).get("completion_reason")
    return reason.value if hasattr(reason, "value") else reason


def _answer_is_complete(candidate):
    return bool(
        candidate.is_trajectory_complete
        and (candidate.raw_text or candidate.text).strip()
        and getattr(candidate.output, "finish_reason", None) != "length"
        and _completion_reason(candidate)
        not in ("context_limit", "repetition_detected", "garbage_detected")
    )


class StrategyBELT(StrategyBase):
    """One reasoning trajectory per request, with explicit BELT stopping."""

    def __init__(
        self, step_generator, scorer, policy, max_steps=128, min_steps=1, query_stride=1
    ):
        if step_generator is None or not hasattr(scorer, "score_prefix"):
            raise ValueError("BELT requires a step generator and a BELT prefix scorer")
        for name in (
            "generate_step_candidates_batch",
            "generate_answer_candidates_batch",
        ):
            if not callable(getattr(step_generator, name, None)) or getattr(
                type(step_generator), name, None
            ) is getattr(StepCandidateGeneratorBase, name):
                raise ValueError(
                    "BELT stopping requires implemented batched step and answer APIs; "
                    "use VLLMStepGenerator or a compatible generator"
                )
        if (
            type(max_steps) is not int
            or type(min_steps) is not int
            or not 1 <= min_steps <= max_steps
        ):
            raise ValueError("Require 1 <= min_steps <= max_steps")
        if type(query_stride) is not int or query_stride < 1:
            raise ValueError("query_stride must be a positive integer")
        if not isinstance(policy, BeltStoppingPolicy):
            policy = BeltStoppingPolicy.from_file(policy)
        self.step_generator = step_generator
        self.scorer = scorer
        self.policy = policy
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.query_stride = query_stride
        self.thinking_mode = bool(
            getattr(
                step_generator,
                "thinking_mode",
                getattr(step_generator, "use_thinking_mode", False),
            )
        )

    def generate_trajectories_batch(
        self, requests, sample_indices=None, save_callback=None
    ):
        if sample_indices is None:
            sample_indices = list(range(len(requests)))
        if len(sample_indices) != len(requests):
            raise ValueError("sample_indices must match requests")
        if len(set(sample_indices)) != len(sample_indices):
            raise ValueError("sample_indices must be unique within a BELT batch")
        if not requests:
            return []
        self._check_cancelled()
        self.step_generator.reset_per_sample_stats()
        model = getattr(self.step_generator, "model", None)
        if model is not None and hasattr(model, "reset_hs_step_cache"):
            model.reset_hs_step_cache()
        trajectories = [[] for _ in requests]
        decisions = [[] for _ in requests]
        reasons = [None for _ in requests]
        completed = [False for _ in requests]
        answer_text = [None for _ in requests]
        reasoning_steps = [0 for _ in requests]
        injected_control_text = [[] for _ in requests]
        active = list(range(len(requests)))

        for step in range(1, self.max_steps + 1):
            if not active:
                break
            self._check_cancelled()
            outputs = self.step_generator.generate_step_candidates_batch(
                requests=[requests[i] for i in active],
                trajectories=[trajectories[i] for i in active],
                candidates_per_step=1,
                compute_uncertainty=False,
                sample_ids=[sample_indices[i] for i in active],
            )
            if len(outputs) != len(active):
                raise RuntimeError("BELT generator returned the wrong batch size")
            continuing, answering = [], []
            for i, candidates in zip(active, outputs):
                self._check_cancelled()
                if not candidates:
                    reasons[i] = "generation_failed"
                    continue
                if len(candidates) != 1:
                    raise RuntimeError("BELT expects one candidate per active sample")
                candidate = candidates[0]
                trajectories[i].append(candidate)
                reasoning_steps[i] += 1
                reason = _completion_reason(candidate)
                if reason in (
                    "context_limit",
                    "repetition_detected",
                    "garbage_detected",
                ):
                    reasons[i] = reason
                    continue
                text = candidate.raw_text or candidate.text
                closed_thinking = (
                    "</think>" in text and not text.split("</think>", 1)[1].strip()
                )
                if (
                    candidate.is_thinking_complete
                    and not candidate.is_trajectory_complete
                ) or (self.thinking_mode and closed_thinking):
                    reasons[i] = "thinking_complete"
                    answering.append(i)
                    continue
                if candidate.is_trajectory_complete:
                    completed[i] = _answer_is_complete(candidate)
                    reasons[i] = (
                        "natural_completion" if completed[i] else "answer_incomplete"
                    )
                    continue
                if candidate.is_thinking_complete:
                    reasons[i] = "thinking_complete"
                    answering.append(i)
                    continue
                if step >= self.min_steps and step % self.query_stride == 0:
                    signal = self.scorer.score_prefix(requests[i], trajectories[i])
                    value = self.policy.decision_logit(signal)
                    stop = value >= self.policy.threshold
                    decisions[i].append(
                        {
                            "step": step,
                            "prefix_tokens": signal.prefix_tokens,
                            "beliefs": dict(signal.beliefs),
                            "policy_logit": value,
                            "threshold": self.policy.threshold,
                            "stop": stop,
                            "checkpoint_sha256": signal.checkpoint_sha256,
                        }
                    )
                    if stop:
                        reasons[i] = "belt_stop"
                        answering.append(i)
                        continue
                continuing.append(i)
            if answering:
                self._check_cancelled()
                for i in answering:
                    if (
                        self.thinking_mode
                        and "</think>"
                        not in convert_trajectory_to_string(trajectories[i])
                    ):
                        trajectories[i].append(
                            StepCandidate(
                                text="</think>",
                                token_ids=[],
                                is_complete=True,
                                is_trajectory_complete=False,
                                is_thinking_complete=True,
                                other_data={"belt_control_text": True},
                            )
                        )
                        injected_control_text[i].append("</think>")
                answers = self.step_generator.generate_answer_candidates_batch(
                    requests=[requests[i] for i in answering],
                    trajectories=[trajectories[i] for i in answering],
                    candidates_per_step=1,
                    sample_ids=[sample_indices[i] for i in answering],
                )
                if len(answers) != len(answering):
                    raise RuntimeError(
                        "BELT answer generator returned the wrong batch size"
                    )
                for i, candidates in zip(answering, answers):
                    if not candidates:
                        reasons[i] = "answer_generation_failed"
                        continue
                    if len(candidates) != 1:
                        raise RuntimeError("BELT expects one answer per sample")
                    answer = candidates[0]
                    trajectories[i].append(answer)
                    answer_text[i] = answer.raw_text or answer.text
                    completed[i] = _answer_is_complete(answer)
                    if not completed[i]:
                        reasons[i] = "answer_incomplete"
            active = continuing

        for i in active:
            reasons[i] = "max_steps"
        results = []
        for i, trajectory in enumerate(trajectories):
            text = convert_trajectory_to_string(trajectory)
            results.append(
                {
                    "trajectory": text,
                    "extracted_answer": extract_answer(text),
                    "steps": trajectory,
                    "answer_step": answer_text[i],
                    "reasoning_steps": reasoning_steps[i],
                    "validity_scores": [d["beliefs"]["q"] for d in decisions[i]],
                    "completed": completed[i],
                    "completion_reason": reasons[i],
                    "context_limit_hit": any(
                        _completion_reason(s) == "context_limit" for s in trajectory
                    ),
                    "max_steps_hit": reasons[i] == "max_steps",
                    "token_stats": self.step_generator.get_sample_stats_for(
                        sample_indices[i]
                    ),
                    "metadata": {
                        "strategy": "belt",
                        "feature_space": self.policy.feature_space,
                        "hidden_dtype": self.policy.hidden_dtype,
                        "checkpoint_sha256": self.policy.checkpoint_sha256,
                        "decisions": decisions[i],
                        "injected_control_text": injected_control_text[i],
                    },
                }
            )
        if save_callback is not None:
            save_callback(results, phase="belt")
        return results

    def cleanup(self):
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()
