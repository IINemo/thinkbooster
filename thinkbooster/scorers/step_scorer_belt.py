"""BELT belief scoring through the published native probe runtime."""

import threading

from thinkbooster.belt.runtime import (
    DEFAULT_CHECKPOINT,
    BeltRuntime,
    checkpoint_manifest,
)

from .step_scorer_base import CandidateScore, StepScorerBase


class StepScorerBELT(StepScorerBase):
    """Rank candidates by one belief and expose full signals for stopping."""

    def __init__(
        self,
        runtime=None,
        checkpoint_repo=DEFAULT_CHECKPOINT,
        seed=0,
        device="cuda:0",
        dtype="bfloat16",
        cache_dir=None,
        hs_chunk_size=8192,
        target="q",
    ):
        super().__init__(name="belt")
        if target not in ("q", "v", "r", "stability"):
            raise ValueError("BELT ranking target must be q, v, r or stability")
        self.target = target
        self._runtime = runtime
        self._load_lock = threading.Lock()
        self._runtime_options = dict(
            checkpoint_repo=checkpoint_repo,
            seed=seed,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
            hs_chunk_size=hs_chunk_size,
        )

    @property
    def checkpoint_sha256(self):
        if self._runtime is not None:
            return self._runtime.checkpoint_sha256
        _, record = checkpoint_manifest(
            self._runtime_options["checkpoint_repo"],
            self._runtime_options["seed"],
            self._runtime_options["cache_dir"],
        )
        return record["sha256"]

    @property
    def runtime(self):
        with self._load_lock:
            if self._runtime is None:
                self._runtime = BeltRuntime.from_pretrained(**self._runtime_options)
            return self._runtime

    def score_prefix(self, chat, trajectory):
        return self.runtime.score_prefix(chat, trajectory)

    def score_candidates_detailed(self, chat, candidates, **kwargs):
        history = list(kwargs.get("trajectory") or [])
        result = []
        for candidate in candidates:
            signal = self.score_prefix(chat, history + [candidate])
            result.append(
                CandidateScore(
                    candidate_text=candidate,
                    claim_scores=[signal.beliefs[self.target]],
                    aggregate_scores={},
                    metadata={
                        "scorer_type": "belt",
                        "beliefs": dict(signal.beliefs),
                        "prefix_tokens": signal.prefix_tokens,
                        "checkpoint_sha256": signal.checkpoint_sha256,
                        "ranking_target": self.target,
                    },
                )
            )
        return result

    def score_trajectory(self, chat, trajectory, **kwargs):
        return [
            self.score_prefix(chat, trajectory[: i + 1]).beliefs[self.target]
            for i in range(len(trajectory))
        ]

    def score_trajectories_batch(
        self, chats, trajectories, sample_ids=None, trajectory_ids=None, **kwargs
    ):
        if len(chats) != len(trajectories):
            raise ValueError("BELT needs one chat per trajectory")
        return [
            self.score_trajectory(chat, trajectory, **kwargs)
            for chat, trajectory in zip(chats, trajectories)
        ]

    def cleanup(self):
        if self._runtime is not None:
            self._runtime.cleanup()
            self._runtime = None
