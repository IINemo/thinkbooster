"""Build the request run_tts_eval sends, for prompt checks in several tests."""

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


class _Captured(Exception):
    pass


def built_request(cfg, question):
    """The first request run_tts_eval builds for this config and question."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    runner = importlib.import_module("run_tts_eval")
    captured = []

    def capture(requests, indices, save_callback=None):
        captured.extend(requests)
        raise _Captured

    field = cfg.dataset.question_field
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(_Captured):
            runner._generate_trajectories_batch(
                results=[],
                save_path=Path(tmp),
                strategy=SimpleNamespace(generate_trajectories_batch=capture),
                dataset=[{field: question, cfg.dataset.answer_field: "4"}],
                processed_indices=set(),
                prompt_template=runner.dataset_prompt_template(cfg),
                system_prompt=cfg.dataset.system_prompt,
                question_field=field,
                answer_field=cfg.dataset.answer_field,
                phase1_evaluators={},
                save_path_file=Path(tmp) / "results.json",
                sample_metrics_path=Path(tmp) / "metrics.jsonl",
                checkpoint_batch_size=1,
            )
    return captured[0]
