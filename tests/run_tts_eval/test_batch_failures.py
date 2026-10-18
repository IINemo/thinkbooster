import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def runner():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("run_tts_eval")


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    import wandb

    monkeypatch.setattr(wandb, "run", None)


def output():
    return {
        "trajectory": "answer",
        "extracted_answer": "4",
        "steps": [],
        "completed": True,
        "metadata": {},
        "token_stats": {"tflops": 0.0, "prm_tflops": 0.0},
    }


def invoke(runner, path, generate, count=1, checkpoint_batch_size=1):
    return runner._generate_trajectories_batch(
        results=[],
        save_path=path,
        strategy=SimpleNamespace(generate_trajectories_batch=generate),
        dataset=[{"question": f"q{i}", "answer": "4"} for i in range(count)],
        processed_indices=set(),
        prompt_template="",
        system_prompt="",
        question_field="question",
        answer_field="answer",
        phase1_evaluators={},
        save_path_file=path / "results.json",
        sample_metrics_path=path / "metrics.jsonl",
        checkpoint_batch_size=checkpoint_batch_size,
    )


def test_generation_failure_escapes_with_original_exception(runner, tmp_path):
    failure = RuntimeError("generation failed")

    def generate(*args, **kwargs):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        invoke(runner, tmp_path, generate)
    assert caught.value is failure
    assert json.loads((tmp_path / "results.json").read_text()) == []


@pytest.mark.parametrize("progressive", [False, True])
def test_failure_preserves_completed_and_progressive_results(
    runner, tmp_path, progressive
):
    failure = RuntimeError("second chunk failed")

    def generate(requests, indices, save_callback):
        if indices == [0]:
            return [output()]
        if progressive:
            save_callback([output()], phase="post_generation")
        raise failure

    with pytest.raises(RuntimeError) as caught:
        invoke(runner, tmp_path, generate, count=2)
    assert caught.value is failure
    saved = json.loads((tmp_path / "results.json").read_text())
    assert [row["index"] for row in saved] == ([0, 1] if progressive else [0])
    assert saved[0]["generated_trajectory"] == "answer"
    if progressive:
        assert saved[1]["scoring_phase"] == "post_generation"
        assert saved[1]["is_correct"] is None
    batch_rows = [
        json.loads(line)
        for line in (tmp_path / "batch_results.jsonl").read_text().splitlines()
    ]
    assert [row["index"] for row in batch_rows] == [0]


def test_first_chunk_progressive_snapshot_is_not_overwritten(runner, tmp_path):
    failure = RuntimeError("scoring failed after generation")

    def generate(requests, indices, save_callback):
        save_callback([output()], phase="post_generation")
        raise failure

    with pytest.raises(RuntimeError) as caught:
        invoke(runner, tmp_path, generate)
    assert caught.value is failure
    saved = json.loads((tmp_path / "results.json").read_text())
    assert len(saved) == 1
    assert saved[0]["scoring_phase"] == "post_generation"
    assert saved[0]["generated_trajectory"] == "answer"


def test_checkpoint_save_failure_does_not_replace_generation_error(
    runner, tmp_path, monkeypatch
):
    failure = RuntimeError("generation failed")

    def generate(*args, **kwargs):
        raise failure

    def save(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(runner, "save_results_json", save)
    with pytest.raises(RuntimeError) as caught:
        invoke(runner, tmp_path, generate)
    assert caught.value is failure


@pytest.mark.parametrize("result_count", [0, 2])
def test_result_count_mismatch_is_an_error(runner, tmp_path, result_count):
    def generate(*args, **kwargs):
        return [output() for _ in range(result_count)]

    with pytest.raises(RuntimeError, match=f"{result_count} results but expected 1"):
        invoke(runner, tmp_path, generate)
    assert not (tmp_path / "batch_results.jsonl").exists()
    assert not (tmp_path / "metrics.jsonl").exists()


@pytest.mark.parametrize("batch_size", [0, -1, True])
def test_invalid_checkpoint_batch_size_fails_before_generation(
    runner, tmp_path, batch_size
):
    calls = []
    with pytest.raises(ValueError, match="positive integer"):
        invoke(
            runner,
            tmp_path,
            lambda *args, **kwargs: calls.append(args),
            checkpoint_batch_size=batch_size,
        )
    assert calls == []


def test_uncaught_generation_failure_exits_nonzero(tmp_path):
    script = """
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from run_tts_eval import _generate_trajectories_batch
def generate(*args, **kwargs):
    raise RuntimeError("batch-generation-subprocess-failure")
path = Path(sys.argv[2])
_generate_trajectories_batch(
    results=[], save_path=path,
    strategy=SimpleNamespace(generate_trajectories_batch=generate),
    dataset=[{"question": "q", "answer": "4"}], processed_indices=set(),
    prompt_template="", system_prompt="", question_field="question",
    answer_field="answer", phase1_evaluators={},
    save_path_file=path / "results.json",
    sample_metrics_path=path / "metrics.jsonl", checkpoint_batch_size=1,
)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve().parents[2] / "scripts"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode != 0
    assert "RuntimeError: batch-generation-subprocess-failure" in result.stderr
