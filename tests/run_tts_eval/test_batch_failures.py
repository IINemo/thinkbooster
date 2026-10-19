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


def test_resume_regenerates_samples_left_as_progressive_snapshots(runner, tmp_path):
    # Offline BoN saves unscored candidates before scoring; if scoring fails,
    # --resume must not treat that snapshot as a finished result.
    def fail_after_snapshot(requests, indices, save_callback):
        if indices == [0]:
            return [output()]
        save_callback([output()], phase="post_generation")
        raise RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        invoke(runner, tmp_path, fail_after_snapshot, count=2)
    results, processed = runner.load_results_json(tmp_path / "results.json")
    assert processed == {0}
    assert [row["index"] for row in results] == [0]

    regenerated = []

    def generate(requests, indices, save_callback):
        regenerated.extend(indices)
        return [output() for _ in indices]

    final = runner._generate_trajectories_batch(
        results=results,
        save_path=tmp_path,
        strategy=SimpleNamespace(generate_trajectories_batch=generate),
        dataset=[{"question": f"q{i}", "answer": "4"} for i in range(2)],
        processed_indices=processed,
        prompt_template="",
        system_prompt="",
        question_field="question",
        answer_field="answer",
        phase1_evaluators={},
        save_path_file=tmp_path / "results.json",
        sample_metrics_path=tmp_path / "metrics.jsonl",
        checkpoint_batch_size=1,
    )
    assert regenerated == [1]
    assert sorted(row["index"] for row in final) == [0, 1]
    assert all("scoring_phase" not in row for row in final)


def test_empty_dataset_with_default_batch_size_is_a_no_op(runner, tmp_path):
    # generate_trajectories() defaults checkpoint_batch_size to len(dataset).
    calls = []
    results = invoke(
        runner,
        tmp_path,
        lambda *args, **kwargs: calls.append(args),
        count=0,
        checkpoint_batch_size=0,
    )
    assert results == [] and calls == []


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


def test_resume_after_kill_leaves_one_jsonl_line_per_sample(
    runner, tmp_path, monkeypatch
):
    # A hard kill after a chunk's JSONL append but before results.json is saved
    # leaves a line for a sample that resume will regenerate and append again.
    real_save = runner.save_results_json

    def killed_on_second_chunk(results, path):
        if any(row.get("index") == 1 for row in results):
            raise KeyboardInterrupt
        real_save(results, path)

    monkeypatch.setattr(runner, "save_results_json", killed_on_second_chunk)
    with pytest.raises(KeyboardInterrupt):
        invoke(runner, tmp_path, lambda r, i, save_callback: [output()], count=2)
    monkeypatch.setattr(runner, "save_results_json", real_save)
    jsonl = tmp_path / "batch_results.jsonl"
    with open(jsonl, "a") as stream:
        stream.write('{"index": 1, "traj')  # torn by the kill
    results, processed = runner.load_results_json(tmp_path / "results.json")
    assert processed == {0}

    assert runner.prune_jsonl_rows(jsonl, processed) == 2
    final = runner._generate_trajectories_batch(
        results=results,
        save_path=tmp_path,
        strategy=SimpleNamespace(
            generate_trajectories_batch=lambda r, i, save_callback: [output()]
        ),
        dataset=[{"question": f"q{i}", "answer": "4"} for i in range(2)],
        processed_indices=processed,
        prompt_template="",
        system_prompt="",
        question_field="question",
        answer_field="answer",
        phase1_evaluators={},
        save_path_file=tmp_path / "results.json",
        sample_metrics_path=tmp_path / "metrics.jsonl",
        checkpoint_batch_size=1,
    )
    lines = [json.loads(x) for x in jsonl.read_text().splitlines()]
    assert sorted(row["index"] for row in final) == [0, 1]
    assert [row["index"] for row in lines] == [0, 1]


def test_prune_keeps_last_line_per_kept_sample(runner, tmp_path):
    path = tmp_path / "sample_metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sample_index": 0, "v": "old"},
                {"sample_index": 2, "v": "stale"},
                {"sample_index": 0, "v": "new"},
            ]
        )
        + "\n"
    )
    assert runner.prune_jsonl_rows(path, {0}, "sample_index") == 2
    assert [json.loads(x) for x in path.read_text().splitlines()] == [
        {"sample_index": 0, "v": "new"}
    ]
    assert runner.prune_jsonl_rows(path, {0}, "sample_index") == 0


def test_truncated_results_file_is_not_read_as_nothing_processed(runner, tmp_path):
    path = tmp_path / "results.json"
    runner.save_results_json([{"index": 0}], path)
    assert runner.results_file_readable(path)
    assert not path.with_name("results.json.tmp").exists()
    path.write_text('[{"index": 0}, {"ind')  # a kill mid-write, before the fix
    assert not runner.results_file_readable(path)
    assert runner.results_file_readable(tmp_path / "missing.json")
