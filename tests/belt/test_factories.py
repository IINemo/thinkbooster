import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from thinkbooster.belt.runtime import BACKBONE_REVISION
from thinkbooster.scorers import StepScorerBELT
from thinkbooster.strategies import StrategyBELT


@pytest.fixture(scope="module")
def factories():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("run_tts_eval")


def config():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        return compose(config_name="experiments/belt/math500_qwen3_1p7b")


@pytest.fixture(autouse=True)
def manifest_identity(monkeypatch):
    monkeypatch.setattr(
        "thinkbooster.scorers.step_scorer_belt.checkpoint_manifest",
        lambda *args: ({}, {"sha256": "a" * 64}),
    )


@pytest.fixture
def fitted_config(tmp_path):
    cfg = config()
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "schema": "thinkbooster-belt-policy/v1",
                "checkpoint_sha256": "a" * 64,
                "feature_space": "H",
                "weights": [1],
                "intercept": 0,
                "means": [0],
                "stds": [1],
                "threshold": 0,
            }
        )
    )
    cfg.strategy.policy_path = str(path)
    return cfg


def test_hydra_example_has_pinned_model_and_no_paid_judge():
    cfg = config()
    assert cfg.model.revision == BACKBONE_REVISION
    assert cfg.scorer.checkpoint_repo.endswith("ReProbeInit")
    assert cfg.scorer.seed == 0
    assert cfg.evaluation.batch_evaluators == []


def test_missing_policy_fails_before_loading_a_model(factories):
    with pytest.raises(ValueError, match="policy_path"):
        factories.create_scorer(config())


def test_belt_factories_create_lazy_scorer_and_strategy(factories, fitted_config):
    cfg = fitted_config
    scorer = factories.create_scorer(cfg)
    generator = SimpleNamespace(
        generate_step_candidates_batch=lambda **kw: [],
        generate_answer_candidates_batch=lambda **kw: [],
    )
    strategy = factories.create_tts_strategy(cfg, object(), generator, scorer)
    assert isinstance(scorer, StepScorerBELT)
    assert scorer._runtime is None
    assert isinstance(strategy, StrategyBELT)
    assert strategy.scorer is scorer


def test_belt_generation_passes_revisions_and_bypasses_uncertainty_wrapper(
    factories, monkeypatch, fitted_config
):
    cfg = fitted_config
    engine = SimpleNamespace()
    seen = {}

    def llm(**kwargs):
        seen.update(kwargs)
        return engine

    monkeypatch.setattr(factories, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(factories, "VLLM_GENERATOR_AVAILABLE", True)
    monkeypatch.setattr(factories, "POLYGRAPH_UNCERTAINTY_AVAILABLE", False)
    monkeypatch.setattr(factories, "LLM", llm, raising=False)
    monkeypatch.setattr(
        factories, "SamplingParams", lambda **kw: SimpleNamespace(**kw), raising=False
    )
    monkeypatch.setattr(
        factories, "WhiteboxModelvLLM", lambda **kw: SimpleNamespace(), raising=False
    )
    monkeypatch.setattr(
        factories,
        "VLLMStepGenerator",
        lambda **kw: SimpleNamespace(**kw),
        raising=False,
    )
    model, generator = factories.create_model(cfg)
    assert model.vllm_engine is engine
    assert generator.model is engine
    assert seen["revision"] == seen["tokenizer_revision"] == BACKBONE_REVISION


def test_wrong_policy_fails_before_vllm_allocation(
    factories, fitted_config, monkeypatch
):
    path = Path(fitted_config.strategy.policy_path)
    data = json.loads(path.read_text())
    data["checkpoint_sha256"] = "b" * 64
    path.write_text(json.dumps(data))
    calls = []
    monkeypatch.setattr(factories, "LLM", lambda **kw: calls.append(kw), raising=False)
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        factories.create_model(fitted_config)
    assert calls == []


def test_evaluation_files_preserve_belt_decisions_and_cost_scope(
    factories, tmp_path, monkeypatch
):
    import wandb

    monkeypatch.setattr(wandb, "run", None)
    metadata = {
        "strategy": "belt",
        "checkpoint_sha256": "a" * 64,
        "decisions": [{"step": 1, "stop": True, "beliefs": {"q": 0.7}}],
        "injected_control_text": ["\n</think>\n\n"],
        "scoring_cost_in_token_stats": False,
    }
    output = {
        "trajectory": "reason\\boxed{4}",
        "extracted_answer": "4",
        "steps": ["reason", "\\boxed{4}"],
        "validity_scores": [0.7, None],
        "completed": True,
        "completion_reason": "belt_stop",
        "metadata": metadata,
        "token_stats": {"tflops": 0.0, "prm_tflops": 0.0},
    }

    def generate(requests, sample_indices, save_callback):
        save_callback([output], phase="belt")
        return [output]

    original_save = factories.save_results_json
    snapshots = []

    def save(rows, path):
        original_save(rows, path)
        snapshots.append(json.loads(path.read_text()))

    monkeypatch.setattr(factories, "save_results_json", save)
    path = tmp_path / "results.json"
    results = factories._generate_trajectories_batch(
        results=[],
        save_path=tmp_path,
        strategy=SimpleNamespace(generate_trajectories_batch=generate),
        dataset=[{"question": "test question", "answer": "4"}],
        processed_indices=set(),
        prompt_template="",
        system_prompt="",
        question_field="question",
        answer_field="answer",
        phase1_evaluators={},
        save_path_file=path,
        sample_metrics_path=tmp_path / "metrics.jsonl",
    )
    batch = json.loads((tmp_path / "batch_results.jsonl").read_text())
    for record in [results[0], batch, *(rows[0] for rows in snapshots)]:
        assert record["metadata"] == metadata
        assert record["completion_reason"] == "belt_stop"
    resumed, indices = factories.load_results_json(path)
    assert indices == {0}
    assert resumed[0]["metadata"] == metadata
    metrics = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert metrics["scoring_cost_in_token_stats"] is False
