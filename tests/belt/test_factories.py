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


def test_hydra_example_has_pinned_model_and_no_paid_judge():
    cfg = config()
    assert cfg.model.revision == BACKBONE_REVISION
    assert cfg.scorer.checkpoint_repo.endswith("ReProbeInit")
    assert cfg.scorer.seed == 0
    assert cfg.evaluation.batch_evaluators == []


def test_missing_policy_fails_before_loading_a_model(factories):
    with pytest.raises(ValueError, match="policy_path"):
        factories.create_scorer(config())


def test_belt_factories_create_lazy_scorer_and_strategy(factories, tmp_path):
    cfg = config()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
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
    cfg.strategy.policy_path = str(policy_path)
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
    factories, monkeypatch
):
    cfg = config()
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
