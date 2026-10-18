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


def test_opt_in_vllm_example_uses_one_gpu_and_preserves_hf_default():
    from thinkbooster.belt.vllm_backend import DEFAULT_CHECKPOINT_BATCH_SIZE

    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        cfg = compose(config_name="experiments/belt/math500_qwen3_1p7b_vllm")
    assert cfg.scorer.backend == "vllm" and cfg.scorer.device == "cuda:0"
    assert cfg.generation.checkpoint_batch_size == DEFAULT_CHECKPOINT_BATCH_SIZE
    assert config().scorer.backend == "hf"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("scorer.backend", "other", "backend"),
        ("model.type", "local", "model.type=vllm"),
        ("model.model_path", "Qwen/Qwen3-8B", "Qwen3-1.7B"),
        ("model.revision", "main", "revision"),
        ("model.tokenizer_revision", "main", "tokenizer revision"),
        ("model.tensor_parallel_size", 2, "TP=1"),
        ("model.quantization", "fp8", "quantization"),
        ("model.dtype", "float16", "bfloat16"),
        ("system.torch_dtype", "float32", "bfloat16"),
        ("generation.checkpoint_batch_size", 0, "checkpoint_batch_size"),
        ("generation.checkpoint_batch_size", -1, "checkpoint_batch_size"),
        ("generation.checkpoint_batch_size", True, "checkpoint_batch_size"),
        ("generation.checkpoint_batch_size", 17, "checkpoint_batch_size"),
        ("generation.checkpoint_batch_size", None, "checkpoint_batch_size"),
    ],
)
def test_vllm_backend_rejects_unsupported_config_before_allocation(
    factories, fitted_config, monkeypatch, field, value, message
):
    from omegaconf import OmegaConf

    cfg = fitted_config
    OmegaConf.update(cfg, "scorer.backend", "vllm", force_add=True)
    OmegaConf.update(cfg, field, value, force_add=True)
    calls = []
    monkeypatch.setattr(factories, "VLLM_AVAILABLE", False)
    monkeypatch.setattr(factories, "LLM", lambda **kw: calls.append(kw), raising=False)
    with pytest.raises(ValueError, match=message):
        factories.create_model(cfg)
    assert calls == []


def test_vllm_backend_hydra_routes_generation_and_binds_lazy_scorer(
    factories, fitted_config, monkeypatch
):
    from thinkbooster.belt.vllm_backend import BeltVLLMBackend

    cfg = fitted_config
    cfg.scorer.backend = "vllm"
    seen = {}
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(collective_rpc=lambda method: [None])
        ),
        get_tokenizer=lambda: "tokenizer",
        model_config=SimpleNamespace(hf_config=object()),
    )
    monkeypatch.setattr(factories, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(factories, "VLLM_GENERATOR_AVAILABLE", True)
    monkeypatch.setattr(
        factories, "LLM", lambda **kw: seen.update(kw) or engine, raising=False
    )
    monkeypatch.setattr(
        factories, "SamplingParams", lambda **kw: SimpleNamespace(**kw), raising=False
    )
    monkeypatch.setattr(
        factories, "WhiteboxModelvLLM", lambda **kw: SimpleNamespace(), raising=False
    )

    def generator(**kwargs):
        return SimpleNamespace(
            **kwargs,
            generate_step_candidates_batch=lambda **kw: [],
            generate_answer_candidates_batch=lambda **kw: [],
        )

    monkeypatch.setattr(factories, "VLLMStepGenerator", generator, raising=False)
    scorer = factories.create_scorer(cfg)
    model, step_generator = factories.create_model(cfg)
    assert isinstance(step_generator.model, BeltVLLMBackend)
    assert (
        getattr(step_generator.model, "llm", step_generator.model)
        is step_generator.model
    )
    assert seen["enforce_eager"] is True
    assert seen["dtype"] == "bfloat16"
    assert (
        seen["worker_extension_cls"]
        == "thinkbooster.belt.vllm_worker.BeltHiddenStatesExtension"
    )
    strategy = factories.create_tts_strategy(cfg, model, step_generator, scorer)
    assert strategy.scorer._runtime is None
    assert scorer._runtime_options["feature_provider"] is step_generator.model
    assert scorer._runtime_options["tokenizer"] == "tokenizer"
    assert scorer._runtime_options["base_config"] is engine.model_config.hf_config


@pytest.mark.parametrize("backend,expected_batch", [("vllm", 4), ("hf", 19)])
def test_backend_default_checkpoint_batch_size(
    factories, tmp_path, monkeypatch, fitted_config, backend, expected_batch
):
    cfg = fitted_config
    cfg.scorer.backend = backend
    if "checkpoint_batch_size" in cfg.generation:
        del cfg.generation["checkpoint_batch_size"]
    monkeypatch.setattr(factories, "build_evaluators", lambda config: {})
    seen = {}
    monkeypatch.setattr(
        factories,
        "_generate_trajectories_batch",
        lambda **kwargs: seen.update(kwargs) or [],
    )
    factories.generate_trajectories(
        results=[],
        save_path=tmp_path,
        strategy=object(),
        dataset=[{"question": "q", "answer": "4"}] * 19,
        processed_indices=set(),
        prompt_template="",
        data_name="math500",
        config=cfg,
    )
    assert seen["checkpoint_batch_size"] == expected_batch
