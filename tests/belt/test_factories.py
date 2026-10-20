import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from belt_requests import built_request
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

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


@pytest.fixture(autouse=True)
def native_reader_present(monkeypatch):
    monkeypatch.setattr(
        "thinkbooster.belt.runtime.require_native_reader", lambda: (None, None)
    )


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_missing_belt_checkout_fails_before_vllm_allocation(
    factories, monkeypatch, fitted_config, backend
):
    def missing():
        raise ImportError("BELT inference needs the pinned BELT source checkout")

    monkeypatch.setattr("thinkbooster.belt.runtime.require_native_reader", missing)
    calls = []
    monkeypatch.setattr(factories, "LLM", lambda **kw: calls.append(kw), raising=False)
    fitted_config.scorer.backend = backend
    if backend == "vllm":
        fitted_config.scorer.device = "cuda:0"
    with pytest.raises(ImportError, match="BELT source checkout"):
        factories.create_model(fitted_config)
    assert calls == []


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
                "hidden_dtype": "float32",
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
    assert cfg.generation.checkpoint_batch_size == 4


def test_missing_policy_fails_before_loading_a_model(factories):
    with pytest.raises(ValueError, match="policy_path"):
        factories.create_scorer(config())


def test_belt_factories_create_lazy_scorer_and_strategy(factories, fitted_config):
    cfg = fitted_config
    scorer = factories.create_scorer(cfg)
    generator = SimpleNamespace(
        thinking_mode=True,
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
        # 4 x 32768 already fills the 131072-row capture buffer.
        ("generation.checkpoint_batch_size", 5, "between 1 and 4"),
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
    with open_dict(cfg):
        cfg.generation.pop("checkpoint_batch_size", None)
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


def test_vllm_backend_batch_cap_scales_with_context_length(factories, fitted_config):
    from omegaconf import OmegaConf

    cfg = fitted_config
    OmegaConf.update(cfg, "scorer.backend", "vllm", force_add=True)
    OmegaConf.update(cfg, "model.max_context_budget", 8192, force_add=True)
    OmegaConf.update(cfg, "generation.checkpoint_batch_size", 16, force_add=True)
    assert factories._belt_backend(cfg) == "vllm"
    OmegaConf.update(cfg, "model.max_context_budget", 262144, force_add=True)
    with pytest.raises(ValueError, match="max_model_len"):
        factories._belt_backend(cfg)


def test_non_belt_metadata_is_not_written_to_result_files(
    factories, tmp_path, monkeypatch
):
    import wandb

    monkeypatch.setattr(wandb, "run", None)
    heavy = {"strategy": "deepconf", "all_traces": [{"token_confs": [0.1] * 1000}]}
    output = {
        "trajectory": "\\boxed{4}",
        "extracted_answer": "4",
        "steps": ["\\boxed{4}"],
        "completed": True,
        "metadata": heavy,
        "token_stats": {"tflops": 0.0, "prm_tflops": 0.0},
    }

    def generate(requests, sample_indices, save_callback):
        save_callback([output], phase="post_generation")
        return [output]

    results = factories._generate_trajectories_batch(
        results=[],
        save_path=tmp_path,
        strategy=SimpleNamespace(generate_trajectories_batch=generate),
        dataset=[{"question": "q", "answer": "4"}],
        processed_indices=set(),
        prompt_template="",
        system_prompt="",
        question_field="question",
        answer_field="answer",
        phase1_evaluators={},
        save_path_file=tmp_path / "results.json",
        sample_metrics_path=tmp_path / "metrics.jsonl",
    )
    batch = json.loads((tmp_path / "batch_results.jsonl").read_text())
    saved = json.loads((tmp_path / "results.json").read_text())
    for row in [results[0], batch, *saved]:
        assert "metadata" not in row


def test_reasoning_step_average_counts_belt_rows_without_queries(
    factories, monkeypatch, tmp_path
):
    monkeypatch.setattr(factories, "build_evaluators", lambda config: {})
    row = dict(
        question="q",
        gold_answer="4",
        generated_answer="4",
        trajectory="t",
        extracted_answer="4",
        steps=["a", "b"],
        completed=True,
        token_stats={},
    )
    results = [
        dict(row, index=0, reasoning_steps=4, validity_scores=[None, 0.5, None]),
        # Closed thinking before its first query: every step is unqueried.
        dict(
            row,
            index=1,
            reasoning_steps=2,
            validity_scores=[None, None],
            metadata={"strategy": "belt"},
        ),
        # Other strategies keep the old rule.
        dict(row, index=2, reasoning_steps=9, validity_scores=[None]),
    ]
    factories.evaluate_results(config(), results, str(tmp_path))
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["avg_reasoning_steps_per_trajectory"] == 3.0


def test_example_prompt_reaches_the_request_builder():
    from thinkbooster.strategies.strategy_belt import _belt_chat

    request = built_request(config(), "Q?")
    assert _belt_chat(request) == [
        {
            "role": "user",
            "content": "Answer the following question. Reply with the answer only. "
            "Answer enclosed with \\boxed{}.\n<Question>: Q?",
        }
    ]


def test_prompt_file_still_wins_over_inline_template(factories, tmp_path):
    cfg = config()
    path = tmp_path / "prompt.txt"
    path.write_text("From file: {question}")
    with open_dict(cfg):
        cfg.dataset.prompt_file = str(path)
    assert factories.dataset_prompt_template(cfg) == "From file: {question}"
