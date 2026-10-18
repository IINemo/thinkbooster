from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinkbooster.belt.runtime import BACKBONE, BACKBONE_REVISION
from thinkbooster.belt.vllm_worker import BeltHiddenStatesExtension


class Qwen3DecoderLayer(torch.nn.Module):
    def forward(self, positions, hidden_states, residual):
        hidden_states.add_(100)
        if residual is not None:
            residual.zero_()
        return hidden_states, residual


def worker(req_ids=("a", "b"), starts=(0, 2, 3)):
    layers = torch.nn.ModuleList([Qwen3DecoderLayer() for _ in range(28)])
    obj = BeltHiddenStatesExtension()
    obj.model_runner = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(model_type="qwen3", hidden_size=2048),
            model=SimpleNamespace(layers=layers),
        ),
        model_config=SimpleNamespace(
            enforce_eager=True,
            model=BACKBONE,
            revision=BACKBONE_REVISION,
            dtype=torch.bfloat16,
            quantization=None,
            enable_prompt_embeds=False,
        ),
        input_batch=SimpleNamespace(req_ids=list(req_ids)),
        query_start_loc=SimpleNamespace(cpu=torch.tensor(starts)),
        kv_cache_dtype=torch.bfloat16,
    )
    obj.parallel_config = SimpleNamespace(
        tensor_parallel_size=1, pipeline_parallel_size=1
    )
    obj.cache_config = SimpleNamespace(cache_dtype="auto")
    obj.lora_config = None
    obj.speculative_config = None
    return obj, layers[16]


def test_residual_capture_is_copied_before_mutation_and_attributed_by_position():
    obj, layer = worker()
    obj.belt_setup_capture()
    hidden, residual = torch.ones(3, 2048), torch.full((3, 2048), 2.0)
    layer(torch.tensor([5, 6, 11]), hidden, residual)
    result = obj.belt_get_capture()
    assert result["a"]["positions"] == [5, 6]
    assert result["b"]["positions"] == [11]
    np.testing.assert_array_equal(result["a"]["features"], np.full((2, 2048), 3.0))
    assert result["a"]["features"].dtype == np.float32
    assert obj.belt_get_capture() == {}


def test_setup_reuses_one_hook_and_reset_releases_owned_buffers():
    obj, layer = worker(("a",), (0, 1))
    obj.belt_setup_capture()
    obj.belt_setup_capture()
    assert len(layer._forward_pre_hooks) == 1
    layer(torch.tensor([0]), torch.ones(1, 2048), None)
    obj.belt_reset_capture()
    assert obj.belt_get_capture() == {}


@pytest.mark.parametrize("starts", [(0, 1), (1, 2, 3), (0, 3, 2), (0, 2, 4)])
def test_bad_segmentation_fails_explicitly(starts):
    obj, layer = worker(starts=starts)
    obj.belt_setup_capture()
    with pytest.raises(RuntimeError, match="attribution"):
        layer(torch.tensor([0, 1, 0]), torch.ones(3, 2048), None)


def test_missing_layer_and_unsupported_parallelism_fail():
    obj, _ = worker()
    obj.model_runner.model.model.layers = []
    with pytest.raises(ValueError, match="layer 16"):
        obj.belt_setup_capture()
    obj, _ = worker()
    obj.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="TP=1"):
        obj.belt_setup_capture()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("model", "Qwen/Other", "pinned"),
        ("revision", "main", "pinned"),
        ("dtype", torch.float16, "bfloat16"),
        ("quantization", "fp8", "quantization"),
        ("enforce_eager", False, "eager"),
        ("enable_prompt_embeds", True, "prompt embeddings"),
    ],
)
def test_direct_api_rejects_unsupported_model_config(field, value, message):
    obj, _ = worker()
    setattr(obj.model_runner.model_config, field, value)
    with pytest.raises(ValueError, match=message):
        obj.belt_setup_capture()


def test_exact_local_snapshot_pin_is_supported_without_revision_argument():
    obj, _ = worker()
    obj.model_runner.model_config.model = f"/cache/snapshots/{BACKBONE_REVISION}"
    obj.model_runner.model_config.revision = None
    obj.belt_setup_capture()


def test_capture_buffer_is_bounded():
    obj, layer = worker()
    obj._belt_max_capture_tokens = 2
    obj.belt_setup_capture()
    with pytest.raises(RuntimeError, match="buffer token limit"):
        layer(torch.tensor([0, 1, 0]), torch.ones(3, 2048), None)


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_direct_api_rejects_explicit_fp8_kv_cache(cache_dtype):
    obj, layer = worker()
    obj.cache_config.cache_dtype = cache_dtype
    with pytest.raises(ValueError, match="KV cache"):
        obj.belt_setup_capture()
    assert not layer._forward_pre_hooks


def test_direct_api_rejects_incompatible_resolved_kv_cache_dtype():
    obj, _ = worker()
    obj.model_runner.kv_cache_dtype = torch.float8_e4m3fn
    with pytest.raises(ValueError, match="KV cache"):
        obj.belt_setup_capture()


@pytest.mark.parametrize(
    "field,message",
    [
        ("lora_config", "LoRA"),
        ("speculative_config", "speculative"),
    ],
)
def test_direct_api_rejects_lora_and_speculative_configs(field, message):
    obj, layer = worker()
    setattr(obj, field, SimpleNamespace())
    with pytest.raises(ValueError, match=message):
        obj.belt_setup_capture()
    assert not layer._forward_pre_hooks
