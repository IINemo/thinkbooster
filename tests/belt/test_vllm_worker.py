import importlib
from concurrent.futures import Future
from importlib.metadata import PackageNotFoundError, version
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thinkbooster.belt.runtime import BACKBONE, BACKBONE_REVISION
from thinkbooster.belt.vllm_wire import decode_features
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
    obj.compilation_config = SimpleNamespace(mode=0)
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
    features = decode_features(result["a"]["features"])
    np.testing.assert_array_equal(features, np.full((2, 2048), 3.0))
    assert features.dtype == np.float32
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


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_compiled_model_is_rejected_even_when_eager(mode):
    obj, _ = worker()
    obj.compilation_config.mode = mode
    with pytest.raises(ValueError, match="compilation mode"):
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
    # Capacity failure must not escape the model hook or allocate residual
    # clones: the engine should finish normally, then fail through utility RPC.
    hidden = torch.ones(3, 2048)
    layer(torch.tensor([0, 1, 0]), hidden, None)
    assert hidden[0, 0] == 101
    assert not obj._belt_capture
    with pytest.raises(RuntimeError, match="buffer token limit"):
        obj.belt_get_capture()
    assert obj.belt_get_capture() == {}


def test_capture_overflow_is_sticky_and_rpc_failure_releases_buffers_for_reuse():
    obj, layer = worker(("a",), (0, 1))
    obj._belt_max_capture_tokens = 1
    obj.belt_setup_capture()
    layer(torch.tensor([0]), torch.ones(1, 2048), None)
    layer(torch.tensor([1]), torch.ones(1, 2048), None)
    layer(torch.tensor([2]), torch.ones(1, 2048), None)
    assert obj._belt_capture_tokens == 1
    assert obj._belt_capture_seen_tokens == 3
    with pytest.raises(RuntimeError, match="observed=3.*retained=1"):
        obj.belt_get_capture()
    assert not obj._belt_capture and obj._belt_capture_tokens == 0
    assert obj._belt_capture_seen_tokens == 0
    layer(torch.tensor([5]), torch.full((1, 2048), 2.0), None)
    result = obj.belt_get_capture()
    assert result["a"]["positions"] == [5]
    assert decode_features(result["a"]["features"])[0, 0] == 2


def test_real_vllm_utility_error_reaches_client_and_next_call_succeeds():
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        pytest.skip("Real vLLM package required for utility exception regression")
    if installed != "0.12.0":
        pytest.skip("Utility exception regression targets supported vLLM 0.12.0")
    core = importlib.import_module("vllm.v1.engine.core")
    client = importlib.import_module("vllm.v1.engine.core_client")
    obj, layer = worker(("a",), (0, 1))
    obj._belt_max_capture_tokens = 1
    obj.belt_setup_capture()
    layer(torch.tensor([0]), torch.ones(1, 2048), None)
    layer(torch.tensor([1]), torch.ones(1, 2048), None)

    def collective_rpc(method):
        assert method == "belt_get_capture"
        return [obj.belt_get_capture()]

    # Exercise the actual utility dispatcher/client future handling without
    # constructing an engine or loading GPU weights.
    dispatcher = SimpleNamespace(
        collective_rpc=collective_rpc,
        output_queue=Queue(),
        _convert_msgspec_args=core.EngineCoreProc._convert_msgspec_args,
    )

    def invoke(call_id):
        core.EngineCoreProc._handle_client_request(
            dispatcher,
            core.EngineCoreRequestType.UTILITY,
            (0, call_id, "collective_rpc", ("belt_get_capture",)),
        )
        _, outputs = dispatcher.output_queue.get_nowait()
        future = Future()
        client._process_utility_output(outputs.utility_output, {call_id: future})
        return future

    with pytest.raises(Exception, match="buffer token limit.*observed=2"):
        invoke(1).result()
    layer(torch.tensor([5]), torch.ones(1, 2048), None)
    result = invoke(2).result()
    assert result[0]["a"]["positions"] == [5]


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
