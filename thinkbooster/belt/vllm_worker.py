"""BELT's full layer-16 residual capture for eager, TP=1 vLLM 0.12 Qwen3.

This extension deliberately does not share the generic upstream MLP hooks.
"""

from pathlib import Path

import torch

from .runtime import BACKBONE, BACKBONE_REVISION


class BeltHiddenStatesExtension:
    """Methods injected into a vLLM V1 worker by worker_extension_cls."""

    _belt_max_capture_tokens = 131072

    def belt_setup_capture(self):
        runner = self.model_runner
        parallel = self.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("BELT vLLM requires TP=1 and PP=1")
        if not runner.model_config.enforce_eager:
            raise ValueError("BELT vLLM requires enforce_eager=True")
        model_config = runner.model_config
        model_ref = str(model_config.model).rstrip("/")
        pinned_repo = (
            model_ref == BACKBONE and model_config.revision == BACKBONE_REVISION
        )
        pinned_snapshot = (
            Path(model_ref).is_absolute() and Path(model_ref).name == BACKBONE_REVISION
        )
        if not (pinned_repo or pinned_snapshot):
            raise ValueError("BELT vLLM requires the pinned Qwen3-1.7B model revision")
        if model_config.dtype != torch.bfloat16:
            raise ValueError("BELT vLLM initially supports only dtype=bfloat16")
        if (
            self.cache_config.cache_dtype != "auto"
            or runner.kv_cache_dtype != torch.bfloat16
        ):
            raise ValueError("BELT vLLM requires auto bfloat16 KV cache storage")
        if self.lora_config is not None:
            raise ValueError("BELT vLLM does not support LoRA configuration")
        if self.speculative_config is not None:
            raise ValueError("BELT vLLM does not support speculative decoding")
        if model_config.enable_prompt_embeds:
            raise ValueError("BELT vLLM does not support prompt embeddings")
        if getattr(model_config, "quantization", None) is not None:
            raise ValueError("BELT vLLM does not support quantization")
        model = runner.model
        if getattr(model, "quant_config", None) is not None:
            raise ValueError("BELT vLLM does not support quantization")
        config = model.config
        if config.model_type != "qwen3" or config.hidden_size != 2048:
            raise ValueError("BELT vLLM supports only Qwen3-1.7B (width 2048)")
        layers = model.model.layers
        if len(layers) != 28 or type(layers[16]).__name__ != "Qwen3DecoderLayer":
            raise ValueError("BELT requires Qwen3 decoder layer 16 of 28")
        layer = layers[16]
        if getattr(self, "_belt_hooked_layer", None) is not layer:
            old = getattr(self, "_belt_capture_hook", None)
            if old is not None:
                old.remove()
            self._belt_capture_hook = layer.register_forward_pre_hook(
                self._belt_capture_input, with_kwargs=True
            )
            self._belt_hooked_layer = layer
        self.belt_reset_capture()

    def belt_reset_capture(self):
        self._belt_capture = {}
        self._belt_capture_tokens = 0

    def _belt_capture_input(self, module, args, kwargs):
        names = ("positions", "hidden_states", "residual")
        values = dict(zip(names, args))
        values.update(kwargs)
        if not all(name in values for name in names):
            raise RuntimeError(
                "BELT requires Qwen3 (positions, hidden_states, residual)"
            )
        positions, hidden, residual = (values[name] for name in names)
        if hidden.ndim != 2 or hidden.shape[1] != 2048 or positions.ndim != 1:
            raise RuntimeError("BELT unsupported residual/position shape")
        if residual is not None and residual.shape != hidden.shape:
            raise RuntimeError("BELT residual shape mismatch")
        # Fused add+RMSNorm mutates both input branches immediately after this
        # pre-hook. Detaching alone aliases storage; take an independent copy.
        full = (hidden if residual is None else hidden + residual).detach().clone()
        position_copy = positions.detach().clone()
        req_ids = list(self.model_runner.input_batch.req_ids)
        starts = self.model_runner.query_start_loc.cpu
        n = len(req_ids)
        if (
            not n
            or len(set(req_ids)) != n
            or any(not isinstance(key, str) or not key for key in req_ids)
            or len(starts) < n + 1
        ):
            raise RuntimeError(
                "BELT request attribution failed: invalid request IDs/offsets"
            )
        offsets = [int(value) for value in starts[: n + 1]]
        if (
            offsets[0] != 0
            or offsets[-1] != hidden.shape[0]
            or len(position_copy) != hidden.shape[0]
            or any(b < a for a, b in zip(offsets, offsets[1:]))
        ):
            raise RuntimeError("BELT request attribution failed: invalid token slices")
        if self._belt_capture_tokens + hidden.shape[0] > self._belt_max_capture_tokens:
            raise RuntimeError("BELT capture buffer token limit exceeded")
        # Keep bounded independent GPU copies until the flush RPC. Copying to
        # CPU here would synchronize the GPU on every single decode token.
        for key, start, end in zip(req_ids, offsets, offsets[1:]):
            if start == end:
                continue
            self._belt_capture.setdefault(key, []).append(
                (position_copy[start:end], full[start:end])
            )
        self._belt_capture_tokens += hidden.shape[0]

    def belt_get_capture(self):
        result = {}
        for key, chunks in self._belt_capture.items():
            positions = torch.cat([positions for positions, _ in chunks]).cpu()
            if bool((positions < 0).any()):
                raise RuntimeError(
                    "BELT request attribution failed: negative positions"
                )
            result[key] = {
                "positions": positions.tolist(),
                "features": torch.cat([features for _, features in chunks])
                .to(device="cpu", dtype=torch.float32)
                .numpy(),
            }
        self.belt_reset_capture()
        return result
