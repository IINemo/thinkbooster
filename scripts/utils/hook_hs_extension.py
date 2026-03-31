"""
Hook-based hidden states worker extension for vLLM v1.

Replaces HiddenStatesWorkerExtension's forward-patching approach with
register_forward_hook — works correctly with enforce_eager=True where
the compilation wrapper bypasses the patched forward method.

Usage:
    llm = LLM(
        model="...",
        enforce_eager=True,
        worker_extension_cls="hook_hs_extension.HookHiddenStatesExtension",
    )
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List

import torch

log = logging.getLogger(__name__)


class HookHiddenStatesExtension:
    """
    Captures hidden states via register_forward_hook on transformer layers.

    Injected into vllm.v1.worker.gpu_worker.Worker via worker_extension_cls.
    In that context self == Worker instance, so self.model_runner.model is
    the loaded model (Qwen2ForCausalLM / LlamaForCausalLM / …).
    """

    # ------------------------------------------------------------------ #
    # RPC-callable methods (called via engine_core.collective_rpc)        #
    # ------------------------------------------------------------------ #

    def _get_base_model(self):
        """Return the transformer body (the nn.Module that owns .layers)."""
        model = self.model_runner.model  # type: ignore[attr-defined]

        # vision-language: model.get_language_model().model
        try:
            return model.get_language_model().model
        except AttributeError:
            pass

        # plain text model: model.model
        try:
            return model.model
        except AttributeError:
            pass

        raise AttributeError(
            f"Cannot find base model with .layers. "
            f"Available attrs on model: {[a for a in dir(model) if not a.startswith('_')]}"
        )

    # ------------------------------------------------------------------ #
    # RPC-callable methods (called via engine_core.collective_rpc)        #
    # ------------------------------------------------------------------ #

    def _setup_hidden_states_capture(self, layer_ids: List[int]) -> None:
        """
        Register forward hooks on the requested transformer layers.
        Safe to call multiple times — old hooks are removed first.
        """
        # Remove stale hooks from a previous call
        for handle in getattr(self, "_hs_hook_handles", []):
            handle.remove()

        self._hs_hook_handles: list = []
        self._hs_captured: Dict[int, Dict] = {}
        self._hs_target_layers = frozenset(layer_ids)

        base_model = self._get_base_model()

        for layer_idx, layer in enumerate(base_model.layers):
            if layer_idx not in self._hs_target_layers:
                continue

            def _make_hook(idx: int):
                def _hook(module, inp, output):
                    # vLLM layers return bare hidden_states tensor;
                    # some return (hidden_states, ...) — handle both.
                    hs = output[0] if isinstance(output, tuple) else output
                    # Only TP rank 0 captures (all ranks have identical values)
                    try:
                        from vllm.distributed import get_tensor_model_parallel_rank

                        if get_tensor_model_parallel_rank() != 0:
                            return
                    except Exception:
                        pass

                    # Initialize buffer on first access
                    if idx not in self._hs_captured:
                        self._hs_captured[idx] = {
                            "buffer": None,
                            "size": 0,
                            "capacity": 0,
                        }

                    buf_info = self._hs_captured[idx]

                    # Flatten all hidden states and append to buffer
                    log.info(
                        f"Capturing hidden states from layer {idx}: shape={hs.shape}, dtype={hs.dtype}, dim={hs.dim()}"
                    )
                    log.info(
                        f"Input to hook: module={module.__class__.__name__}, inp_shapes={[i.shape for i in inp]}, output_shape={output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]}"
                    )
                    if hs.dim() == 3:
                        # [batch, seq, hidden] -> flatten all tokens
                        hs_flat = hs.reshape(-1, hs.shape[-1])  # [batch*seq, hidden]
                        log.info(
                            f"  Prefill step: reshaped {hs.shape} -> {hs_flat.shape}"
                        )
                    else:
                        # [batch, hidden] -> already 2D
                        hs_flat = hs
                        log.info(f"  Decode step: shape {hs_flat.shape}")

                    num_tokens = hs_flat.shape[0]
                    hidden_size = hs_flat.shape[1]
                    log.info(
                        f"  Adding {num_tokens} tokens to buffer (current size: {buf_info['size']})"
                    )

                    # Initialize buffer if needed
                    if buf_info["buffer"] is None:
                        buf_info["buffer"] = torch.empty(
                            (1024, hidden_size),
                            device=hs_flat.device,
                            dtype=hs_flat.dtype,
                        )
                        buf_info["capacity"] = 1024

                    buffer = buf_info["buffer"]
                    size = buf_info["size"]
                    capacity = buf_info["capacity"]

                    # Expand buffer if needed
                    if size + num_tokens > capacity:
                        new_capacity = max(capacity * 2, size + num_tokens)
                        new_buffer = torch.empty(
                            (new_capacity, hidden_size),
                            device=hs_flat.device,
                            dtype=hs_flat.dtype,
                        )
                        # Copy only the used portion of old buffer
                        if size > 0:
                            new_buffer[:size] = buffer[:size]
                        buf_info["buffer"] = new_buffer
                        buf_info["capacity"] = new_capacity
                        buffer = new_buffer

                    # Append all tokens to buffer
                    buffer[size : size + num_tokens] = hs_flat
                    buf_info["size"] = size + num_tokens

                return _hook

            handle = layer.register_forward_hook(_make_hook(layer_idx))
            self._hs_hook_handles.append(handle)

        log.info(
            "HookHiddenStatesExtension: hooks registered for layers %s "
            "(%d hooks total)",
            sorted(layer_ids),
            len(self._hs_hook_handles),
        )

    def _reset_capture(self) -> None:
        """Clear the capture buffer before a new generate() call."""
        self._hs_captured = {}

    def _get_captured_states(self) -> Dict[int, bytes]:
        """
        Return captured hidden states as flat concatenated tensor.
        Returns: {layer_id: pickle_dumps(numpy_array)}
        """
        result: Dict[int, bytes] = {}
        for layer_id, buf_info in self._hs_captured.items():
            buffer = buf_info["buffer"]
            size = buf_info["size"]
            # Extract actual data and convert to numpy, then pickle
            arr = buffer[:size].cpu().float().numpy()
            log.info(
                f"Captured hidden states for layer {layer_id}: shape={arr.shape}, dtype={arr.dtype}"
            )
            result[layer_id] = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
        self._hs_captured = {}
        return result

    # Kept for API compatibility — not used in hook approach
    def _store_captured_states(self, states) -> None:
        pass
