"""
Hook-based hidden states worker extension for vLLM v1 — with per-request attribution.

Captures hidden states via register_forward_hook on transformer layers and
uses model_runner.input_batch.req_ids + model_runner.query_start_loc to
split the flat batch tensor into per-request segments.

Usage:
    llm = LLM(
        model="...",
        enforce_eager=True,
        worker_extension_cls="utils.hook_hs_extension.HookHiddenStatesExtension",
    )
"""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


class HookHiddenStatesExtension:
    """
    Captures hidden states via register_forward_hook on transformer layers.

    Injected into vllm.v1.worker.gpu_worker.Worker via worker_extension_cls.
    In that context self == Worker instance, so self.model_runner is available.
    """

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
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

    def _get_request_segmentation(self):
        """
        Read model_runner.input_batch and query_start_loc to figure out
        which slice of the flat token tensor belongs to which request.

        Returns:
            req_ids:  list[str]          -- request IDs in batch order
            slices:   list[tuple[int,int]] -- (start, end) for each request
        """
        model_runner = self.model_runner  # type: ignore[attr-defined]
        input_batch = model_runner.input_batch

        req_ids: list[str] = list(input_batch.req_ids)
        num_reqs = len(req_ids)

        if num_reqs == 0:
            return [], []

        # query_start_loc is a CpuGpuBuffer; .cpu gives the CPU tensor
        # shape: (num_reqs + 1,) -- cumulative token offsets
        try:
            qsl = model_runner.query_start_loc.cpu[: num_reqs + 1]
        except Exception:
            # Fallback: try .numpy or direct attribute
            qsl_raw = getattr(model_runner, "query_start_loc", None)
            if qsl_raw is None:
                log.warning("query_start_loc not found on model_runner")
                return req_ids, []
            if hasattr(qsl_raw, "cpu"):
                qsl = qsl_raw.cpu[: num_reqs + 1]
            else:
                qsl = qsl_raw[: num_reqs + 1]

        slices = []
        for i in range(num_reqs):
            start = int(qsl[i])
            end = int(qsl[i + 1])
            slices.append((start, end))

        return req_ids, slices

    def _track_prefill_chunk(self, req_id: str, chunk_size: int) -> None:
        """
        Track prefill chunks per request.
        A chunk is considered "prefill" if it has more than 1 token
        (decode steps are always 1 token per request).
        """
        if req_id not in self._hs_prefill_tokens:
            self._hs_prefill_tokens[req_id] = 0
        # Prefill chunks have > 1 token; decode chunks have exactly 1
        if chunk_size > 1:
            self._hs_prefill_tokens[req_id] += chunk_size
        elif self._hs_prefill_tokens[req_id] == 0:
            # First chunk is 1 token -- single-token prefill or
            # the request was fully cached except 1 token
            self._hs_prefill_tokens[req_id] = chunk_size

    # ------------------------------------------------------------------ #
    # RPC-callable methods (called via engine_core.collective_rpc)        #
    # ------------------------------------------------------------------ #

    def _setup_hidden_states_capture(self, layer_ids: List[int]) -> None:
        """
        Register forward hooks on the requested transformer layers.
        Safe to call multiple times -- old hooks are removed first.
        """
        # Remove stale hooks from a previous call
        for handle in getattr(self, "_hs_hook_handles", []):
            handle.remove()

        self._hs_hook_handles: list = []
        # Per-request captured states: {layer_id: {req_id: [tensor_chunks]}}
        self._hs_captured: Dict[int, Dict[str, list]] = {}
        self._hs_target_layers = frozenset(layer_ids)
        # Track per-request token counts for prefix-cache gap detection
        self._hs_req_meta: Dict[str, Dict] = {}
        # Track first chunk size per request (prefill size, for gap filling)
        self._hs_prefill_tokens: Dict[str, int] = {}

        base_model = self._get_base_model()

        for layer_idx, layer in enumerate(base_model.layers):
            if layer_idx not in self._hs_target_layers:
                continue

            def _make_hook(idx: int):
                def _hook(module, inp, output):
                    # Only TP rank 0 captures
                    try:
                        from vllm.distributed import get_tensor_model_parallel_rank

                        if get_tensor_model_parallel_rank() != 0:
                            return
                    except Exception:
                        pass

                    hs = output[0] if isinstance(output, tuple) else output

                    # Flatten to 2D if needed: [tokens, hidden]
                    if hs.dim() == 3:
                        hs = hs.reshape(-1, hs.shape[-1])

                    # Get per-request segmentation from model_runner
                    req_ids, slices = self._get_request_segmentation()

                    if not slices:
                        # Fallback: no segmentation available
                        log.warning(
                            "Layer %d: no request segmentation available, "
                            "storing %d tokens as __unknown__",
                            idx,
                            hs.shape[0],
                        )
                        if idx not in self._hs_captured:
                            self._hs_captured[idx] = {}
                        self._hs_captured[idx].setdefault("__unknown__", []).append(
                            hs.detach()
                        )
                        return

                    if idx not in self._hs_captured:
                        self._hs_captured[idx] = {}

                    layer_store = self._hs_captured[idx]

                    for req_id, (start, end) in zip(req_ids, slices):
                        if start >= end:
                            continue  # no tokens for this request in this step
                        chunk = hs[start:end].detach()
                        layer_store.setdefault(req_id, []).append(chunk)
                        self._track_prefill_chunk(req_id, end - start)

                    # Update request metadata
                    try:
                        for i, req_id in enumerate(req_ids):
                            s, e = slices[i]
                            computed_this_step = e - s
                            if req_id not in self._hs_req_meta:
                                self._hs_req_meta[req_id] = {
                                    "total_computed": 0,
                                }
                            self._hs_req_meta[req_id][
                                "total_computed"
                            ] += computed_this_step
                    except Exception:
                        pass

                return _hook

            handle = layer.register_forward_hook(_make_hook(layer_idx))
            self._hs_hook_handles.append(handle)

        log.info(
            "HookHiddenStatesExtension v2: hooks registered for layers %s "
            "(%d hooks total)",
            sorted(layer_ids),
            len(self._hs_hook_handles),
        )

    def _reset_capture(self) -> None:
        """Clear the capture buffers before a new generate() call."""
        self._hs_captured = {}
        self._hs_req_meta = {}
        self._hs_prefill_tokens = {}

    def _get_captured_states(self) -> Dict[int, Dict[str, bytes]]:
        """
        Return captured hidden states, split per request.

        Returns:
            {layer_id: {request_id: pickle_dumps(numpy_array)}}

        Each numpy array has shape (num_captured_tokens, hidden_size).
        """
        # Concatenate chunks per request per layer
        combined_states: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_id, req_store in self._hs_captured.items():
            combined_states[layer_id] = {}
            for req_id, chunks in req_store.items():
                combined_states[layer_id][req_id] = torch.cat(chunks, dim=0)

        # Serialize to numpy + pickle
        result: Dict[int, Dict[str, bytes]] = {}
        for layer_id, req_tensors in combined_states.items():
            result[layer_id] = {}
            for req_id, tensor in req_tensors.items():
                arr = tensor.cpu().float().numpy()
                log.info(
                    "Layer %d, req %s: final shape=%s",
                    layer_id,
                    req_id,
                    arr.shape,
                )
                result[layer_id][req_id] = pickle.dumps(
                    arr, protocol=pickle.HIGHEST_PROTOCOL
                )

        self._hs_captured = {}
        return result

    def _get_capture_metadata(self) -> Dict[str, Dict]:
        """
        Return per-request capture metadata.

        Returns:
            {request_id: {"total_computed": int, "prefill_tokens": int}}

        Use prefill_tokens to detect prefix-cache gaps:
        if prefill_tokens < prompt_len, the difference was cached.
        """
        meta = {}
        for req_id, info in self._hs_req_meta.items():
            meta[req_id] = {
                **info,
                "prefill_tokens": self._hs_prefill_tokens.get(req_id, 0),
            }
        self._hs_req_meta = {}
        self._hs_prefill_tokens = {}
        return meta

    # Kept for API compatibility
    def _store_captured_states(self, states) -> None:
        pass


def fill_prefix_gaps(
    captured: Dict[str, np.ndarray],
    metadata: Dict[str, Dict],
    prompt_groups: Dict[str, List[str]],
    prompt_tokens: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Fill prefix-cache gaps in two phases:

    Phase 1 -- Within-group fill: for requests sharing the exact same prompt,
    copy the missing prefix from the group's donor (the request that computed
    the most prefill tokens).

    Phase 2 -- Cross-group fill (requires prompt_tokens): vLLM also caches the
    shared chat-template prefix across *different* prompts.  Using token-level
    longest-common-prefix (LCP) with a global donor, fill the remaining gap.

    Args:
        captured:  {req_id: numpy_array}
        metadata:  {req_id: {"total_computed": int, "prefill_tokens": int}}
        prompt_groups: {group_name: [req_id, ...]} -- requests with identical prompts
        prompt_tokens: {req_id: [token_ids]} -- full prompt token IDs per request.
                       Required for cross-group gap filling.

    Returns:
        {req_id: numpy_array} -- with gaps filled
    """
    result = dict(captured)

    # Track effective prefill count per request (updated as we fill)
    eff_prefill: Dict[str, int] = {
        rid: metadata.get(rid, {}).get("prefill_tokens", 0) for rid in result
    }

    # -- Phase 1: within-group fill (identical prompts) --
    for _, req_ids in prompt_groups.items():
        group_reqs = [rid for rid in req_ids if rid in result]
        if len(group_reqs) < 2:
            continue

        donor_id = max(group_reqs, key=lambda rid: eff_prefill.get(rid, 0))
        donor_prefill = eff_prefill[donor_id]
        donor_arr = result[donor_id]

        for req_id in group_reqs:
            if req_id == donor_id:
                continue
            gap = donor_prefill - eff_prefill[req_id]
            if gap > 0:
                result[req_id] = np.concatenate(
                    [donor_arr[:gap], result[req_id]], axis=0
                )
                eff_prefill[req_id] = donor_prefill

    # -- Phase 2: cross-group fill (shared token prefix) --
    if not prompt_tokens:
        return result

    all_req_ids = list(result.keys())
    if len(all_req_ids) < 2:
        return result

    # Global donor: pick from requests with FULL prefill, then longest prompt
    full_prefill_reqs = [
        rid
        for rid in all_req_ids
        if eff_prefill.get(rid, 0) >= len(prompt_tokens.get(rid, []))
    ]
    if not full_prefill_reqs:
        return result  # no request has full prefill

    global_donor = max(
        full_prefill_reqs,
        key=lambda rid: len(prompt_tokens.get(rid, [])),
    )
    gd_prefill = eff_prefill[global_donor]
    gd_tokens = prompt_tokens.get(global_donor, [])
    gd_arr = result[global_donor]

    for req_id in all_req_ids:
        if req_id == global_donor:
            continue

        req_tokens = prompt_tokens.get(req_id, [])
        req_prompt_len = len(req_tokens)
        cur_prefill = eff_prefill[req_id]

        if cur_prefill >= req_prompt_len:
            continue  # already full

        # Longest common prefix with global donor
        lcp = 0
        for a, b in zip(gd_tokens, req_tokens):
            if a == b:
                lcp += 1
            else:
                break

        # Number of leading tokens still missing for this request
        missing = req_prompt_len - cur_prefill

        # We can fill up to min(missing, lcp, gd_prefill) tokens from global donor
        fillable = min(missing, lcp, gd_prefill)
        if fillable > 0:
            result[req_id] = np.concatenate([gd_arr[:fillable], result[req_id]], axis=0)
            eff_prefill[req_id] = cur_prefill + fillable

    return result
