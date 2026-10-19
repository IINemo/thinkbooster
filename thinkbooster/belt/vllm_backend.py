"""Single-engine vLLM generation and exact-token BELT feature provider."""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from .vllm_wire import FEATURE_WIRE_SCHEMA, MAX_CAPTURE_TOKENS, decode_features

DEFAULT_CACHE_ENTRIES = 16
DEFAULT_CHECKPOINT_BATCH_SIZE = 4


def _repair_sampling_params():
    from vllm import SamplingParams

    return SamplingParams(n=1, max_tokens=1, temperature=0, seed=0, stop=None)


@dataclass
class _FeatureEntry:
    tokens: tuple
    features: np.ndarray
    mask: np.ndarray


class BeltVLLMBackend:
    """Capture generation residuals; repair exact token prefixes through vLLM.

    The shared LLM remains owned by its caller. No cleanup method destroys it.
    Gaps from automatic prefix caching are represented by masks, never values.
    """

    chunk_size = 0
    backend = "vllm"

    def __init__(self, llm, tokenizer=None, max_cache_entries=DEFAULT_CACHE_ENTRIES):
        if type(max_cache_entries) is not int or max_cache_entries < 1:
            raise ValueError("max_cache_entries must be a positive integer")
        self._engine = llm
        self._tokenizer = tokenizer
        self.max_cache_entries = max_cache_entries
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._stats = self._new_stats()
        self._rpc("belt_setup_capture")

    def __getattr__(self, name):
        # VLLMStepGenerator unwraps .llm when compute_uncertainty=False.
        # This wrapper must remain the generation entry point on that route.
        if name == "llm":
            raise AttributeError(name)
        return getattr(self._engine, name)

    @staticmethod
    def _new_stats():
        return dict(
            generation_calls=0,
            generation_captured_tokens=0,
            generation_seconds=0.0,
            queries=0,
            cache_hits=0,
            repair_calls=0,
            repair_tokens=0,
            repair_captured_tokens=0,
            repair_requested_tokens=0,
            repair_seconds=0.0,
            query_seconds=0.0,
            prefix_cache_resets=0,
            discarded_overrun_rows=0,
        )

    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)

    @property
    def metadata(self):
        return {
            "backend": self.backend,
            "feature_layer": 16,
            "feature_kind": "full_residual_input",
            "feature_width": 2048,
            "feature_dtype": "float32",
            "feature_wire_schema": FEATURE_WIRE_SCHEMA,
            "feature_cache_ownership": "first_cached_causal_prefix",
            "capture_token_limit": MAX_CAPTURE_TOKENS,
            "max_cache_entries": self.max_cache_entries,
            "cache_entries": len(self._cache),
            "stats": self.stats,
            "scoring_cost_in_token_stats": False,
        }

    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._engine.get_tokenizer()
        return self._tokenizer

    def _rpc(self, method):
        replies = self._engine.llm_engine.engine_core.collective_rpc(method)
        if not isinstance(replies, (list, tuple)) or len(replies) != 1:
            raise RuntimeError("BELT vLLM requires exactly one worker RPC reply (TP=1)")
        reply = replies[0]
        if method != "belt_get_capture":
            return reply
        if not isinstance(reply, dict):
            raise RuntimeError("BELT worker returned an invalid capture mapping")
        decoded, count = {}, 0
        for key, record in reply.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(record, dict)
                or set(record) != {"positions", "features"}
                or not isinstance(record["positions"], list)
                or any(type(pos) is not int or pos < 0 for pos in record["positions"])
            ):
                raise RuntimeError("BELT worker returned invalid request positions")
            rows = decode_features(
                record["features"], max_rows=MAX_CAPTURE_TOKENS - count
            )
            if len(record["positions"]) != len(rows):
                raise RuntimeError("BELT feature wire position/row count mismatch")
            count += len(rows)
            decoded[key] = {"positions": record["positions"], "features": rows}
        return decoded

    def reset_hs_step_cache(self, reset_prefix_cache=True):
        """Start a strategy batch with fresh feature/capture ownership and counters."""
        with self._lock:
            self._rpc("belt_reset_capture")
            self._cache.clear()
            self._stats = self._new_stats()
            if reset_prefix_cache:
                if self._engine.reset_prefix_cache() is not True:
                    raise RuntimeError("BELT vLLM prefix cache reset failed")
                self._stats["prefix_cache_resets"] += 1

    @staticmethod
    def _token_tuple(ids):
        values = tuple(ids.tolist() if isinstance(ids, torch.Tensor) else ids)
        if not values or any(type(value) is not int or value < 0 for value in values):
            raise ValueError("BELT requires a nonempty flat sequence of token IDs")
        return values

    def _collect_prefix(self, tokens):
        features = np.empty((len(tokens), 2048), dtype=np.float32)
        mask = np.zeros(len(tokens), dtype=bool)
        # Causality permits position p to transfer only when tokens[:p+1]
        # match exactly. A changed token invalidates that position and its tail.
        for entry in self._cache.values():
            common = 0
            for left, right in zip(tokens, entry.tokens):
                if left != right:
                    break
                common += 1
            # Keep the first cached row for each exact causal prefix. Later
            # captures may fill holes but must not change a retained owner.
            known = np.flatnonzero(entry.mask[:common] & ~mask[:common])
            features[known] = entry.features[known]
            mask[known] = True
        return features, mask

    def _record_outputs(self, outputs, captured):
        if not isinstance(captured, dict):
            raise RuntimeError("BELT worker returned an invalid capture mapping")
        ids = [getattr(item, "request_id", None) for item in outputs]
        if (
            any(not isinstance(key, str) or not key for key in ids)
            or len(set(ids)) != len(ids)
            or any(key not in ids for key in captured)
        ):
            raise RuntimeError(
                "BELT cannot unambiguously attribute captured request IDs"
            )
        count = 0
        for item in outputs:
            if len(item.outputs) != 1:
                raise ValueError("BELT vLLM supports only n=1")
            tokens = self._token_tuple(
                list(item.prompt_token_ids) + list(item.outputs[0].token_ids)
            )
            features, mask = self._collect_prefix(tokens)
            data = captured.get(item.request_id)
            if data is not None:
                positions = np.asarray(data["positions"])
                rows = np.asarray(data["features"])
                if (
                    positions.ndim != 1
                    or not np.issubdtype(positions.dtype, np.integer)
                    or rows.shape != (len(positions), 2048)
                    or rows.dtype != np.float32
                    or not np.isfinite(rows).all()
                    or np.any(positions < 0)
                ):
                    raise RuntimeError(
                        "BELT capture has invalid absolute positions/features"
                    )
                # Count all measured rows before filtering or deduplication,
                # including overruns and repeated/already-owned positions.
                count += len(positions)
                # vLLM V1 matches stop strings in the frontend, so the engine
                # core can run more decode steps before the abort lands. Rows
                # past the returned sequence feed tokens that were never
                # returned; discard them. The last returned token's row is a
                # valid causal feature and is kept for later extensions.
                overrun = positions >= len(tokens)
                self._stats["discarded_overrun_rows"] += int(overrun.sum())
                positions, rows = positions[~overrun], rows[~overrun]
                # Recomputations can repeat a position in one capture. Retain
                # its first row, and inherit existing owners before eviction.
                unique_positions, first_rows = np.unique(positions, return_index=True)
                unknown = ~mask[unique_positions]
                added = unique_positions[unknown]
                features[added] = rows[first_rows[unknown]]
                mask[added] = True
            self._cache[tokens] = _FeatureEntry(tokens, features, mask)
            self._cache.move_to_end(tokens)
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)
        return count

    def _max_model_len(self):
        engine = getattr(self._engine, "llm_engine", None)
        config = getattr(engine, "model_config", None)
        value = getattr(config, "max_model_len", None)
        return value if type(value) is int and value > 0 else None

    def _preflight_capacity(self, args, kwargs):
        prompts = kwargs.get("prompts", args[0] if args else None)
        batch = prompts if isinstance(prompts, (list, tuple)) else [prompts]
        if not batch:
            raise ValueError("BELT vLLM requires a nonempty prompt batch")
        if len(batch) > self.max_cache_entries:
            raise ValueError(
                "BELT vLLM batch exceeds max_cache_entries; reduce the batch "
                "or explicitly increase feature-cache capacity"
            )
        params = kwargs.get("sampling_params", args[1] if len(args) > 1 else None)
        if params is None:
            params = self._engine.get_default_sampling_params()
        if isinstance(params, (list, tuple)):
            if len(params) != len(batch):
                raise ValueError("BELT sampling_params must match the prompt batch")
        else:
            params = [params] * len(batch)
        max_len = self._max_model_len()
        bound = 0
        for prompt, param in zip(batch, params):
            if getattr(param, "n", None) != 1:
                raise ValueError("BELT vLLM supports only n=1")
            maximum = getattr(param, "max_tokens", None)
            if type(maximum) is not int or maximum < 1:
                raise ValueError(
                    "BELT vLLM requires a finite positive integer max_tokens"
                )
            if isinstance(prompt, dict):
                if set(prompt) - {"prompt", "prompt_token_ids", "cache_salt"}:
                    raise ValueError("BELT vLLM supports only text/token-ID prompts")
                if "prompt_token_ids" in prompt:
                    size = len(self._token_tuple(prompt["prompt_token_ids"]))
                elif isinstance(prompt.get("prompt"), str):
                    size = len(self._engine.get_tokenizer().encode(prompt["prompt"]))
                else:
                    raise ValueError("BELT vLLM requires a text/token-ID prompt")
            elif isinstance(prompt, str):
                # Pinned Qwen3 uses encode() defaults in vLLM's preprocessor.
                # Ignore optional truncation to keep this an upper bound.
                size = len(self._engine.get_tokenizer().encode(prompt))
            else:
                raise ValueError("BELT vLLM requires a text/token-ID prompt")
            # vLLM never computes more positions than max_model_len for one
            # request. Recompute after preemption can repeat rows; the worker
            # still reports any real overflow through the capture RPC.
            request_bound = size + maximum
            bound += min(request_bound, max_len) if max_len else request_bound
        if bound > MAX_CAPTURE_TOKENS:
            raise ValueError(
                f"BELT capture token limit exceeded by conservative generation "
                f"bound: {bound} > {MAX_CAPTURE_TOKENS}; reduce batch/prefix/max_tokens"
            )

    def _generate(self, args, kwargs, repair=False):
        self._preflight_capacity(args, kwargs)
        self._rpc("belt_reset_capture")
        start = time.perf_counter()
        counter = "repair_calls" if repair else "generation_calls"
        self._stats[counter] += 1
        try:
            outputs = self._engine.generate(*args, **kwargs)
            captured = self._rpc("belt_get_capture")
            count = self._record_outputs(outputs, captured)
            if repair:
                # Every computed layer-16 row is captured, including the last
                # prompt token, even though HeadStateFn only uses ids[:-1].
                self._stats["repair_tokens"] += count
                self._stats["repair_captured_tokens"] += count
            else:
                self._stats["generation_captured_tokens"] += count
            return outputs
        except Exception:
            self._rpc("belt_reset_capture")
            raise
        finally:
            key = "repair_seconds" if repair else "generation_seconds"
            self._stats[key] += time.perf_counter() - start

    def generate(self, *args, **kwargs):
        if kwargs.get("lora_request") is not None:
            raise ValueError("BELT vLLM does not support LoRA requests")
        prompts = kwargs.get("prompts", args[0] if args else None)
        batch = prompts if isinstance(prompts, (list, tuple)) else [prompts]
        if any(
            isinstance(prompt, dict) and "prompt_embeds" in prompt for prompt in batch
        ):
            raise ValueError("BELT vLLM does not support prompt embeddings")
        params = kwargs.get("sampling_params", args[1] if len(args) > 1 else None)
        params_list = params if isinstance(params, (list, tuple)) else [params]
        if any(getattr(param, "n", 1) != 1 for param in params_list):
            raise ValueError("BELT vLLM supports only n=1")
        with self._lock:
            return self._generate(args, kwargs)

    def get_features(self, sequences, device):
        """Return the canonical next-token-aligned [1,T-1,2048] features."""
        if len(sequences) != 1:
            raise ValueError("BELT vLLM feature queries require a singleton sequence")
        tokens = self._token_tuple(sequences[0])
        if len(tokens) < 2:
            raise ValueError("BELT feature queries require at least two tokens")
        with self._lock:
            start = time.perf_counter()
            self._stats["queries"] += 1
            try:
                features, mask = self._collect_prefix(tokens)
                if mask[:-1].all():
                    self._stats["cache_hits"] += 1
                else:
                    for attempt in range(2):
                        if attempt:
                            success = self._engine.reset_prefix_cache()
                            if success is not True:
                                raise RuntimeError(
                                    "BELT vLLM prefix cache reset failed"
                                )
                            self._stats["prefix_cache_resets"] += 1
                        self._stats["repair_requested_tokens"] += len(tokens)
                        self._generate(
                            ([{"prompt_token_ids": list(tokens)}],),
                            {
                                "sampling_params": _repair_sampling_params(),
                                "use_tqdm": False,
                            },
                            repair=True,
                        )
                        features, mask = self._collect_prefix(tokens)
                        if mask[:-1].all():
                            break
                    if not mask[:-1].all():
                        missing = np.flatnonzero(~mask[:-1]).tolist()
                        raise RuntimeError(
                            f"BELT missing feature positions after cold retry: {missing}"
                        )
                tensor = torch.from_numpy(features[:-1].copy()).unsqueeze(0).to(device)
                attention = torch.ones(
                    (1, len(tokens) - 1), dtype=torch.long, device=device
                )
                return tensor, attention, [0], []
            finally:
                self._stats["query_seconds"] += time.perf_counter() - start
