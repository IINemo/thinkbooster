"""Optional bridge to the released BELT native inference implementation."""

import bisect
import hashlib
import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from .blocks import block_end_offsets
from .policy import TARGETS, BeltSignal

BACKBONE = "Qwen/Qwen3-1.7B"
BACKBONE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_CHECKPOINT = "JingweiNi/BELT-Qwen3-1.7B-ReProbeInit"
CHECKPOINT_REVISIONS = {
    DEFAULT_CHECKPOINT: "1ccc94337a9672dee94f79bc3c3a98a2f994220b",
    "JingweiNi/BELT-Qwen3-1.7B": "15d1c4d82b13b47f3aaaa403687cf88d91d8052f",
}
_NATIVE_IMPORT_LOCK = threading.RLock()


def require_native_reader():
    """Import the pinned BELT reader and locate its native tree, or fail now.

    The runtime loads lazily at the first query; callers that allocate a
    generation model first use this to fail before that allocation.
    """
    try:
        from rl_controller.rollout_contract import resolve_native_root
        from rl_controller.scripts.rl_train import HeadStateFn
    except ImportError as error:
        raise ImportError(
            "BELT inference needs the pinned BELT source checkout installed with "
            "pip install --no-deps -e /path/to/BELT; see docs/strategies/belt.md"
        ) from error
    return resolve_native_root().resolve(), HeadStateFn


@contextmanager
def _native_reader_imports(native_root):
    """Restore public reader imports after the pinned factory finishes loading.

    Native inference retains its module references and performs no further
    luh/train_luh imports. BELT initializers are serialized; applications must
    avoid importing those packages concurrently with initialization.
    """

    def reader_modules():
        return {
            name: module
            for name, module in sys.modules.copy().items()
            if name.split(".")[0] in ("luh", "train_luh")
        }

    with _NATIVE_IMPORT_LOCK:
        previous = reader_modules()
        previous_path = sys.path[:]
        for name in previous:
            sys.modules.pop(name)
        sys.path.insert(0, str(native_root))
        try:
            yield
        finally:
            for name in reader_modules():
                sys.modules.pop(name)
            sys.modules.update(previous)
            sys.path[:] = previous_path


def _file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checkpoint_manifest(repo_id=DEFAULT_CHECKPOINT, seed=0, cache_dir=None):
    """Read pinned checkpoint identity without loading weights or a backbone."""
    if repo_id not in CHECKPOINT_REVISIONS:
        raise ValueError(f"Supported Qwen BELT packages: {tuple(CHECKPOINT_REVISIONS)}")
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("BELT seed must be 0, 1 or 2")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id,
        "CHECKPOINTS.json",
        revision=CHECKPOINT_REVISIONS[repo_id],
        cache_dir=cache_dir,
    )
    manifest = json.loads(Path(path).read_text())
    if manifest["backbone"] != BACKBONE or manifest["revision"] != BACKBONE_REVISION:
        raise ValueError("Checkpoint manifest uses a different backbone revision")
    record = next(c for c in manifest["checkpoints"] if c["seed"] == seed)
    return manifest, record


def download_checkpoint(repo_id=DEFAULT_CHECKPOINT, seed=0, cache_dir=None):
    """Download and verify one seed and its native reader template."""
    from huggingface_hub import hf_hub_download

    manifest, record = checkpoint_manifest(repo_id, seed, cache_dir)

    def fetch(filename):
        return Path(
            hf_hub_download(
                repo_id,
                filename,
                revision=CHECKPOINT_REVISIONS[repo_id],
                cache_dir=cache_dir,
            )
        )

    weight = fetch(record["checkpoint"])
    if _file_sha256(weight) != record["sha256"]:
        raise ValueError("BELT checkpoint SHA256 mismatch")
    summary = json.loads(fetch(f"seed-{seed}/summary.json").read_text())
    if summary["config"]["seed"] != seed:
        raise ValueError("Checkpoint summary seed mismatch")
    if tuple(summary["config"]["targets"]) != TARGETS:
        raise ValueError("Checkpoint summary must describe all four BELT targets")
    template = None
    template_files = set()
    for asset in manifest["additional_assets"]:
        if asset["destination"].startswith("reader-template/"):
            path = fetch(asset["destination"])
            if _file_sha256(path) != asset["sha256"]:
                raise ValueError("BELT reader-template SHA256 mismatch")
            template = path.parent
            template_files.add(path.name)
    if template_files != {"config.yaml", "weights.pth"}:
        raise ValueError(
            "BELT package requires the exact reader configuration and weights"
        )
    return weight, template, record


class BeltRuntime:
    """Four-target Qwen probe, evaluated by the original BELT runtime.

    Install the pinned BELT source checkout before using from_pretrained().
    Importing ThinkBooster does not download or import that runtime.
    """

    def __init__(self, tokenizer, state_fn, readout, checkpoint_sha256, metadata=None):
        self.tokenizer = tokenizer
        self.state_fn = state_fn
        self.readout = readout
        self.checkpoint_sha256 = checkpoint_sha256
        self._metadata = metadata or {}
        self.feature_provider = getattr(state_fn, "provider", None)
        self._lock = threading.RLock()

    @property
    def metadata(self):
        metadata = dict(self._metadata)
        metadata.setdefault("scoring_backend", "hf")
        provider = self.feature_provider
        if provider is not None and hasattr(provider, "stats"):
            metadata["backend_stats"] = dict(provider.stats)
        if provider is not None and hasattr(provider, "metadata"):
            metadata["backend_metadata"] = provider.metadata
        return metadata

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_repo=DEFAULT_CHECKPOINT,
        seed=0,
        device="cuda:0",
        dtype="bfloat16",
        cache_dir=None,
        hs_chunk_size=8192,
        feature_provider=None,
        tokenizer=None,
        base_config=None,
    ):
        if feature_provider is not None and tokenizer is None:
            raise ValueError("An injected BELT feature provider requires a tokenizer")
        native_root, HeadStateFn = require_native_reader()
        import torch

        weight, template, record = download_checkpoint(checkpoint_repo, seed, cache_dir)
        payload = torch.load(weight, map_location="cpu", weights_only=True)
        if (
            tuple(payload.get("targets", ())) != TARGETS
            or payload.get("head_mode", "targets") != "targets"
        ):
            raise ValueError("Expected a four-target BELT checkpoint")
        args = SimpleNamespace(
            model_path=None,
            model=BACKBONE,
            revision=BACKBONE_REVISION,
            head_checkpoint=str(template),
            hf_cache=cache_dir,
            dtype=dtype,
            hs_chunk_size=hs_chunk_size,
        )
        with _native_reader_imports(native_root):
            if feature_provider is None:
                from rl_controller.scripts.rl_train import _load_head_runtime

                tokenizer, provider, head, frozen, _ = _load_head_runtime(
                    args, torch.device(device)
                )
                scoring_backend = "hf"
            else:
                from luh import AutoUncertaintyHead

                if base_config is None:
                    from transformers import AutoConfig

                    base_config = AutoConfig.from_pretrained(
                        BACKBONE, revision=BACKBONE_REVISION, cache_dir=cache_dir
                    )
                if (
                    base_config.model_type != "qwen3"
                    or base_config.hidden_size != 2048
                    or base_config.num_hidden_layers != 28
                ):
                    raise ValueError("BELT reader requires the Qwen3-1.7B base config")
                base = SimpleNamespace(config=base_config)
                head = AutoUncertaintyHead.from_pretrained(str(template), base).to(
                    device
                )
                frozen = AutoUncertaintyHead.from_pretrained(str(template), base).to(
                    device
                )
                frozen.eval()
                for parameter in frozen.parameters():
                    parameter.requires_grad_(False)
                provider = feature_provider
                scoring_backend = getattr(provider, "backend", "injected")
        head.load_state_dict(payload["head"], strict=True)
        head.eval()
        state_fn = HeadStateFn(
            provider=provider,
            head=head,
            frozen=frozen,
            device=torch.device(device),
            raw2048=False,
        )
        state_fn.incremental = False
        w = payload["readout"]["weight"]
        if w.shape[0] != 4:
            raise ValueError("BELT readout must contain all four targets")
        readout = torch.nn.Linear(w.shape[1], 4).to(device).eval()
        readout.load_state_dict(payload["readout"], strict=True)
        return cls(
            tokenizer,
            state_fn,
            readout,
            record["sha256"],
            {
                "checkpoint_repo": checkpoint_repo,
                "checkpoint_revision": CHECKPOINT_REVISIONS[checkpoint_repo],
                "seed": seed,
                "backbone": BACKBONE,
                "backbone_revision": BACKBONE_REVISION,
                "device": str(device),
                "dtype": dtype,
                "scoring_backend": scoring_backend,
            },
        )

    def score_token_ids(self, prompt_ids, completion_ids, block_ends):
        """Score a prefix with all exclusive completion-token block ends."""
        if not prompt_ids or not completion_ids:
            raise ValueError("BELT needs nonempty prompt and completion token IDs")
        if not block_ends or block_ends[-1] != len(completion_ids):
            raise ValueError("Last block end must equal completion length")
        if any(type(e) is not int or e < 1 for e in block_ends) or any(
            b <= a for a, b in zip(block_ends, block_ends[1:])
        ):
            raise ValueError("Block ends must be strictly increasing positive integers")
        import torch

        with self._lock, torch.inference_mode():
            try:
                output = self.state_fn(
                    prompt_ids, completion_ids, len(block_ends), block_ends=block_ends
                )
                hidden = output["repr"].float()
                probabilities = torch.sigmoid(self.readout(hidden)).tolist()
                return BeltSignal(
                    dict(zip(TARGETS, probabilities)),
                    tuple(hidden.tolist()),
                    len(completion_ids),
                    self.checkpoint_sha256,
                )
            finally:
                self.state_fn.clear_store()

    def score_prefix(self, chat, steps):
        """Score the joined step text with an end marker at every BELT block end.

        Blocks are found in the text by BELT's own separator rule, not taken
        from how the steps were cut, so a step holding several blocks or a cut
        inside a block places the same markers. The prefix tip is always the
        last end; trailing whitespace after a block end belongs to that block.
        Generators rebuild the next prompt from the joined step text, so the
        prefix is tokenized once, as the model reads it, and a block ends after
        the last token that starts inside it (a ".\\n\\n" token closes its block).
        Call score_token_ids for exact replay of an existing tokenized bank.
        """
        if not chat or chat[-1].get("role") == "assistant":
            raise ValueError("Pass chat before the assistant turn and explicit steps")
        prompt_ids = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, enable_thinking=True
        )
        pieces = []
        for step in steps:
            piece = step if isinstance(step, str) else (step.raw_text or step.text)
            if not isinstance(piece, str) or not piece:
                raise ValueError("BELT steps must contain nonempty text")
            pieces.append(piece)
        if not pieces:
            raise ValueError("BELT needs at least one reasoning step")
        text = "".join(pieces)
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        completion_ids = list(encoded["input_ids"])
        starts = [start for start, _ in encoded["offset_mapping"]]
        cuts = block_end_offsets(text)
        if cuts and not text[cuts[-1] :].strip():
            cuts[-1] = len(text)
        else:
            cuts.append(len(text))
        ends = []
        for cut in cuts:
            end = bisect.bisect_left(starts, cut)
            # A block swallowed whole by the previous token adds no new end.
            if end > (ends[-1] if ends else 0):
                ends.append(end)
        return self.score_token_ids(list(prompt_ids), completion_ids, ends)

    def cleanup(self):
        with self._lock:
            self.state_fn.clear_store()
