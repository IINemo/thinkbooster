"""Utilities for managing evaluation results (reading, writing, resuming)."""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

from thinkbooster.generators import StepCandidate

log = logging.getLogger(__name__)


def safe_serialize(obj):
    """Convert arbitrary Python objects (including tensors, numpy) to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_serialize(v) for v in obj]
    if isinstance(obj, StepCandidate):
        return {
            "text": obj.text,
            "token_ids": list(obj.token_ids) if obj.token_ids is not None else None,
            "is_complete": obj.is_complete,
            "is_trajectory_complete": obj.is_trajectory_complete,
            "generation_scores": safe_serialize(obj.generation_scores),
            "raw_text": obj.raw_text,
            "other_data": (
                safe_serialize(
                    {k: v for k, v in obj.other_data.items() if k != "raw_logprobs"}
                )
                if obj.other_data
                else None
            ),
        }
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return safe_serialize(vars(obj))

    # Fallback to string representation
    return str(obj)


def save_results_json(results, json_path: Path):
    """
    Save results to a JSON file with safe serialization.

    Args:
        results: List of result dictionaries
        json_path: Path to save the JSON file
    """
    json_path = Path(json_path)
    json_data = [safe_serialize(r) for r in results]
    # Write then rename: a kill mid-write must not leave a truncated file that
    # resume would read as "nothing processed".
    tmp_path = json_path.with_name(json_path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, json_path)


def results_file_readable(json_path: Path) -> bool:
    """False only when the file exists but is not a JSON list of results."""
    json_path = Path(json_path)
    if not json_path.exists():
        return True
    try:
        with open(json_path, encoding="utf-8") as f:
            return isinstance(json.load(f), list)
    except (OSError, ValueError):
        return False


def load_results_json(json_path: Path):
    """
    Load existing results from a JSON file for resuming evaluation.

    Args:
        json_path: Path to the results.json file

    Returns:
        Tuple of (results, processed_indices)
    """
    json_path = Path(json_path)

    if not json_path.exists():
        return [], set()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # Progressive snapshots (e.g. offline BoN before scoring) carry a
        # scoring_phase and are kept on failure for inspection only. Treating
        # them as processed would turn unscored rows into final results.
        snapshots = [r for r in results if "scoring_phase" in r]
        if snapshots:
            log.warning(
                f"Ignoring {len(snapshots)} unfinished progressive-snapshot rows "
                f"in {json_path}; those samples will be regenerated"
            )
            results = [r for r in results if "scoring_phase" not in r]

        # Extract processed indices
        processed_indices = {r["index"] for r in results if "index" in r}

        log.info(f"Loaded {len(results)} existing results from {json_path}")
        log.info(f"Resuming from {len(processed_indices)} processed samples")

        return results, processed_indices

    except Exception as e:
        log.warning(f"Failed to load existing results from {json_path}: {e}")
        return [], set()


def prune_jsonl_rows(jsonl_path: Path, keep_indices, key: str = "index"):
    """Keep the last line for each sample in keep_indices and drop the rest.

    Per-chunk JSONL files are appended before results.json is saved. On resume,
    samples missing from results.json are regenerated and appended again, so
    their earlier lines (and a line torn by a hard kill) must go.

    Returns:
        Number of dropped lines
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return 0
    rows, dropped = {}, 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                index = json.loads(line).get(key)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if index in keep_indices:
                dropped += index in rows
                rows[index] = line if line.endswith("\n") else line + "\n"
            else:
                dropped += 1
    if dropped:
        tmp_path = jsonl_path.with_name(jsonl_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(rows.values())
        os.replace(tmp_path, jsonl_path)
        log.warning(
            f"Dropped {dropped} stale or duplicate lines from {jsonl_path}; "
            f"kept {len(rows)} samples present in results.json"
        )
    return dropped


def find_latest_output_dir():
    """Find the most recent output directory."""
    outputs_root = Path("outputs")
    if not outputs_root.exists():
        return None

    # Find all timestamped directories (YYYY-MM-DD/HH-MM-SS)
    all_dirs = []
    for date_dir in outputs_root.iterdir():
        if date_dir.is_dir() and date_dir.name.count("-") == 2:  # YYYY-MM-DD format
            for time_dir in date_dir.iterdir():
                if (
                    time_dir.is_dir() and time_dir.name.count("-") == 2
                ):  # HH-MM-SS format
                    all_dirs.append(time_dir)

    if not all_dirs:
        return None

    # Sort by modification time, return most recent
    latest_dir = max(all_dirs, key=lambda p: p.stat().st_mtime)
    return latest_dir


def parse_resume_arguments():
    """
    Parse custom --resume and --resume-from arguments.

    Returns the resume directory path if found, otherwise None.
    Modifies sys.argv to remove custom arguments and inject hydra.run.dir override.
    """
    resume_dir = None
    args_to_remove = []

    for i, arg in enumerate(sys.argv):
        if arg == "--resume":
            # Find latest directory
            resume_dir = find_latest_output_dir()
            if not resume_dir:
                print("ERROR: No previous output directories found to resume from.")
                sys.exit(1)
            args_to_remove.append(i)
            print(f"Resuming from latest directory: {resume_dir}")

        elif arg == "--resume-from":
            # Get the next argument as the directory path
            if i + 1 >= len(sys.argv):
                print("ERROR: --resume-from requires a directory path argument")
                sys.exit(1)
            resume_dir = Path(sys.argv[i + 1])
            if not resume_dir.exists():
                print(f"ERROR: Resume directory does not exist: {resume_dir}")
                sys.exit(1)
            args_to_remove.extend([i, i + 1])
            print(f"Resuming from specified directory: {resume_dir}")

    # Remove custom arguments from sys.argv
    for idx in sorted(args_to_remove, reverse=True):
        sys.argv.pop(idx)

    # If resuming, inject hydra.run.dir override
    if resume_dir:
        hydra_override = f"hydra.run.dir={resume_dir}"
        sys.argv.append(hydra_override)

    return resume_dir
