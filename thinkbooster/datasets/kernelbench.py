"""
KernelBench dataset loader for thinkbooster.

This module loads KernelBench dataset and generates prompts using KernelAct's
prompt generation system to maintain consistency with the original evaluation.

Usage in run_tts_eval.py:
    from thinkbooster.datasets.kernelbench import load_kernelbench_with_prompts

    kb_data = load_kernelbench_with_prompts(
        level=1,
        prompt_type="improve",
        trial=1,
        subset_size=None
    )
    dataset = Dataset.from_list(kb_data)
"""

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

# Load KernelAct modules from their directory to avoid conflicts with scripts/utils.
# KernelAct internally does "from utils import ..." so we must temporarily register
# modules under bare names, then restore the originals to avoid polluting sys.modules.
_kernelact_dir = Path(__file__).parent / "KernelAct" / "kernelact"

_KERNELACT_MODULE_NAMES = ["utils", "utils_inference", "prompts_kb", "prompts_v2"]


def _load_kernelact_module(module_name: str, register_as: str = None):
    """Load a module from KernelAct directory and register it in sys.modules."""
    key = register_as or module_name
    spec = importlib.util.spec_from_file_location(
        key, _kernelact_dir / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE executing so internal imports find it
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def _load_kernelact_modules():
    """Load KernelAct modules, then restore original sys.modules entries."""
    # Save any existing modules that would be shadowed
    saved = {name: sys.modules.get(name) for name in _KERNELACT_MODULE_NAMES}

    # Load KernelAct modules (registers under bare names for internal imports)
    modules = {}
    modules["utils"] = _load_kernelact_module("utils", register_as="utils")
    modules["utils_inference"] = _load_kernelact_module(
        "utils_inference", register_as="utils_inference"
    )
    modules["prompts_kb"] = _load_kernelact_module(
        "prompts_kb", register_as="prompts_kb"
    )
    modules["prompts_v2"] = _load_kernelact_module(
        "prompts_v2", register_as="prompts_v2"
    )

    # Restore original modules (or remove if they didn't exist before)
    for name in _KERNELACT_MODULE_NAMES:
        if saved[name] is not None:
            sys.modules[name] = saved[name]
        else:
            sys.modules.pop(name, None)

    return modules


try:
    _ka = _load_kernelact_modules()
    choose_prompt = _ka["prompts_v2"].choose_prompt
    extract_code = _ka["utils_inference"].extract_code
except (FileNotFoundError, ModuleNotFoundError):
    # KernelAct not installed — these will fail at use time with a clear error
    choose_prompt = None
    extract_code = None

log = logging.getLogger(__name__)


def load_kernelbench_with_prompts(
    level: int = 1,
    prompt_type: str = "improve",
    trial: int = 1,
    subset_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load KernelBench dataset and generate prompts using KernelAct's choose_prompt.

    This generates prompts that match the format used in:
    python kernelact/run_inference_test_time_scaling.py \\
        --model_name "openai/gpt-oss-120b" \\
        --tts_service_url http://localhost:8001/v1 \\
        --tts_strategy offline_bon \\
        --prompt_type improve \\
        --trial 1

    Args:
        level: Dataset level (1, 2, or 3)
        prompt_type: Type of prompt ("improve", "kernelbench", "normal", etc.)
        trial: Trial number (affects prompt generation for TTS)
        subset_size: If provided, only load first N examples

    Returns:
        List of dicts with formatted data for thinkbooster evaluation pipeline
    """
    if choose_prompt is None:
        raise ImportError(
            "KernelAct is required for KernelBench dataset. "
            "Install it via: ./setup.sh"
        )

    log.info(
        f"Loading KernelBench level_{level} with prompt_type={prompt_type}, trial={trial}..."
    )

    data_repo = "ai-nikolai/KernelBench"
    split = f"level_{level}"

    try:
        dataset = load_dataset(data_repo)
        dataset = dataset[split]
    except Exception as e:
        log.error(f"Failed to load KernelBench: {e}")
        raise

    formatted_data = []

    for item in dataset:
        # Extract fields from dataset
        reference_code = item.get("code", "")
        problem_id = item.get("problem_id", "")
        name = item.get("name", "")

        # Create sample dict that mimics KernelAct's sample format
        sample = {
            "code": reference_code,
            "problem_id": problem_id,
            "name": name,
        }

        # Generate prompt using KernelAct's choose_prompt
        # This matches the logic in run_inference_test_time_scaling.py:
        # prompt_func, prompt_category = choose_prompt(sample, trial, prompt_type)
        # prompt = prompt_func(sample, **kwargs)
        try:
            prompt_func, prompt_category = choose_prompt(sample, trial, prompt_type)
            prompt = prompt_func(sample)
            log.debug(
                f"Generated prompt for problem_id={problem_id}: {prompt_category}"
            )
        except Exception as e:
            log.warning(f"Failed to generate prompt for problem_id={problem_id}: {e}")
            # Fallback to basic prompt
            prompt = _create_fallback_prompt(reference_code)

        formatted = {
            # Standard fields for thinkbooster evaluation pipeline
            "question": prompt,  # This is what thinkbooster uses as the prompt
            "answer": reference_code,  # Reference implementation for comparison
            # KernelBench specific fields (for evaluation and debugging)
            "problem_id": problem_id,
            "name": name,
            "level": level,
            "reference_code": reference_code,
            "prompt_category": prompt_category,  # For tracking which prompt was used
        }
        formatted_data.append(formatted)

        if subset_size and len(formatted_data) >= subset_size:
            break

    log.info(f"Loaded {len(formatted_data)} KernelBench problems (level {level})")
    return formatted_data


def _create_fallback_prompt(reference_code: str) -> str:
    """
    Create a fallback prompt if KernelAct prompt generation fails.
    """
    return f"""You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation of a Model(), your job will be to rewrite it using efficient CUDA Kernels.

Here is the target pytorch implementation:
```python
{reference_code}
```

You need to output an inline CUDA kernel that can be compiled with pytorch and a pytorch nn.Module that you must call `ModelNew`. Use torch.utils.cpp_extension.load_inline for JIT compilation.

Now implement a more efficient solution.
"""


def extract_code_from_response(response: str, model_name: Optional[str] = None) -> str:
    """
    Extract Python/CUDA code from model response.

    This is a wrapper around KernelAct's extract_code function.
    Handles various formats including gpt-oss thinking mode.

    Args:
        response: Model's response text
        model_name: Optional model name for special handling (e.g., gpt-oss)

    Returns:
        Extracted code string
    """
    return extract_code(response, model_name=model_name)


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing KernelBench loader ===\n")

    # Load small subset from level 1
    data = load_kernelbench_with_prompts(
        level=1, prompt_type="improve", trial=1, subset_size=3
    )

    print(f"Loaded {len(data)} problems\n")

    for i, item in enumerate(data[:3]):
        print(f"Problem {i + 1}:")
        print(f"  Problem ID: {item['problem_id']}")
        print(f"  Name: {item['name']}")
        print(f"  Level: {item['level']}")
        print(f"  Prompt Category: {item.get('prompt_category', 'N/A')}")
        print(f"  Prompt preview: {item['question'][:200]}...")
        print()
