"""
KernelBench dataset loader for llm-tts.

This module loads KernelBench dataset and generates prompts using KernelAct's
prompt generation system to maintain consistency with the original evaluation.

Usage in run_tts_eval.py:
    from llm_tts.datasets.kernelbench import load_kernelbench_with_prompts

    kb_data = load_kernelbench_with_prompts(
        level=1,
        prompt_type="improve",
        trial=1,
        subset_size=None
    )
    dataset = Dataset.from_list(kb_data)
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add KernelAct to path to import utilities
KERNELACT_PATH = Path(__file__).parents[4] / "KernelAct"
if str(KERNELACT_PATH) not in sys.path:
    sys.path.insert(0, str(KERNELACT_PATH))

from datasets import load_dataset

# Import from KernelAct
from .kernelact.prompts_v2 import choose_prompt
from .kernelact.utils_inference import extract_code

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
        List of dicts with formatted data for llm-tts evaluation pipeline
    """
    log.info(f"Loading KernelBench level_{level} with prompt_type={prompt_type}, trial={trial}...")

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
            log.debug(f"Generated prompt for problem_id={problem_id}: {prompt_category}")
        except Exception as e:
            log.warning(f"Failed to generate prompt for problem_id={problem_id}: {e}")
            # Fallback to basic prompt
            prompt = _create_fallback_prompt(reference_code)

        formatted = {
            # Standard fields for llm-tts evaluation pipeline
            "question": prompt,  # This is what llm-tts uses as the prompt
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
        level=1,
        prompt_type="improve",
        trial=1,
        subset_size=3
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
