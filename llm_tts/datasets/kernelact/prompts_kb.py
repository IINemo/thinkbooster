# src/prompt_constructor_toml.py | toml based prompt constructor
"""
This is the KernelBench Prompt...
"""
import os
import runpy
import tomli  
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

def read_file(file_path) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

"""
TOML-based prompt constructor for managing prompt templates and configurations.
This module provides a way to load and compose prompt templates from a TOML configuration file.

You can easily check some of the prompt templates we have provided and create your own.
"""

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROMPTS_TOML = os.path.join(REPO_TOP_PATH, "kernelact/kb_prompts/prompts.toml")

assert os.path.exists(PROMPTS_TOML), f"Prompts.toml not found at {PROMPTS_TOML}" 
GPU_SPECS_PY = "kernelact/kb_prompts/hardware/gpu_specs.py"
HARDWARE_COMPONENT_KEYS = [
    "hardware_header",
    "hardware_specs",
    "hardware_definitions",
    "hardware_best_practices",
]

def _abs_path(rel: str) -> str:
    """Convert relative path to absolute path from repo root."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(REPO_TOP_PATH, rel)

@dataclass
class PromptConfig:
    """
    Configuration wrapper for prompts.toml data.
    
    This class holds the parsed TOML file data and provides ways to navigate 
    the nested structure and compose prompt templates.
    
    The TOML file has a  structure like:
        [backends.cuda]
        [options.few_shot]
        [templates.common.arch_block]
    
    This class makes it easy to look up values in that hierarchy.
    """
    data: Dict[str, Any]  # The raw parsed TOML data as nested dictionaries

    @classmethod
    def from_toml(cls, path: str) -> "PromptConfig":
        """
        Load and parse a TOML configuration file.
        
        Args:
            path: Filesystem path to the prompts.toml file
            
        Returns:
            PromptConfig instance with parsed data
        """
        with open(path, "rb") as f:
            data = tomli.load(f)
        return cls(data)

    def compose_blocks(self, keys: List[str]) -> str:
        """
        Look up and concatenate multiple template blocks using dotted key paths.
        
        This method navigates the nested TOML structure using dotted notation
        (e.g., "templates.common.arch_block") to find template strings, then
        concatenates them together with newlines.
        
        Args:
            keys: List of dotted key paths (e.g., ["templates.common.arch_block"])
                  Each key is split on "." and used to traverse the nested dict.
                  
        Returns:
            Concatenated string of all template blocks, each separated by newlines
        """
        text_parts = []
        for key in keys:
            # Navigate through the nested dictionary structure
            node: Any = self.data
            for part in key.split("."):
                if part not in node:
                    raise KeyError(f"compose key not found: {key}")
                node = node[part]
            
            # Ensure we found a string template, not another dict/list
            if not isinstance(node, str):
                raise TypeError(f"compose key must resolve to string: {key}")
            
            text_parts.append(node.strip() + "\n")
        
        return "\n".join(text_parts).strip() + "\n"

def _gpu_context_from_gpu_specs(py_path: str, gpu_name: str) -> Dict[str, str]:
    """
    Load GPU_* dicts from the GPU specs file (no exec of raw strings; use runpy).
    Expected globals:
      - GPU_SPEC_INFO: dict[str, dict]
      - GPU_DEFINITIONS: dict[str, str]
      - GPU_BEST_PRACTICES: list[str]  OR {"list": [...]} for compatibility
    """
    mod = runpy.run_path(py_path)
    spec_info = mod.get("GPU_SPEC_INFO", {})
    definitions = mod.get("GPU_DEFINITIONS", {})
    best = mod.get("GPU_BEST_PRACTICES", [])

    if not spec_info or not definitions or best is None:
        raise ValueError("GPU_SPEC_INFO / GPU_DEFINITIONS / GPU_BEST_PRACTICES missing in gpu specs .py")

    if isinstance(best, dict) and "list" in best:
        best = best["list"]

    if gpu_name not in spec_info:
        raise KeyError(f"GPU name {gpu_name} not found in GPU_SPEC_INFO")

    curr = spec_info[gpu_name]
    gpu_architecture = curr.get("GPU Architecture", "Unknown")
    specs_bullets = "\n".join([f"- We have {v} of {k}." for k, v in curr.items() if k != "GPU Architecture"])
    defs_bullets = "\n".join([f"- {k}: {v}" for k, v in definitions.items()])
    best_bullets = "\n".join([f"- {x}" for x in (best or [])])

    return {
        "gpu_name": gpu_name,
        "gpu_architecture": gpu_architecture,
        "gpu_specs_bullets": specs_bullets,
        "gpu_definitions_bullets": defs_bullets,
        "gpu_best_practices_bullets": best_bullets,
    }

def render_prompt_by_option(
    *,
    prompts_toml: str,
    backend: str,
    option: str,
    context: Dict[str, str],
    gpu_specs_py: Optional[str] = None,
    gpu_name: Optional[str] = None,
    precision: Optional[str] = None,
    include_hardware: bool = False,
    components_override: Optional[List[str]] = None,
) -> str:
    """
    Render a prompt using backends.X and options.Y structure from TOML.
    
    Args:
        prompts_toml: Path to the prompts.toml file
        backend: The kernel backend (triton, cuda, cute, tilelang)
        option: The prompt option (zero_shot, one_shot, few_shot)
                - zero_shot: No examples (model learns from description only)
                - one_shot: Single example
                - few_shot: Multiple examples if available for backend, otherwise falls back to one_shot
        context: Variables to fill in the prompt template
        gpu_specs_py: Optional path to GPU specs Python file (required if hardware info is included)
        gpu_name: Optional GPU name (required if hardware info is included)
        precision: Optional precision string (fp32, fp16, bf16) - defaults to fp32 if not provided
        include_hardware: Whether to inject hardware guidance blocks after the examples section
        components_override: When provided, users can arrange prompt components from the toml
                             file in any order they want.
                             Components must exist under templates.common or be hardware_* entries.
    
    Returns:
        The rendered prompt string
    """
    cfg = PromptConfig.from_toml(prompts_toml)
    
    # Get backend-specific content
    try:
        backend_data = cfg.data["backends"][backend]
    except KeyError:
        raise KeyError(f"Unknown backend: {backend}")
    
    # Get option configuration
    try:
        option_data = cfg.data["options"][option]
    except KeyError:
        raise KeyError(f"Unknown option: {option}")

    component_sequence = list(components_override or option_data["components"])
    if include_hardware:
        if components_override is None:
            insert_idx = component_sequence.index("arch_block") if "arch_block" in component_sequence else len(component_sequence)
            component_sequence[insert_idx:insert_idx] = HARDWARE_COMPONENT_KEYS
        else:
            # Custom sequences must explicitly have hardware blocks present in their prompt if they 
            # have set they are including hardware info. 
            if not any(component in HARDWARE_COMPONENT_KEYS for component in component_sequence):
                raise ValueError(
                    "components_override must contain at least one hardware_* entry when include_hardware=True"
                )
    
    # Get shared templates
    shared = cfg.data.get("shared", {})
    backend_display = backend_data.get("backend_display", backend.upper())
    
    # Fill in shared templates with backend-specific terms
    problem_statement = shared.get("problem_statement", "").format(backend_display=backend_display)
    instruction = shared.get("instruction", "").format(backend_display=backend_display)
    
    # Add backend-specific content to context
    context = {
        **context,
        "backend": backend.upper() if backend in ["cuda", "cute"] else backend.capitalize(),
        "backend_display": backend_display,
        "problem_statement": problem_statement,
        "instruction": instruction,
    }
    
    # Load precision details if provided
    if precision:
        try:
            precision_data = cfg.data["precision"][precision]
            context["precision_display"] = precision_data.get("precision_display", precision.upper())
        except KeyError:
            raise KeyError(f"Unknown precision: {precision}. Must be one of: fp32, fp16, bf16")
    else:
        # Default to fp32 if not specified
        default_precision = cfg.data.get("meta", {}).get("default_precision", "fp32")
        precision_data = cfg.data["precision"].get(default_precision, {})
        context["precision_display"] = precision_data.get("precision_display", "FP32 (32-bit floating point)")
    
    # Load example files if requested. Supports loading one shot or few shot examples. 
    requires_example = option_data.get("requires_example")
    if requires_example:
        example_entry_template = cfg.compose_blocks(["templates.common.example_entry_template"]).strip()
        intro_one_shot = cfg.compose_blocks(["templates.common.example_intro_one_shot"]).strip()
        intro_few_shot = cfg.compose_blocks(["templates.common.example_intro_few_shot"]).strip()
        intro_one_shot = intro_one_shot.format(
            backend_display=backend_display
        )
        intro_few_shot = intro_few_shot.format(
            backend_display=backend_display
        )

        def render_example_entry(input_code: str, output_code: str, example_label: str) -> str:
            return example_entry_template.format(
                example_label=example_label,
                input_code=input_code,
                output_code=output_code,
                backend_display=backend_display,
            )

        examples_entries: List[str] = []
        examples_intro = intro_one_shot

        if requires_example == "few_shot":
            # Try to load few-shot examples if available
            few_shot_examples = backend_data.get("few_shot_examples")

            if few_shot_examples and len(few_shot_examples) > 0:
                # Use multiple examples (true few-shot)
                examples_intro = intro_few_shot
                for i, (input_path, output_path) in enumerate(few_shot_examples, 1):
                    input_code = read_file(_abs_path(input_path))
                    output_code = read_file(_abs_path(output_path))
                    examples_entries.append(
                        render_example_entry(input_code, output_code, f"Example {i}:")
                    )
            else:
                # Fall back to one-shot
                ex_arch_path = _abs_path(
                    backend_data.get("few_shot_example_arch") or shared.get("few_shot_example_arch")
                )
                ex_new_path = _abs_path(backend_data["one_shot_new_arch"])
                input_code = read_file(ex_arch_path)
                output_code = read_file(ex_new_path)
                examples_entries.append(
                    render_example_entry(input_code, output_code, "Example:")
                )

        elif requires_example == "one_shot":
            # Always use one-shot
            ex_arch_path = _abs_path(
                backend_data.get("few_shot_example_arch") or shared.get("few_shot_example_arch")
            )
            ex_new_path = _abs_path(backend_data["one_shot_new_arch"])
            input_code = read_file(ex_arch_path)
            output_code = read_file(ex_new_path)
            examples_entries.append(
                render_example_entry(input_code, output_code, "Example:")
            )

        if not examples_entries:
            raise ValueError(f"No example entries could be constructed for option '{option}'.")

        context["examples_intro"] = examples_intro
        context["examples_entries"] = "\n\n".join(examples_entries).strip()
    
    # Load GPU details if requested
    if option_data.get("requires_gpu") or include_hardware:
        if not (gpu_specs_py and gpu_name):
            raise ValueError(
                f"Hardware info requested for option '{option}'; provide gpu_specs_py and gpu_name"
            )
        context = {**context, **_gpu_context_from_gpu_specs(_abs_path(gpu_specs_py), gpu_name)}
    
    # Builds the prompt from the components in the toml file. 
    prompt_parts = []
    for component in component_sequence:
        if component == "problem_statement":
            # Use the already-formatted problem_statement from context
            prompt_parts.append(context["problem_statement"])
        elif component == "instruction":
            # Use the already-formatted instruction from context
            prompt_parts.append(context["instruction"])
        elif component.startswith("hardware_"):
            # Hardware components from templates.hardware
            template_key = f"templates.hardware.{component}"
            prompt_parts.append(cfg.compose_blocks([template_key]))
        else:
            # Other components from templates.common
            template_key = f"templates.common.{component}"
            prompt_parts.append(cfg.compose_blocks([template_key]))
    
    prompt_text = "\n".join(prompt_parts).strip() + "\n"
    
    try:
        return prompt_text.format(**context).strip() + "\n"
    except KeyError as e:
        raise KeyError(f"Missing placeholder in context: {e.args[0]}. Available: {list(context.keys())}") from e

# -------------------------------------------------------------------------
# High-level convenience functions
# -------------------------------------------------------------------------

def get_prompt_for_backend(
    ref_arch_src: str,
    backend: str = "triton",
    option: str = "one_shot",
    precision: Optional[str] = None,
    include_hardware: bool = False,
    gpu_name: Optional[str] = None,
) -> str:
    """
    Generate a prompt for a specific backend and option.
    
    Args:
        ref_arch_src: The reference architecture source code
        backend: The kernel backend (triton, cuda, cute, tilelang)
        option: The prompt option (zero_shot, one_shot, few_shot)
        precision: Optional precision (fp32, fp16, bf16) - defaults to fp32 if not provided
        include_hardware: When True, append hardware guidance blocks (requires gpu_name)
        gpu_name: GPU identifier used when include_hardware is True (e.g., "A100")
    """
    return render_prompt_by_option(
        prompts_toml=PROMPTS_TOML,
        backend=backend.lower(),
        option=option.lower(),
        context={"ref_arch_src": ref_arch_src},
        precision=precision,
        include_hardware=include_hardware,
        gpu_specs_py=GPU_SPECS_PY if include_hardware else None,
        gpu_name=gpu_name,
    )


def get_custom_prompt(
    custom_key: str,
    *,
    ref_arch_src: str,
    backend: str,
    option: str,
    precision: Optional[str] = None,
    include_hardware: bool = False,
    gpu_name: Optional[str] = None,
    prompts_toml: str = PROMPTS_TOML,
) -> str:
    """
    Render a prompt defined under [custom_prompts.<custom_key>] in prompts.toml.
    Must still provide backend/option/precision settings just like
    get_prompt_for_backend. 
    """
    if not ref_arch_src:
        raise ValueError(f"Custom prompt '{custom_key}' requires ref_arch_src.")
    cfg = PromptConfig.from_toml(prompts_toml)
    try:
        custom_cfg: Dict[str, Any] = cfg.data["custom_prompts"][custom_key]
    except KeyError as exc:
        raise KeyError(f"Unknown custom prompt: {custom_key}") from exc

    components_override = custom_cfg.get("components")

    return render_prompt_by_option(
        prompts_toml=prompts_toml,
        backend=backend.lower(),
        option=option.lower(),
        context={"ref_arch_src": ref_arch_src},
        precision=precision,
        include_hardware=include_hardware,
        gpu_specs_py=GPU_SPECS_PY if include_hardware else None,
        gpu_name=gpu_name,
        components_override=components_override,
    )

__all__ = [
    "get_prompt_for_backend",
    "get_custom_prompt",
    "get_prompt_with_hardware",
    "render_prompt_by_option",
    "PromptConfig",
]


def log_prompt(prompt: str, dir_path: str, file_name: str):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, file_name), "w") as f:
        f.write(prompt)

def test_prompt():
    """
    Demonstrate baseline, few-shot, DSL, hardware-aware, and custom prompt
    generation. Customize the reference architecture or custom_prompt_key
    if you want to try different inputs.
    """
    REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ref_arch_src = read_file(os.path.join(REPO_TOP_PATH, "archive", "kb_solutions", "level1","2_Standard_matrix_multiplication_.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"   
    
    scratch_dir = os.path.join(REPO_TOP_PATH, "scratch")
    # baseline prompt
    baseline_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="cuda",
        option="one_shot",
        precision="fp32",
        # GPU platform agnostic for baseline
    )
    log_prompt(baseline_prompt, os.path.join(scratch_dir), "baseline_prompt.txt")

    # few shot prompt
    few_shot_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="cuda",
        option="few_shot",
        precision="fp32",
    )
    log_prompt(few_shot_prompt, os.path.join(scratch_dir), "few_shot_prompt.txt")

    # DSL prompt
    dsl_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="triton",
        option="one_shot",
        precision="fp32",
    )
    log_prompt(dsl_prompt, os.path.join(scratch_dir), "dsl_prompt.txt")

    # hardware prompt
    hardware_prompt = get_prompt_for_backend(
        ref_arch_src=ref_arch_src,
        backend="cute",
        option="one_shot",
        precision="fp32",
        include_hardware=True,
        gpu_name="L40S",
    )
    log_prompt(hardware_prompt, os.path.join(scratch_dir), "hardware_prompt.txt")

    # custom prompt defined in prompts.toml
    custom_prompt = get_custom_prompt(
        # the key is whatever you name the prompt in the custom_prompts section of the toml file
        custom_key="custom",
        
        ref_arch_src=ref_arch_src,
        backend="triton",
        option="one_shot",
        precision="fp32",
        include_hardware=True,
        gpu_name="L40S",
    )
    log_prompt(custom_prompt, os.path.join(scratch_dir), "custom_prompt.txt")




# TTS PROMPTS FOR KERNELBENCH STYLE PROMPTs...
def kb_add_instruction(custom_instruction=None):
    """Simple add on"""
    prompt = "Note: The kernels should be optimized for FP32 (32-bit floating point) precision."
    prompt += "\n\n"
    if custom_instruction:
        prompt += custom_instruction
    else:
        prompt += "Optimize the architecture named Model with custom CUDA operators!"
    
    prompt+=" Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!"
    
    prompt += "\n\n"
    return prompt


def kb_instruction(custom_instruction=None):
    if custom_instruction:
        instruction = custom_instruction 
    else:
        instruction = "You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups."

    prompt = instruction + "\n\n" + "You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA operators and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination."

    return prompt


def kb_tts_base(code_src, generate_instruction=False, custom_instruction=None):
    """Base Prompt for KB TTS, that should be equivalent to the normal KB prompt... (therefore do not use explicitly)"""
    prompt = kb_instruction(custom_instruction)
    prompt += "\n\n"
    prompt += \
'''Here's an example to show you the syntax of inline embedding custom CUDA operators in PyTorch:

Example:

Input architecture:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []


Optimized with CUDA operators:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)

You are given the following architecture:


'''
    prompt += code_src
    prompt += "\n\n"

    if generate_instruction:
        if False: #NOTE: it is added now explicitly...
            prompt += kb_add_instruction()
    
    return prompt


def get_prompt_kb_tts(code_src, generated_code, case, error_message=None, score=None):
    """Base Prompt for KB TTS, that should be equivalent to the normal KB prompt... (therefore do not use explicitly)"""

    prompt = kb_tts_base(code_src)
    prompt += "\n\n\n"

    prompt +="Here is your previous attempt:" #Here are your previous attempts:
    prompt += "\n\n"

    prompt += generated_code
    prompt += "\n\n"

    if case == "compilation_error":
        prompt += f"Your previous answer failed to compile. Here is the error message:\n\n{error_message}"

    elif case == "runtime_error":
        prompt += f"Your previous answer compiled successfully but had runtime errors. Here is the error message:\n\n{error_message}"

    elif case == "incorrect":
        prompt += f"Your previous answer was incorrect. Here is the error message:\n\n{error_message}"
    
    elif case == "improve":
        prompt += f"Your previous answer was correct but can be made faster. Here is the speedup you achieved relative to the baseline: {score}"
    else:
        raise Exception(f"Something is going wrong with generating KB TTS prompt. Case does not exist: {case}")

    prompt += "\n\n"
    prompt += "Restart your reasoning process and generate new, complete code.\n\n"
    prompt += kb_add_instruction()

    return prompt


def get_prompt_kb_tts_multi_stage(code_src, generated_code, case, error_message=None, score=None):
    """Base Prompt for KB TTS, that should be equivalent to the normal KB prompt... (therefore do not use explicitly)"""

    if case == "compilation_error":
        custom_instruction = "You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups."
        appendix = f"Unfortunately, your previous answer failed to compile. Think carefully about how to solve the compilation issues. Here is the error message:\n\n{error_message}"
        custom_final_instruction = "Remember your key objective now is to debug the code and arrive at a correct CUDA Implementation."
    #La illaha il Allah, Muhammadan Rasul Allah, sal ALlah alyahi wa salim
    elif case == "runtime_error":
        custom_instruction = "You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups."
        appendix = f"Your previous answer compiled successfully but had runtime errors. Think carefully how to solve the runtime issues. Here is the error message:\n\n{error_message}"
        custom_final_instruction = "Remember your key objective now is to debug the code and arrive at a correct CUDA Implementation."
    # La illaha il Allah, Muhammadan Rasul Allah, sal Allah alyahi wa salim
    elif case == "incorrect":
        custom_instruction = "You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups."
        appendix = f"Your previous answered compiled and ran without problems. However, your previous code produced the wrong answer. I.e. either the shape or the numbers were wrong. Think carefully how to make your code produce the correct code. Here is the error message:\n\n{error_message}"
        custom_final_instruction = "Remember your key objective now is to make sure your code produces the correct answer."
    elif case == "improve":
        custom_instruction = "You write custom CUDA operators to replace the pytorch operators in the given architecture to get speedups."
        if score>1:
            appendix = f"Your previous answer was correct but can be made faster. Think carefully how to make the solution faster. Here is the speedup you achieved relative to the baseline: {score:.4f} times faster."
        else:
            appendix = f"Your previous answer was correct but can be made faster. Think carefully how to make the solution faster. Your solution is slower, you got the following slowdown relative to the baseline: {1/(score+1e-8):.4f} times slower."

        custom_final_instruction = "Remember your key objective now is to make the code run faster."
    
    else:
        raise Exception(f"Something is going wrong with generating KB TTS prompt. Case does not exist: {case}")

    prompt = kb_tts_base(code_src, custom_instruction)
    prompt += "\n\n\n"

    prompt +="Here is your previous attempt:" #Here are your previous attempts:
    prompt += "\n\n"

    prompt += generated_code
    prompt += "\n\n"

    prompt += appendix
    prompt += "\n\n"

    prompt += kb_add_instruction(custom_final_instruction)

    return prompt


# def get_reflection_kb(code_src, generated_code, case, error_message=None, score=None):
#     """Base Prompt for KB TTS, that should be equivalent to the normal KB prompt... (therefore do not use explicitly)"""

#     prompt = kb_tts_base(code_src)
#     prompt += "\n\n\n"
#     prompt +="Here is your previous attempt:" #Here are your previous attempts:
#     prompt += "\n\n"
#     prompt += generated_code
#     prompt += "\n\n"
#     if case == "compilation_error":
#         prompt += f"Your previous answer failed to compile. Here is the error message:\n\n{error_message}"

#     elif case == "runtime_error":
#         prompt += f"Your previous answer compiled successfully but had runtime errors. Here is the error message:\n\n{error_message}"

#     elif case == "incorrect":
#         prompt += f"Your previous answer was incorrect. Here is the error message:\n\n{error_message}"
    
#     elif case == "improve":
#         prompt += f"Your previous answer was correct but can be made faster. Here is the speedup you achieved relative to the baseline: {score}"
#     else:
#         raise Exception(f"Something is going wrong with generating KB TTS prompt. Case does not exist: {case}")

#     prompt += "\n\n"
#     prompt += "Restart your reasoning process and generate new, complete code.\n\n"
#     prompt += kb_add_instruction()

#     return prompt


if __name__ == "__main__":
    test_prompt()





# ORIGINAL PROMPTS for TTS, we will provide the updated prompts from KB repo...

# def tts_base_prompt():
#     '''
#     You are given the following architecture:
# 2 import torch
# 3 import torch.nn as nn
# 4
# 5 class Model(nn.Module):
# 6 """
# 7 Simple model that performs Layer Normalization.
# 8 """
# 9 def __init__(self, normalized_shape: tuple):
# 10 """
# 11 Initializes the LayerNorm layer.
# 12
# 13 Args:
# 14 normalized_shape (tuple): Shape of the input tensor to be normalized.
# 15 """
# 16 super(Model, self).__init__()
# 17 self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
# 18
# 19 def forward(self, x: torch.Tensor) -> torch.Tensor:
# 20 """
# 21 Applies Layer Normalization to the input tensor.
# 22
# 23 Args:
# 24 x (torch.Tensor): Input tensor of shape (*, normalized_shape).
# 25
# 26 Returns:
# 27 torch.Tensor: Output tensor with Layer Normalization applied, same
# shape as input.
# 28 """
# 29 return self.ln(x)
# 30
# 31 Replace pytorch operators in the given architecture with raw CUDA kernels,
# optimizing for performance on NVIDIA H100 (e.g. shared memory, kernel fusion,
# warp primitives, vectorization,...). Use torch.utils.cpp_extension.load_inline
# and name your optimized output architecture ModelNew. You are not allowed to
# 18
# use torch.nn (except for Parameter, containers, and init). The input and
# output have to be on CUDA device. Your answer must be the complete new
# architecture (no testing code, no other code): it will be evaluated and you
# will be given feedback on its correctness and speedup so you can keep
# iterating, trying to maximize the speedup. After your answer, summarize your
# changes in a few sentences.Here is an example:
# 32
# 33 import torch.nn as nn
# 34 from torch.utils.cpp_extension import load_inline
# 35
# 36 # Define the custom CUDA kernel for element-wise addition
# 37 elementwise_add_source = """
# 38 #include <torch/extension.h>
# 39 #include <cuda_runtime.h>
# 40
# 41 __global__ void elementwise_add_kernel(const float* a, const float* b, float* out,
# int size) {
# 42 int idx = blockIdx.x * blockDim.x + threadIdx.x;
# 43 if (idx < size) {
# 44 out[idx] = a[idx] + b[idx];
# 45 }
# 46 }
# 47
# 48 torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
# 49 auto size = a.numel();
# 50 auto out = torch::zeros_like(a);
# 51
# 52 const int block_size = 256;
# 53 const int num_blocks = (size + block_size - 1) / block_size;
# 54
# 55 elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(),
# b.data_ptr<float>(), out.data_ptr<float>(), size);
# 56
# 57 return out;
# 58 }
# 59 """
# 60
# 61 elementwise_add_cpp_source = (
# 62 "torch::Tensor␣elementwise_add_cuda(torch::Tensor␣a,␣torch::Tensor␣b);"
# 63 )
# 64
# 65 # Compile the inline CUDA code for element-wise addition
# 66 elementwise_add = load_inline(
# 67 name="elementwise_add",
# 68 cpp_sources=elementwise_add_cpp_source,
# 69 cuda_sources=elementwise_add_source,
# 70 functions=["elementwise_add_cuda"],
# 71 verbose=True,
# 72 extra_cflags=[""],
# 73 extra_ldflags=[""],
# 74 )
# 75
# 76
# 77 class ModelNew(nn.Module):
# 78 def __init__(self) -> None:
# 79 super().__init__()
# 80 self.elementwise_add = elementwise_add
# 81
# 82 def forward(self, a, b):
# 83 return self.elementwise_add.elementwise_add_cuda(a, b)
#     '''
#     pass

# def tts_prompt():
#     """
# <Base prompt containing pytorch architecture and instruction>
# 2
# 3 Here are your previous attempts:
# 4
# 5 < for each (i) previously generated kernel >
# 6 <Previously generated kernel G[i]>
# 7
# 8 <Summary of CoT[i]>
# 9
# 10 <if parsing error>
# 11
# 12 Your previous answer failed to be parsed due to not adhering to the desired
# formatting. Here is the error message: <error_message>
# 13
# 14 <elif compilation error>
# 15
# 16 Your previous answer failed to compile. Here is the error message:
# <error_message>
# 17
# 18 <elif run error>
# 19
# 20 Your previous answer compiled successfully but had runtime errors. Here is
# the error message: <error_message>
# 21
# 22 <elif correctness error>
# 23
# 24 Your previous answer was incorrect. Here is the error message:
# <error_message>
# 25
# 26 <elif correct>
# 27
# 28 Your previous answer was correct but can be made faster. Here is the
# speedup you achieved relative to the baseline: <speedup>
# 29
# 30 Restart your reasoning process and generate new, complete code.
#     """
#     pass