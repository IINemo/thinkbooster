import os
import pandas as pd

from datasets import load_from_disk, load_dataset
from datasets import Dataset

from functools import partial

from pprint import pprint

from .utils import (
    get_folder_path,
    get_file_name
)

from .utils_inference import (
    create_joined_dataset,
    extract_code
)

from .prompts_kb import (
    get_prompt_for_backend,
    get_prompt_kb_tts,
    get_prompt_kb_tts_multi_stage,
)

EXAMPLE_IMPLEMENTATION='''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


# The actual CUDA KERNEL Inline source code.
cuda_src = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void custom_kernel(// this will include the relevant function signature
) {
//The required more efficient CUDA KERNEL.
}

torch::Tensor custom_cuda_forward(// this will include the relevant `torch::Tensor`
) {
//The required torch CUDA function. That will call `custom_kernel`
}
"""


# The CPP Source that declares the CUDA KERNEL.
cpp_src="""
torch::Tensor custom_cuda_forward(//with the same function signature as the CUDA code.
 );
"""


# JIT compile the CUDA kernel
custom_kernel = load_inline(
    name="custom_cuda",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["custom_cuda_forward"],
    verbose=True, 
    extra_cuda_cflags=[], #additional flags
)


class ModelNew(nn.Module):
    """
    The required class ModelNew.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(# The required function signature.
    ) -> torch.Tensor:
        """
        The required forward function.
        """
        # this function should call the CUDA Kernel. 
        #e.g. custom_kernel.custom_cuda_forward( #with correct inputs )  
'''
INSTRUCTION = "You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation of a Model(), your job will be to rewrite it using efficient CUDA Kernels in the exact format as the 'example output' provided below."

REF_CODE_TORCH = "\n\nHere is the target pytorch implementation:\n```python\n{reference_code}\n```"
CODE_TORCH = "\n\nHere is the target pytorch implementation:\n```python\n{reference_code}\n```"
CODE_TORCH_CUDA = "\n\nHere is the target pytorch + cuda implementation:\n```python\n{reference_code}\n```"
BUG_CODE_TORCH_CUDA = "\n\nHere is the buggy pytorch + cuda implementation:\n```python\n{reference_code}\n```"

OUTPUT_FORMAT_CUDA = "\n\nYou need to output an inline CUDA kernel that can be compiled with pytorch and a pytorch nn.Module that you must call `ModelNew`. "
OUTPUT_FORMAT_CUDA += f"Here is an 'example output' with comments:\n```python\n{EXAMPLE_IMPLEMENTATION}\n```"
FINAL_INSTRUCTION = "\n\nNow implement a more efficient solution."

def generate_prompt(sample, **kwargs):
    """Generates a prompt from the reference code..."""
    reference_code = sample["code"]
    output_prompt = ""
    output_prompt += INSTRUCTION
    output_prompt += REF_CODE_TORCH.format(reference_code=reference_code)
    output_prompt += OUTPUT_FORMAT_CUDA
    output_prompt += FINAL_INSTRUCTION

    return output_prompt


INSTRUCTION_TRANSLATE_CUDA = "You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation of a Model(), your job will be to translate it into CUDA."
INSTRUCTION_TRANSLATE_TRITON = "You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation of a Model(), your job will be to translate it into the Triton Language."
INSTRUCTION_TRANSLATE_PYTORCH_FUNC = "You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation of a Model(), your job will be to translate it into the Pytorch functional format."

def generate_translation_prompt(reference_code, translation_lang="cuda", input_lang="torch",**kwargs):
    """This can be used as many pipelines including:
    Pytorch -> PyFunc
    PyFunc (as reference_code) -> CUDA
    (from Sakana's AI CUDA ENGINEER)

    func_code = generate_translation_prompt(vanilla_torch_code, "pytorch_functional")
    # clean_code = clean(func_code)
    cuda_code = generate_translation_prompt(clean_code, "cuda")
    """
    output_prompt = ""
    appendix = ""
    if translation_lang == "cuda":
        output_prompt += INSTRUCTION_TRANSLATE_CUDA
        appendix += OUTPUT_FORMAT_CUDA
    elif translation_lang == "triton":
        output_prompt += INSTRUCTION_TRANSLATE_TRITON
    elif translation_lang == "pytorch_functional":
        output_prompt += INSTRUCTION_TRANSLATE_PYTORCH_FUNC
    else:
        raise Exception(f"The following Translation Lang is not supported yet: {translation_lang}")
    if input_lang == "torch":
        output_prompt += REF_CODE_TORCH.format(reference_code=reference_code)
    elif input_lang == "torch_cuda":
        output_prompt += CODE_TORCH_CUDA.format(reference_code=reference_code)
    else:
        print(f"[WARNING] Don't recognise input lang: {input_lang}")
        output_prompt += CODE_TORCH.format(reference_code=reference_code)

    output_prompt += appendix
    output_prompt += FINAL_INSTRUCTION

    return output_prompt


def get_compile_error_info(sample):
    """Gets info from sample"""
    bug_info = sample.get("compilation_output_runtime", "")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("compilation_output_error","")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("main_output","")
    
    return bug_info


def get_runtime_error_info(sample):
    """Gets info from sample"""
    bug_info = sample.get("correctness_output_error", "")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("correctness_output_traceback","")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("correctness_output","")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("main_output","")

    return bug_info


def get_correctness_error_info(sample):
    """Gets info from sample"""
    bug_info = sample.get("correctness_output_error", "")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("correctness_output_traceback","")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("correctness_output","")

    if not bug_info or (str(bug_info) == "nan"):
        bug_info = sample.get("main_output","")

    return bug_info


INSTRUCTION_DEBUG = "You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation and a buggy CUDA implementation. The problem is that you will see a CUDA implementation that does not compile. Your job is implement a bug-free version."
BUG_INFORMATION = """\nHere is some additional debugging information: 
```bug_info
{bug_info}
```"""

FINAL_INSTRUCTION_DEBUG = "\n\nNow implement a correct solution."

def generate_debug_compilation_prompt(sample, include_code_format=True, **kwargs):
    """
    Generates a prompt for debugging compilation
    """
    reference_code = sample["code"]
    generation_code = extract_code(sample["generation"])

    bug_info = get_compile_error_info(sample)
        
    output_prompt = ""
    output_prompt += INSTRUCTION_DEBUG
    output_prompt += REF_CODE_TORCH.format(reference_code=reference_code)
    output_prompt += BUG_CODE_TORCH_CUDA.format(reference_code=generation_code)
    if bug_info and not str(bug_info)=="nan":
        output_prompt += BUG_INFORMATION.format(bug_info=bug_info)
    if include_code_format:
        output_prompt += OUTPUT_FORMAT_CUDA
    output_prompt += FINAL_INSTRUCTION_DEBUG

    return output_prompt


INSTRUCTION_DEBUG_RUNTIME="You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation and a buggy CUDA implementation. The problem is that there is a runtime issue when running the CUDA implementation. Your job is to implement a bug-free version of the CUDA implementation."

def generate_debug_runtime_prompt(sample, include_code_format=True, **kwargs):
    """
    Generate prompt for when it compiled but fails during correctness check (i.e. runtime error)
    """
    reference_code = sample["code"]
    generation_code = extract_code(sample["generation"])

    bug_info = get_runtime_error_info(sample)

    output_prompt = ""
    output_prompt += INSTRUCTION_DEBUG_RUNTIME
    output_prompt += REF_CODE_TORCH.format(reference_code=reference_code)
    output_prompt += BUG_CODE_TORCH_CUDA.format(reference_code=generation_code)
    if bug_info and not str(bug_info)=="nan":
        output_prompt += BUG_INFORMATION.format(bug_info=bug_info)
    if include_code_format:
        output_prompt += OUTPUT_FORMAT_CUDA
    output_prompt += FINAL_INSTRUCTION_DEBUG

    return output_prompt



INSTRUCTION_DEBUG_CORRECTNESS="You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation and a reference, but wrong CUDA implementation. The problem is that the output of the CUDA programme does not match the target pytorch implementation. Your job is to implement a version of the CUDA implementation that matches the output of the target pytorch implementation."

def generate_correctness_prompt(sample, include_code_format=False, **kwargs):
    """Generates a prompt from the reference code..."""
    reference_code = sample["code"]
    generation_code = extract_code(sample["generation"])
    
    bug_info = get_correctness_error_info(sample)

    output_prompt = ""
    output_prompt += INSTRUCTION_DEBUG_CORRECTNESS
    output_prompt += REF_CODE_TORCH.format(reference_code=reference_code)
    output_prompt += BUG_CODE_TORCH_CUDA.format(reference_code=generation_code)
    if bug_info and not str(bug_info)=="nan":
        output_prompt += BUG_INFORMATION.format(bug_info=bug_info)
    if include_code_format:
        output_prompt += OUTPUT_FORMAT_CUDA
    output_prompt += FINAL_INSTRUCTION_DEBUG

    return output_prompt


INSTRUCTION_IMPROVE = "You are an amazing CUDA Kernel Engineer. You will see a target pytorch implementation and a reference CUDA implementation. Your job is to improve the {param_to_improve} of the reference CUDA Kernel."
PREVIOUS_SCORE = "\nThe {param_to_improve} of this implementation was: **{previous_score}** {measure}"
BASELINE_SCORE = "\nThe baseline {param_to_improve} you need to achieve at least is: **{baseline_score}** {measure}"

def generate_improvement_prompt(sample, param_to_improve="speed", include_code_format=True, **kwargs):
    """Generates a prompt from the reference code..."""

    if param_to_improve == "speed":
        measure = "milliseconds"
    elif param_to_improve == "memory":
        measure = "gigabytes"
    else:
        measure = ""

    reference_code = sample["code"]
    generation_code = extract_code(sample["generation"])

    previous_score = sample.get("runtime")
    baseline_score = sample.get("runtime_original")

    output_prompt = ""
    output_prompt += INSTRUCTION_IMPROVE.format(param_to_improve=param_to_improve)
    output_prompt += REF_CODE_TORCH.format(reference_code=reference_code)
    output_prompt += CODE_TORCH_CUDA.format(reference_code=generation_code)

    if previous_score and previous_score > 0:
        output_prompt += PREVIOUS_SCORE.format(param_to_improve=param_to_improve, previous_score=previous_score, measure=measure)
    if baseline_score and baseline_score > 0:
        output_prompt += BASELINE_SCORE.format(param_to_improve=param_to_improve, baseline_score=baseline_score, measure=measure)
    if include_code_format:
        output_prompt += OUTPUT_FORMAT_CUDA
    output_prompt += FINAL_INSTRUCTION

    return output_prompt


SINGLE_STAGE_IMPROVEMENT_INSTRUCTION = "You are an amazing CUDA Kernel Engineer. You will see a reference pytorch & CUDA implementation. Your job is to improve the code."

def generate_improvement_prompt_single_stage(sample,  include_code_format=True, **kwargs):
    """Generates a prompt from the reference code..."""

    reference_code = sample["code"]
    generation_code = extract_code(sample["generation"])

    # previous_score = sample.get("runtime")
    # baseline_score = sample.get("runtime_original")

    output_prompt = ""
    output_prompt += SINGLE_STAGE_IMPROVEMENT_INSTRUCTION
    output_prompt += REF_CODE_TORCH.format(reference_code=reference_code)
    output_prompt += CODE_TORCH_CUDA.format(reference_code=generation_code)

    bug_info = sample.get("main_output", "")

    if bug_info and not str(bug_info)=="nan":
        output_prompt += BUG_INFORMATION.format(bug_info=bug_info)

    if include_code_format:
        output_prompt += OUTPUT_FORMAT_CUDA
    output_prompt += FINAL_INSTRUCTION

    return output_prompt


def generate_kb_prompt(sample,**kwargs):
    """Official KernelBench Prompt"""
    reference_code = sample["code"]

    baseline_prompt = get_prompt_for_backend(
            ref_arch_src=reference_code,
            backend="cuda",
            option="one_shot",
            precision="fp32",
            # GPU platform agnostic for baseline
        )
    
    return baseline_prompt

def generate_kb_tts_prompt(sample, case, multi_stage=False, **kwargs):
    """ (more or less) Official KB TTS Prompt"""
    reference_code = sample["code"]
    generation_code = extract_code(sample["generation"])


    if case == "compilation_error":
        bug_info = get_compile_error_info(sample)

    elif case == "runtime_error":
        bug_info = get_runtime_error_info(sample)

    elif case == "incorrect":
        bug_info = get_correctness_error_info(sample)
    
    else:
        bug_info = None

    if case=="improve":
        previous_score = sample.get("runtime")
        baseline_score = sample.get("runtime_original")

        # print(f"previous_score: {previous_score}")
        # print(f"baseline_score: {baseline_score}")
        # pprint(sample)

        EPSILON = 1e-8
        if previous_score:
            if baseline_score:
                # score = previous_score / (baseline_score+EPSILON)
                score = baseline_score / (previous_score+EPSILON)

    else:
        score = None

    if multi_stage:
        prompt = get_prompt_kb_tts_multi_stage(reference_code, generation_code, case=case, error_message=bug_info, score=score)

    else:
        prompt = get_prompt_kb_tts(reference_code, generation_code, case=case, error_message=bug_info, score=score)

    return prompt


# def generate_kb_prompt(sample,**kwargs):
#     """Official KernelBench Prompt"""
#     reference_code = sample["code"]

#     baseline_prompt = get_prompt_for_backend(
#             ref_arch_src=ref_arch_src,
#             backend="cuda",
#             option="one_shot",
#             precision="fp32",
#             # GPU platform agnostic for baseline
#         )

def choose_prompt(sample, trial, prompt_type="normal"):
    """Create a prompt func based on the inputs..."""
    prompt_category = prompt_type
    print(f"\n----\n[Prompt] Entering... trial={trial}, prompt_type={prompt_type}")

    # TRIAL == 1
    if trial == 1:
        print(f"[Prompt] We are in TRIAL 1, and we are using: prompt_type:{prompt_type}")
        if prompt_type=="normal" or prompt_type == "single_stage" or prompt_type == "multi_stage":
            prompt_func = generate_prompt
        elif prompt_type=="kernelbench":
            prompt_func = generate_kb_prompt
        # elif prompt_type=="kernelbench_fewshot":
        #     prompt_func = prompts.generate_improvement_prompt
        # elif prompt_type=="accelopt":
        #     prompt_func = prompts.generate_improvement_prompt
        else:
            print(f"[Prompt] This prompt_type does not exist: {prompt_type} falling back to normal")
            prompt_func = generate_prompt

    # TRIAL > 1
    elif trial > 1:
        print(f"[Prompt] We are in TRIAL >1 ({trial}), and we are using: prompt_type:{prompt_type}")
        if prompt_type=="normal":
            prompt_func = generate_prompt

        # Single Stage
        elif prompt_type=="single_stage":
            prompt_func = generate_improvement_prompt_single_stage


        # KernelBench TTS
        elif prompt_type == "kernelbench":
            if not sample.get("model_new_available", False):
                prompt_category = "kb_debug_error"
                prompt_func = partial(generate_kb_tts_prompt,case="compilation_error")
            elif not sample.get("second_compile_success",False):
                prompt_category = "kb_debug2_error"
                prompt_func = partial(generate_kb_tts_prompt,case="compilation_error")
            elif not sample.get("correctness_success",False):
                prompt_category = "kb_runtime_error"
                prompt_func = partial(generate_kb_tts_prompt,case="runtime_error")
            elif not sample.get("correctness",False):
                prompt_category = "kb_correctness_error"
                prompt_func = partial(generate_kb_tts_prompt,case="incorrect")
            else:
                prompt_category = "kb_improve"
                prompt_func = partial(generate_kb_tts_prompt,case="improve")

        # KernelBench TTS
        elif prompt_type == "kb_multi_stage":
            if not sample.get("model_new_available", False):
                prompt_category = "kb_debug_error"
                prompt_func = partial(generate_kb_tts_prompt,case="compilation_error",multi_stage=True)
            elif not sample.get("second_compile_success",False):
                prompt_category = "kb_debug2_error"
                prompt_func = partial(generate_kb_tts_prompt,case="compilation_error",multi_stage=True)
            elif not sample.get("correctness_success",False):
                prompt_category = "kb_runtime_error"
                prompt_func = partial(generate_kb_tts_prompt,case="runtime_error",multi_stage=True)
            elif not sample.get("correctness",False):
                prompt_category = "kb_correctness_error"
                prompt_func = partial(generate_kb_tts_prompt,case="incorrect",multi_stage=True)
            else:
                prompt_category = "kb_improve"
                prompt_func = partial(generate_kb_tts_prompt,case="improve",multi_stage=True)


        # Multi Stage
        elif prompt_type=="multi_stage": #previous improve
            if not sample.get("model_new_available", False):
                prompt_category = "improve_compilation"
                prompt_func = generate_debug_compilation_prompt
            elif not sample.get("second_compile_success",False):
                prompt_category = "improve_compilation_v2"
                prompt_func = generate_debug_compilation_prompt
            elif not sample.get("correctness_success",False):
                prompt_category = "improve_debug"
                prompt_func = generate_debug_runtime_prompt
            elif not sample.get("correctness",False):
                prompt_category = "improve_correctness"
                prompt_func = generate_correctness_prompt
            else:
                prompt_category = "improve_runtime"
                prompt_func = generate_improvement_prompt
        
        else:
            print(f"[Prompt] This prompt_type does not exist: {prompt_type} falling back to normal")
            prompt_func = generate_prompt
        
    else:
        raise Exception(f"Trial needs to be >= 1. Actually it is: {trial}")
    print(f"[Prompt] Exiting... prompt_category={prompt_category}\n----")
    return prompt_func, prompt_category




if __name__=="__main__":
    EXPERIMENT_NAME = "exp_v4_5_q3b"
    data_repo = "ai-nikolai/KernelBench"

    dataset = load_dataset(data_repo)
    level = 1
    split = f"level_{level}"
    dataset = dataset[split]

    previous_folder_path = get_folder_path(EXPERIMENT_NAME, 1)
    generations_path = os.path.join(previous_folder_path,"generations.csv")
    previous_generations = pd.read_csv(generations_path)

    evaluations_path = os.path.join(previous_folder_path,"evaluations.csv")
    previous_evaluations = pd.read_csv(evaluations_path)

    compilations_path = os.path.join(previous_folder_path,"compilations.csv")
    previous_compilations = pd.read_csv(compilations_path)

    dataset = create_joined_dataset(
        dataset,
        previous_generations = previous_generations,
        previous_compilations = previous_compilations,
        previous_evaluations = previous_evaluations,
    )

    # COUNT FINDER
    # count = 0
    # for sample in dataset:
    #     if sample['second_compile_success'] == True and sample["correctness_success"]==False:
    #     # if sample["problem_id"] = 96:
    #         break
    #     count += 1
    # print(count)

    idx = 0 #correctness #pid=100
    idx = 1 #compilation #pid=10
    idx = 4 #runtime #pid=13
    idx = 6 #compilation buggy code blocks
    idx = 7 #debug pid = 16 (more than one code block) #second compile failed.. very interesting...
    idx = 26 #debug pid = 16 (more than one code block)
    idx = 95 #compilation pid=96 has feedback

    correctness_sample = dataset[0]
    compilation_sample = dataset[95]
    improvement_sample = dataset[4]
    compilation_bonus_sample = dataset[6]
    second_compile_sample = dataset[7] #where second compile failed
    debug_sample = dataset[26]

    first_sample = dataset[idx]

    prompt_type = "multi_stage"

    _, x1 = choose_prompt(correctness_sample, 2, prompt_type)
    _, x2 = choose_prompt(compilation_sample, 2, prompt_type)
    _, x3 = choose_prompt(improvement_sample, 2, prompt_type)
    _, x4 = choose_prompt(debug_sample, 2, prompt_type)
    _, x5 = choose_prompt(second_compile_sample, 2, prompt_type)

    assert x1 == "improve_correctness", f"Choose Prompt is failing x1, {x1}"
    assert x2 == "improve_compilation", f"Choose Prompt is failing x2, {x2}"
    assert x3 == "improve_runtime", f"Choose Prompt is failing x3, {x3}"
    assert x4 == "improve_debug", f"Choose Prompt is failing x4, {x4}"
    assert x5 == "improve_compilation_v2", f"Choose Prompt is failing x5, {x5}"

    sample_to_choose = improvement_sample
    prompt_func, prompt_category = choose_prompt(sample_to_choose, 2, "kernelbench") #kb_multi_stage
    # prompt_func, prompt_category = choose_prompt(sample_to_choose, 2, "kb_multi_stage") #kb_multi_stage

    print(f"[Main] The prompt category is: {prompt_category} and problem_id {sample_to_choose['problem_id']}")

    output = prompt_func(sample_to_choose)

    if True:
        print(f"[Main] The kernelbench prompt for problem_id {sample_to_choose['problem_id']} is:")
        print(output)

        # print(f"\n\n=====\n[Main] Debug Compilation Prompt:")
        # print(generate_debug_compilation_prompt(compilation_sample))

        # print(f"\n\n=====\n[Main] Debug Runtime Prompt:")
        # print(generate_debug_runtime_prompt(debug_sample))
        # pprint(debug_sample)

        # print(f"\n\n=====\n[Main] Correctness Prompt:")
        # print(generate_correctness_prompt(correctness_sample))
        # pprint(correctness_sample)

        # print(f"\n\n=====\n[Main] Improvement Prompt:")
        # print(generate_improvement_prompt(improvement_sample))

        # print(f"\n\n=====\n[Main] Debug Runtime Prompt:")
        # print(generate_debug_compilation_prompt(second_compile_sample))
        # pprint(debug_sample)


    # OTHER STUFF WE USED TO TEST BEFORE... OBSOLETE
    # if False:
    #     ref_code = """import torch
    # from torch import nn
    # class MyModel(nn.Module):
    #     def forward(self,A):
    #         return A
    # """
    #     print("\n\n\n=====\nGenerate Prompt:----\n", generate_prompt(ref_code) )
    #     print("\n\n\n=====\nGenerate Translation Prompt:----\n", generate_translation_prompt(ref_code, translation_lang = "cuda", input_lang = "cuda"))
    #     print("\n\n\n=====\nGenerate Debug Prompt:----\n", generate_debug_prompt(ref_code, "[COMPILER INFO]: Line 73, there was no bug reason.") )
    #     print("\n\n\n=====\nGenerate Improvement Prompt:----\n", generate_improvement_prompt(ref_code, "64ms", "5ms", "speed") )

# todos:
# metadata.json during inference
# prompts: debug, improvement, improvement scientist, self-evaluation prompt, reflection prompt, improve with reflection
# eval: memory measurement
# pipeline: parallel, sequential, tree sampling (K,T)