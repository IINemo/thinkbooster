import os
import pandas as pd

from datasets import load_from_disk, load_dataset
from datasets import Dataset

from pprint import pprint

from .utils import (
    get_folder_path,
    get_file_name
)


def extract_code(generated_response, model_name=None):
    """Function to extract code from the responses...."""
    try:
        # Handle gpt-oss thinking mode format
        is_oss_model = model_name and any(keyword in model_name.lower() for keyword in ["oss", "k2"])

        if is_oss_model:
            # Remove analysis section if present
            if "analysis" in generated_response:
                parts = generated_response.split("analysis")
                if len(parts) > 1:
                    after_analysis = parts[1]
                    if "```python" not in after_analysis:
                        print("NOTE: No python code block found after analysis, using the full response.")
                        pass
                    elif "assistant" in after_analysis:
                        generated_response = after_analysis.split("assistant")[-1]
                    else:
                        generated_response = after_analysis

        if "Optimized with CUDA operators:" in generated_response:
            resp = generated_response.split("Optimized with CUDA operators:")[-1]
            if resp:
                return resp
        
        num_code = generated_response.count("```python")
        num_cpp_code = generated_response.count("```cpp")
        num_cuda_code = generated_response.count("```cuda")
        num_code_code = generated_response.count("```code")
        special_blocks = generated_response.count("```")
        if not num_code:
            print("NOTE, there were no python tags found, attempting split by [```cpp,```cuda, ```]")
            if num_cpp_code:
                answer = generated_response.split('```cpp')[-1]
                answer = answer.split("```")[0]
                if answer:
                    return answer

            if num_cuda_code:
                answer = generated_response.split('```cuda')[-1]
                answer = answer.split("```")[0]
                if answer:
                    return answer

            if num_code_code:
                answer = generated_response.split('```code')[-1]
                answer = answer.split("```")[0]
                if answer:
                    return answer

            if special_blocks:
                potential_answer = generated_response.split('```')
                answer = ""
                if potential_answer[-1]:
                    answer = potential_answer[-1]
                elif potential_answer[-2]:
                    answer = potential_answer[-2]
                if answer:            
                    return answer

            print("WARNING, no matching code blocks found / it's empty, returning the full response.")
            return generated_response
        
        if num_code > 1:
            print("Note there is more than one code block!")

        # For gpt-oss models: collect all blocks and choose the best one
        if is_oss_model and num_code > 1:
            code_blocks = []
            parts = generated_response.split("```python")
            for part in parts[1:]:
                if "```" in part:
                    code = part.split("```")[0].strip()
                    if code:
                        code_blocks.append(code)

            # Look for block with cuda_src
            for code in code_blocks:
                if "cuda_src" in code or 'cuda_src = r"""' in code:
                    print("[gpt-oss] Found block with cuda_src, using it")
                    return code
            # If no cuda_src found, look for CUDA kernel patterns
            for code in code_blocks:
                if "__global__" in code or "<<<" in code:
                    print("[gpt-oss] Found block with CUDA kernel, using it")
                    return code
            # Fall back to first block
            print("[gpt-oss] No cuda_src/CUDA patterns found, using first block")
            if code_blocks:
                return code_blocks[0]

        if special_blocks % 2 == 1:
            print("WARNING: Code blocks are not matching, it could be buggy.")

        answer = generated_response.split('```python')[-1]
        answer = answer.split("```")[0]
        if answer:
            return answer
        else:
            return generated_response
    except Exception as e:
        print("[Extracting Code Error]")
        print(e)
        return generated_response


def create_joined_dataset(
    dataset,    
    previous_generations = None,
    previous_compilations = None,
    previous_evaluations = None,
    ):
    """Creates a new joined dataset with sample_id as a separate key..."""
    if previous_generations is None and previous_compilations is None and previous_evaluations is None:
        # First trial - just return the original dataset
        return dataset

    # Create a mapping from (problem_id, sample_id) to additional data
    problem_data = {}

    # Add generation data if available
    if previous_generations is not None:
        for _, row in previous_generations.iterrows():
            problem_id = row['problem_id']
            generations = eval(row['generations']) if isinstance(row['generations'], str) else row['generations']
            for sample_id, generation in enumerate(generations):
                key = (problem_id, sample_id)
                if key not in problem_data:
                    problem_data[key] = {}
                problem_data[key]['generation'] = generation
    # Add compilation data if available
    if previous_compilations is not None:
        for _, row in previous_compilations.iterrows():
            problem_id = row['problem_id']
            sample_id = row['sample_id']
            key = (problem_id, sample_id)
            if key not in problem_data:
                problem_data[key] = {}
            problem_data[key]['model_new_available'] = row.get('model_new_available',False)
            problem_data[key]['compilation_output'] = row.get('main_output',"")

            problem_data[key]['compilation_output_runtime'] = row.get('main_output_runtime',"")
            if type(problem_data[key]['compilation_output_runtime']) == float:
                problem_data[key]['compilation_output_runtime'] = ""
            
            problem_data[key]['compilation_output_error'] = row.get('main_error', "")
            if type(problem_data[key]['compilation_output_error']) == float:
                problem_data[key]['compilation_output_error'] = ""

            problem_data[key]['main_output'] = row.get('main_traceback', "")
            if type(problem_data[key]['main_output']) == float:
                problem_data[key]['main_output'] = ""

            # extra analysis
            # if not row['model_new_available']:
            #     from pprint import pprint
            #     pprint(row)
            #     input()

    # Add evaluation data if available
    if previous_evaluations is not None:
        for _, row in previous_evaluations.iterrows():
            problem_id = row['problem_id']
            sample_id = row['sample_id']
            key = (problem_id, sample_id)
            if key not in problem_data:
                problem_data[key] = {}
            problem_data[key]['second_compile_success'] = row.get('compiled', False)
            problem_data[key]['correctness_success'] = row.get('correctness_success', False)
            problem_data[key]['correctness'] = row.get('correctness', False)

            problem_data[key]['runtime'] = row['runtime']
            problem_data[key]['runtime_original'] = row['runtime_original']


            problem_data[key]['correctness_output'] = row.get('main_output',"")
            if type(problem_data[key]['correctness_output']) == float:
                problem_data[key]['correctness_output'] = ""

            problem_data[key]['correctness_output_traceback'] = row.get('main_traceback',"")
            if type(problem_data[key]['correctness_output_traceback']) == float:
                problem_data[key]['correctness_output_traceback'] = ""
            
            problem_data[key]['correctness_output_error'] = row.get('main_error', "")
            if type(problem_data[key]['correctness_output_error']) == float:
                problem_data[key]['correctness_output_error'] = ""

            problem_data[key]['main_output'] = row.get('main_error', "")
            if type(problem_data[key]['main_output']) == float:
                problem_data[key]['main_output'] = ""
            
            # Extra analysis
            # if not row['correctness_success']:
            #     from pprint import pprint
            #     pprint(row)
            #     input()

    # Create expanded dataset with sample_id
    expanded_examples = []
    for example in dataset:
        problem_id = example['problem_id']
        # Determine how many samples we have for this problem
        matching_keys = [key for key in problem_data.keys() if key[0] == problem_id]
        if matching_keys:
            sample_ids = sorted(set(key[1] for key in matching_keys))
            for sample_id in sample_ids:
                new_example = example.copy()
                new_example['sample_id'] = sample_id
                key = (problem_id, sample_id)
                if key in problem_data:
                    new_example.update(problem_data[key])
                expanded_examples.append(new_example)
        else:
            # No previous data for this problem, keep original example with sample_id = 0
            new_example = example.copy()
            new_example['sample_id'] = 0
            expanded_examples.append(new_example)

    # Convert list of dicts to HuggingFace dataset
    joined_dataset = Dataset.from_list(expanded_examples)
    return joined_dataset


if __name__ == "__main__":
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

    print(dataset)