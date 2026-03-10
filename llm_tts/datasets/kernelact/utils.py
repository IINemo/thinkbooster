import os
from pathlib import Path


VERSION="0.0.2"

def get_folder_path(experiment_name, trial=None, postfix=None, prefix="experiments"):
    """Creates the folder and returns path"""    
    if trial:
        trial_name = f"trial_{trial}"
        folder_path = os.path.join(experiment_name,trial_name)
    else:
        folder_path = os.path.join(experiment_name)
    
    if postfix:
        folder_path = os.path.join(folder_path, postfix)
    
    if prefix:
        folder_path = os.path.join(prefix, folder_path)

    Path(folder_path).mkdir(parents=True, exist_ok=True)

    return folder_path

def get_file_name(problem_id, sample_id=None):
    """Creates the filename"""
    if sample_id:
        file_name = f"{problem_id}__sample_{sample_id}"
    else:
        file_name = f"{problem_id}__sample_"
              
    return file_name


def get_problem_id_from_file_name(file_name):
    """Returns problem_id"""
    problem_id = int(file_name.split("__")[0])
    return problem_id