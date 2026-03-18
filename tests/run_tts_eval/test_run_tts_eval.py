import os
import subprocess

import pytest


@pytest.mark.integration
def test_run_tts_eval():
    print("Current directory:", os.getcwd())
    cmd = (
        "PYTHONPATH=./ python scripts/run_tts_eval.py "
        "--config-path=../config "
        "--config-name=experiments/offline_best_of_n/math500/offline_bon_openrouter_gpt4o_mini_math500_entropy "
        "dataset.subset=1 report_to=''"
    )
    exec_result = subprocess.run(cmd, shell=True)
    assert exec_result.returncode == 0, f"running {cmd} failed!"
