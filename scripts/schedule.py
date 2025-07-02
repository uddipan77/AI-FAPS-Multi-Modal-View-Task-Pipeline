"""
Script Name: schedule.py
Author: Vishnudev Krishnadas
Date: 31/10/2023
Version: 1.0
Description: Auto schedules slurm jobs based on dynamic user input.
"""

from rich.panel import Panel
from rich import print
import pyrootutils
import subprocess
import textwrap
import inquirer
import glob
import os
import re

# Define available GPUs and number of devices
AVAILABLE_GPUS = [
    ('RTX 2080 Ti (11 GB)', 'work'),
    ('Geforce RTX 3080 (10 GB)', 'rtx3080'),
    ('Tesla V100 (32 GB)', 'v100'),
    ('A100 (40 GB)', 'a100'),
]
AVAILABLE_NUM_DEVICES = [1, 2, 4]

# Find the project root
ROOT = pyrootutils.find_root(search_from=__file__, indicator=".project-root")


def get_user_input():
    """Function to get user input for scheduling slurm jobs.
    Get the slurm script and experiment to run.
    Also get the GPU and number of devices to run the job on.

    :return: User's custom input
    :rtype: dict
    """
    
    # Get all slurm scripts
    slurm_scripts = glob.glob(os.path.join(ROOT, "scripts", "slurm", "**", "*.slurm"), recursive=True)
    
    # Filter out unused scripts and get names
    pprint_slurm_scripts = []
    for script in slurm_scripts:
        if 'unused' not in script:
            rel_path = os.path.relpath(script, os.path.join(ROOT, "scripts", "slurm"))
            script_name = '    '.join(rel_path.replace('.slurm', '').split(os.sep))
            script_name = script_name.replace('_', ' ').title()
            pprint_slurm_scripts.append((script_name, script))
    
    pprint_slurm_scripts = list(sorted(pprint_slurm_scripts, key=lambda x: x[0]))
    
    # Get user input for slurm script to run
    slurm_answers = inquirer.prompt([
        inquirer.List('slurm_script', message="Choose a SLURM template", choices=pprint_slurm_scripts)
    ])
    
    # Get all available experiments or sweep experiments
    if "sweep" in slurm_answers['slurm_script']:
        exp_path = os.path.join(ROOT, "configs", "hparams_search", "wandb_sweep")
    else:
        exp_path = os.path.join(ROOT, "configs", "experiment")
        
    # Get all available experiments
    available_experiments = glob.glob(os.path.join(exp_path, "**", "*.yaml"), recursive=True)
    
    # Filter out unused scripts and get names
    pprint_experiments = []
    for exp in available_experiments:
        rel_path = os.path.relpath(exp, exp_path)
        pprint_experiments.append(('    '.join(rel_path.replace('.yaml', '').split(os.sep)), rel_path))
    
    # Get user input for experiment to run, GPU devoce and number of devices
    questions = [
        inquirer.List('experiment', message="Choose an experiment", choices=pprint_experiments, carousel=True),
        inquirer.List('gpu', message="Select a GPU", choices=AVAILABLE_GPUS),
        inquirer.List('num_devices', message="Select number of GPU devices", choices=AVAILABLE_NUM_DEVICES),
    ]
    
    # Return user input
    return inquirer.prompt(questions) | slurm_answers

def generate_slurm_script(user_input):
    """Function to generate a slurm script based on user input and template.

    :param user_input: User's custom input
    :type user_input: dict
    :return: Path to the generated slurm script
    :rtype: str
    """
    # Get the slurm script path from user input
    slurm_script = os.path.join(ROOT, "scripts", "slurm", user_input['slurm_script'])

    # Read the slurm script
    with open(slurm_script, "r") as f:
        script_str = f.read()
    
    warning_and_comments = textwrap.dedent("""
    ### ----------------------------------------------------------------------
    ### !! THIS IS A GENERATED SLURM SCRIPT !! DO NOT EDIT !! DO NOT DELETE !!
    ### Lines like "#SBATCH" configure the job resources
    ### (even though they look like bash comments)
    ### ----------------------------------------------------------------------
    """).strip()
    
    script_str = re.sub(r"#!/bin/bash -l", f"#!/bin/bash -l\n\n{warning_and_comments}\n", script_str)
    
    # Custom syntax for work GPU nodes
    if user_input['gpu'] == 'work':
        script_str = re.sub(r"#SBATCH --partition=\w+\n", "", script_str)
        script_str = re.sub(r"#SBATCH --gres=gpu:(.+)", f"#SBATCH --gres=gpu:{user_input['num_devices']}", script_str)
    else:
        script_str = re.sub(r"#SBATCH --gres=gpu:(.+)", f"#SBATCH --gres=gpu:{user_input['gpu']}:{user_input['num_devices']}", script_str)
        script_str = re.sub(r"#SBATCH --partition=(.+)", f"#SBATCH --partition={user_input['gpu']}", script_str)
    
    # Replace PLACEHOLDER with experiment name from user input
    script_str = script_str.replace("PLACEHOLDER_JOB_NAME", os.path.splitext(user_input['experiment'])[0])
    
    # Generate a temporary slurm script for scheduling
    temp_slurm_script = os.path.join(ROOT, "scripts", "generated_slurm_script")
    with open(temp_slurm_script, "w") as f:
        f.write(script_str)
    
    return temp_slurm_script


def submit_job(script_path):
    """Function to submit a slurm job using sbatch.

    :param script_path: Path to the slurm script
    :type script_path: str
    :return: Output of the sbatch command
    :rtype: subprocess.CompletedProcess
    """
    after_previous = inquirer.confirm("Do you want to schedule this run after previous job?")
    
    if not after_previous:
        print("Scheduling job now.")
    
    previous_job_id = subprocess.run(["squeue -u $USER | tail -1| awk '{print $1}'"], shell=True, capture_output=True).stdout.decode().strip()
    print(f"Previous job id: {previous_job_id}")
    
    if previous_job_id == "JOBID" :
        print("No running jobs found.")
        dependency = []
    else:
        dependency = [f"--dependency=afterany:{previous_job_id}"] if after_previous else []
    
    cmd = ["sbatch"] + dependency + [script_path]
    return subprocess.run(cmd)


if __name__ == "__main__":
    
    os.system('clear')
    
    print(
        Panel.fit(
            "Schedule SLURM jobs on HPC based on dynamic user input",
            title="INTERACTIVE EXPERIMENT SCHEDULER",
            subtitle="Vishnudev Krishnadas",
            border_style="green",
            padding=1
        )
    )
    
    user_input = get_user_input()
    script_path = generate_slurm_script(user_input)
    output = submit_job(script_path)
    print(output)
