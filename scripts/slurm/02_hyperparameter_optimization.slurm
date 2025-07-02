#!/bin/bash -l
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00

export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Root log folder
#LOG_ROOT="/home/hpc/iwfa/iwfa028h/dev/faps/AI-FAPS_Vishnudev_Krishnadas/logs/hpc/"
LOG_ROOT="/home/hpc/iwfa/iwfa110h/Uddipan/AI-FAPS_Vishnudev_Krishnadas/logs/hpc/"

# Create a subfolder named after the job inside the root log folder
JOB_LOG_DIR="${LOG_ROOT}/multitask_sweep_densenet_rescnn/"
mkdir -p ${JOB_LOG_DIR}

exec &> "${JOB_LOG_DIR}/${SLURM_JOB_ID}.log"

echo "Experiment: PLACEHOLDER_JOB_NAME"

conda activate faps

#cd /home/hpc/iwfa/iwfa028h/dev/faps/AI-FAPS_Vishnudev_Krishnadas/
cd /home/hpc/iwfa/iwfa110h/Uddipan/AI-FAPS_Vishnudev_Krishnadas/

SWEEP_ID=$(wandb sweep configs/hparams_search/wandb_sweep/multitask/multitask_densenet_rescnn.yaml 2>&1 | awk '/ID:/ {print $NF}')
echo "Sweep ID: $SWEEP_ID"

srun wandb agent --count=80 "uddipan-bir-basu-Friedrich-Alexander-Universität Erlange/ai-faps-project/$SWEEP_ID"
