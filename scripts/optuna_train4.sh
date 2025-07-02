#!/bin/bash -l
#
#SBATCH --gres=gpu:a100:1                  # Request 1 RTX3080 GPU
#SBATCH --partition=a100                   # GPU partition
#SBATCH --time=24:00:00                        # Maximum runtime (12 hours)
#SBATCH --export=NONE                          # Don't export current environment variables to the job
#SBATCH --job-name=densenet_ITplus            # Job name
#SBATCH --output=/home/hpc/iwfa/iwfa110h/Uddipan/AI-FAPS_Vishnudev_Krishnadas/logs/optuna/ITplus/optuna_train_densenet121_ITplus.out  # Output log file
#SBATCH --error=/home/hpc/iwfa/iwfa110h/Uddipan/AI-FAPS_Vishnudev_Krishnadas/logs/optuna/ITplus/optuna_train_densenet121_ITplus.err   # Error log file

export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

# Load necessary modules
module load python/3.10-anaconda

# Activate the conda environment
conda activate /home/woody/iwfa/iwfa110h/software/private/conda/envs/faps  # Activate conda environment

# Export PROJECT_ROOT
export PROJECT_ROOT=/home/hpc/iwfa/iwfa110h/Uddipan/AI-FAPS_Vishnudev_Krishnadas/

# Set PYTHONPATH to include the project root
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
echo "PYTHONPATH set to: $PYTHONPATH"

# Navigate to the project directory
cd $PROJECT_ROOT

# Print current directory and PYTHONPATH (for sanity checks)
pwd
echo "PYTHONPATH is: $PYTHONPATH"

# Run the Optuna hyperparameter optimization script
python src/train_optuna4.py \
  --config configs/hyperparameter_search/optuna_train4.yaml \
  --n_trials 75 \
  --study_name multimodal_multiview_multitask_Dense121_and_Inception_Timeplus_withpruner \
  --storage sqlite:///optuna_multitask_multiview_multimodal.db


