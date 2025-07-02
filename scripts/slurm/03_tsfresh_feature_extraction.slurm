#!/bin/bash -l
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

source ~/.profile
conda activate faps

#cd /home/hpc/iwfa/iwfa028h/dev/faps/AI-FAPS_Vishnudev_Krishnadas/
cd /home/hpc/iwfa/iwfa110h/Uddipan/AI-FAPS_Vishnudev_Krishnadas/

srun python scripts/tsfresh_feature_validation.py
