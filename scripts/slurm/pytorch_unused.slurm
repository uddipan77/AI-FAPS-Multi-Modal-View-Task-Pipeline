#!/bin/bash -l
#SBATCH --job-name=multimodal_tiny
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=12:00:00

conda activate faps

#cd /home/hpc/iwfa/iwfa028h/dev/faps/AI-FAPS_Vishnudev_Krishnadas/
cd /home/hpc/iwfa/iwfa110h/Uddipan/AI-FAPS_Vishnudev_Krishnadas/

python src/train.py experiment=mm_tiny logger=tensorboard
