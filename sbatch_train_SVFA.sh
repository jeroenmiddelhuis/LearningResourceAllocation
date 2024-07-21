#!/bin/bash
#SBATCH -t 0-23:58
#SBATCH -A def-dkrass
#SBATCH --mem 10000
source /home/eliransc/projects/def-dkrass/eliransc/queues1/queues/bin/activate
python /home/eliransc/projects/def-dkrass/eliransc/LearningResourceAllocation/train_SVFA.py
