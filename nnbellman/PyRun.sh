#!/bin/bash
#SBATCH --job-name=PyBatch
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --output=python_current.out

#            d-hh:mm:ss
#SBATCH --time=40:00:00

module load 2023 
module load CUDA/12.1.1 
module load cuDNN/8.9.2.26-CUDA-12.1.1 

# Environment (Snellius specific)
source NNFun/bin/activate

echo "Script: PyRun.sh File: $1"
date=$(date)
echo "$date Started run"
python -u $1
echo "$date Finished run"


