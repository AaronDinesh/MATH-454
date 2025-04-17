#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=1:0:0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free
#SBATCH --account=math-454
#SBATCH --reservation=Course-math-454

module purge
module load gcc cuda

srun nvprof ./matrixmul
