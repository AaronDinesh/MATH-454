#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=math-454
#SBATCH --account=math-454
#SBATCH --output=./hello_world.out

module purge
module load gcc cuda
module list

make clean
make

#nvprof is used to profile our code
srun nvprof ./hello
