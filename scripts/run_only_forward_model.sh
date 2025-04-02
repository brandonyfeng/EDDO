#!/bin/bash
##SBATCH --account=p32465  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --account=b1094  ## YOUR ACCOUNT pXXXX or bXXXX
## SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --partition=ciera-gpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:a100:1
##SBATCH --constraint=rhel8
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=1 ## how many cpus or processors do you need on each computer
#SBATCH --time=00:30:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=1G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=job2015  ## When you run squeue -u NETID this is how you can identify the job

module purge all
##module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate jwst_model

CUDA_VISIBLE_DEVICES=0 python forward_model.py 