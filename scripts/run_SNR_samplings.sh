#!/bin/bash
##SBATCH --account=p32465  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --account=b1094  ## YOUR ACCOUNT pXXXX or bXXXX
## SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --partition=ciera-gpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:a100:1
##SBATCH --constraint=rhel8
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=5 ## how many cpus or processors do you need on each computer
#SBATCH --time=00:30:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=6G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=job2015  ## When you run squeue -u NETID this is how you can identify the job

module purge all
##module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate jwst_model

CUDA_VISIBLE_DEVICES=0 python run_JWST_broadband_OPDandPosition_referenceStar_then_targetv2.py \
    --data_dir ./data \
    --measurement_file justdata_bothintegrations.npy \
    --reference_file reference_00001.npy \
    --scene_name referenceStar_thenTarget \
    --exp_name over2_wl10_snr_calc \
    --iters 1000 \
    --star_offset_x 0.4 --star_offset_y -0.8 \
    --oversample 2 --num_wl 10 