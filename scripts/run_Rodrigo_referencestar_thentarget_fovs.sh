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
#SBATCH --mem-per-cpu=12G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=job2015  ## When you run squeue -u NETID this is how you can identify the job

## For NOTMEDIAN simulated data: lr 0.0000000003
module purge all
##module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate jwst_model

CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV.py \
    --data_dir ./4050_data_for_diff_modeling/group_2_before  \
    --measurement_file TWA-23A/target_0_ints.npy  \
    --reference_file REF-HD-89063/reference_7_ints.npy \
    --scene_name SIM_DATA \
    --exp_name reference_TWA_diffinitconds_NOttv_NOMEDIAN3_IEC_1_2\
    --iters 1500 \
    --star_offset_x 0.4 --star_offset_y -0.8 \
    --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 2 --lr 0.0000000003 --use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt