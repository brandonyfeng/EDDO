#!/bin/bash
##SBATCH --account=p32465  ## YOUR ACCOUNT pXXXX or bXXXX
##SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --account=b1094  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=ciera-gpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:a30:1
##SBATCH --gres=gpu:h100:1
##SBATCH --constraint=rhel8
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=2 ## how many cpus or processors do you need on each computer
#SBATCH --time=08:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=10G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=smooth50  ## When you run squeue -u NETID this is how you can identify the job
## cd /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/

## For NOTMEDIAN simulated data: lr 0.0000000003
## For NONMEDIAN real data: lr ~ 0.000000001
module purge all
##module load python-miniconda3
eval "$(conda shell.bash hook)"
conda activate jwst_model

## TWA 
## CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV.py \
##     --data_dir ./4050_data_for_diff_modeling/group_2_before  \
##     --measurement_file TWA-23A/target_0_ints.npy  \
##     --reference_file REF-HD-89063/reference_7_ints.npy \
##     --scene_name REAL_SMOOTH_BOTH_PLANES_wMEDIAN2\
##     --exp_name smooth_06\
##     --iters 1000 --ref_cutoff_iter 1800\
##     --star_offset_x 0.4 --star_offset_y -0.8\
##     --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 2 --lr 0.000000003 --smooth 0.6 ##--stage_fluxpos_cutoff_iter 800 #--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt


## HR8799
## Reference decent LR: 0.000000001 
## reference LR: decent too for full param is 0.0000000035 (tried 10 times this)
## reference LR: decent too for full param is 0.000000005 was good too
## Attempt one at super small fov:
##CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter.py \

## used 5000, 3000 before
##        --smooth 1.2 \
for ((i=0; i<=4; i++))
do
     CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter_tryincnoise.py \
        --data_dir ./HR8799stuff/before_data  \
        --measurement_file HR8799_original/target_006_00001.npy  \
        --reference_file REF-HD220657_original/reference_00001.npy \
        --scene_name HR8799_ref_long_allints_SMOOTHfiftypointone\
        --px_mask_file REF-HD220657_original/pixel_masks.npy \
        --ref_which_int $i --sci_which_int 0 \
        --exp_name "ref1_dith1_int$i"\
        --iters 3000 --sci_targ_name HR8799 --ref_cutoff_iter 20000\
        --star_offset_x 0.4 --star_offset_y -1.2 --no_median \
        --smooth 50.1 \
        --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 1 --lr 0.0000000075 --stage_fluxpos_cutoff_iter 200  #--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt --use_ptt --use_linear_interp_opd
done

##    --reference_noisemap REF-HD220657_original/noisemap_reference_00001.npy\
    ##     --px_mask_file REF-HD220657_original/pixel_masks.npy \