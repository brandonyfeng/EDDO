#!/bin/bash
##SBATCH --account=p32465  ## YOUR ACCOUNT pXXXX or bXXXX
##SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --account=b1094  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=ciera-gpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:a30:1
##SBATCH --gres=gpu:l40s:1
##SBATCH --gres=gpu:h100:1
##SBATCH --constraint=rhel8
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=2 ## how many cpus or processors do you need on each computer
#SBATCH --time=08:00:00 ## how long does this need to run (remember different partitions have restrictions on this param)
#SBATCH --mem-per-cpu=80G ## how much RAM do you need per CPU, also see --mem=<XX>G for RAM per node/computer (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=twag  ## When you run squeue -u NETID this is how you can identify the job
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
## lr = 0.000000004 was okay, lr = 0.000000008 not bad! maybe even better?
## Attempt one at super small fov:
##CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter.py \

## CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter_tryincnoise.py \
##         --data_dir ./HR8799stuff/before_data_newversion2_SHIFTEDLYOT  \
##         --measurement_file HR8799_original/target_006_00001.npy  \
##         --reference_file REF-HD220657_original/reference_00001.npy \
##         --scene_name HR8799_ref_only_freezepos\
##         --px_mask_file REF-HD220657_original/pixel_masks.npy \
##         --ref_which_int 0 --sci_which_int 0 \
##         --exp_name ref1_dith1_int0_constrain_2kiters_poslr1\
##         --iters 2000 --sci_targ_name HR8799 --ref_cutoff_iter 30000 \
##         --star_offset_x 0.9 --star_offset_y 0.3 --no_median --freeze_position --be_normal\
##         --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 1 --lr 0.000000004 --stage_fluxpos_cutoff_iter 200  #--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt --use_ptt --use_linear_interp_opd

## Group 2 was on the ~14th, group 4 was on the ~16th

CUDA_VISIBLE_DEVICES=0 python run_JWST_LATEST_twodeltawfes_trackinjected.py \
        --data_dir ./HIP65426_newmodel/beforeOPD  \
        --measurement_file full_im/target_002_00001.npy  \
        --reference_file full_im/reference_00001.npy \
        --scene_name group4_cleansmooth_16OPD/HIP65426_YEAHH_TRACK \
        --px_mask_file pixel_masks/pixel_masks.npy \
        --ref_which_int 0 --sci_which_int 0 --blur_annulus_SNR --blur_before_signal \
        --exp_name ref1000_sciint0_3kiters_lr04_scilr1_smoothNone_biglr7_WITHINITIALCONDITIONS_freezeOTEOPDsci_inj3mjy_17px --big_lr_factor 7.0 \
        --other_OPD_meas /projects/b1094/rodrigoyeah/optics_jwst/EDDO_repo/EDDO/HIP65426_newmodel/afterOPD/masks_2048/observation_opd.npy \
        --iters 3000 --sci_targ_name HIP65426 --ref_cutoff_iter 1000 --sci_lr_weight 1.0 --insert_initial_delta_OTE_OPD --insert_initial_delta_NIRCam_OPD\
        --star_offset_x 0.9 --star_offset_y 0.3 --no_median --freeze_position --primaryOPD_basis grid --freeze_OTE_OPD_sci --track_injected_planet\
        --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 1 --lr 0.000000004 --stage_fluxpos_cutoff_iter 300 ##--smooth 5.1 ##--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt --use_ptt --use_linear_interp_opd


## group4/TWA-47/full_im/target_035_00001.npy
## --lr 0.000000003 good for TWA! Science targets at least
## maybe try customizing when OTE OPD is tuned and when NIRCam OPD is tuned????
## --freeze_OTE_OPD_sci if you want to freeze the entrance OPD!!
## used 5000, 3000 before  --fit_second_wfe_offsets grid --ref_cutoff_iter 1000 --iters 2000
##        --smooth 1.2 \--OPD_loss_weight 5.0
## for ((i=0; i<=4; i++))
## GOOD VALUES: --star_offset_x 0.9 --star_offset_y 0.3 
## do
##     CUDA_VISIBLE_DEVICES=0 python run_JWST_current_smaller_FOV_decenter_tryincnoise.py \
##         --data_dir ./HR8799stuff/before_data  \
##         --measurement_file HR8799_original/target_006_00001.npy  \
##         --reference_file REF-HD220657_original/reference_00001.npy \
##         --scene_name HR8799_ref_long_allints_SMOOTHonepointtwojajanew\
##         --px_mask_file REF-HD220657_original/pixel_masks.npy \
##         --ref_which_int $i --sci_which_int 0 \
##         --exp_name "ref1_dith1_int$i"\
##         --iters 3000 --sci_targ_name HR8799 --ref_cutoff_iter 20000\
##         --star_offset_x 0.4 --star_offset_y -1.2 --no_median \
##         --smooth 1.2 \
##         --oversample 2 --num_wl 21 --num_det_px 80 --num_ints 1 --lr 0.0000000075 --stage_fluxpos_cutoff_iter 200  #--use_simulated_data  #--lr 0.00000003 #--smooth 0.6 #--use_linear_interp_opd  # --OPD_loss_weight 1 --use_ptt --use_ptt --use_linear_interp_opd
## done

##    --reference_noisemap REF-HD220657_original/noisemap_reference_00001.npy\
    ##     --px_mask_file REF-HD220657_original/pixel_masks.npy \