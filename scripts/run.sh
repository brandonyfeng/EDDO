CUDA_VISIBLE_DEVICES=0 python run_JWST_broadband_OPDandPosition.py \
    --data_dir ./data \
    --measurement_file justdata_bothintegrations.npy \
    --exp_name no_wfe_3K_iters \
    --iters 3000 \
    --star_offset_x 1.5 --star_offset_y -3.0
