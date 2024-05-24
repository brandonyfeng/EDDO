CUDA_VISIBLE_DEVICES=1 python run_JWST.py \
    --iters 1000 --scene_name test/JWST --num_t 1 \
    --lr 1e-11 \
    --log_progress \
    --error_ratio 0.01 --contrast_mult 1 --contrast_exp -8 \
    --photon_noise 0.0001