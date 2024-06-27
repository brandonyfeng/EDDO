CUDA_VISIBLE_DEVICES=1 python run_JWST.py \
    --iters 1000 --scene_name test/JWST --num_t 1 \
    --offset_r 0.6 --offset_t 0.3 --read_noise 42.0 \
    --drift_ratio 1.0 --contrast_mult 1 --contrast_exp -6