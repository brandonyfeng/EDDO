CUDA_VISIBLE_DEVICES=1 python sweep_lyot_JWST.py \
    --iters 1000 --scene_name exp/lyot --num_t 1 \
    --error_ratio 0.0001 --contrast_exp -9