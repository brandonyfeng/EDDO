CUDA_VISIBLE_DEVICES=1 python sweep_lyot.py \
    --iters 200 --scene_name exp/lyot --num_t 32 \
    --error_ratio 0.01 --contrast_exp -9 \
    --photon_noise 0.0005
