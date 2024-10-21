CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.001 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.002 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.004 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.008 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.01 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.02 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.04 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.08 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

CUDA_VISIBLE_DEVICES=1 python test_guidestar.py \
    --iters 200 --scene_name exp/curve/error_rate --num_t 32 \
    --error_ratio 0.1 --contrast_mult 1 --contrast_exp -9 \
    --photon_noise 0.0005

