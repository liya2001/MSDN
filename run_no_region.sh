#!/usr/bin/env bash
nohup python train_hdn.py \
    --load_RPN --saved_model_path=./output/RPN/RPN_region_full_best.h5  --dataset_option=normal --enable_clip_gradient --step_size=2 --MPS_iter=2 \
    --log_interval=1000 --disable_language_model \
> nohup/msdn_original_no_region_load_rpn_normal_step2_iter2.out 2>&1 &
