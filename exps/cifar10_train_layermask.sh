#!/usr/bin/env bash

python3 train_trial.py \
-gen_bs 100 \
-dis_bs 50 \
--max_epoch 200 \
--dataset cifar10 \
--bottom_width 8 \
--img_size 32 \
--max_iter 500000 \
--gen_model TransGAN_8_8_1_layermask \
--dis_model ViT_8_8 \
--mask 8_8 \
--mask_type 0 \
--df_dim 384 \
--d_depth 7 \
--g_depth 5 \
--latent_dim 1024 \
--gf_dim 1024 \
--num_workers 36 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss wgangp-eps \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 100 \
--num_eval_imgs 5000 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--diff_aug translation,cutout,color \
--output_dir layermask_8_8_checkpoint
