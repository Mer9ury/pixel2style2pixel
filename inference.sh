CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
--exp_dir=experiments/1208_e4e \
--checkpoint_path=/workspace/pixel2style2pixel/experiments/1130_e4e_synt/checkpoints/best_model.pt \
--data_path=/workspace/celeba_test_crop \
--load_latent_avg \
--test_batch_size=1 \
--test_workers=4 \
--couple_outputs  \