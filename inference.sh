CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
--exp_dir=experiments/res_1120_enc_celeba_e4e \
--checkpoint_path=experiments/1027_e4e_realign/checkpoints/best_model.pt \
--data_path=/workspace/celeba_test_crop \
--load_latent_avg \
--test_batch_size=1 \
--test_workers=4 \
--couple_outputs  \