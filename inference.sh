CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
--exp_dir=experiments/res_1114_enc_celeba_psp \
--checkpoint_path=experiments/1109_psp_realign/checkpoints/best_model.pt \
--data_path=/workspace/celeba_test_crop \
--load_latent_avg \
--test_batch_size=1 \
--test_workers=4 \
--couple_outputs  \