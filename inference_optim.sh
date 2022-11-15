CUDA_VISIBLE_DEVICES=0 python scripts/inference_optim.py \
--exp_dir=experiments/res_1024_optim \
--checkpoint_path=experiments/1012_psp_focal/checkpoints/best_model.pt \
--data_path=/workspace/celeba_test_crop \
--load_latent_avg \
--test_batch_size=1 \
--test_workers=4 \
--couple_outputs  \