CUDA_VISIBLE_DEVICES=1 python scripts/inference_optim.py \
--exp_dir=experiments/res_1014_optim \
--checkpoint_path=experiments/1012_psp_focal/checkpoints/best_model.pt \
--data_path=/workspace/ffhq_512_mirrored \
--load_latent_avg \
--test_batch_size=1 \
--test_workers=4 \
--couple_outputs  \