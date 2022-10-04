CUDA_VISIBLE_DEVICES=1 python scripts/inference.py \
--exp_dir=experiments/rescheck \
--checkpoint_path=experiments/0930_test3_orig/checkpoints/best_model.pt \
--data_path=/workspace/ffhq_512_mirrored \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs