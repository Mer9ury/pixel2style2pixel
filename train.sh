OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=experiments/1004_64_0.5 \
--workers=4 \
--batch_size=6 \
--test_batch_size=4 \
--test_workers=4 \
--val_interval=5000 \
--save_interval=10000 \
--encoder_type=GradualStyleEncoder \
--render_resolution=64 \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1 \
--cams_lambda=0.5 \
--dataset_path=/workspace/ffhq_512_mirrored \
--distributed=True 

