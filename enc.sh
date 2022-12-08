CUDA_VISIBLE_DEVICES=1 python3 scripts/enc_samples.py \
--network /workspace/pixel2style2pixel/experiments/1130_e4e_synt/checkpoints/best_model.pt \
--outdir enc_output \
--image_path /workspace/ffhq_output/ \
--c_path 0 \

