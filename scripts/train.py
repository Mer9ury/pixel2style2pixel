"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.eg3d_coach import Coach



def setup_progressive_steps(opts):
	num_style_layers = 14
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

def main():

	local_rank = int(os.environ['LOCAL_RANK'])
	torch.cuda.empty_cache()

	# torch.autograd.set_detect_anomaly(True)
	opts = TrainOptions().parse()
	# if os.path.exists(opts.exp_dir):
	# 	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	setup_progressive_steps(opts)
	os.makedirs(opts.exp_dir,exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	torch.cuda.set_device(local_rank)
	print(local_rank)

	# os.environ['MASTER_ADDR'] = 'localhost'
	# os.environ['MASTER_PORT'] = '25455'
	dist.init_process_group(backend = "nccl", init_method='env://')

	rank = dist.get_rank()
	size = dist.get_world_size()

	opts.num_gpus = size
	opts.rank = local_rank

	dist.barrier()
	coach = Coach(opts)
	coach.train()

	# if opts.distributed:
	# 	mp.spawn(main_worker, nprocs = opts.num_gpus, args = (opts.num_gpus, opts), join = True)
	# else:
	# 	main_worker(0, opts.num_gpus, opts)


if __name__ == '__main__':
	main()
