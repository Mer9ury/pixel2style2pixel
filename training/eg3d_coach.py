import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import json

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss, geodesic_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.eg3d_dataset import EG3DDataset
from datasets.synthetic_dataset import SyntheticDataset
from models.latent_codes_pool import LatentCodesPool
from models.encoders.psp_encoders import ProgressiveStage
from models.discriminator import LatentCodesDiscriminator
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from training.ranger import Ranger
import torch.distributed as dist
import copy

from tqdm import tqdm
from datetime import datetime
import time
from itertools import cycle



class Coach:
	def __init__(self, opts):
		self.opts = opts
 
		self.global_step = 0

		self.device = torch.device(opts.rank)# TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device
		torch.cuda.set_device(self.opts.rank)


		if self.opts.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.opts)

		# Initialize network

		self.net = pSp(self.opts)
		# self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
		self.net.to(self.device)
		

		if self.opts.distributed:

			self.opts.batch_size = int(self.opts.batch_size / self.opts.num_gpus)
			self.opts.test_batch_size = int(self.opts.test_batch_size / self.opts.num_gpus)

			self.net_ddp = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.opts.rank], output_device=self.opts.rank)
			self.net = self.net_ddp.module

			
		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			latent_in = torch.randn(int(1e5),self.net.decoder.z_dim).to(self.device)
			c_in = torch.zeros((int(1e5),25)).to(self.device)

			self.net.latent_avg = self.net.decoder.mapping(latent_in,c_in).mean(0,keepdim=True)[0].detach()
		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

		self.mse_loss = nn.MSELoss().cuda(self.opts.rank).eval()
		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex',rank = self.opts.rank).cuda(self.opts.rank).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss(self.opts.rank).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss().cuda(self.opts.rank).eval()
		if self.opts.cams_lambda > 0:
			self.geodesic_loss = geodesic_loss.GeodesicLoss().cuda()
		
		if self.opts.w_discriminator_lambda > 0:
			self.discriminator = LatentCodesDiscriminator(512, 4).to(self.opts.rank)
			self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
															lr=opts.w_discriminator_lr)
			self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
			self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset, self.synthetic_dataset = self.configure_datasets(use_synthetic = True)

		
		
		if opts.distributed:
			self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,num_replicas=self.opts.num_gpus,
        rank=self.opts.rank)
			self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset,num_replicas=self.opts.num_gpus,
        rank=self.opts.rank),
			self.synthetic_sampler = torch.utils.data.distributed.DistributedSampler(self.synthetic_dataset,num_replicas=self.opts.num_gpus,
        rank=self.opts.rank)
		else:
			self.train_sampler, self.test_sampler, self.synthetic_sampler = None, None, None
			
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle= self.train_sampler is None,
										   num_workers=int(self.opts.workers),
										   sampler = self.train_sampler,
										   pin_memory=True,
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle= self.test_sampler is None,
										  num_workers=int(self.opts.test_workers),
										  sampler = self.test_sampler,
										  pin_memory=True,
										  drop_last=True)
		self.synthetic_dataloader = DataLoader(self.synthetic_dataset,
											batch_size=self.opts.batch_size,
											shuffle= self.synthetic_sampler is None,
											num_workers=int(self.opts.workers),
											sampler = self.synthetic_sampler,
											pin_memory=True,
											drop_last=True)
			
		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		print(f"GPU {self.opts.rank} intialization is done.")

	def train(self):
		print(f"At GPU {self.opts.rank}, Train starts.")
		self.net.train()
		epoch = 0
		if self.opts.progressive_steps:
			self.check_for_progressive_training_update()
		while self.global_step < self.opts.max_steps:
			self.train_dataloader.sampler.set_epoch(epoch)
			
			for batch_idx, batches in enumerate(zip(tqdm(self.train_dataloader),self.synthetic_dataloader)):
				x, y_cams = batches[0]
				can_x, rand_x, can_x_cams, rand_x_cams = batches[1]
				y = copy.deepcopy(x)
				loss_dict = {}
				if self.is_training_discriminator():
					loss_dict = self.train_discriminator(batch)

				x, y, y_cams = x.to(self.device),y.to(self.device).float(), y_cams.to(self.device)

				y_hat, cams, latent = self.net.forward(x, return_latents=True)
				loss, enc_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, cams, y_cams)

				can_x, can_x_cams, rand_x, rand_x_cams = can_x.to(self.device), rand_x.to(self.device), can_x_cams.to(self.device), rand_x_cams.to(self.device)
		
				rand_codes, rand_cams_hat = self.net.encoder(rand_x)
				rand_codes = rand_codes + self.net.latent_avg.repeat(rand_codes.shape[0], 1, 1)

				can_y_hat = self.net.decoder.synthesis(rand_codes, can_x_cams)['image']
				can_y_hat = self.net.face_pool(can_y_hat)
				
				syn_loss, syn_loss_dict, syn_logs = self.calc_loss(rand_x,can_y_hat,can_x,rand_codes,rand_cams_hat,rand_x_cams)


				self.optimizer.zero_grad()
				total_loss = loss + syn_loss
				total_loss.backward()
				self.optimizer.step()

				loss_dict = {**loss_dict, **enc_loss_dict, **syn_loss_dict}

				if self.opts.rank == 0:
					# Logging related
					if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
						self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
					if self.global_step % self.opts.board_interval == 0:
						self.print_metrics(loss_dict, prefix='train')
						self.log_metrics(loss_dict, prefix='train')

					# Log images of first batch to wandb
					if self.opts.use_wandb and batch_idx == 0:
						self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="train", step=self.global_step, opts=self.opts)

					# Validation related
					val_loss_dict = None
					if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
						val_loss_dict = self.validate()
						if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
							self.best_val_loss = val_loss_dict['loss']
							self.checkpoint_me(val_loss_dict, is_best=True)
					if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
						if val_loss_dict is not None:
							self.checkpoint_me(val_loss_dict, is_best=False)
						else:
							self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1
				if self.opts.progressive_steps:
					self.check_for_progressive_training_update()
				
			epoch += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, y_cams = batch
			y = copy.deepcopy(x)
			cur_loss_dict = {}
			if self.is_training_discriminator():
				cur_loss_dict = self.validate_discriminator(batch)

			with torch.no_grad():
				x, y, y_cams = x.to(self.device).float(),y.to(self.device).float(), y_cams.to(self.device).float()
				y_hat, cams, latent = self.net.forward(x, return_latents=True)
				loss, enc_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent, cams, y_cams)
				cur_loss_dict = {**cur_loss_dict,**enc_loss_dict}
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			self.parse_and_log_images(id_logs, x, y, y_hat,
									  title='images/test/faces',
									  subscript='{:04d}'.format(batch_idx))

			# Log images of first batch to wandb
			if self.opts.use_wandb and batch_idx == 0:
				self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="test", step=self.global_step, opts=self.opts)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self, use_synthetic = False):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
		train_dataset = EG3DDataset(  dataset_path = self.opts.dataset_path,
									  transform=transforms_dict['transform_gt_train'],
									  opts=self.opts,
									  metadata = 'ffhq_dataset.json')
		test_dataset = EG3DDataset(  dataset_path = self.opts.dataset_path,
									  transform=transforms_dict['transform_gt_train'],
									  opts=self.opts,
									  metadata = 'ffhq_dataset.json',
									  is_train = False)

		if self.opts.use_wandb:
			self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
			self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
			
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")

		if use_synthetic:
			synthetic_dataset = SyntheticDataset( dataset_path = self.opts.synthetic_dataset_path,
												transform=transforms_dict['transform_gt_train'],
												opts=self.opts,
												metadata = 'synthetic_dataset.json',
			)
			if self.opts.use_wandb:
				self.wb_logger.log_dataset_wandb(synthetic_dataset, dataset_name = "Synthetic")
			print(f"Number of synthetic samples: {len(synthetic_dataset)}")

			return train_dataset, test_dataset, synthetic_dataset
		return train_dataset, test_dataset

	def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
		for i in range(len(self.opts.progressive_steps)):
			if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))
			if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
				self.net.encoder.set_progressive_stage(ProgressiveStage(i))

	def calc_loss(self, x, y, y_hat, latent, cams, y_cams):
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.is_training_discriminator():  # Adversarial loss
			loss_disc = 0.
			dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
				list(range(self.net.decoder.n_latent))

			for i in dims_to_discriminate:
				w = latent[:, i, :]
				fake_pred = self.discriminator(w)
				loss_disc += F.softplus(-fake_pred).mean()
			loss_disc /= len(dims_to_discriminate)
			loss_dict['encoder_discriminator_loss'] = float(loss_disc)
			loss += self.opts.w_discriminator_lambda * loss_disc
			
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:
			loss_l2 = F.mse_loss(y_hat, y)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, x)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda
		if self.opts.cams_lambda > 0:
			extrinsics = cams[:,:16].reshape(-1,4,4)
			y_extrinsics = y_cams[:,:16].reshape(-1,4,4)
			loss_angle = self.geodesic_loss(extrinsics[:,:3,:3],y_extrinsics[:,:3,:3])
			loss_trans = F.mse_loss(extrinsics[:,:3,3], y_extrinsics[:,:3,3])
			loss_cams = loss_angle + loss_trans
			loss_dict['loss_cams'] = float(loss_cams)
			loss += loss_cams * self.opts.cams_lambda

		if self.opts.progressive_steps and self.net.encoder.progressive_stage.value != 14:  # delta regularization loss
			total_delta_loss = 0
			deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()

			first_w = latent[:, 0, :]
			for i in range(1, self.net.encoder.progressive_stage.value + 1):
				curr_dim = deltas_latent_dims[i]
				delta = latent[:, curr_dim, :] - first_w
				delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
				loss_dict[f"delta{i}_loss"] = float(delta_loss)
				total_delta_loss += delta_loss
			loss_dict['total_delta_loss'] = float(total_delta_loss)
			loss += self.opts.delta_norm_lambda * total_delta_loss

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
		if self.opts.use_wandb:
			self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=1):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict

	def get_dims_to_discriminate(self):
		deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
		return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

	def is_progressive_training(self):
		return self.opts.progressive_steps is not None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

	def is_training_discriminator(self):
		return self.opts.w_discriminator_lambda > 0

	@staticmethod
	def discriminator_loss(real_pred, fake_pred, loss_dict):
		real_loss = F.softplus(-real_pred).mean()
		fake_loss = F.softplus(fake_pred).mean()

		loss_dict['d_real_loss'] = float(real_loss)
		loss_dict['d_fake_loss'] = float(fake_loss)

		return real_loss + fake_loss

	@staticmethod
	def discriminator_r1_loss(real_pred, real_w):
		grad_real, = autograd.grad(
			outputs=real_pred.sum(), inputs=real_w, create_graph=True
		)
		grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

		return grad_penalty

	@staticmethod
	def requires_grad(model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag

	def train_discriminator(self, batch):
		loss_dict = {}
		x, _ = batch
		x = x.to(self.device).float()
		self.requires_grad(self.discriminator, True)

		with torch.no_grad():
			real_w, fake_w = self.sample_real_and_fake_latents(x)
		real_pred = self.discriminator(real_w)
		fake_pred = self.discriminator(fake_w)
		loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
		loss_dict['discriminator_loss'] = float(loss)

		self.discriminator_optimizer.zero_grad()
		loss.backward()
		self.discriminator_optimizer.step()

		# r1 regularization
		d_regularize = self.global_step % self.opts.d_reg_every == 0
		if d_regularize:
			real_w = real_w.detach()
			real_w.requires_grad = True
			real_pred = self.discriminator(real_w)
			r1_loss = self.discriminator_r1_loss(real_pred, real_w)

			self.discriminator.zero_grad()
			r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
			r1_final_loss.backward()
			self.discriminator_optimizer.step()
			loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

		# Reset to previous state
		self.requires_grad(self.discriminator, False)

		return loss_dict

	def validate_discriminator(self, test_batch):
		with torch.no_grad():
			loss_dict = {}
			x, _ = test_batch
			x = x.to(self.device).float()
			real_w, fake_w = self.sample_real_and_fake_latents(x)
			real_pred = self.discriminator(real_w)
			fake_pred = self.discriminator(fake_w)
			loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
			loss_dict['discriminator_loss'] = float(loss)
			return loss_dict

	def sample_real_and_fake_latents(self, x):
		sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
		sample_c = torch.zeros(self.opts.batch_size,25).to(self.device)
		real_w = self.net.decoder.mapping(sample_z,sample_c)[0]
		fake_w, _ = self.net.encoder(x)
		if self.opts.start_from_latent_avg:
			fake_w = fake_w + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
		if self.is_progressive_training():  # When progressive training, feed only unique w's
			dims_to_discriminate = self.get_dims_to_discriminate()
			fake_w = fake_w[:, dims_to_discriminate, :]
		if self.opts.use_w_pool:
			real_w = self.real_w_pool.query(real_w)
			fake_w = self.fake_w_pool.query(fake_w)
		if fake_w.ndim == 3:
			fake_w = fake_w[:, 0, :]
		return real_w, fake_w
