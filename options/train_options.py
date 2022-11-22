from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str, help='Type of dataset/experiment to run')
		self.parser.add_argument('--dataset_path', default=None, type=str, help='dataset_path')
		self.parser.add_argument('--synthetic_dataset_path', default=None, type=str, help='dataset_path')
		self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use')
		self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
		self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
		self.parser.add_argument('--render_resolution', default=128, type=int, help='Resoloution of neural rendering result')
		self.parser.add_argument('--output_size', default=512, type=int, help='Output size of generator')

		self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=4, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.000075, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
		self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
		self.parser.add_argument('--load_latent_avg', action='store_true', help='Whether to use average latent vector to generate codes from pretrained model.')
		self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')
		self.parser.add_argument('--norm', default = 'GN', type=str, help='which normalization to use at encoder model')

		self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
		self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
		self.parser.add_argument('--moco_lambda', default=0, type=float, help='Moco-based feature similarity loss multiplier factor')
		self.parser.add_argument('--cams_lambda', default=0, type=float, help='Camera parameter Loss factor')

		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')

		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

		# Discriminator flags
		self.parser.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
		self.parser.add_argument('--w_discriminator_lr', default=1.5e-5, type=float, help='Dw learning rate')
		self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
		self.parser.add_argument("--d_reg_every", type=int, default=16,
									help="interval for applying r1 regularization")
		self.parser.add_argument('--use_w_pool', action='store_true',
									help='Whether to store a latnet codes pool for the discriminator\'s training')
		self.parser.add_argument("--w_pool_size", type=int, default=50,
									help="W\'s pool size, depends on --use_w_pool")
		# arguments for weights & biases support
		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None, help='For super-res, comma-separated resize factors to use for inference.')

		# e4e specific
		self.parser.add_argument('--delta_norm', type=int, default=2, help="norm type of the deltas")
		self.parser.add_argument('--delta_norm_lambda', type=float, default=2e-4, help="lambda for delta norm loss")

        # Progressive training
		self.parser.add_argument('--progressive_steps', nargs='+', type=int, default=None,
                                 help="The training steps of training new deltas. steps[i] starts the delta_i training")
		self.parser.add_argument('--progressive_start', type=int, default=None,
                                 help="The training step to start training the deltas, overrides progressive_steps")
		self.parser.add_argument('--progressive_step_every', type=int, default=2_000,
                                 help="Amount of training steps for each progressive step")

		# arguments for data distributed learning
		self.parser.add_argument('--distributed', type=bool, default=False, help ='Whether distributed learning is used')
		self.parser.add_argument('--num_gpus', type=int, default=4, help ='the number of process(GPU)')
		self.parser.add_argument('--dist_backend', type=str, default='nccl', help ='dist_backend')
		self.parser.add_argument('--rank', type=int, default= 0, help ='rank')
	def parse(self):
		opts = self.parser.parse_args()
		return opts
