import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from models.eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics


def run():
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 512
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []

    latents = []
    orig_latents = []
    device = torch.device('cuda')
    intrinsics = FOV_to_intrinsics(18.837, device=device)

    for i in range(1000):

        z = torch.from_numpy(np.random.randn(1,512)).to(device)

        cam_pivot = torch.tensor(net.decoder.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = net.decoder.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        ws = net.decoder.mapping(z, conditioning_params, truncation_psi=1, truncation_cutoff=14)
        orig_latents.append(ws.cpu().detach().numpy())
    orig_latents = np.concatenate(orig_latents)
    np.save(os.path.join(test_opts.exp_dir,'orig_latents.npy'),orig_latents)
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():

            input_cuda = input_batch.cuda().float()
            tic = time.time()
            img, cp, latent = net(input_cuda, return_latents= True,randomize_noise=False, resize=opts.resize_outputs)
            latents.append(latent.cpu().detach().numpy())
    latents = np.concatenate(latents)
    np.save(os.path.join(test_opts.exp_dir,'latents.npy'),latents)

if __name__ == '__main__':
    run()
