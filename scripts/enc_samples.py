
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""
import sys

sys.path.append(".")
sys.path.append("..")

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import torch
from torchvision.transforms import transforms
from models.eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from options.test_options import TestOptions
from PIL import Image
from glob import glob
from argparse import Namespace
from models.psp import pSp
from tqdm import tqdm
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
              default='w', show_default=True)
@click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int,
              help='Multiplier for depth sampling in volume rendering', default=500, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type:str,
        image_path:str,
        c_path:str,
        num_steps:int
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    os.makedirs(outdir, exist_ok=True)

    test_opts = TestOptions().parse()

    ckpt = torch.load(network_pkl, map_location='cpu')
    opts = ckpt['opts']
    
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 512
    opts = Namespace(**opts)
    device = torch.device('cuda')
    net = pSp(opts)
    net.eval()
    net.cuda()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((512,512))
    ])
    intrinsics = FOV_to_intrinsics(18.837).cuda()
    cs = []
    ws = []
    G = net.decoder
    for angle_p in [-0.2,0,0.2]:
        for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            cs.append(camera_params)

    im_paths = glob(image_path+'/*.png')
    print(len(im_paths))
    for im_path in tqdm(im_paths):
        orig_image = Image.open(im_path).convert('RGB')
        image_name = im_path[:-4]
        temp_ws = []
        for i, c in enumerate(cs):
            image = orig_image.crop(/Users/mer9ury/Desktop/ciplab/Encoder_for_3DGAN/latent_distribution/result1.png(i*512,0,(i+1)*512,512))
            # c = torch.FloatTensor(c)
            from_im = trans(image).cuda().reshape(1,3,512,512)
            print(from_im.shape)
            id_image = torch.squeeze((from_im.cuda() + 1) / 2) * 255
            
            w, cams = net.encoder(from_im)
            w = w + net.latent_avg.repeat(codes.shape[0], 1)
            print(w.shape)
            w = w.detach().cpu().numpy()
            temp_ws.append(w)
        temp_ws = np.concatenate(temp_ws)    
        ws.append(temp_ws)
    ws = np.concatenate(ws)

    np.save(f'{outdir}/results.npy', ws)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------



