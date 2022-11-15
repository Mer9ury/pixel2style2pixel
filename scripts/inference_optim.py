import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import json

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from configs import eg3d_config
from configs.paths_config import model_paths
from datasets.eg3d_dataset import EG3DDataset
from datasets.celeba_dataset import CELEBADataset
from utils.common import tensor2im, log_input_image
import torch.nn.functional as F
from criteria import id_loss


from options.test_options import TestOptions
from training.triplane import TriPlaneGenerator
from models.w_plus_projector import project
from models.eg3d.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from configs.eg3d_config import init_kwargs,rendering_kwargs

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

    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 512
    opts = Namespace(**opts)

    sampling_multiplier = 2
    G = TriPlaneGenerator(*(), **init_kwargs).eval()
    G.rendering_kwargs = eg3d_config.rendering_kwargs
    G_ckpt = torch.load(model_paths['eg3d_pth'], map_location='cpu')['G_ema']
    G.load_state_dict(G_ckpt, strict=False)

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    G.eval()
    G.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = CELEBADataset(dataset_path=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts,
							    metadata = 'celebahq_dataset.json',
                                is_train = False)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    sim_loss = id_loss.IDLoss(opts.device,use_curricular=True).eval()

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    similarity = 0
    mse = 0
    score_dict = {}
    transform = transforms_dict['transform_inference']
    w_vectors = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break

        images, cams = input_batch
        tic = time.time()

        result_batch, recon_img, result_ws = run_on_batch_samples(images, cams, G, opts)
        w_vectors.append(result_ws)
        toc = time.time()
        global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset.metadata[global_i]['path']
            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(images[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                          np.array(result)], axis=1)
                

                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

                gt = transform(input_im).unsqueeze(0).to(opts.device)

                temp_sim = sim_loss(recon_img,gt,gt,True)[2][0]['diff_target']
                similarity += temp_sim
                temp_mse = F.mse_loss(gt,recon_img) 
                mse += temp_mse
                score_dict[im_path] = {'sim':temp_sim, 'mse':temp_mse.item()}
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))
                print(temp_sim,temp_mse)
            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1
            
        if global_i > 1000:
            break
    w_vectors = np.concatenate(w_vectors)
    np.save(os.path.join(opts.exp_dir, 'w_result.npy'),w_vectors)

    print(f'mean sim is {similarity / 1000}')
    print(f'mean mse is {mse / 1000}')

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(os.path.join(opts.exp_dir,'scores.json'), 'w') as f : 
	    json.dump(score_dict, f, indent=4)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net.encoder(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch

def run_on_batch_samples(images, cams, G, opts):
    images = images.to(opts.device)
    cams = cams.to(opts.device)
    images = torch.squeeze((images + 1) / 2) * 255
    ws = project(G, c = cams, target = images, device=opts.device, w_avg_samples=600, num_steps = 500)

    with torch.no_grad():

        angle_p = -0.2
        intrinsics = FOV_to_intrinsics(18.837, device=opts.device)
        recon_img = G.synthesis(ws, cams)['image']
        imgs = [recon_img]

        for angle_y, angle_p in [(.3, angle_p), (0, angle_p), (-.3, angle_p)]:
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=opts.device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=opts.device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=opts.device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            img = G.synthesis(ws, camera_params)['image']
            imgs.append(img)
    img = torch.cat(imgs, dim=3)


    return img, recon_img, ws.detach().cpu().numpy()


if __name__ == '__main__':
    run()
