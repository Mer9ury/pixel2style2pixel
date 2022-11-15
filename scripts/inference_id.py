import sys
from argparse import ArgumentParser
sys.path.append(".")
sys.path.append("..")
import os
from PIL import Image
from criteria import curricular_loss
from glob import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from SixDRepNet import SixDRepNet
import json
import cv2

def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--path', type=str,  default='results')
	args = parser.parse_args()
	return args

if __name__ == "__main__":
    args = parse_args()
    
    sim_loss = curricular_loss.IDLoss(0,use_curricular=True).eval()

    image_folder = glob(os.path.join(args.path,'inference_coupled','*.jpg'))
    with open(os.path.join(args.path,'scores.json'), "r") as st_json:
        scores = json.load(st_json)

    sim = 0
    res_dict = {}
    model = SixDRepNet()
    recon_sims = 0
    for image_path in tqdm(sorted(image_folder)):
        image = Image.open(image_path)

        image_name = '/workspace/celeba_test_crop/' +image_path[-9:]
        # image_name = image_path[-9:]
        img = cv2.imread(image_path)
        
        pitch, yaw, _ = model.predict(img[:,:512])
        res_dict[image_name] = [pitch.item(),yaw.item(),scores[image_name]['sim'],scores[image_name]['mse']]
        orig = image.crop((0,0,512,512))
        recon = image.crop((512,0,1024,512))
        left = image.crop((1024,0,1536,512))
        center = image.crop((1536,0,2048,512))
        right = image.crop((2048,0,2560,512))

        recon_sim = sim_loss(orig,recon,True)
        recon_sims += recon_sim
        temp_sim1 = sim_loss(orig,left,True)
        temp_sim2 = sim_loss(orig,center,True)
        temp_sim3 = sim_loss(orig,right,True)
        temp_sim = (temp_sim1 + temp_sim3) / 2
        print(image_name)
        print(recon_sim)
        sim += temp_sim
        res_dict[image_name] = [pitch.item(),yaw.item(),scores[image_name]['sim'],scores[image_name]['mse'],temp_sim]
    print(sim / len(image_folder))
    print(recon_sims /len(image_folder))

    with open(os.path.join(args.path,'results.json'), 'w') as f : 
	    json.dump(res_dict, f, indent=4)