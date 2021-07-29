import torch
from unet import *
from os.path import isfile, join
import glob
import argparse
import os
import sys
import re
from PIL import *
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--images_add', '-i', default = '../signals/data/frames/23', type = str,
                    help = 'Path to images are required to segment.')
    parser.add_argument('--output_add', '-o', default = './outputs/23', type = str,
                    help = 'Path to the folder that outpust are going to be stored.')
    parser.add_argument('--model_add', '-m', default = "/home/lunet/wsmo6/SemanticSegmentation/checkpoints/new_gpu/best.pth", type = str,
                    help = 'Path to the model.')

    args = parser.parse_args()
    inference_unet(args.images_add, args.output_add, args.model_add)



def inference_unet(main_dir, output_add, model_add):

    img_w = 224
    img_h = 224

    preprocess_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((img_w, img_h))
    ])

    files = glob.glob(join(main_dir, "*.jpg"))
    files.sort(key = lambda f: int(re.sub('\D', '', f)))
    files = files[19800:]

    model = UNet(out_channels = 3)
    model.load_state_dict(torch.load(model_add)['model_state_dict'])
    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    model = model.to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for path in files:

        base_img = Image.open(path)
        pre_base_img = preprocess_in(base_img)
        start.record()
        output = model((pre_base_img.unsqueeze(0)).to(device))
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))  # milliseconds
        save_image(output[0], output_add + "/" + path.split("/")[-1])
        #output = (((output.squeeze(0).detach().cpu().numpy()) * 255).astype(np.uint8)).reshape((img_h, img_w, 3))
        #output = Image.fromarray(output)
        #output.save(output_add + "/" + path.split("/")[-1])




if __name__ == "__main__":
    main()
