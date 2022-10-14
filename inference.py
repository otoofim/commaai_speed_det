import torch
from arch import *
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
import cv2




def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--images_add', '-i', default = '../signals/data/signal dataset/veh_yaw_latt_25.txt', type = str,
                    help = 'Path to images are required to segment.')
    parser.add_argument('--output_add', '-o', default = './output/sp_yaw_latt_winter_25pred.txt', type = str,
                    help = 'Path to the folder that outpust are going to be stored.')
    parser.add_argument('--model_add', '-m', default = "./checkpoints/winter_veh_latt_yaw/best.pth", type = str,
                    help = 'Path to the model.')

    args = parser.parse_args()
    inference_unet(args.images_add, args.output_add, args.model_add)



def inference_unet(main_dir, output_add, model_add):

    img_w = 256
    img_h = 256

    preprocess_in = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((img_w, img_h))
    ])

    #files = glob.glob(join(main_dir, "*.jpg"))
    #files.sort(key = lambda f: int(re.sub('\D', '', f)))
    #files = files[:-4]
    files = open(main_dir).readlines()[1:-4]
    files = [line.split(',')[0].strip() for line in files]


    #model =MyConv(img_w)
    model = half_UNet((img_h, img_w), 3,  out_channels = 3)
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

    f = open(output_add, "w")

    for i, path in enumerate(files):


        if i == 0:
            i_1 = 0
        else:
            i_1 = i -1

        img = load_data(files[i_1], files[i], preprocess_in)

        start.record()
        output = model(img.to(device))
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))  # milliseconds
        ext_output = output.detach().cpu().numpy()[0]

        if isinstance(ext_output, np.ndarray):
            for out in ext_output:
                f.write(str(out)  + ",")
        else:
            f.write(str(ext_output))
        f.write("\n")


    f.close()


def load_data(image1, image2, transform_in):

    brightness_factor = 0.2 + np.random.uniform()

    base_img = cv2.imread(image2)
    base_img = cv2.resize(base_img, (200,66), interpolation = cv2.INTER_AREA)
    base_img_br = change_brightness(base_img, brightness_factor)

    base_img__1 = cv2.imread(image1)
    base_img__1 = cv2.resize(base_img__1, (200,66), interpolation = cv2.INTER_AREA)
    base_img_br__1 = change_brightness(base_img__1, brightness_factor)

    optical_rgb = calc_dense_optical_flow(base_img_br__1, base_img_br)
    if transform_in:
        optical_rgb = transform_in(optical_rgb)


    return optical_rgb.unsqueeze(0)



def change_brightness(image, bright_factor):

    """augment brightness"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image_rgb


def calc_dense_optical_flow(prev_frame, curr_frame):

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_frame)
    hsv[:,:,1] = 255
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 2, 5, 1.3, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow











if __name__ == "__main__":
    main()
