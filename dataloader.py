from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os, os.path
import torch
#from PIL import *
from PIL import Image
import re
from torchvision.models import *
from optFlow import *
from torchvision import transforms


class commaai(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform_in = None):

        self.base_add = root_dir

        self.samples = next(os.walk(root_dir + "/frames_tra"))[2]
        self.samples.sort(key=lambda f: int(re.sub('\D', '', f)))

        self.labels = open(root_dir + "/train.txt").readlines()

        self.transform_in = transform_in

        self.transform_flow = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((200, 200)),
            ])

        #self.method = cv2.optflow.calcOpticalFlowSparseToDense


    def __len__(self):
        return len(self.labels)

    # def __getitem__(self, idx):
    #
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #
    #     if idx == 0:
    #         idx = 1
    #     if idx == len(self.labels) - 1:
    #         idx = len(self.labels) - 2
    #
    #     base_img__1 = Image.open(self.base_add + "/frames_tra/" + self.samples[idx - 1])
    #     label_img__1 = self.labels[idx - 1]
    #
    #
    #     base_img = Image.open(self.base_add + "/frames_tra/" + self.samples[idx])
    #     label_img = self.labels[idx]
    #
    #
    #     #base_img_1 = Image.open(self.base_add + "/frames_tra/" + self.samples[idx + 1])
    #     #label_img_1 = self.labels[idx + 1]
    #
    #
    #     optical_flow = self.dense_optical_flow(self.method, base_img__1, base_img, to_gray = True)
    #
    #     if self.transform_in:
    #         base_img__1 = self.transform_in(base_img__1)
    #         base_img = self.transform_in(base_img)
    #         #base_img_1 = self.transform_in(base_img_1)
    #
    #     if self.transform_flow:
    #         optical_flow = self.transform_flow(optical_flow)
    #
    #     #img = torch.cat([base_img__1, base_img, base_img_1], dim=0)
    #     img = torch.cat([base_img__1, base_img], dim=0)
    #
    #
    #     sample = {'image': img, 'optical': optical_flow, 'label': float(label_img)}
    #     #sample = {'image': img, 'label': float(label_img_1)}
    #
    #     return sample




    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_img = Image.open(self.base_add + "/frames_tra/" + self.samples[idx])
        label_img = self.labels[idx]


        if self.transform_in:
            base_img = self.transform_in(base_img)

        #img = torch.cat([base_img__1, base_img, base_img_1], dim=0)

        sample = {'image': base_img, 'label': float(label_img)}

        return sample



    def dense_optical_flow(self, method, old_frame, new_frame, params=[], to_gray=False):

        # crate HSV & make Value a constant
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255

        # Preprocessing for exact method
        if to_gray:
            old_frame = cv2.cvtColor(np.array(old_frame.convert('RGB')), cv2.COLOR_BGR2GRAY)

        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(np.array(new_frame.convert('RGB')), cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return Image.fromarray(bgr)
