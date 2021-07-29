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



class winterdata(Dataset):

    def __init__(self, root_dir, transform_in = None):

        self.base_add = root_dir

        self.samples_23 = next(os.walk(root_dir + "/23"))[2]
        self.samples_23.sort(key=lambda f: int(re.sub('\D', '', f)))

        self.samples_25 = next(os.walk(root_dir + "/25"))[2]
        self.samples_25.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.samples_25 = self.samples_25[:-4]

        self.labels_23 = open(root_dir + "/23times_signals.txt").readlines()
        self.labels_25 = open(root_dir + "/25times_signals.txt").readlines()[:-4]

        self.transform_in = transform_in


    def __len__(self):
        return len(self.labels_23) + len(self.labels_25)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx > len(self.labels_23)-1:

            if idx == len(self.labels_23):
                idx_1 = len(self.labels_23)
            else:
                idx_1 = idx - 1

            idx = idx - len(self.labels_23)
            idx_1 = idx_1 - len(self.labels_23)
            return self.load_data("25", self.samples_25, self.labels_25, idx, idx_1)

        else:

            if idx == 0:
                idx_1 = 0
            else:
                idx_1 = idx - 1

            return self.load_data("23", self.samples_23, self.labels_23, idx, idx_1)


    def load_data(self, file, samples, labels, idx, idx_1):

        brightness_factor = 0.2 + np.random.uniform()

        base_img = cv2.imread(self.base_add + "/" + file + "/" + samples[idx])
        base_img = cv2.resize(base_img, (200,66), interpolation = cv2.INTER_AREA)
        base_img_br = self.change_brightness(base_img, brightness_factor)
        label_img = labels[idx]

        base_img__1 = cv2.imread(self.base_add + "/" + file + "/" + samples[idx_1])
        base_img__1 = cv2.resize(base_img__1, (200,66), interpolation = cv2.INTER_AREA)
        base_img_br__1 = self.change_brightness(base_img__1, brightness_factor)
        base_img__1 = labels[idx_1]

        optical_rgb = self.calc_dense_optical_flow(base_img_br__1, base_img_br)
        if self.transform_in:
            optical_rgb = self.transform_in(optical_rgb)

        #img = torch.cat([base_img__1, base_img, base_img_1], dim=0)

        sample = {'image': optical_rgb, 'label': abs(int(float(label_img)))}

        return sample


    def change_brightness(self, image, bright_factor):

        """augment brightness"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
        image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return image_rgb


    def calc_dense_optical_flow(self, prev_frame, curr_frame):

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




class commaai(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform_in = None):

        self.base_add = root_dir

        self.samples = next(os.walk(root_dir + "/frames_tra"))[2]
        self.samples.sort(key=lambda f: int(re.sub('\D', '', f)))

        self.labels = open(root_dir + "/train.txt").readlines()

        self.transform_in = transform_in

        # self.transform_flow = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Resize((200, 200)),
        #     ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx == 0:
            idx_1 = 0
        else:
            idx_1 = idx - 1


        brightness_factor = 0.2 + np.random.uniform()

        base_img = cv2.imread(self.base_add + "/frames_tra/" + self.samples[idx])
        base_img_br = self.change_brightness(base_img, brightness_factor)
        label_img = self.labels[idx]

        base_img__1 = cv2.imread(self.base_add + "/frames_tra/" + self.samples[idx_1])
        base_img_br__1 = self.change_brightness(base_img__1, brightness_factor)
        base_img__1 = self.labels[idx_1]

        optical_rgb = self.calc_dense_optical_flow(base_img_br__1, base_img_br)
        if self.transform_in:
            optical_rgb = self.transform_in(optical_rgb)

        #if self.transform_flow:
        #    optical_flow = self.transform_flow(optical_flow)

        #img = torch.cat([base_img__1, base_img, base_img_1], dim=0)

        sample = {'image': optical_rgb, 'label': float(label_img)}

        return sample



    def change_brightness(self, image, bright_factor):

        """augment brightness"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
        image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return image_rgb


    def calc_dense_optical_flow(self, prev_frame, curr_frame):

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
