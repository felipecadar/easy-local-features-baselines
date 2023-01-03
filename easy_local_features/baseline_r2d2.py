import sys, os
r2d2_folder = os.path.dirname(os.path.realpath(__file__)) + '/submodules/git_r2d2/'
sys.path.insert(0, r2d2_folder)
import os, pdb
from PIL import Image
import numpy as np
import torch

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *
from extract import load_network, NonMaxSuppression, extract_multiscale

import torch
import numpy as np
from functools import partial
import cv2

# Pretrained models
# -----------------
# For your convenience, we provide five pre-trained models in the `models/` folder:
#  - `r2d2_WAF_N16.pt`: this is the model used in most experiments of the paper (on HPatches `MMA@3=0.686`). It was trained with Web images (`W`), Aachen day-time images (`A`) and Aachen optical flow pairs (`F`)
#  - `r2d2_WASF_N16.pt`: this is the model used in the visual localization experiments (on HPatches `MMA@3=0.721`). It was trained with Web images (`W`), Aachen day-time images (`A`), Aachen day-night synthetic pairs (`S`), and Aachen optical flow pairs (`F`).
#  - `r2d2_WASF_N8_big.pt`: Same than previous model, but trained with `N=8` instead of `N=16` in the repeatability loss. In other words, it outputs a higher density of keypoints. This can be interesting for certain applications like visual localization, but it implies a drop in MMA since keypoints gets slighlty less reliable.
#  - `faster2d2_WASF_N16.pt`: The Fast-R2D2 equivalent of r2d2_WASF_N16.pt
#  - `faster2d2_WASF_N8_big.pt`: The Fast-R2D2 equivalent of r2d2_WASF_N8.pt

# model name	model size  (#weights)	number of keypoints	MMA@3 on HPatches
# r2d2_WAF_N16.pt	0.5M	5K	0.686
# r2d2_WASF_N16.pt	0.5M	5K	0.721
# r2d2_WASF_N8_big.pt	1.0M	10K	0.692
# faster2d2_WASF_N8_big.pt	1.0M	5K	0.650

RGB_mean = [0.485, 0.456, 0.406]
RGB_std  = [0.229, 0.224, 0.225]

class R2D2_baseline():
    def __init__(self,
                max_keypoints=2048,
                model_name='r2d2_WASF_N16.pt',
                rel_thr=0.7,
                rep_thr=0.7,
                scale_f=2**0.25,
                min_scale=0.0,
                max_scale=1,
                min_size=256,
                max_size=1024):

        self.top_k = max_keypoints
        self.model_name = model_name
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
        self.scale_f = scale_f
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_size = min_size
        self.max_size = max_size

        self.DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CPU   = torch.device('cpu')

        model_path = r2d2_folder + 'models/' + model_name

        if torch.cuda.is_available():
            iscuda = common.torch_set_gpu([0])
        else:
            iscuda = False

        # load the network...
        self.net = load_network(model_path)
        if iscuda: self.net = self.net.cuda()

        # create the non-maxima detector
        self.detector = NonMaxSuppression(
            rel_thr = rel_thr, 
            rep_thr = rep_thr)

    def _toTorch(self, img):
        return img

    def detectAndCompute(self, img, op=None):
        img = self._toTorch(img)
        img = norm_RGB(img)[None]
        img = img.to(self.DEV)
        
        # extract keypoints/descriptors for a single image
        xys, desc, scores = extract_multiscale(self.net, img, self.detector,
            scale_f   = self.scale_f, 
            min_scale = self.min_scale, 
            max_scale = self.max_scale,
            min_size  = self.min_size, 
            max_size  = self.max_size, 
            verbose = False)

        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-self.top_k or None:]
        
        keypoints = xys[idxs]
        descriptors = desc[idxs]
        scores = scores[idxs]

        cv_kps = [cv2.KeyPoint(kp[0], kp[1], 1, -1, s, 0, -1) for kp, s in zip(keypoints, scores)]

        return cv_kps, descriptors

    def detect(self, img, op=None):
        cv_kps, descriptors = self.detectAndCompute(img)
        return cv_kps

    def compute(self, img, cv_kps):
        raise NotImplemented

if __name__ == "__main__":
    img = cv2.imread("assets/notredame.png")

    extractor = R2D2_baseline()

    keypoints, descriptors = extractor.detectAndCompute(img)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))

    cv2.imwrite("r2d2_test.png", output_image)