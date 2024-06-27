import os
import numpy as np
import torch
import wget
import cv2

from easy_local_features.submodules.git_r2d2.tools import common
from easy_local_features.submodules.git_r2d2.tools.dataloader import norm_RGB
from easy_local_features.submodules.git_r2d2.nets.patchnet import *
from easy_local_features.submodules.git_r2d2.extract import load_network, NonMaxSuppression, extract_multiscale


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

models_URLs = {
    'faster2d2_WASF_N8_big': 'https://github.com/naver/r2d2/raw/master/models/faster2d2_WASF_N16.pt',
    'faster2d2_WASF_N16': 'https://github.com/naver/r2d2/raw/master/models/faster2d2_WASF_N8_big.pt',
    'r2d2_WAF_N16': 'https://github.com/naver/r2d2/raw/master/models/r2d2_WAF_N16.pt',
    'r2d2_WASF_N8_big': 'https://github.com/naver/r2d2/raw/master/models/r2d2_WASF_N16.pt',
    'r2d2_WASF_N16': 'https://github.com/naver/r2d2/raw/master/models/r2d2_WASF_N8_big.pt',
}

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils.download import downloadModel
from ..utils import ops

class R2D2_baseline(BaseExtractor):
    
    default_conf = {
        'top_k': 2048,
        'pretrained_weigts': 'r2d2_WASF_N16',
        'model_path': None,
        'rel_thr': 0.7,
        'rep_thr': 0.7,
        'scale_f': 2**0.25,
        'min_scale': 0.0,
        'max_scale': 1,
        'min_size': 256,
        'max_size': 1024
    }
    
    def __init__(self,conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        
        self.top_k = conf.top_k
        self.rel_thr = conf.rel_thr
        self.rep_thr = conf.rep_thr
        self.scale_f = conf.scale_f
        self.min_scale = conf.min_scale
        self.max_scale = conf.max_scale
        self.min_size = conf.min_size
        self.max_size = conf.max_size
        pretrained_weigts = conf.pretrained_weigts

        self.DEV   = torch.device('cpu')

        url = models_URLs[pretrained_weigts]
        model_path = downloadModel('r2d2', pretrained_weigts, url)

        # load the network...
        self.model = load_network(model_path)

        # create the non-maxima detector
        self.detector = NonMaxSuppression(
            rel_thr = self.rel_thr, 
            rep_thr = self.rep_thr)
        
        self.matcher = NearestNeighborMatcher()

    def _toTorch(self, img):
        return img

    def detectAndCompute(self, img, return_dict=False):
        img = ops.prepareImage(img, imagenet=True).to(self.DEV)
        
        # extract keypoints/descriptors for a single image
        xys, desc, scores = extract_multiscale(self.model, img, self.detector,
            scale_f   = self.scale_f, 
            min_scale = self.min_scale, 
            max_scale = self.max_scale,
            min_size  = self.min_size, 
            max_size  = self.max_size, 
            verbose = False)

        # idxs = scores.argsort()[-self.top_k or None:]
        idxs = torch.argsort(scores, descending=True)[:self.top_k]
        
        keypoints = xys[idxs]
        keypoints, scales = keypoints.split([2,1], dim=1)
        descriptors = desc[idxs]
        scores = scores[idxs]

        if return_dict:
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores,
                'scales': scales,
            }


        return keypoints, descriptors
    
    def detect(self, img):
        return self.detectAndCompute(img)[0]

    def compute(self, image, keypoints):
        raise NotImplemented

    def to(self, device):
        self.model.to(device)
        self.DEV = device

    def match(self, image1, image2):
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        
        data = {
            "descriptors0": desc0.unsqueeze(0),
            "descriptors1": desc1.unsqueeze(0),
        }
        
        response = self.matcher(data)
        
        m0 = response['matches0'][0]
        valid = m0 > -1
        
        mkpts0 = kp0[valid]
        mkpts1 = kp1[m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }
        
    @property
    def has_detector(self):
        return True