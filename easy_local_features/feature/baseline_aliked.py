import sys, os
from easy_local_features.submodules.git_aliked.aliked import ALIKED
from .basemodel import BaseExtractor
from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import download, ops
import torch
import numpy as np
from functools import partial
import cv2
import wget
from omegaconf import OmegaConf


class ALIKED_baseline(BaseExtractor):
    """ALIKED baseline implementation.
        model_name: str = 'aliked-n32', Choose from ['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32']
        top_k: int = -1, # -1 for threshold based mode, >0 for top K mode.
        scores_th: float = 0.2, # Threshold for top K = -1 mode
    """
    default_conf = {
        "model_name": "aliked-n16",
        "top_k": -1,
        "detection_threshold": 0.2,
        "force_num_keypoints": False,
        "nms_radius": 2,
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        
        self.DEV = torch.device('cpu')        

        self.model = ALIKED({
            "model_name": conf.model_name,
            "max_num_keypoints": conf.top_k,
            "detection_threshold": conf.detection_threshold,
            "force_num_keypoints": conf.force_num_keypoints,
            "pretrained": True,
            "nms_radius": conf.nms_radius,
        })
        
        self.model = self.model.to(self.DEV)
        self.model.eval()

        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        image = ops.prepareImage(img, batch=True).to(self.DEV)
        with torch.no_grad():
            res = self.model({'image': image})

        keypoints = res['keypoints']
        descriptors = res['descriptors']
        
        if return_dict:
            return res

        return keypoints, descriptors

    def detect(self, img):
        keypoints, descriptors = self.detectAndCompute(img)
        return keypoints

    def compute(self, img, keypoints):
        raise NotImplemented
    
    def match(self, image1, image2):
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        
        data = {
            "descriptors0": desc0,
            "descriptors1": desc1,
        }
        
        response = self.matcher(data)
        
        m0 = response['matches0'][0]
        valid = m0 > -1
        
        mkpts0 = kp0[0][valid]
        mkpts1 = kp1[0][m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }

    def to(self, device):
        self.model.to(device)
        self.DEV = device
    
    @property
    def has_detector(self):
        return True