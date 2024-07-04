import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import cv2, os, wget

from kornia.feature.tfeat import TFeat

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils import ops

class XFeat_baseline(BaseExtractor):
    default_conf = {
        'top_k': 2048,
        'detection_threshold': 0.2,
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device   = torch.device('cpu')
        self.model = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = conf.top_k, detection_threshold=conf.detection_threshold)
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=None):
        img = ops.prepareImage(img).to(self.device)
        response = self.model.detectAndCompute(img, top_k = self.conf.top_k)[0]
        
        if return_dict:
            return response
        
        return response['keypoints'].unsqueeze(0), response['descriptors'].unsqueeze(0)

    def detect(self, img, op=None):
        raise NotImplemented

    def compute(self, img, keypoints):
        raise NotImplemented

    def to(self, device):
        self.model.to(device)
        self.device = device

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
        
        mkpts0 = kp0[0, valid]
        mkpts1 = kp1[0, m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }
        
    @property
    def has_detector(self):
        return True