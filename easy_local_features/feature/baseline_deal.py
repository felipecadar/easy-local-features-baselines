import torch
import numpy as np
import cv2, os

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils.download import getCache
from ..utils import ops

class DEAL_baseline(BaseExtractor):
    default_conf = {
        'top_k': 2048,
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        
        self.max_kps = conf.top_k
        self.DEV = torch.device('cpu')        
        cache = getCache('DEAL')
        self.sift = cv2.SIFT_create(nfeatures=self.max_kps, contrastThreshold=0.04, edgeThreshold=10)
        self.deal = torch.hub.load('verlab/DEAL_NeurIPS_2021', 'DEAL', True, cache)
        self.deal.device = self.DEV
        self.deal.net.eval()
        
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        gray = ops.to_cv(ops.prepareImage(img), to_gray=True)

        with torch.no_grad():
            kps = self.detect(gray)
            kps, desc = self.compute(gray, kps)

        kps = torch.tensor([kp.pt for kp in kps]).to(self.DEV)
        desc = torch.from_numpy(desc).to(self.DEV)
        
        if return_dict:
            return {
                'keypoints': kps,
                'descriptors': desc
            }

        return kps, desc

    def detect(self, img):
        gray = ops.to_cv(ops.prepareImage(img), to_gray=True)

        with torch.no_grad():
            kps = self.sift.detect(gray, None)

        return kps

    def compute(self, img, kps):
        gray = ops.to_cv(ops.prepareImage(img), to_gray=True)

        if not isinstance(kps[0], cv2.KeyPoint):
            kps = [cv2.KeyPoint(kp[0], kp[1], 0) for kp in kps]

        with torch.no_grad():
            desc = self.deal.compute(gray, kps)

        return kps, desc
    
    def to(self, device):
        self.deal.device = device
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
