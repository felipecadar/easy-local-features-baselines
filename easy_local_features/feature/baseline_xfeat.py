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
        self.model.eval()
        self.model.dev = self.device
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=None):
        img = ops.prepareImage(img).to(self.device)
        response = self.model.detectAndCompute(img, top_k = self.conf.top_k)[0]
        
        if return_dict:
            return response
        
        return response['keypoints'].unsqueeze(0), response['descriptors'].unsqueeze(0)

    def detect(self, img, op=None):
        return self.detectAndCompute(img, return_dict=True)['keypoints']

    # def match(self, image1, image2):
    #     mkpts0, mkpts1 = self.model.match_xfeat(image1, image2)
    #     return {
    #         'mkpts0': mkpts0,
    #         'mkpts1': mkpts1,
    #     }

    def match_xfeat(self, image1, image2):
        mkpts0, mkpts1 = self.model.match_xfeat(image1, image2)
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }

    def match_xfeat_star(self, image1, image2):
        mkpts0, mkpts1 = self.model.match_xfeat_star(image1, image2)
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }

    def compute(self, img, keypoints):
        raise NotImplemented

    def to(self, device):
        self.model.to(device)
        self.model.dev = device
        self.device = device
        return self
    
    @property
    def has_detector(self):
        return True
    
    
if __name__ == "__main__":
    from easy_local_features.utils import io, vis, ops
    method = XFeat_baseline()
    
    img0 = io.fromPath("test/assets/megadepth0.jpg")
    img1 = io.fromPath("test/assets/megadepth1.jpg")
    
    nn_matches = method.match(img0, img1)
    xfeat_matches = method.match_xfeat(img0, img1)
    xfeat_star_matches = method.match_xfeat_star(img0, img1)
    
    vis.plot_pair(img0, img1)
    vis.plot_matches(nn_matches['mkpts0'], nn_matches['mkpts1'])
    vis.add_text("")

    vis.plot_pair(img0, img1)
    vis.plot_matches(xfeat_matches['mkpts0'], xfeat_matches['mkpts1'])

    vis.plot_pair(img0, img1)
    vis.plot_matches(xfeat_star_matches['mkpts0'], xfeat_star_matches['mkpts1'])
    
    vis.show()