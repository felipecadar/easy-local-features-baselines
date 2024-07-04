import torch
import numpy as np
import cv2, os

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils.download import downloadModel
from ..utils import ops

from ..submodules.git_sfd2.sfd2 import SFD2

weights_link = 'https://github.com/feixue94/sfd2/raw/dev/weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth'


class SFD2_baseline(BaseExtractor):
    default_conf = {
        'top_k': 2048,
        "config_name":"ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1600",
        "use_stability":False,
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        
        self.max_kps = conf.top_k
        self.DEV = torch.device('cpu')        
        self.model = SFD2(
            config_name=conf.config_name, 
            use_stability=conf.use_stability,
            top_k=conf.top_k,
        )
        
        weights = downloadModel("sfd2", "20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth", weights_link)
        self.model.model.load_state_dict(torch.load(weights, map_location='cpu')['model'], strict=False)
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        img = ops.prepareImage(img)

        with torch.no_grad():
            pred = self.model({
                'image': img.to(self.DEV),
                'original_size': torch.tensor(img.shape[-2:]).to(self.DEV),
            })

        # make all into tensor and move to device
        for k in pred:
            pred[k] = torch.tensor(pred[k]).to(self.DEV).unsqueeze(0)

        if return_dict:
            return pred
        
        return pred['keypoints'], pred['descriptors']

    def detect(self, img):
        raise NotImplementedError("This method is not implemented in this class")

    def compute(self, img, kps):
        raise NotImplementedError("This method is not implemented in this class")
    
    def to(self, device):
        self.model = self.model.to(device)
        self.DEV = device
        return self

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
