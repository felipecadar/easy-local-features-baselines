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

class ORB_baseline(BaseExtractor):
    default_conf = {
        'top_k': 2048,
        "scaleFactor": 1.2,
        "nlevels": 8,
        "edgeThreshold": 31,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": cv2.ORB_HARRIS_SCORE,
        "patchSize": 31,
        "fastThreshold": 20
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.model = cv2.ORB_create(
            nfeatures=conf.top_k,
            scaleFactor=conf.scaleFactor,
            nlevels=conf.nlevels,
            edgeThreshold=conf.edgeThreshold,
            firstLevel=conf.firstLevel,
            WTA_K=conf.WTA_K,
            scoreType=conf.scoreType,
            patchSize=conf.patchSize,
            fastThreshold=conf.fastThreshold
        )
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        img = ops.prepareImage(img, gray=True, batch=False)
        img = ops.to_cv(img)

        keypoints, descriptors = self.model.detectAndCompute(img, None)
        pred = {
            'keypoints': np.array([kp.pt for kp in keypoints], dtype=np.float32),
            'descriptors': np.array(descriptors, dtype=np.float32)
        }

        if return_dict:
            return pred
        
        return pred['keypoints'], pred['descriptors']

    def detect(self, img):
        img = ops.prepareImage(img, gray=True, batch=False)
        img = ops.to_cv(img)

        keypoints = self.model.detect(img, None)
        pred = {
            'keypoints': np.array([kp.pt for kp in keypoints], dtype=np.float32),
        }

        return pred['keypoints']

    def compute(self, img, kps):
        raise NotImplementedError("This method is not implemented in this class")
    
    def to(self, device):
        return self

    @property
    def has_detector(self):
        return True
