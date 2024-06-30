import sys, os
from easy_local_features.submodules.git_alike.alike import ALike
import torch
import numpy as np
import cv2
import wget


from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils.download import downloadModel
from ..utils import ops

models = {
    'alike-t': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-t.pth',
    'alike-s': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-s.pth',
    'alike-n': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-n.pth',
    'alike-l': 'https://github.com/Shiaoming/ALIKE/raw/main/models/alike-l.pth',
}

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': ""},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': ""},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': ""},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': ""},
}

class ALIKE_baseline(BaseExtractor):
    """ALIKE baseline implementation.
    model_name: str = 'alike-t' | 'alike-s' | 'alike-n' | 'alike-l'
    top_k: int = -1. Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)
    scores_th: float = 0.2. Detector score threshold (default: 0.2).
    n_limit: int = 5000. Maximum number of keypoints to be detected (default: 5000).
    no_sub_pixel: bool = False. Do not detect sub-pixel keypoints (default: False).
    device: int = -1. Device to run the model on. -1 for CPU, >=0 for GPU. (default: -1)
    """
    
    default_conf = {
        'model_name': 'alike-t',
        'top_k': -1,
        'scores_th': 0.2,
        'n_limit': 2048,
        'sub_pixel': True,
        'model_path': None
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        model_name = conf.model_name
        top_k = conf.top_k
        scores_th = conf.scores_th
        n_limit = conf.n_limit
        sub_pixel = conf.sub_pixel
        model_path = conf.model_path
        self.DEV = torch.device('cpu')

        if model_name not in models:
            raise ValueError(f"Model name {model_name} not found in {models.keys()}")

        if model_path is None:
            url = models[model_name]
            model_path = downloadModel('alike', model_name, url)

        config = configs[model_name]
        config['model_path'] = model_path

        self.model = ALike(
            **config,
            top_k=top_k,
            scores_th=scores_th,
            n_limit=n_limit
        ).to(self.DEV)

        self.sub_pixel = sub_pixel
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, image, return_dict=False):   
        image = ops.prepareImage(image)
        image = image.to(self.DEV)
        pred = self.model(image, sub_pixel=self.sub_pixel)
        keypoints = pred['keypoints'] # (N, 2)
        descriptors = pred['descriptors'] # (N, 64)
        scores = pred['scores'] # (N,)
        
        if return_dict:
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores
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