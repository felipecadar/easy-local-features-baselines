import sys, os
from easy_local_features.submodules.git_aliked.nets.aliked import ALIKED
from .basemodel import BaseExtractor
from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import download, ops
import torch
import numpy as np
from functools import partial
import cv2
import wget
import pyrootutils
root = pyrootutils.find_root()
from omegaconf import OmegaConf

models = {
    "aliked-n16": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
    "aliked-n16rot": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16rot.pth",
    "aliked-n32": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n32.pth",
    "aliked-t16": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-t16.pt"
}

class ALIKED_baseline(BaseExtractor):
    """ALIKED baseline implementation.
        model_name: str = 'aliked-n32', Choose from ['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32']
        top_k: int = -1, # -1 for threshold based mode, >0 for top K mode.
        scores_th: float = 0.2, # Threshold for top K = -1 mode
        n_limit: int = 5000, # Maximum number of keypoints to be detected
        load_pretrained: bool = True, load pretrained model or not
        device=-1, -1 for CPU, >=0 for GPU
        model_path=None, use custom model path instead of the default one. 
    """
    default_conf = {
        'model_name': 'aliked-n32',
        'top_k': -1,
        'scores_th': 0.2,
        'n_limit': 2048,
        'load_pretrained': True,
        'model_path': None
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        
        model_name = conf.model_name
        top_k = conf.top_k
        scores_th = conf.scores_th
        n_limit = conf.n_limit
        load_pretrained = conf.load_pretrained
        model_path = conf.model_path
        self.DEV = torch.device('cpu')        

        if model_path is None:
            if model_name not in models:
                raise ValueError(f"Model name {model_name} not found in {models.keys()}")
            model_path = download.downloadModel('aliked', model_name, models[model_name])

        self.model = ALIKED(
            model_name=model_name,
            top_k=top_k,
            scores_th=scores_th,
            n_limit=n_limit,
            load_pretrained=load_pretrained,
            pretrained_path=model_path,
        )
        self.model = self.model.to(self.DEV)
        self.model.eval()

        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img):
        image = ops.prepareImage(img, batch=True).to(self.DEV)
        with torch.no_grad():
            res = self.model.run(image)

        keypoints = res['keypoints']
        descriptors = res['descriptors']
        # scores = res['scores']

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

    def to(self, device):
        self.model.to(device)
        self.DEV = device
    
    @property
    def has_detector(self):
        return True

if __name__ == "__main__":
    img = cv2.imread(str(root / "assets" / "notredame.png"))
    extractor = ALIKED_baseline()

    keypoints, descriptors = extractor.detectAndCompute(img)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))
    cv2.imshow('superpoint', output_image)
    cv2.waitKey(0)