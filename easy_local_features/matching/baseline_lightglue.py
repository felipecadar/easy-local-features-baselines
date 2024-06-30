from kornia.feature import LightGlueMatcher, laf_from_center_scale_ori

import cv2

import kornia
import torch
import functools
import numpy as np

AVAILABLE_FEATURES = ["superpoint", "disk"]

@functools.lru_cache(maxsize=1)
def getLightGlueMatcher(features="superpoint"):
    return LightGlueMatcher(feature_name=features)

class LightGlue_baseline:
    def __init__(self, features="superpoint"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = getLightGlueMatcher(features=features).to(self.device)

    def match(self, keypoints0, keypoints1, descriptors0, descriptors1):

        if len(keypoints0) == 0 or len(keypoints1) == 0:
            return [], [], []
        
        og_keypoints0 = keypoints0
        og_keypoints1 = keypoints1
        
        if isinstance(keypoints0[0], cv2.KeyPoint):
            keypoints0 = np.array([k.pt for k in keypoints0])

        if isinstance(keypoints1[0], cv2.KeyPoint):
            keypoints1 = np.array([k.pt for k in keypoints1])

        if isinstance(keypoints0, np.ndarray):
            keypoints0 = torch.from_numpy(keypoints0).to(self.device)

        if isinstance(keypoints1, np.ndarray):
            keypoints1 = torch.from_numpy(keypoints1).to(self.device)

        if isinstance(descriptors0, np.ndarray):
            descriptors0 = torch.from_numpy(descriptors0).to(self.device)

        if isinstance(descriptors1, np.ndarray):
            descriptors1 = torch.from_numpy(descriptors1).to(self.device)

        # check shape
        if len(keypoints0.shape) == 2:
            keypoints0 = keypoints0.unsqueeze(0)

        if len(keypoints1.shape) == 2:
            keypoints1 = keypoints1.unsqueeze(0)

        # set to float
        if keypoints0.dtype != torch.float32:
            keypoints0 = keypoints0.float()

        if keypoints1.dtype != torch.float32:
            keypoints1 = keypoints1.float()

        # to lafs
        lafs0 = laf_from_center_scale_ori(keypoints0)
        lafs1 = laf_from_center_scale_ori(keypoints1)

        # match
        input = {
            "lafs1": lafs0,
            "lafs2": lafs1,
            "desc1": descriptors0,
            "desc2": descriptors1,
        }
        with torch.no_grad():
            mscores, matches = self.matcher(**input)

        mscores = mscores.squeeze(-1).detach().cpu().numpy()
        matches = matches.detach().cpu().numpy().astype(int)

        cv2_matches = [ 
            cv2.DMatch(_imgIdx=0, _trainIdx=i, _queryIdx=j, _distance=1-s) 
            for [i, j], s in 
            zip(matches, mscores) 
        ]

        return og_keypoints0, og_keypoints1, cv2_matches
    
