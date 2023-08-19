from kornia.feature import LightGlueMatcher, laf_from_center_scale_ori

import pyrootutils
import cv2

import kornia
import torch
import functools
import numpy as np

root = pyrootutils.find_root()

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
    
if __name__ == "__main__":
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
    
    img = cv2.imread(str(root / "assets" / "notredame.png"))
    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)

    extractor = SuperPoint_baseline()
    keypoints0, descriptors0 = extractor.detectAndCompute(img)
    keypoints1, descriptors1 = extractor.detectAndCompute(img)

    matcher = LightGlue_baseline()
    cv2_mkpts1, cv2_mkpts2, cv2_matches = matcher.match(
        keypoints0=keypoints0,
        keypoints1=keypoints1,
        descriptors0=descriptors0,
        descriptors1=descriptors1,
    )

    img = cv2.drawMatches(img, cv2_mkpts1, img, cv2_mkpts2, cv2_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    