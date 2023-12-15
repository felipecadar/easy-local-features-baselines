import pyrootutils
root = pyrootutils.find_root()

import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import cv2, os, wget

from kornia.feature.tfeat import TFeat

class TFeat_baseline():
    def __init__(self, device=-1):
        self.CPU   = torch.device('cpu')
        self.DEV   = torch.device(f'cuda:{device}' if (torch.cuda.is_available() and device >= 0) else 'cpu')
        self.model = TFeat().to(self.DEV)

    def detectAndCompute(self, img, op=None):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def compute(self, img, cv_kps):
        # make sure image is gray
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # patches ["B", "1", "32", "32"]
        patches = []
        for kp in cv_kps:
            patch = cv2.getRectSubPix(img, (32, 32), kp.pt)
            patches.append(patch)

        patches = np.stack(patches, axis=0)
        patches = torch.from_numpy(patches).to(self.DEV).unsqueeze(1).float() / 255.0
        desc = self.model(patches)
        desc = desc.squeeze().detach().cpu().numpy()
        return cv_kps, desc



if __name__ == "__main__":
    img1 = cv2.imread(str(root / "assets" / "notredame.png"))
    img2 = cv2.imread(str(root / "assets" / "notredame2.jpeg"))
    # img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    # img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    sift = cv2.SIFT_create()

    tfeat = TFeat_baseline(device=0)

    kps1 = sift.detect(img1, None)
    kps2 = sift.detect(img2, None)

    _, desc1 = tfeat.compute(img1, kps1)
    _, desc2 = tfeat.compute(img2, kps2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)

    # ransac
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matches = [m for m,msk in zip(matches, mask) if msk == 1]

    img3 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)

    cv2.imwrite("tfeat.png", img3)

