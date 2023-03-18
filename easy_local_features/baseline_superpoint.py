import sys, os
sp_folder = os.path.dirname(os.path.realpath(__file__)) + '/submodules/git_superglue/'
sys.path.insert(0, sp_folder)
from models.superpoint import SuperPoint

import torch
import numpy as np
from functools import partial
import cv2

class SuperPoint_baseline():
    def __init__(self, max_keypoints=2048, nms_radius=4, keypoint_threshold=0.005, remove_borders=4, descriptor_dim=256):
        self.DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CPU   = torch.device('cpu')
    
        config = {
            'descriptor_dim': descriptor_dim,
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints,
            'remove_borders': remove_borders,
        }

        self.model = SuperPoint(config)
        self.model = self.model.to(self.DEV)
        self.model.eval()

    def _toTorch(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(img/255.).float()[None, None].to(self.DEV)

    def detectAndCompute(self, img, op=None):
        image = self._toTorch(img)
        with torch.no_grad():
            res = self.model({'image': image})

        keypoints   = res['keypoints'][0].to(self.CPU).numpy()
        descriptors = res['descriptors'][0].to(self.CPU).numpy().T
        scores      = res['scores'][0].to(self.CPU).numpy()

        cv_kps = [cv2.KeyPoint(kp[0], kp[1], 1, -1, s, 0, -1) for kp, s in zip(keypoints, scores)]

        return cv_kps, descriptors

    def detect(self, img, op=None):
        cv_kps, descriptors = self.detectAndCompute(img)
        return cv_kps

    def compute(self, img, cv_kps):
        raise NotImplemented

if __name__ == "__main__":
    import pdb
    img = cv2.imread("../assets/notredame.png")
    extractor = SuperPoint_baseline()

    keypoints, descriptors = extractor.detectAndCompute(img)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))

    cv2.imwrite("sp_test.png", output_image)