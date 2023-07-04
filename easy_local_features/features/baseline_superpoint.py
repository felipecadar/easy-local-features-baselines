import sys, os
from easy_local_features.submodules.git_superglue.models.superpoint import SuperPoint

import torch
import numpy as np
from functools import partial
import cv2
import wget
import pyrootutils
root = pyrootutils.find_root()

class SuperPoint_baseline():
    def __init__(self, max_keypoints=2048, nms_radius=4, keypoint_threshold=0.005, remove_borders=4, descriptor_dim=256, device=-1, model_path=None):
        self.DEV   = torch.device('cuda' if (torch.cuda.is_available() and device>=0) else 'cpu')
        self.CPU   = torch.device('cpu')

        if model_path is None:
            url = 'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth'
            cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'SuperPoint')
            model_name = url.split('/')[-1]
            
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)

            cache_path = os.path.join(cache_path, model_name)
            if not os.path.exists(cache_path):
                print(f'Downloading SuperPoint model...')
                wget.download(url, cache_path)
                print('Done.')

            model_path = cache_path

        config = {
            'descriptor_dim': descriptor_dim,
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints,
            'remove_borders': remove_borders,
            'model_path': model_path,
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
    img = cv2.imread(str(root / "assets" / "notredame.png"))
    extractor = SuperPoint_baseline()

    keypoints, descriptors = extractor.detectAndCompute(img)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))
    cv2.imshow('superpoint', output_image)
    cv2.waitKey(0)