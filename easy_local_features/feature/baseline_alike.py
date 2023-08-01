import sys, os
from easy_local_features.submodules.git_alike.alike import ALike

import torch
import numpy as np
import cv2
import wget
import pyrootutils
root = pyrootutils.find_root()

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

class ALIKE_baseline():
    """ALIKE baseline implementation.
    model_name: str = 'alike-t' | 'alike-s' | 'alike-n' | 'alike-l'
    top_k: int = -1. Detect top K keypoints. -1 for threshold based mode, >0 for top K mode. (default: -1)
    scores_th: float = 0.2. Detector score threshold (default: 0.2).
    n_limit: int = 5000. Maximum number of keypoints to be detected (default: 5000).
    no_sub_pixel: bool = False. Do not detect sub-pixel keypoints (default: False).
    device: int = -1. Device to run the model on. -1 for CPU, >=0 for GPU. (default: -1)
    """
    def __init__(self, 
                model_name: str = 'alike-t',
                top_k: int = -1,
                scores_th: float = 0.2,
                n_limit: int = 5000,
                no_sub_pixel=False,
                device=-1,
                model_path=None):

        self.DEV   = torch.device('cuda' if (torch.cuda.is_available() and device>=0) else 'cpu')
        self.CPU   = torch.device('cpu')

        if model_name not in models:
            raise ValueError(f"Model name {model_name} not found in {models.keys()}")

        if model_path is None:
            url = models[model_name]
            cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'ALIKE')
            file_model_name = url.split('/')[-1]
            
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)

            cache_path = os.path.join(cache_path, file_model_name)
            if not os.path.exists(cache_path):
                print(f'Downloading ALIKE model...')
                wget.download(url, cache_path)
                print('Done.')

            model_path = cache_path

        config = configs[model_name]
        config['model_path'] = model_path

        self.model = ALike(
            **config,
            device=self.DEV,
            top_k=top_k,
            scores_th=scores_th,
            n_limit=n_limit
        )

        self.no_sub_pixel = no_sub_pixel

    def _toTorch(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detectAndCompute(self, img, op=None):   
        img = self._toTorch(img)
        pred = self.model(img, sub_pixel=not self.no_sub_pixel)
        kpts = pred['keypoints'] # (N, 2)
        desc = pred['descriptors'] # (N, 64)
        scores = pred['scores'] # (N,)

        cv2_kpts = []
        for i in range(kpts.shape[0]):
            cv2_kpts.append(cv2.KeyPoint(kpts[i, 0], kpts[i, 1], 1, response=scores[i] ))

        return cv2_kpts, desc

    def detect(self, img, op=None):
        kps = self.detectAndCompute(img, op)[0]
        return kps

    def compute(self, img, cv_kps):
        raise NotImplemented

if __name__ == "__main__":
    img = cv2.imread(str(root / "assets" / "notredame.png"))
    extractor = ALIKE_baseline()

    keypoints, descriptors = extractor.detectAndCompute(img)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))
    cv2.imshow('alike', output_image)
    cv2.waitKey(0)