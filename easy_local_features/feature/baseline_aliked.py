import sys, os
from easy_local_features.submodules.git_aliked.nets.aliked import ALIKED

import torch
import numpy as np
from functools import partial
import cv2
import wget
import pyrootutils
root = pyrootutils.find_root()


models = {
    "aliked-n16": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
    "aliked-n16rot": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16rot.pth",
    "aliked-n32": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n32.pth",
    "aliked-t16": "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-t16.pt"
}

class ALIKED_baseline():
    """ALIKED baseline implementation.
        model_name: str = 'aliked-n32', Choose from ['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32']
        top_k: int = -1, # -1 for threshold based mode, >0 for top K mode.
        scores_th: float = 0.2, # Threshold for top K = -1 mode
        n_limit: int = 5000, # Maximum number of keypoints to be detected
        load_pretrained: bool = True, load pretrained model or not
        device=-1, -1 for CPU, >=0 for GPU
        model_path=None, use custom model path instead of the default one. 
    """
    def __init__(self, 
                model_name: str = 'aliked-n32',
                top_k: int = -1, # -1 for threshold based mode, >0 for top K mode.
                scores_th: float = 0.2,
                n_limit: int = 5000, # Maximum number of keypoints to be detected
                load_pretrained: bool = True,
                device=-1, model_path=None):

        self.DEV   = torch.device('cuda' if (torch.cuda.is_available() and device>=0) else 'cpu')
        self.CPU   = torch.device('cpu')

        if model_path is None:
            if model_name not in models:
                raise ValueError(f"Model name {model_name} not found in {models.keys()}")
            url = models[model_name]
            cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'ALIKED')
            file_model_name = url.split('/')[-1]
            
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)

            cache_path = os.path.join(cache_path, file_model_name)
            if not os.path.exists(cache_path):
                print(f'Downloading ALIKED model...')
                wget.download(url, cache_path)
                print('Done.')

            model_path = cache_path

        self.model = ALIKED(
            model_name=model_name,
            device=self.DEV,
            top_k=top_k,
            scores_th=scores_th,
            n_limit=n_limit,
            load_pretrained=load_pretrained,
            pretrained_path=model_path,
        )
        self.model = self.model.to(self.DEV)
        self.model.eval()

    def _toTorch(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detectAndCompute(self, img, op=None):
        image = self._toTorch(img)
        with torch.no_grad():
            res = self.model.run(image)

        keypoints = res['keypoints']
        descriptors = res['descriptors']
        scores = res['scores']

        cv_kps = [cv2.KeyPoint(kp[0], kp[1], 1, -1, s, 0, -1) for kp, s in zip(keypoints, scores)]

        return cv_kps, descriptors

    def detect(self, img, op=None):
        cv_kps, descriptors = self.detectAndCompute(img)
        return cv_kps

    def compute(self, img, cv_kps):
        raise NotImplemented

if __name__ == "__main__":
    img = cv2.imread(str(root / "assets" / "notredame.png"))
    extractor = ALIKED_baseline()

    keypoints, descriptors = extractor.detectAndCompute(img)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))
    cv2.imshow('superpoint', output_image)
    cv2.waitKey(0)