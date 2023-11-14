import pyrootutils
root = pyrootutils.find_root()

import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import cv2, os, wget
import scipy

from easy_local_features.submodules.git_d2net.models import D2Net, preprocess_image, process_multiscale

class D2Net_baseline():
    def __init__(self, top_kps = 2048, use_relu=True, device=-1):
        self.CPU   = torch.device('cpu')
        self.DEV   = torch.device(f'cuda:{device}' if (torch.cuda.is_available() and device >= 0) else 'cpu')
        self.model = D2Net( use_relu=use_relu,
                            use_cuda=(torch.cuda.is_available() and device >= 0))
        self.model.eval()
        self.model.to(self.DEV)
        self.top_kps = top_kps

        self.max_edge = 1600
        self.max_sum_edges = 2800
        self.multiscale = False
        self.use_relu = use_relu
        self.preprocessing = 'torch' # 'caffe' or 'torch'

    def compute(self, img, cv_kps):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def detectAndCompute(self, image, op=None):
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        if max(resized_image.shape) > self.max_edge:
            resized_image = scipy.misc.imresize(
                resized_image,
                self.max_edge / max(resized_image.shape)
            ).astype('float')
        if sum(resized_image.shape[: 2]) > self.max_sum_edges:
            resized_image = scipy.misc.imresize(
                resized_image,
                self.max_sum_edges / sum(resized_image.shape[: 2])
            ).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=self.preprocessing
        )

        with torch.no_grad():
            if self.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=self.DEV
                    ),
                    self.model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=self.DEV
                    ),
                    self.model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        keypoints = [cv2.KeyPoint(kp[0], kp[1], kp[2]) for kp in keypoints]

        argsort = np.argsort(-scores)

        keypoints = [keypoints[idx] for idx in argsort[:self.top_kps]]
        scores = scores[argsort[:self.top_kps]]
        descriptors = descriptors[argsort[:self.top_kps]]


        return keypoints, descriptors        



if __name__ == "__main__":
    img1 = cv2.imread(str(root / "assets" / "notredame.png"))
    img2 = cv2.imread(str(root / "assets" / "notredame2.jpeg"))
    # img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    # img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    model = D2Net_baseline(2048, device=0)

    kps1, desc1 = model.detectAndCompute(img1)
    kps2, desc2 = model.detectAndCompute(img2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)

    # ransac
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matches = [m for m,msk in zip(matches, mask) if msk == 1]

    img3 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)

    cv2.imwrite("d2net.png", img3)

