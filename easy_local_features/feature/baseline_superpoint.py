import sys, os
from easy_local_features.submodules.git_superglue.models.superpoint import *

import torch
import numpy as np
from functools import partial
import cv2
import wget
import pyrootutils
root = pyrootutils.find_root()

def dense_forward(self, data):
    """ Compute keypoints, scores, descriptors for image """
    # Shared Encoder
    x = self.relu(self.conv1a(data['image']))
    x = self.relu(self.conv1b(x))
    x = self.pool(x)
    x = self.relu(self.conv2a(x))
    x = self.relu(self.conv2b(x))
    x = self.pool(x)
    x = self.relu(self.conv3a(x))
    x = self.relu(self.conv3b(x))
    x = self.pool(x)
    x = self.relu(self.conv4a(x))
    x = self.relu(self.conv4b(x))

    # Compute the dense keypoint scores
    cPa = self.relu(self.convPa(x))
    scores = self.convPb(cPa)
    scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
    b, _, h, w = scores.shape
    scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
    scores = simple_nms(scores, self.config['nms_radius'])

    # Extract keypoints
    keypoints = [
        torch.nonzero(s > self.config['keypoint_threshold'])
        for s in scores]
    scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

    # Discard keypoints near the image borders
    keypoints, scores = list(zip(*[
        remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
        for k, s in zip(keypoints, scores)]))

    # Keep the k keypoints with highest score
    if self.config['max_keypoints'] >= 0:
        keypoints, scores = list(zip(*[
            top_k_keypoints(k, s, self.config['max_keypoints'])
            for k, s in zip(keypoints, scores)]))

    # Convert (h, w) to (x, y)
    keypoints = [torch.flip(k, [1]).float() for k in keypoints]

    # Compute the dense descriptors
    cDa = self.relu(self.convDa(x))
    descriptors = self.convDb(cDa)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    dense_descriptors = torch.clone(descriptors)

    # Extract descriptors
    descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                    for k, d in zip(keypoints, descriptors)]

    return {
        'keypoints': keypoints,
        'scores': scores,
        'descriptors': descriptors,
        'dense_descriptors': dense_descriptors,
    }


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
        self.model.forward = dense_forward.__get__(self.model, SuperPoint)
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
        image = self._toTorch(img)
        with torch.no_grad():
            res = self.model({'image': image})

        # convert cv_kps to torch_kps
        torch_kps = torch.from_numpy(np.array([[kp.pt[0], kp.pt[1]] for kp in cv_kps])).float().to(self.DEV)
        descriptors = res['dense_descriptors']
        descriptors = sample_descriptors(torch_kps, descriptors, s=8)[0]
        descriptors = descriptors.T.to(self.CPU).numpy()

        return cv_kps, descriptors

if __name__ == "__main__":
    img = cv2.imread(str(root / "assets" / "notredame.png"))
    extractor = SuperPoint_baseline()

    keypoints = extractor.detect(img)
    descriptors = extractor.compute(img, keypoints)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))
    cv2.imshow('superpoint', output_image)
    cv2.waitKey(0)

