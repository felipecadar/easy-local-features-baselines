from easy_local_features.submodules.git_disk.disk import DISK

import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import cv2, os, wget

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils.download import downloadModel
from ..utils import ops


class Image:
    def __init__(self, bitmap, orig_shape=None):
        self.bitmap     = bitmap
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    def resize_to(self, shape):
        return Image(
            self._pad(self._interpolate(self.bitmap, shape), shape),
            orig_shape=self.bitmap.shape[1:],
        )

    def to_image_coord(self, xys):
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        scaled = xys / f

        h, w = self.orig_shape
        x, y = scaled

        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)

        return scaled, mask

    def _compute_interpolation_size(self, shape):
        x_factor = self.orig_shape[0] / shape[0]
        y_factor = self.orig_shape[1] / shape[1]

        f = 1 / max(x_factor, y_factor)

        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        return f, new_size

    def _interpolate(self, image, shape):
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)
    
    def _pad(self, image, shape):
        x_pad = shape[0] - image.shape[1]
        y_pad = shape[1] - image.shape[2]

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        return F.pad(image, (0, y_pad, 0, x_pad))


class DISK_baseline(BaseExtractor):
    default_conf = {
        'window': 8,
        'desc_dim': 128,
        'mode': 'rng',
        'top_k': 2048,
        'auto_resize': True,
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        window = conf.window
        desc_dim = conf.desc_dim
        mode = conf.mode
        max_feat = conf.top_k
        auto_resize = conf.auto_resize
        
        self.DEV   = torch.device('cpu')
        self.auto_resize = auto_resize
        self.ratio = 0

        self.model = DISK(window=window, desc_dim=desc_dim)

        url = 'https://github.com/cvlab-epfl/disk/raw/master/depth-save.pth'
        model_path = downloadModel('disk', 'depth-save', url)

        state_dict = torch.load(model_path, map_location=self.DEV)

        # print(state_dict.keys())
        self.model.load_state_dict(state_dict['extractor'])
        self.model = self.model.to(self.DEV)
        self.model.eval()

        if mode == 'nms':
            self.extract = partial(
                self.model.features,
                kind='nms',
                window_size=5,
                cutoff=0.,
                n=max_feat
            )
        else:
            self.extract = partial(self.model.features, kind='rng')


        self.matcher = NearestNeighborMatcher()

    def _toImage(self, img):
        bitmap = ops.prepareImage(img, batch=False).to(self.DEV)
        img_shape = bitmap.shape[1:]
        image = Image(bitmap)

        if self.auto_resize:
            new_shape = [0,0]

            if (img_shape[0] % 16) != 0 or (img_shape[1] % 16) != 0:
                new_shape[0] = (img_shape[0] // 16) * 16
                new_shape[1] = (img_shape[1] // 16) * 16

                image = image.resize_to(new_shape)

        return image

    def detectAndCompute(self, img, return_dict=False):
        image = self._toImage(img)
        with torch.no_grad():
            try:
                features = self.extract(image.bitmap.unsqueeze(0)).flat[0] #batch
            except RuntimeError as e:
                if 'U-Net failed' in str(e):
                    msg = ('Please use input size which is multiple of 16.'
                           'This is because we internally use a U-Net with 4'
                           'downsampling steps, each by a factor of 2'
                           'therefore 2^4=16.')
                    raise RuntimeError(msg) from e
                else:
                    raise

        kps_crop_space = features.kp.T
        kps_img_space, mask = image.to_image_coord(kps_crop_space)

        keypoints   = kps_img_space.T[mask]
        descriptors = features.desc[mask]
        scores      = features.kp_logp[mask]

        # order = np.argsort(scores)[::-1]
        order = torch.argsort(scores, descending=True)

        keypoints   = keypoints[order].unsqueeze(0)
        descriptors = descriptors[order].unsqueeze(0)
        scores      = scores[order].unsqueeze(0)
        
        if return_dict:
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores
            }

        return keypoints, descriptors
    
    def detect(self, img):
        return self.detectAndCompute(img)[0]

    def compute(self, image, keypoints):
        raise NotImplemented
    
    def to(self, device):
        self.model.to(device)
        self.DEV = device

    def match(self, image1, image2):
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        
        data = {
            "descriptors0": desc0,
            "descriptors1": desc1,
        }
        
        response = self.matcher(data)
        
        m0 = response['matches0'][0]
        valid = m0 > -1
        
        mkpts0 = kp0[0][valid]
        mkpts1 = kp1[0][m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }
        
    @property
    def has_detector(self):
        return True