import sys, os
disk_folder = os.path.dirname(os.path.realpath(__file__)) + '/submodules/git_disk/'
sys.path.insert(0, disk_folder)
from disk import DISK, Features

import torch, h5py, imageio, os, argparse
import numpy as np
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import cv2

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


class DISK_baseline():
    def __init__(self, window=8, desc_dim=128, mode='nms', max_feat=2048, model_path=disk_folder+'depth-save.pth', auto_resize=True):
        self.DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CPU   = torch.device('cpu')
        self.auto_resize = auto_resize
        self.ratio = 0

        self.model = DISK(window=window, desc_dim=desc_dim)
        state_dict = torch.load(model_path, map_location='cpu')

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


    def _toImage(self, img):
        tensor = torch.from_numpy(img).to(torch.float32)
        if len(tensor.shape) == 2: # some images may be grayscale
            tensor = tensor.unsqueeze(-1).expand(-1, -1, 3)

        bitmap = (tensor.permute(2, 0, 1) / 255.).to(self.DEV)
        image = Image(bitmap)

        if self.auto_resize:
            img_shape = img.shape[:2]
            new_shape = [0,0]

            if (img_shape[0] % 16) != 0 or (img_shape[1] % 16) != 0:
                new_shape[0] = (img_shape[0] // 16) * 16
                new_shape[1] = (img_shape[1] // 16) * 16

                image = image.resize_to(new_shape)

        return image

    def detectAndCompute(self, img, op=None):
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

        features = features.to(self.CPU)

        kps_crop_space = features.kp.T
        kps_img_space, mask = image.to_image_coord(kps_crop_space)

        keypoints   = kps_img_space.numpy().T[mask]
        descriptors = features.desc.numpy()[mask]
        scores      = features.kp_logp.numpy()[mask]

        order = np.argsort(scores)[::-1]

        keypoints   = keypoints[order]
        descriptors = descriptors[order]
        scores      = scores[order]

        cv_kps = [cv2.KeyPoint(kp[0], kp[1], 1, -1, s, 0, -1) for kp, s in zip(keypoints, scores)]

        return cv_kps, descriptors

    def detect(self, img, op=None):
        cv_kps, descriptors = self.detectAndCompute(img)
        return cv_kps

    def compute(self, img, cv_kps):
        raise NotImplemented

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

        features = features.to(self.CPU)
        descriptors = features.desc.numpy()

        # TODO convert KeyPoints to mask

if __name__ == "__main__":
    import pdb
    img = cv2.imread("../assets/notredame.png")
    img = cv2.resize(img, None, fx=0.9, fy=1)

    extractor = DISK_baseline()

    keypoints, descriptors = extractor.detectAndCompute(img)

    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255))

    cv2.imwrite("disk_test.png", output_image)