logpolar_weight = 'https://github.com/cvlab-epfl/log-polar-descriptors/raw/master/weights/log-polar.pth'
cartesian_weight = 'https://github.com/cvlab-epfl/log-polar-descriptors/raw/master/weights/cartesian.pth'


import pyrootutils
ROOT = pyrootutils.find_root()

import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import cv2, os, wget

from easy_local_features.submodules.git_logpolar.models import HardNet
from easy_local_features.submodules.git_logpolar.defaults import _C as cfg

class LogPolar_baseline():
    def __init__(self, use_log_polar=True, device=-1):
        self.CPU   = torch.device('cpu')
        self.DEV   = torch.device(f'cuda:{device}' if (torch.cuda.is_available() and device >= 0) else 'cpu')
        this_file = os.path.abspath(__file__)
        this_dir = os.path.dirname(this_file)

        if use_log_polar:
            config_path = os.path.join(this_dir, '../submodules/git_logpolar/init_one_example_ptn_96.yml')
            weights = logpolar_weight
        else:
            config_path = os.path.join(this_dir, '../submodules/git_logpolar/init_one_example_stn_16.yml')
            weights = cartesian_weight

        cfg.merge_from_file(config_path)

        self.model = HardNet(transform=cfg.TEST.TRANSFORMER,
                    coords=cfg.TEST.COORDS,
                    patch_size=cfg.TEST.IMAGE_SIZE,
                    scale=cfg.TEST.SCALE,
                    is_desc256=cfg.TEST.IS_DESC_256,
                    orientCorrect=cfg.TEST.ORIENT_CORRECTION)

        state_dict = torch.hub.load_state_dict_from_url(weights, progress=True)['state_dict']
        self.model.load_state_dict(state_dict)
        self.model.to(self.DEV)

    def detectAndCompute(self, img, op=None):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def compute(self, img, cv_kps):
        # make sure image is gray
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pts = np.array([kp.pt for kp in cv_kps])
        scales = np.array([kp.size for kp in cv_kps])
        oris = np.array([kp.angle for kp in cv_kps])

        # Mirror-pad the image to avoid boundary effects
        if any([s > cfg.TEST.PAD_TO for s in img.shape[:2]]):
            raise RuntimeError(
                "Image exceeds acceptable size ({}x{}), please downsample".format(
                    cfg.TEST.PAD_TO, cfg.TEST.PAD_TO))

        fillHeight = cfg.TEST.PAD_TO - img.shape[0]
        fillWidth = cfg.TEST.PAD_TO - img.shape[1]

        padLeft = int(np.round(fillWidth / 2))
        padRight = int(fillWidth - padLeft)
        padUp = int(np.round(fillHeight / 2))
        padDown = int(fillHeight - padUp)

        img = np.pad(img,
                    pad_width=((padUp, padDown), (padLeft, padRight)),
                    mode='reflect')

        # Normalize keypoint locations
        kp_norm = []
        for i, p in enumerate(pts):
            _p = 2 * np.array([(p[0] + padLeft) / (cfg.TEST.PAD_TO),
                            (p[1] + padUp) / (cfg.TEST.PAD_TO)]) - 1
            kp_norm.append(_p)

        theta = [
            torch.from_numpy(np.array(kp_norm)).float().squeeze(),
            torch.from_numpy(scales).float(),
            torch.from_numpy(np.array([np.deg2rad(o) for o in oris])).float()
        ]


        # Extract descriptors
        imgs, img_keypoints = torch.from_numpy(img).unsqueeze(0).to(self.DEV), \
            [theta[0].to(self.DEV), theta[1].to(self.DEV), theta[2].to(self.DEV)]

        # import pdb; pdb.set_trace()   
        input_filename = 'test'
        descriptors, patches = self.model({input_filename: imgs}, img_keypoints,
                                 [input_filename] * len(img_keypoints[0]))

        descriptors = descriptors.squeeze().detach().cpu().numpy()
        return cv_kps, descriptors


if __name__ == "__main__":
    img1 = cv2.imread(str(ROOT / "assets" / "notredame.png"))
    img2 = cv2.imread(str(ROOT / "assets" / "notredame2.jpeg"))
    # img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    # img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    sift = cv2.SIFT_create()

    tfeat = LogPolar_baseline(device=0)

    kps1 = sift.detect(img1, None)
    kps2 = sift.detect(img2, None)

    desc1 = tfeat.compute(img1, kps1)
    desc2 = tfeat.compute(img2, kps2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)

    # ransac
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matches = [m for m,msk in zip(matches, mask) if msk == 1]

    img3 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)

    cv2.imwrite("logpolar.png", img3)

