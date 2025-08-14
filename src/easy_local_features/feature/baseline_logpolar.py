# logpolar_weight = 'https://github.com/cvlab-epfl/log-polar-descriptors/raw/master/weights/log-polar.pth'
# cartesian_weight = 'https://github.com/cvlab-epfl/log-polar-descriptors/raw/master/weights/cartesian.pth'


# import torch
# import numpy as np
# import torch.nn.functional as F
# from functools import partial
# import cv2, os, wget

# from easy_local_features.submodules.git_logpolar.models import HardNet
# from easy_local_features.submodules.git_logpolar.defaults import _C as cfg

# from ..matching.nearest_neighbor import NearestNeighborMatcher
# from omegaconf import OmegaConf
# from .basemodel import BaseExtractor
# from ..utils.download import downloadModel
# from ..utils import ops

# class LogPolar_baseline(BaseExtractor):
#     default_conf = {
#         'use_log_polar': True,
#     }
    
#     def __init__(self, conf={}):
#         self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        
#         self.DEV   = torch.device('cpu')
#         use_log_polar = conf.use_log_polar
#         this_file = os.path.abspath(__file__)
#         this_dir = os.path.dirname(this_file)

#         if use_log_polar:
#             config_path = os.path.join(this_dir, '../submodules/git_logpolar/init_one_example_ptn_96.yml')
#             weights = logpolar_weight
#         else:
#             config_path = os.path.join(this_dir, '../submodules/git_logpolar/init_one_example_stn_16.yml')
#             weights = cartesian_weight

#         cfg.merge_from_file(config_path)

#         self.model = HardNet(transform=cfg.TEST.TRANSFORMER,
#                     coords=cfg.TEST.COORDS,
#                     patch_size=cfg.TEST.IMAGE_SIZE,
#                     scale=cfg.TEST.SCALE,
#                     is_desc256=cfg.TEST.IS_DESC_256,
#                     orientCorrect=cfg.TEST.ORIENT_CORRECTION)

#         state_dict = torch.hub.load_state_dict_from_url(weights, progress=True)['state_dict']
#         self.model.load_state_dict(state_dict)
#         self.model.to(self.DEV)

#     def detectAndCompute(self, img, return_dict=False):
#         raise NotImplemented

#     def detect(self, img):
#         raise NotImplemented

#     def compute(self, img, kps, scales, orientations, return_dict=False):
#         # make sure image is gray
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         pts = np.array([kp.pt for kp in cv_kps])
#         scales = np.array([kp.size for kp in cv_kps])
#         oris = np.array([kp.angle for kp in cv_kps])

#         # Mirror-pad the image to avoid boundary effects
#         if any([s > cfg.TEST.PAD_TO for s in img.shape[:2]]):
#             raise RuntimeError(
#                 "Image exceeds acceptable size ({}x{}), please downsample".format(
#                     cfg.TEST.PAD_TO, cfg.TEST.PAD_TO))

#         fillHeight = cfg.TEST.PAD_TO - img.shape[0]
#         fillWidth = cfg.TEST.PAD_TO - img.shape[1]

#         padLeft = int(np.round(fillWidth / 2))
#         padRight = int(fillWidth - padLeft)
#         padUp = int(np.round(fillHeight / 2))
#         padDown = int(fillHeight - padUp)

#         img = np.pad(img,
#                     pad_width=((padUp, padDown), (padLeft, padRight)),
#                     mode='reflect')

#         # Normalize keypoint locations
#         kp_norm = []
#         for i, p in enumerate(pts):
#             _p = 2 * np.array([(p[0] + padLeft) / (cfg.TEST.PAD_TO),
#                             (p[1] + padUp) / (cfg.TEST.PAD_TO)]) - 1
#             kp_norm.append(_p)

#         theta = [
#             torch.from_numpy(np.array(kp_norm)).float().squeeze(),
#             torch.from_numpy(scales).float(),
#             torch.from_numpy(np.array([np.deg2rad(o) for o in oris])).float()
#         ]


#         # Extract descriptors
#         imgs, img_keypoints = torch.from_numpy(img).unsqueeze(0).to(self.DEV), \
#             [theta[0].to(self.DEV), theta[1].to(self.DEV), theta[2].to(self.DEV)]

#         # import pdb; pdb.set_trace()   
#         input_filename = 'test'
#         descriptors, patches = self.model({input_filename: imgs}, img_keypoints,
#                                  [input_filename] * len(img_keypoints[0]))

#         descriptors = descriptors.squeeze().detach().cpu().numpy()
#         return cv_kps, descriptors


#     def detect(self, img):
#         return self.detectAndCompute(img)[0]

#     def compute(self, image, keypoints):
#         raise NotImplemented

#     def to(self, device):
#         self.model.to(device)
#         self.DEV = device

#     def match(self, image1, image2):
#         kp0, desc0 = self.detectAndCompute(image1)
#         kp1, desc1 = self.detectAndCompute(image2)
        
#         data = {
#             "descriptors0": desc0.unsqueeze(0),
#             "descriptors1": desc1.unsqueeze(0),
#         }
        
#         response = self.matcher(data)
        
#         m0 = response['matches0'][0]
#         valid = m0 > -1
        
#         mkpts0 = kp0[valid]
#         mkpts1 = kp1[m0[valid]]
        
#         return {
#             'mkpts0': mkpts0,
#             'mkpts1': mkpts1,
#         }
        
#     @property
#     def has_detector(self):
#         return True
