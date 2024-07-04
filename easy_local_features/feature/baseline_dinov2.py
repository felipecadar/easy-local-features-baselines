import torch
import torch.nn.functional as F

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils import ops

class DINOv2_baseline(BaseExtractor):
    available_weights = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14',
        'dinov2_vits14_reg',
        'dinov2_vitb14_reg',
        'dinov2_vitl14_reg',
        'dinov2_vitg14_reg',
    ]
        
    default_conf = {"weights": "dinov2_vits14", "allow_resize": True}
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device   = torch.device('cpu')
        self.matcher = NearestNeighborMatcher()
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", conf.weights)
        except:
            self.model = torch.hub.load("facebookresearch/dinov2", conf.weights, force_reload=True)

    def sample_features(self, keypoints, features, s=14, mode="bilinear"):
        if s is None:
            s = self.vit_size
        b, c, h, w = features.shape
        keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        features = torch.nn.functional.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
        )
        features = torch.nn.functional.normalize(
            features.reshape(b, c, -1), p=2, dim=1
        )
        
        features = features.permute(0, 2, 1)
        return features

    def detectAndCompute(self, img, return_dict=None):
        raise NotImplemented
    
    def detect(self, img, op=None):
        raise NotImplemented

    def compute(self, img, keypoints=None, return_dict=False):
        img = ops.prepareImage(img).to(self.device)

        if self.conf.allow_resize:
            img = F.interpolate(img, [int(x // 14 * 14) for x in img.shape[-2:]])

        desc, cls_token = self.model.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]
        
        if keypoints is not None:
            descriptors = self.sample_features(keypoints, desc)
        if return_dict:
            return {
                "descriptors": descriptors,
                "keypoints": keypoints,
                "global_descriptor": cls_token,
                "feature_map": desc,
            }
        return keypoints, descriptors

    def to(self, device):
        self.model.to(device)
        self.device = device

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
        
        mkpts0 = kp0[0, valid]
        mkpts1 = kp1[0, m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }
        
    @property
    def has_detector(self):
        return False