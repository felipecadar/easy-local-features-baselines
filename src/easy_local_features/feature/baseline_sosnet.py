import torch
from einops import rearrange
from kornia.feature.sosnet import SOSNet
from omegaconf import OmegaConf

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from .basemodel import BaseExtractor, MethodType
from typing import TypedDict


class SOSNetConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    desc_dim: int


class SOSNet_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    default_conf: SOSNetConfig = {
        "model_name": "sosnet",
        "top_k": 2048,
        "detection_threshold": 0.2,
        "nms_radius": 4,
        "desc_dim": 128,
    }

    def __init__(self, conf: SOSNetConfig = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.DEV = torch.device("cpu")
        self.model = SOSNet().to(self.DEV)
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        raise NotImplementedError

    def detect(self, img, op=None):
        raise NotImplementedError

    def compute(self, img, keypoints):
        img = ops.prepareImage(img, gray=True)
        do_squeeze = False
        if len(keypoints.shape) == 2:
            keypoints = keypoints.unsqueeze(0)
            do_squeeze = True

        # patches ["B", "1", "32", "32"]
        patches = ops.crop_patches(img, keypoints, patch_size=32)

        B, N, one, PS1, PS2 = patches.shape
        patches = rearrange(patches, "B N one PS1 PS2 -> (B N) one PS1 PS2", B=B)

        desc = self.model(patches)

        desc = rearrange(desc, "(B N) D -> B N D", B=B, N=N, D=128)

        return keypoints, desc

    def to(self, device):
        self.model.to(device)
        self.DEV = device

    @property
    def has_detector(self):
        return False
