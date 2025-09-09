import torch
from einops import rearrange
from kornia.feature.tfeat import TFeat
from omegaconf import OmegaConf

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from .basemodel import BaseExtractor, MethodType
from typing import TypedDict


class TFeatConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    desc_dim: int


class TFeat_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    default_conf: TFeatConfig = {
        "model_name": "tfeat",
        "top_k": 2048,
        "detection_threshold": 0.2,
        "nms_radius": 4,
        "desc_dim": 128,
    }

    def __init__(self, conf: TFeatConfig = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.DEV = torch.device("cpu")
        self.model = TFeat().to(self.DEV)
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, op=None):
        raise NotImplementedError

    def detect(self, img, op=None):
        raise NotImplementedError

    def compute(self, img, keypoints):
        # make sure image is gray
        image = ops.prepareImage(img, gray=True)

        do_squeeze = False
        if len(keypoints.shape) == 2:
            keypoints = keypoints.unsqueeze(0)
            do_squeeze = True

        patches = ops.crop_patches(image, keypoints, 32)
        B, N, one, PS1, PS2 = patches.shape
        patches = rearrange(patches, "B N one PS1 PS2 -> (B N) one PS1 PS2", B=B)

        # # patches ["B", "1", "32", "32"]
        desc = self.model(patches)

        desc = rearrange(desc, "(B N) D -> B N D", B=B, N=N, D=128)

        return keypoints, desc

    def to(self, device):
        self.model.to(device)
        self.DEV = device

    @property
    def has_detector(self):
        return False
