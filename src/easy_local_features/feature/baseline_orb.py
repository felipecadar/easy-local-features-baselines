import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from .basemodel import BaseExtractor, MethodType
from typing import TypedDict


class ORBConfig(TypedDict):
    top_k: int
    scaleFactor: float
    nlevels: int
    edgeThreshold: int
    firstLevel: int
    WTA_K: int
    scoreType: int
    patchSize: int
    fastThreshold: int


class ORB_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    default_conf: ORBConfig = {
        "top_k": 2048,
        "scaleFactor": 1.2,
        "nlevels": 8,
        "edgeThreshold": 31,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": cv2.ORB_HARRIS_SCORE,
        "patchSize": 31,
        "fastThreshold": 20,
    }

    def __init__(self, conf: ORBConfig = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.model = cv2.ORB_create(
            nfeatures=conf.top_k,
            scaleFactor=conf.scaleFactor,
            nlevels=conf.nlevels,
            edgeThreshold=conf.edgeThreshold,
            firstLevel=conf.firstLevel,
            WTA_K=conf.WTA_K,
            scoreType=conf.scoreType,
            patchSize=conf.patchSize,
            fastThreshold=conf.fastThreshold,
        )
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        img = ops.prepareImage(img, gray=True, batch=False)
        img = ops.to_cv(img)

        keypoints, descriptors = self.model.detectAndCompute(img, None)
        # Convert to torch tensors with batch dimension
        kps_np = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        desc_np = (
            np.array(descriptors, dtype=np.float32) if descriptors is not None else np.zeros((0, 32), dtype=np.float32)
        )
        kps = torch.from_numpy(kps_np).float().unsqueeze(0)
        desc = torch.from_numpy(desc_np).float().unsqueeze(0)
        pred = {
            "keypoints": kps,
            "descriptors": desc,
        }

        if return_dict:
            return pred
        return pred["keypoints"], pred["descriptors"]

    def detect(self, img):
        img = ops.prepareImage(img, gray=True, batch=False)
        img = ops.to_cv(img)

        keypoints = self.model.detect(img, None)
        kps_np = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        kps = torch.from_numpy(kps_np).float().unsqueeze(0)
        return kps

    def compute(self, img, kps):
        raise NotImplementedError("This method is not implemented in this class")

    def to(self, device):
        return self

    @property
    def has_detector(self):
        return True
