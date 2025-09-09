import torch
import cv2

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor, MethodType
from ..utils.download import getCache
from ..utils import ops
from typing import TypedDict


class DEALConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    force_num_keypoints: bool


class DEAL_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    default_conf: DEALConfig = {
        "model_name": "deal",
        "top_k": 2048,
        "detection_threshold": 0.2,
        "nms_radius": 4,
        "force_num_keypoints": False,
    }

    def __init__(self, conf: DEALConfig = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

        self.max_kps = conf.top_k
        self.DEV = torch.device("cpu")
        cache = getCache("DEAL")
        self.sift = cv2.SIFT_create(nfeatures=self.max_kps, contrastThreshold=0.04, edgeThreshold=10)
        self.deal = torch.hub.load("verlab/DEAL_NeurIPS_2021", "DEAL", True, cache)
        self.deal.device = self.DEV
        self.deal.net.eval()

        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        gray = ops.to_cv(ops.prepareImage(img), to_gray=True)

        with torch.no_grad():
            kps = self.detect(gray)
            kps, desc = self.compute(gray, kps)

        kps = torch.tensor([kp.pt for kp in kps]).to(self.DEV).unsqueeze(0)
        desc = desc.to(self.DEV)

        if return_dict:
            return {"keypoints": kps, "descriptors": desc}

        return kps, desc

    def detect(self, img):
        gray = ops.to_cv(ops.prepareImage(img), to_gray=True)

        with torch.no_grad():
            kps = self.sift.detect(gray, None)

        return kps

    def compute(self, img, kps):
        gray = ops.to_cv(ops.prepareImage(img), to_gray=True)

        if not isinstance(kps[0], cv2.KeyPoint):
            kps = [cv2.KeyPoint(kp[0], kp[1], 0) for kp in kps]

        with torch.no_grad():
            desc = self.deal.compute(gray, kps)

        desc = torch.from_numpy(desc).to(self.DEV).unsqueeze(0)

        return kps, desc

    def to(self, device):
        self.deal.device = device
        self.DEV = device

    @property
    def has_detector(self):
        return True
