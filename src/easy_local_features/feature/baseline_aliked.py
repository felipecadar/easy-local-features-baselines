import numpy as np
import torch
from omegaconf import OmegaConf

from easy_local_features.submodules.git_aliked.aliked import ALIKED

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from .basemodel import BaseExtractor, MethodType
from typing import TypedDict


class ALIKEDConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    force_num_keypoints: bool
    nms_radius: int


class ALIKED_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    """ALIKED baseline implementation.
        model_name: str = 'aliked-n32', Choose from ['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32']
        top_k: int = -1, # -1 for threshold based mode, >0 for top K mode.
        scores_th: float = 0.2, # Threshold for top K = -1 mode
    """
    default_conf: ALIKEDConfig = {
        "model_name": "aliked-n16",
        "top_k": -1,
        "detection_threshold": 0.2,
        "force_num_keypoints": False,
        "nms_radius": 2,
    }

    def __init__(self, conf: ALIKEDConfig = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

        self.DEV = torch.device("cpu")

        self.model = ALIKED(
            {
                "model_name": conf.model_name,
                "max_num_keypoints": conf.top_k,
                "detection_threshold": conf.detection_threshold,
                "force_num_keypoints": conf.force_num_keypoints,
                "pretrained": True,
                "nms_radius": conf.nms_radius,
            }
        )

        self.model = self.model.to(self.DEV)
        self.model.eval()

        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        image = ops.prepareImage(img, batch=True).to(self.DEV)
        with torch.no_grad():
            res = self.model({"image": image})

        keypoints = res["keypoints"]
        descriptors = res["descriptors"]

        if return_dict:
            return res

        return keypoints, descriptors

    def detect(self, img):
        keypoints, descriptors = self.detectAndCompute(img)
        return keypoints

    def compute(self, img, keypoints):
        if isinstance(keypoints, np.ndarray) or isinstance(keypoints, list):
            keypoints = torch.tensor(keypoints)

        keypoints = keypoints.to(self.DEV)

        if len(keypoints) == 0:
            return torch.zeros(0, 128).to(self.DEV)

        if len(keypoints.shape) == 2:
            keypoints = keypoints.unsqueeze(0)

        image = ops.prepareImage(img, batch=True).to(self.DEV)
        with torch.no_grad():
            res = self.model.forward_desc({"image": image}, keypoints)

        return res["descriptors"]

    def to(self, device):
        self.model.to(device)
        self.DEV = device

    @property
    def has_detector(self):
        return True


if __name__ == "__main__":
    from easy_local_features.utils import io, ops

    method = ALIKED_baseline(
        {
            "model_name": "aliked-n16",
            "top_k": 128,
            "detection_threshold": 0.2,
            "force_num_keypoints": False,
            "nms_radius": 2,
        }
    )

    img0 = io.fromPath("test/assets/megadepth0.jpg")

    kpts = method.detect(img0)
    desc = method.compute(img0, kpts)

    kpts2, desc2 = method.detectAndCompute(img0)

    assert torch.allclose(kpts, kpts2)

    # import pdb; pdb.set_trace()
    print(desc)
    print(desc2)
    assert torch.allclose(desc, desc2, atol=1e-5)
