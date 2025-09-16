import warnings
from typing import TypedDict

import torch

from easy_local_features.submodules import git_dad as dad
from easy_local_features.submodules.git_dad.utils import get_best_device

from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.utils.ops import prepareImage

class DadConfig(TypedDict):
    num_keypoints: int
    resize: int
    nms_size: int


class DAD_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    default_conf = DadConfig(
        num_keypoints=1024,
        resize=1024,
        nms_size=3,
    )

    def __init__(self, conf={}):
        self.num_keypoints = conf.get("num_keypoints", self.default_conf["num_keypoints"])
        self.detector = dad.load_DaD()
        self.DEV = get_best_device()
        self.matcher = NearestNeighborMatcher()

    def detect(self, image, return_dict=None):
        img = prepareImage(image, imagenet=True, batch=True)
        img = image.to(self.DEV)
        

        mkpts = self.detector.detect({"image": img}, num_keypoints=self.num_keypoints)["keypoints"]
        mkpts = self.detector.to_pixel_coords(mkpts, img.shape[-2], img.shape[-1])

        if return_dict:
            return {"mkpts": mkpts}
        return mkpts

    def detectAndCompute(self, image, return_dict=None):
        raise NotImplementedError

    def compute(self, image, keypoints):
        raise NotImplementedError

    @property
    def has_detector(self):
        return True

    def to(self, device):
        self.detector.to(device)
        self.DEV = device


if __name__ == "__main__":
    from easy_local_features.utils import io, vis

    detector = DAD_baseline({"num_keypoints": 512})

    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")

    kps0 = detector.detect(img0)
    kps1 = detector.detect(img1)

    vis.plot_pair(img0, img1)
    vis.plot_keypoints(keypoints0=kps0.cpu(), keypoints1=kps1.cpu())
    vis.show()
