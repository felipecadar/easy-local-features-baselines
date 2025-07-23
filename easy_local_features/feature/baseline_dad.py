from easy_local_features.submodules.git_dad.dad import dad
from easy_local_features.submodules.git_dad.dad.dad.utils import get_best_device

from .basemodel import BaseExtractor
from ..matching.nearest_neighbor import NearestNeighborMatcher


def dad_extract(img0, img1):
    H, W = img0.shape[-2:]

    img0 = img0.to("mps")
    img1 = img1.to("mps")
    mkpts0 = detector.detect({"image": img0}, num_keypoints=512)["keypoints"]
    mkpts1 = detector.detect({"image": img1}, num_keypoints=512)["keypoints"]

    mkpts0 = detector.to_pixel_coords(mkpts0, H, W)
    mkpts1 = detector.to_pixel_coords(mkpts1, H, W)

    # vis.plot_pair(img0=img0, img1=img1)
    # vis.plot_keypoints(keypoints0=mkpts0.cpu(), keypoints1=mkpts1.cpu())
    # vis.show()

    return {
        "mkpts0": mkpts0[0],
        "mkpts1": mkpts1[0],
    }


class DAD_baseline(BaseExtractor):
    default_conf = {"num_keypoints": 512}

    def __init__(self, conf={}):
        self.num_keypoints = conf.get("num_keypoints", self.default_conf["num_keypoints"])
        self.detector = dad.load_DaD()
        self.DEV = get_best_device()
        self.matcher = NearestNeighborMatcher()

    def detect(self, img, return_dict=None):
        img = img.to(self.DEV)

        mkpts = self.detector.detect({"image": img}, num_keypoints=self.num_keypoints)["keypoints"]
        mkpts = self.detector.to_pixel_coords(mkpts, img.shape[-2], img.shape[-1])

        if return_dict:
            return {"mkpts": mkpts[0]}
        return mkpts[0]

    def detectAndCompute(self, img, return_dict=None):
        raise NotImplementedError("detectAndCompute is not implemented for DAD_baseline")

    def compute(self, image, keypoints):
        raise NotImplementedError("compute is not implemented for DAD_baseline")

    @property
    def has_detector(self):
        return True

    def to(self, device):
        self.detector.to(device)
        self.DEV = device


if __name__ == "__main__":
    from easy_local_features.utils import io, vis

    detector = DAD_baseline({"num_keypoints": 512})

    img0 = io.fromPath("test/assets/megadepth0.jpg")
    img1 = io.fromPath("test/assets/megadepth1.jpg")

    kps0 = detector.detect(img0)
    kps1 = detector.detect(img1)

    vis.plot_pair(img0, img1)
    vis.plot_keypoints(keypoints0=kps0.cpu(), keypoints1=kps1.cpu())
    vis.show()
