import torch
import numpy as np
import torch.nn.functional as F

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor, MethodType
from ..utils import ops


class XFeat_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    default_conf = {
        "top_k": 2048,
        "detection_threshold": 0.2,
    }

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device = torch.device("cpu")
        self.model = torch.hub.load(
            "verlab/accelerated_features",
            "XFeat",
            pretrained=True,
            top_k=conf.top_k,
            detection_threshold=conf.detection_threshold,
        )
        self.model.eval()
        self.model.dev = self.device
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=None):
        img = ops.prepareImage(img).to(self.device)
        response = self.model.detectAndCompute(img, top_k=self.conf.top_k)

        # batch it
        batch_response = {}
        for im_result in response:
            for key in im_result:
                if key not in batch_response:
                    batch_response[key] = []
                batch_response[key].append(im_result[key])

        for key in batch_response:
            batch_response[key] = torch.stack(batch_response[key])
            if len(batch_response[key].shape) == 2:
                batch_response[key] = batch_response[key].unsqueeze(0)

        if return_dict:
            return batch_response

        return batch_response["keypoints"], batch_response["descriptors"]

    def detect(self, img, op=None):
        return self.detectAndCompute(img, return_dict=True)["keypoints"]

    # def match(self, image1, image2):
    #     mkpts0, mkpts1 = self.model.match_xfeat(image1, image2)
    #     return {
    #         'mkpts0': mkpts0,
    #         'mkpts1': mkpts1,
    #     }

    def match_xfeat(self, image1, image2):
        mkpts0, mkpts1 = self.model.match_xfeat(image1, image2)
        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
        }

    def match_xfeat_star(self, image1, image2):
        mkpts0, mkpts1 = self.model.match_xfeat_star(image1, image2)
        return {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
        }

    def compute(self, img, keypoints):
        """Compute descriptors for keypoints in the image.
        img: np.array, image
        keypoints: np.array | torch.tensor, keypoints
        """

        if isinstance(keypoints, np.ndarray):
            keypoints = torch.tensor(keypoints)
        keypoints = keypoints.to(self.device)

        # if unbatched, add batch dimension
        if len(keypoints.shape) == 2:
            keypoints = keypoints.unsqueeze(0)

        img = ops.prepareImage(img, gray=True).to(self.device)

        x, rh1, rw1 = self.model.preprocess_tensor(img)
        keypoints = keypoints / torch.tensor([rw1, rh1]).to(self.device)
        B, _, _H1, _W1 = x.shape

        M1, K1, H1 = self.model.net(x)
        M1 = F.normalize(M1, dim=1)

        # Interpolate descriptors at kpts positions
        feats = self.model.interpolator(M1, keypoints, H=_H1, W=_W1)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)
        return feats

    def to(self, device):
        self.model.to(device)
        self.model.dev = device
        self.device = device
        return self

    @property
    def has_detector(self):
        return True


if __name__ == "__main__":
    from easy_local_features.utils import io, vis, ops

    method = XFeat_baseline()

    img0 = io.fromPath("test/assets/megadepth0.jpg")

    kpts = method.detect(img0)
    desc = method.compute(img0, kpts)

    kpts2, desc2 = method.detectAndCompute(img0)

    assert torch.allclose(kpts, kpts2)

    # import pdb; pdb.set_trace()
    print(desc)
    print(desc2)
    assert torch.allclose(desc, desc2, atol=1e-5)

    # img1 = io.fromPath("test/assets/megadepth1.jpg")

    # nn_matches = method.match(img0, img1)
    # xfeat_matches = method.match_xfeat(img0, img1)
    # xfeat_star_matches = method.match_xfeat_star(img0, img1)

    # vis.plot_pair(img0, img1)
    # vis.plot_matches(nn_matches['mkpts0'], nn_matches['mkpts1'])
    # vis.add_text("")

    # vis.plot_pair(img0, img1)
    # vis.plot_matches(xfeat_matches['mkpts0'], xfeat_matches['mkpts1'])

    # vis.plot_pair(img0, img1)
    # vis.plot_matches(xfeat_star_matches['mkpts0'], xfeat_star_matches['mkpts1'])

    # vis.show()
