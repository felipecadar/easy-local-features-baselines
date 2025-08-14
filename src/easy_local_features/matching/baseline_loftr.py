from kornia.feature import LoFTR

import kornia
import torch
import functools
import numpy as np


@functools.lru_cache(maxsize=1)
def getLoFTR(pretrained="outdoor"):
    return LoFTR(pretrained=pretrained)

class LoFTR_baseline:
    def __init__(self, pretrained="outdoor"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = getLoFTR(pretrained=pretrained).to(self.device)

    def match(self, img1, img2):

        if isinstance(img1, np.ndarray):
            img1 = kornia.utils.image_to_tensor(img1).to(self.device)
        if isinstance(img2, np.ndarray):
            img2 = kornia.utils.image_to_tensor(img2).to(self.device)

        # check if its float
        if img1.dtype != torch.float32:
            img1 = img1.float()
        if img2.dtype != torch.float32:
            img2 = img2.float()

        # check if its in range 0-1
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0

        # batch dimension
        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)

        if len(img2.shape) == 3:
            img2 = img2.unsqueeze(0)

        # grayscale
        if img1.shape[1] == 3:
            img1 = kornia.color.bgr_to_grayscale(img1)
        if img2.shape[1] == 3:
            img2 = kornia.color.bgr_to_grayscale(img2)

        input = {"image0": img1, "image1": img2}
        correspondences_dict = self.matcher(input)

        mkpts0 = correspondences_dict["keypoints0"].detach().cpu()
        mkpts1 = correspondences_dict["keypoints1"].detach().cpu()
        scores = correspondences_dict.get("confidence", None)
        out = {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
        }
        if scores is not None:
            out["scores"] = scores.detach().cpu()
        return out
