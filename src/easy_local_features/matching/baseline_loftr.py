from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from kornia.feature import LoFTR

import kornia
import torch
import functools
import numpy as np


@functools.lru_cache(maxsize=1)
def getLoFTR(pretrained="outdoor"):
    return LoFTR(pretrained=pretrained)

class LoFTR_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.END2END_MATCHER
    
    def __init__(self, conf=None, pretrained="outdoor"):
        if conf is not None:
            pretrained = conf.get("pretrained", pretrained)
        self.device = torch.device("cpu")
        self.matcher = getLoFTR(pretrained=pretrained).to(self.device)

    @property
    def has_detector(self):
        return False

    def detectAndCompute(self, image, return_dict=False):
        raise NotImplementedError("LoFTR is a matcher only. It does not extract features.")
        
    def compute(self, image, keypoints):
        raise NotImplementedError("LoFTR is a matcher only. It does not extract features.")

    def to(self, device):
        self.device = torch.device(device)
        self.matcher = self.matcher.to(self.device)
        return self
        
    def __call__(self, data):
        # Compatible with generic __call__ signature expecting data dictionary
        if "image1" in data:
            with torch.no_grad():
                res = self.matcher(data)
            return res
        return dict()

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
        with torch.no_grad():
            correspondences_dict = self.matcher(input)

        b_size = img1.shape[0]
        out_list = []
        
        batch_indexes = correspondences_dict["batch_indexes"].detach().cpu()
        all_mkpts0 = correspondences_dict["keypoints0"].detach().cpu()
        all_mkpts1 = correspondences_dict["keypoints1"].detach().cpu()
        all_scores = correspondences_dict.get("confidence", None)
        if all_scores is not None:
            all_scores = all_scores.detach().cpu()

        for b in range(b_size):
            mask = batch_indexes == b
            mkpts0_b = all_mkpts0[mask]
            mkpts1_b = all_mkpts1[mask]
            
            out_dict = {
                "mkpts0": mkpts0_b,
                "mkpts1": mkpts1_b,
            }
            if all_scores is not None:
                out_dict["scores"] = all_scores[mask]
                
            out_list.append(out_dict)

        if b_size == 1:
            return out_list[0]
        return out_list
