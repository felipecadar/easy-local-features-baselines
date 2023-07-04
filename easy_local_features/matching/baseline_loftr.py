from kornia.feature import LoFTR

import pyrootutils
import cv2

import kornia
import torch
import functools
import numpy as np

root = pyrootutils.find_root()

@functools.lru_cache(maxsize=1)
def getLoFTR(pretrained="outdoor"):
    return LoFTR(pretrained=pretrained)

class LoFTR_baseline:
    def __init__(self, pretrained="outdoor"):
        self.matcher = getLoFTR(pretrained=pretrained)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        matches = []
        mkpts1 = correspondences_dict["keypoints0"].cpu().numpy()
        mkpts2 = correspondences_dict["keypoints1"].cpu().numpy()
        mconf = correspondences_dict["confidence"].cpu().numpy()
        batch_idx = correspondences_dict["batch_indexes"].cpu().numpy()
    
        return mkpts1, mkpts2, mconf
    
if __name__ == "__main__":
    img = cv2.imread(str(root / "assets" / "notredame.png"))

    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)


    matcher = LoFTR_baseline()
    res = matcher.match(img, img)

    import pdb; pdb.set_trace()