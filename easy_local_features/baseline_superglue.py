import sys, os
sp_folder = os.path.dirname(os.path.realpath(__file__)) + '/submodules/git_superglue/'
sys.path.insert(0, sp_folder)
from models.superglue import SuperGlue

import torch
import numpy as np
from functools import partial
import cv2

class SuperGlue_baseline():
    def __init__(self, weights='indoor', sinkhorn_iterations=100, match_threshold=0.2, descriptor_dim=256):
        self.DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.CPU   = torch.device('cpu')
    
        config = {
            'descriptor_dim': descriptor_dim,
            'weights': weights,
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }

        self.model = SuperGlue(config)
        self.model = self.model.to(self.DEV)
        self.model.eval()

    def _toTorch(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(img/255.).float()[None, None].to(self.DEV)

    def match(self, img0, kps0, desc0, img1, kps1, desc1):

        if kps0 is None or kps1 is None:
            return [], []

        if len(kps0) == 0 or len(kps1) == 0:
            return [], []

        if isinstance(kps0[0] , cv2.KeyPoint):
            keypoints0 = torch.tensor([kp.pt for kp in kps0]).to(self.DEV).unsqueeze(0).float()
            keypoints1 = torch.tensor([kp.pt for kp in kps1]).to(self.DEV).unsqueeze(0).float()

            scores0 = torch.tensor([kp.response for kp in kps0]).to(self.DEV).unsqueeze(0).float()
            scores1 = torch.tensor([kp.response for kp in kps1]).to(self.DEV).unsqueeze(0).float()

        if isinstance(kps0, np.ndarray):
            keypoints0 = torch.tensor(kps0).to(self.DEV).unsqueeze(0).float()
            keypoints1 = torch.tensor(kps1).to(self.DEV).unsqueeze(0).float()

            scores0 = torch.ones(len(kps0)).to(self.DEV).unsqueeze(0).float()
            scores1 = torch.ones(len(kps1)).to(self.DEV).unsqueeze(0).float()
    
        descriptors0 = torch.tensor(desc0.T).to(self.DEV).unsqueeze(0).float()
        descriptors1 = torch.tensor(desc1.T).to(self.DEV).unsqueeze(0).float()
    
        inp = {
            'image0': self._toTorch(img0),
            'keypoints0': keypoints0,
            'descriptors0':descriptors0,
            'scores0':scores0,
            'image1': self._toTorch(img1),
            'keypoints1':keypoints1,
            'descriptors1':descriptors1,
            'scores1':scores1,
        }

        with torch.no_grad():
            res = self.model(inp)

        matches0 = res['matches0'][0].to(self.CPU).numpy()
        matches1 = res['matches1'][0].to(self.CPU).numpy()
        matching_scores0 = res['matching_scores0'][0].to(self.CPU).numpy()
        matching_scores1 = res['matching_scores1'][0].to(self.CPU).numpy()

        matching_scores0[matching_scores0 == 0] = 1e-9
        matching_scores1[matching_scores1 == 0] = 1e-9

        matches1to2 = [cv2.DMatch(i,j,1/s) for i,[j,s] in enumerate(zip(matches0, matching_scores0))]
        matches2to1 = [cv2.DMatch(i,j,1/s) for i,[j,s] in enumerate(zip(matches1, matching_scores1))]

        return matches1to2, matches2to1

if __name__ == "__main__":
    import pdb
    from baseline_superpoint import SuperPoint_baseline

    img = cv2.imread("../assets/notredame.png")
    extractor = SuperPoint_baseline()
    matcher = SuperGlue_baseline()

    keypoints0, descriptors0 = extractor.detectAndCompute(img)
    
    matches1to2, matches2to1 = matcher.match(img, keypoints0, descriptors0, img, keypoints0, descriptors0)

    matched_img1 = cv2.drawMatches(img, keypoints0, img, keypoints0, matches1to2, None)
    matched_img2 = cv2.drawMatches(img, keypoints0, img, keypoints0, matches2to1, None)

    stacked = np.vstack([matched_img1, matched_img2])

    cv2.imwrite("sg_test.png", stacked)