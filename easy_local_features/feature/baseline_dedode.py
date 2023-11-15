import pyrootutils
root = pyrootutils.find_root()

import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import cv2, os, wget
import scipy

try:
    import DeDoDe
except:
    print("DeDoDe not installed. Please install it with:")
    print("pip install git+https://github.com/Parskatt/DeDoDe.git")
    print("and restart the script.")
    exit()      

from DeDoDe import dedode_detector_L, dedode_descriptor_B, dedode_descriptor_G
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import to_pixel_coords

DETECT_L='https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_detector_L.pth'
DESC_G='https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_G.pth'
DESC_B='https://github.com/Parskatt/DeDoDe/releases/download/dedode_pretrained_models/dedode_descriptor_B.pth'

class DeDoDe_baseline():
    def __init__(self, top_kps = 2048, use_relu=True, device=-1):
        self.CPU   = torch.device('cpu')
        self.DEV   = torch.device(f'cuda:{device}' if (torch.cuda.is_available() and device >= 0) else 'cpu')
        self.detector = dedode_detector_L(
            weights=torch.hub.load_state_dict_from_url(DETECT_L),
        )
        self.detector.eval()
        self.detector.to(self.DEV)

        self.descriptor = dedode_descriptor_G(
            weights=torch.hub.load_state_dict_from_url(DESC_G),
            dinov2_weights=None
        )
        self.descriptor.eval()
        self.descriptor.to(self.DEV)

        self.top_kps = top_kps

    def compute(self, img, cv_kps):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def detectAndCompute(self, image, op=None):
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        tensor_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.DEV) / 255.0

        # pad to be multiple of 14
        h, w = tensor_image.shape[2:]
        h_pad = 14 - h % 14
        w_pad = 14 - w % 14
        tensor_image = F.pad(tensor_image, (0, w_pad, 0, h_pad), mode='constant', value=0)

        detections = self.detector.detect({'image':tensor_image}, self.top_kps)
        keypoints, P = detections["keypoints"], detections["confidence"]

        descriptors = self.descriptor.describe_keypoints({'image':tensor_image}, keypoints)["descriptions"]

        keypoints = to_pixel_coords(keypoints, h, w)

        keypoints = keypoints.squeeze().cpu().numpy().astype(np.float32)
        P = P.squeeze().cpu().numpy().astype(np.float32)
        keypoints = [cv2.KeyPoint(x=keypoints[i][0], y=keypoints[i][1], size=1, response=P[i]) for i in range(len(keypoints))]
        descriptors = descriptors.squeeze().cpu().numpy()

        return keypoints, descriptors        

    # def match(self, kp1, desc1, kp2, desc2, H1, W1, H2, W2):

    #     matches_A, matches_B, batch_ids = matcher.match(kps1, desc1, kps2, desc2, P_A = P1, P_B = P2, normalize = True, inv_temp=20, threshold = 0.1)#Increasing threshold -> fewer matches, fewer outliers

    #     matches_A, matches_B = matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)

    #     mkpts1 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in matches_A.cpu().numpy().astype(np.float32)]
    #     mkpts2 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in matches_B.cpu().numpy().astype(np.float32)]
    #     matches = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(matches_A))]


    def detectAndComputeTorch(self, image, op=None):
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        tensor_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.DEV) / 255.0

        # pad to be multiple of 14
        h, w = tensor_image.shape[2:]
        h_pad = 14 - h % 14
        w_pad = 14 - w % 14
        tensor_image = F.pad(tensor_image, (0, w_pad, 0, h_pad), mode='constant', value=0)

        detections = self.detector.detect({'image':tensor_image}, self.top_kps)
        keypoints, P = detections["keypoints"], detections["confidence"]

        descriptors = self.descriptor.describe_keypoints({'image':tensor_image}, keypoints)["descriptions"]

        return keypoints, P, descriptors        



if __name__ == "__main__":
    img1 = cv2.imread(str(root / "assets" / "notredame.png"))
    img2 = cv2.imread(str(root / "assets" / "notredame2.jpeg"))
    # img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    # img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    model = DeDoDe_baseline(2048, device=0)

    # H_A, W_A = img1.shape[:2]
    # H_B, W_B = img2.shape[:2]

    # kps1, P1, desc1 = model.detectAndComputeTorch(img1)
    # kps2, P2, desc2 = model.detectAndComputeTorch(img2)

    # matcher = DualSoftMaxMatcher()

    # matches_A, matches_B, batch_ids = matcher.match(kps1, desc1, kps2, desc2, P_A = P1, P_B = P2, normalize = True, inv_temp=20, threshold = 0.1)#Increasing threshold -> fewer matches, fewer outliers

    # matches_A, matches_B = matcher.to_pixel_coords(matches_A, matches_B, H_A, W_A, H_B, W_B)

    # mkpts1 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in matches_A.cpu().numpy().astype(np.float32)]
    # mkpts2 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in matches_B.cpu().numpy().astype(np.float32)]
    # matches = [cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(matches_A))]
    # img3 = cv2.drawMatches(img1, mkpts1, img2, mkpts2, matches, None, flags=2)

    # cv2.imwrite("dedode.png", img3)


    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    kps1, desc1 = model.detectAndCompute(img1)
    kps2, desc2 = model.detectAndCompute(img2)
    matches = bf.match(desc1, desc2)

    # ransac
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matches = [m for m,msk in zip(matches, mask) if msk == 1]

    img3 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)

    cv2.imwrite("dedode_cv.png", img3)

