import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
import cv2, os, wget
import scipy


from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils import ops
  
from kornia.feature import DeDoDe

def dual_softmax_matcher(desc_A, desc_B, inv_temperature = 1, normalize = False):
    if len(desc_A.shape) < 3:
        desc_A, desc_B = desc_A[None], desc_B[None]
    if normalize:
        desc_A = desc_A/desc_A.norm(dim=-1,keepdim=True)
        desc_B = desc_B/desc_B.norm(dim=-1,keepdim=True)
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    else:
        corr = torch.einsum("b n c, b m c -> b n m", desc_A, desc_B) * inv_temperature
    P = corr.softmax(dim = -2) * corr.softmax(dim= -1)
    return P

class DualSoftMaxMatcher(torch.nn.Module):        
    @torch.inference_mode()
    def forward(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, 
              normalize = False, inv_temp = 1, threshold = 0.0):
        if isinstance(descriptions_A, list):
            matches = [self.forward(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = dual_softmax_matcher(descriptions_A, descriptions_B, 
                                 normalize = normalize, inv_temperature=inv_temp,
                                 )
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))
        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]
        return matches_A, matches_B, batch_inds

class DeDoDe_baseline(BaseExtractor):
    # detector_weights: The weights to load for the detector. One of 'L-upright', 'L-C4', 'L-SO2'.
    # descriptor_weights: The weights to load for the descriptor. One of 'B-upright', 'B-C4', 'B-SO2', 'G-upright', 'G-C4'.
    default_conf = {
        'top_k': 2048,
        'detector_weights': 'L-upright', # 'L-upright', 'L-C4', 'L-SO2'
        'descriptor_weights': 'B-upright', # 'B-upright', 'B-C4', 'B-SO2', 'G-upright', 'G-C4'
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
         
        self.DEV   = torch.device('cpu')
        self.model = DeDoDe.from_pretrained(detector_weights=conf.detector_weights, descriptor_weights=conf.descriptor_weights)
        self.matcher = DualSoftMaxMatcher()
        self.top_kps = conf.top_k

    def detectAndCompute(self, image, return_dict=False):
        image = ops.prepareImage(image, imagenet=True).to(self.DEV)
        keypoints, scores, descriptors = self.model(image, n=self.top_kps, apply_imagenet_normalization=False)

        if return_dict:
            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores
            }
       
        return keypoints, descriptors        

    def detect(self, img):
        return self.detectAndCompute(img)[0]

    def compute(self, image, keypoints):
        raise NotImplemented
    
    def to(self, device):
        self.model.to(device)
        self.DEV = device

    def match(self, image1, image2):
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        
        matches_A, matches_B, batch_ids = self.matcher(kp0, desc0, kp1, desc1, normalize = True, inv_temp=20, threshold = 0.1)

        return {
            'mkpts0': matches_A,
            'mkpts1': matches_B,
            'batch_ids': batch_ids,
        }
        
    @property
    def has_detector(self):
        return True