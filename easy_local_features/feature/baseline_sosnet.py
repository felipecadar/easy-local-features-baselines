import torch
from kornia.feature.sosnet import SOSNet
from einops import rearrange
from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils.download import downloadModel
from ..utils import ops


class SOSNet_baseline(BaseExtractor):
    default_conf = {}
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.DEV   = torch.device('cpu')
        self.model = SOSNet().to(self.DEV)
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def compute(self, img, keypoints):
        img = ops.prepareImage(img, gray=True)
        do_squeeze = False
        if len(keypoints.shape) == 2:
            keypoints = keypoints.unsqueeze(0)
            do_squeeze = True
        
        # patches ["B", "1", "32", "32"]
        patches = ops.crop_patches(img, keypoints, patch_size=32)
        
        B, N, one, PS1, PS2 = patches.shape
        patches = rearrange(patches, 'B N one PS1 PS2 -> (B N) one PS1 PS2', B=B)
        
        desc = self.model(patches)
        
        desc = rearrange(desc, '(B N) D -> B N D', B=B, N=N, D=128)
        
        if do_squeeze:
            desc = desc.squeeze(0)
            keypoints = keypoints.squeeze(0)
            
        return keypoints, desc
    
    def to(self, device):
        self.model.to(device)
        self.DEV = device

    def match(self, image1, image2):
        try:
            kp0, desc0 = self.detectAndCompute(image1)
            kp1, desc1 = self.detectAndCompute(image2)
        except NotImplemented:
            raise NotImplemented("This method requires detectAndCompute to be implemented. Add a detector to the extractor with addDetector.")
            
        
        data = {
            "descriptors0": desc0.unsqueeze(0),
            "descriptors1": desc1.unsqueeze(0),
        }
        
        response = self.matcher(data)
        
        m0 = response['matches0'][0]
        valid = m0 > -1
        
        mkpts0 = kp0[valid]
        mkpts1 = kp1[m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }
        
    @property
    def has_detector(self):
        return False