import torch
import torch.nn.functional as F
from functools import partial
import cv2, os, wget
from easy_local_features.submodules.git_roma import romatch

from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils import ops

models = {'outdoor': romatch.models.roma_outdoor,
         'indoor': romatch.models.roma_indoor,
         'tiny_outdoor': romatch.models.tiny_roma_v1_outdoor}

class RoMa_baseline(BaseExtractor):
    default_conf = {
        'top_k': 512,
        'model': 'outdoor'
    }
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.num_keypoints = conf.top_k
        self.device  = torch.device('cpu')
        self.model = models[self.conf.model](device=self.device)
        self.model.eval()


    def preprocess_batch_tensor(
        self,
        tensor: torch.Tensor,
        target_size: tuple[int,int],
        mean=(0.485, 0.456, 0.406),
        std =(0.229, 0.224, 0.225),
    ) -> torch.Tensor:
        
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        x = tensor
        hs, ws = target_size
        x = F.interpolate(x, size=(hs,ws), mode='bicubic', align_corners=False)
        dev = x.device
        m = torch.tensor(mean, device=dev).view(1,3,1,1)
        s = torch.tensor(std,  device=dev).view(1,3,1,1)
        x = (x - m) / s
        return x
    
    def detectAndCompute(self, image, return_dict=False):
        raise NotImplementedError("Every BaseExtractor must implement the detectAndCompute method.")
    
    def detect(self, image):
        raise NotImplementedError("Every BaseExtractor must implement the detect method.")
    
    def compute(self, image, keypoints):
        raise NotImplementedError("Every BaseExtractor must implement the compute method.")    
    
    def to(self, device):
        raise NotImplementedError("Every BaseExtractor must implement the to method.")
    
    @property
    def has_detector(self):
        return True

    
    # @abstractmethod
    def match(self, image1, image2):
        hs, ws = self.model.h_resized, self.model.w_resized
        imA_h, imA_w = image1.shape[2:]
        imB_h, imB_w = image2.shape[2:]

        batchA = self.preprocess_batch_tensor(image1, (hs, ws)).to(self.device)
        batchB = self.preprocess_batch_tensor(image2, (hs, ws)).to(self.device)

        warp, certainty = self.model.match(batchA, batchB, batched=True, device=self.device)
        matches, certs = self.model.sample(warp, certainty, num=self.num_keypoints)

        kptsA, kptsB = self.model.to_pixel_coordinates(matches, imA_h, imA_w, imB_h, imB_w)

        return {
            'mkpts0': kptsA,
            'mkpts1': kptsB,
        }    