import tensorflow as tf
import tensorflow_hub as hub

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils import ops

import torch
class DELF_baseline(BaseExtractor):
    default_config = {
        "top_k": 2048,
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_config), conf)
        
        self.model = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
        self.top_k = conf.top_k
        self.DEV = torch.device('cpu')
        self.matcher = NearestNeighborMatcher()

    def compute(self, img, cv_kps):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def run_delf(self, image):
        float_image = tf.image.convert_image_dtype(image, tf.float32)

        return self.model(
            image=float_image,
            score_threshold=tf.constant(100.0),
            image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
            max_feature_num=tf.constant(self.top_k))

    def detectAndCompute(self, img, return_dict=False):
        # make sure image is rgb
        img = ops.to_cv(ops.prepareImage(img))
        
        results = self.run_delf(img)

        locations = results['locations'].numpy()
        descriptors = results['descriptors'].numpy()
        scales = results['scales'].numpy()

        localtions = torch.tensor(locations).to(self.DEV)
        scales = torch.tensor(scales).to(self.DEV)
        descriptors = torch.tensor(descriptors).to(self.DEV)
        
        if return_dict:
            return {
                'keypoints': localtions,
                'descriptors': descriptors,
                'scales': scales
            }

        return localtions, descriptors

    def detect(self, img):
        return self.detectAndCompute(img)[0]

    def compute(self, image, keypoints):
        raise NotImplemented
    
    def to(self, device):
        # self.model.to(device)
        self.DEV = device

    def match(self, image1, image2):
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        
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
        return True