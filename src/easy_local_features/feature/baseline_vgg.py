import torch
import torch.nn.functional as F
import torchvision

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor, MethodType
from ..utils import ops

class VGG_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    available_weights = [
        'vgg11',
        'vgg11_bn',
        'vgg13',
        'vgg13_bn',
        'vgg16',
        'vgg16_bn',
        'vgg19',
        'vgg19_bn',
    ]
        
    default_conf = {"weights": "vgg11", "allow_resize": True}
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device   = torch.device('cpu')
        self.matcher = NearestNeighborMatcher()
        self.model = torchvision.models.vgg11(weights='IMAGENET1K_V1')
        self.model.eval()

    def sample_features(self, keypoints, features, image_shape, mode="bilinear"):

        b, _, h, w = image_shape
        c = features.shape[1]
        keypoints = keypoints / (keypoints.new_tensor([w, h]))
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        features = torch.nn.functional.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
        )
        features = torch.nn.functional.normalize(
            features.reshape(b, c, -1), p=2, dim=1
        )
        
        features = features.permute(0, 2, 1)
        return features

    def detectAndCompute(self, img, return_dict=None):
        raise NotImplementedError
    
    def detect(self, img, op=None):
        raise NotImplementedError

    def compute(self, img, keypoints=None, return_dict=False):
        img = ops.prepareImage(img, imagenet=True).to(self.device)

        if self.conf.allow_resize:
            img = F.interpolate(img, [224,224])

        features = self.model.features(img)
        if keypoints is None:
            return features
        
        descriptors = self.sample_features(keypoints, features, img.shape)
        
        return keypoints, descriptors

    def to(self, device):
        self.model.to(device)
        self.device = device

        
    @property
    def has_detector(self):
        return False
    
if __name__ == "__main__":
    from easy_local_features.utils import io, vis, ops
    from easy_local_features.feature.baseline_xfeat import XFeat_baseline
    method = VGG_baseline({
        'weights': 'vgg11',
        'allow_resize': False,
    })
    detector = XFeat_baseline({
        'top_k': 512,
    })
    
    img0 = io.fromPath("test/assets/megadepth0.jpg")
    img1 = io.fromPath("test/assets/megadepth1.jpg")
    
    # resize to 512x512
    img0 = F.interpolate(img0, [512,512])
    img1 = F.interpolate(img1, [512,512])
    
    kps0 = detector.detect(img0)
    kps1 = detector.detect(img1)
    
    batched_images = torch.cat([img0, img1], dim=0)
    batched_kps = torch.cat([kps0, kps1], dim=0)
    
    _, batched_descriptors = method.compute(batched_images, batched_kps)
    
    descriptors0 = batched_descriptors[0].unsqueeze(0)
    descriptors1 = batched_descriptors[1].unsqueeze(0)
    
    print(descriptors0.shape)
    print(descriptors1.shape)
    
    response = method.matcher({
        "descriptors0": descriptors0,
        "descriptors1": descriptors1,
    })
            
    m0 = response['matches0'][0]
    valid = m0 > -1
    mkpts0 = kps0[valid]
    mkpts1 = kps1[m0[valid]]
        
    vis.plot_pair(img0, img1)
    vis.plot_matches(mkpts0, mkpts1)
    vis.add_text("VGG")
    vis.show()