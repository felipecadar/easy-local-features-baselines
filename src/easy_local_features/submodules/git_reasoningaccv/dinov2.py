import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

class DinoV2(torch.nn.Module):
    default_conf = {"weights": "dinov2_vits14", "allow_resize": True}
    required_data_keys = ["image"]
    
    def __init__(self, conf={}):
        super().__init__()
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.net = torch.hub.load("facebookresearch/dinov2", conf.weights)
        self.vit_size = 14
        
    def forward(self, data):
        assert all(k in data for k in self.required_data_keys), f"Missing keys: {self.required_data_keys}"
        
        img = data["image"]
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        if self.conf.allow_resize:
            img = F.interpolate(img, [int(x // 14 * 14) for x in img.shape[-2:]])

        desc, cls_token = self.net.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]
        
        return {
            "features": desc,
            "global_descriptor": cls_token,
            "descriptors": desc.flatten(-2).transpose(-2, -1),
        }

    def sample_features(self, keypoints, features, s=None, mode="bilinear"):
        if s is None:
            s = self.vit_size
        b, c, h, w = features.shape
        keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        features = torch.nn.functional.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
        )
        features = torch.nn.functional.normalize(
            features.reshape(b, c, -1), p=2, dim=1
        )
        
        features = features.permute(0, 2, 1)
        return features

