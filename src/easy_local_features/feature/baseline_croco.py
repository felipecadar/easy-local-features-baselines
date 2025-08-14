import torch
import torch.nn.functional as F

from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor, MethodType
from ..utils import ops, download

from ..submodules.git_croco.croco import CroCoNet

BASE_LINK = "https://download.europe.naverlabs.com/ComputerVision/CroCo/"  # model.pth


class CroCo_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    available_weights = [
        "CroCo",
        "CroCo_V2_ViTBase_SmallDecoder",
        "CroCo_V2_ViTBase_BaseDecoder",
        "CroCo_V2_ViTLarge_BaseDecoder",
    ]

    default_conf = {"weights": "CroCo", "allow_resize": True}

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device = torch.device("cpu")
        self.matcher = NearestNeighborMatcher()

        # load model
        ckpt_link = BASE_LINK + conf.weights + ".pth"
        ckpt_path = download.downloadModel("croco", conf.weights, ckpt_link)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model = CroCoNet(**ckpt.get("croco_kwargs", {})).to(self.device)
        self.model.eval()
        self.model.load_state_dict(ckpt["model"], strict=True)

    def sample_features(self, keypoints, features, s=14, mode="bilinear"):
        if s is None:
            s = self.vit_size
        b, c, h, w = features.shape
        keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        features = torch.nn.functional.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
        )
        features = torch.nn.functional.normalize(features.reshape(b, c, -1), p=2, dim=1)

        features = features.permute(0, 2, 1)
        return features

    def detectAndCompute(self, img, return_dict=None):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def compute(self, img, keypoints=None, return_dict=False):
        img = ops.prepareImage(img).to(self.device)

        if self.conf.allow_resize:
            img = F.interpolate(img, [int(x // 14 * 14) for x in img.shape[-2:]])

        desc, cls_token = self.model.get_intermediate_layers(img, n=1, return_class_token=True, reshape=True)[0]

        if keypoints is not None:
            descriptors = self.sample_features(keypoints, desc)
        if return_dict:
            return {
                "descriptors": descriptors,
                "keypoints": keypoints,
                "global_descriptor": cls_token,
                "feature_map": desc,
            }
        return keypoints, descriptors

    def to(self, device):
        self.model.to(device)
        self.device = device

    @property
    def has_detector(self):
        return False
