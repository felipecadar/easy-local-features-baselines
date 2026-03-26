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
        raise NotImplementedError("CroCo is descriptor-only; use compute() with keypoints.")

    def detect(self, img, op=None):
        raise NotImplementedError("CroCo is descriptor-only; use compute() with keypoints.")

    def compute(self, img, keypoints=None, return_dict=False):
        if keypoints is None:
            raise ValueError("CroCo requires keypoints (descriptor-only method).")
        img = ops.prepareImage(img).to(self.device)

        # CroCo's PatchEmbed enforces exact img_size; resize to match
        target_h, target_w = self.model.patch_embed.img_size
        if img.shape[2] != target_h or img.shape[3] != target_w:
            # Scale keypoints to match the resized image
            scale_x = target_w / img.shape[3]
            scale_y = target_h / img.shape[2]
            keypoints = keypoints.clone()
            keypoints[..., 0] *= scale_x
            keypoints[..., 1] *= scale_y
            img = F.interpolate(img, size=(target_h, target_w), mode="bilinear", align_corners=False)

        # CroCo uses _encode_image which returns (features, pos, masks)
        # features: [B, N_patches, C] where N_patches = (H/patch_size) * (W/patch_size)
        with torch.no_grad():
            features, pos, masks = self.model._encode_image(img, do_mask=False)

        # Reshape to spatial feature map [B, C, H_patches, W_patches]
        B, N, C = features.shape
        h_patches = img.shape[2] // self.model.patch_embed.patch_size[0]
        w_patches = img.shape[3] // self.model.patch_embed.patch_size[1]
        desc = features.permute(0, 2, 1).reshape(B, C, h_patches, w_patches)

        patch_size = self.model.patch_embed.patch_size[0]
        descriptors = self.sample_features(keypoints, desc, s=patch_size)

        if return_dict:
            return {
                "descriptors": descriptors,
                "keypoints": keypoints,
                "feature_map": desc,
            }
        return keypoints, descriptors

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    @property
    def has_detector(self):
        return False
