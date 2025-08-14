import torch
from omegaconf import OmegaConf

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..submodules.git_sfd2.sfd2 import SFD2
from ..utils import ops
from ..utils.download import downloadModel
from .basemodel import BaseExtractor, MethodType

weights_link = "https://github.com/feixue94/sfd2/raw/dev/weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"


class SFD2_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    default_conf = {
        "top_k": 2048,
        "model_name": "ressegnetv2",  # "ressegnetv2", "ressegnet"
        "use_stability": True,
        "conf_th": 0.001,
        "scales": [1.0],
    }

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

        self.max_kps = conf.top_k
        self.DEV = torch.device("cpu")
        self.model = SFD2(
            model_name=conf.model_name,
            use_stability=conf.use_stability,
            top_k=conf.top_k,
            conf_th=conf.conf_th,
            scales=conf.scales,
        )

        weights = downloadModel("sfd2", "20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth", weights_link)
        self.model.model.load_state_dict(torch.load(weights, map_location="cpu")["model"], strict=False)
        self.matcher = NearestNeighborMatcher()

    def detectAndCompute(self, img, return_dict=False):
        img = ops.prepareImage(img)

        with torch.no_grad():
            pred = self.model(
                {
                    "image": img.to(self.DEV),
                    "original_size": torch.tensor(img.shape[-2:]).to(self.DEV),
                }
            )

        # make all into tensor and move to device
        for k in pred:
            pred[k] = torch.tensor(pred[k]).to(self.DEV).unsqueeze(0)

        if return_dict:
            return pred

        return pred["keypoints"], pred["descriptors"]

    def detect(self, img):
        raise NotImplementedError("This method is not implemented in this class")

    def compute(self, img, kps):
        raise NotImplementedError("This method is not implemented in this class")

    def to(self, device):
        self.model = self.model.to(device)
        self.DEV = device
        return self

    @property
    def has_detector(self):
        return True
