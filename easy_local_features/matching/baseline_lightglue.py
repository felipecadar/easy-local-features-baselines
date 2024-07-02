import torch
from omegaconf import OmegaConf

from kornia.feature import LightGlue
from ..feature.basemodel import BaseExtractor
from .. import getExtractor
from ..utils import io, vis, ops

class LightGlue_baseline(BaseExtractor):
    defalut_conf = {
        "features": "superpoint",
        "top_k": 2048,
    }
    
    variations = {
        "features": list(LightGlue.features.keys())
    }
    
    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.defalut_conf), conf)
        features = conf.get("features", self.defalut_conf["features"])
        
        assert features in LightGlue.features, f"Invalid feature {features}. Available features: {list(LightGlue.features.keys())}"
        
        self.device = torch.device("cpu")
        self.matcher = LightGlue(features).to(self.device)

        self._extractor = None

    @property
    def extractor(self):
        if self._extractor is None:
            if self.conf["features"] == "dedodeg":
                tmp_conf = self.conf.copy()
                tmp_conf["descriptor_weights"] = "G-upright"
                self._extractor = getExtractor("dedode", tmp_conf)
            elif self.conf["features"] == "dedodeb":
                tmp_conf = self.conf.copy()
                tmp_conf["descriptor_weights"] = "B-upright"
                self._extractor = getExtractor("dedode", tmp_conf)
            else:
                self._extractor = getExtractor(self.defalut_conf["features"], self.conf)
            self._extractor.to(self.device)
        return self._extractor        

    def detectAndCompute(self, image, return_dict=False):
        return self.extractor.detectAndCompute(image, return_dict=return_dict)

    def detect(self, image):
        return self.extractor.detect(image)
    
    def compute(self, image, keypoints):
        return self.extractor.compute(image, keypoints)
    
    @property
    def has_detector(self):
        return self.extractor.has_detector

    def match(self, image0, image1):        
        data0 = self.detectAndCompute(image0, return_dict=True)
        data1 = self.detectAndCompute(image1, return_dict=True)
        
        data = {
            "keypoints0": data0["keypoints"],
            "keypoints1": data1["keypoints"],
            "descriptors0": data0["descriptors"],
            "descriptors1": data1["descriptors"],
            "image0": image0,
            "image1": image1,
        }

        return self.match_cached(data)

    def match_cached(self, data):
        """
        data: dict with keys:
        data = 
        {
            "keypoints0": keypoints0,
            "keypoints1": keypoints1,
            "descriptors0": descriptors0,
            "descriptors1": descriptors1,
            "image0": image0,
            "image1": image1,
        }
        """

        keypoints0 = data["keypoints0"]
        keypoints1 = data["keypoints1"]
        descriptors0 = data["descriptors0"]
        descriptors1 = data["descriptors1"]
        
        # match
        input = {
            "image0": {
                "keypoints": keypoints0,
                "descriptors": descriptors0,
                "image_size": torch.tensor(ops.prepareImage(data["image0"]).shape[-2:]).unsqueeze(0),
            },
            "image1": {
                "keypoints": keypoints1,
                "descriptors": descriptors1,
                "image_size": torch.tensor(ops.prepareImage(data["image1"]).shape[-2:]).unsqueeze(0),
            },
        }
        
        with torch.no_grad():
            out = self.matcher(input)

        matches = out["matches"][0].to(self.device)
        mscores = out["matching_scores0"][0].to(self.device)

        mkpts0 = keypoints0[0, matches[:, 0]]
        mkpts1 = keypoints1[0, matches[:, 1]]

        return {
            "matches": matches,
            "scores": mscores,
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
        }
    
    def to(self, device):
        self.matcher = self.matcher.to(device)
        self.device = device
        return self