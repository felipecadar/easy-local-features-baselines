import torch
import numpy as np
import scipy

from easy_local_features.submodules.git_d2net.models import D2Net, preprocess_image, process_multiscale
from ..utils import ops
from .basemodel import BaseExtractor, MethodType
from ..utils.download import downloadModel
from ..matching.nearest_neighbor import NearestNeighborMatcher

from omegaconf import OmegaConf


class D2Net_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE

    default_conf = {
        "top_k": 2048,
        "use_relu": True,
    }

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        top_kps = conf.top_k
        use_relu = conf.use_relu

        self.DEV = torch.device("cpu")
        self.model = D2Net(use_relu=use_relu)

        model_file_url = (
            "https://dusmanu.com/files/d2-net/d2_tf_no_phototourism.pth"  # https://dsmn.ml/files/d2-net/d2_ots.pth
        )
        model_path = downloadModel("d2net", "d2_tf_no_phototourism.pth", model_file_url)
        self.model.load_state_dict(torch.load(model_path, map_location=self.DEV)["model"])

        self.model.eval()
        self.model.to(self.DEV)
        self.top_kps = top_kps

        self.max_edge = 1600
        self.max_sum_edges = 2800
        self.multiscale = False
        self.use_relu = use_relu
        self.preprocessing = "torch"  # 'caffe' or 'torch'

        self.matcher = NearestNeighborMatcher()

    def compute(self, img, cv_kps):
        raise NotImplementedError

    def detect(self, img, op=None):
        raise NotImplementedError

    def detectAndCompute(self, image, return_dict=False):
        image = ops.prepareImage(image)
        image = ops.to_cv(image)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        if max(resized_image.shape) > self.max_edge:
            resized_image = scipy.misc.imresize(resized_image, self.max_edge / max(resized_image.shape)).astype("float")
        if sum(resized_image.shape[:2]) > self.max_sum_edges:
            resized_image = scipy.misc.imresize(
                resized_image, self.max_sum_edges / sum(resized_image.shape[:2])
            ).astype("float")

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(resized_image, preprocessing=self.preprocessing)

        with torch.no_grad():
            if self.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=self.DEV), self.model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=self.DEV),
                    self.model,
                    scales=[1],
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]
        keypoints, scales = keypoints[:, :2], keypoints[:, 2]

        argsort = np.argsort(-scores)

        keypoints = keypoints[argsort[: self.top_kps]].unsqueeze(0)
        scores = scores[argsort[: self.top_kps]].unsqueeze(0)
        descriptors = descriptors[argsort[: self.top_kps]].unsqueeze(0)
        scales = scales[argsort[: self.top_kps]].unsqueeze(0)

        if return_dict:
            return {"keypoints": keypoints, "descriptors": descriptors, "scores": scores, "scales": scales}

        return keypoints, descriptors

    def detect(self, img):
        keypoints, descriptors = self.detectAndCompute(img)
        return keypoints

    def compute(self, img, keypoints):
        raise NotImplemented

    def to(self, device):
        self.model.to(device)
        self.DEV = device

    @property
    def has_detector(self):
        return True
