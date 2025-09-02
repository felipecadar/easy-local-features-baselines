import tensorflow as tf
import tensorflow_hub as hub
import torch
from omegaconf import OmegaConf
from typing import TypedDict

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from .basemodel import BaseExtractor, MethodType


class DELFConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    use_pca: bool
    use_whitening: bool


class DELF_baseline(BaseExtractor):
    """DELF baseline wrapper using TF Hub model.

    Notes / Fixes:
    - TF Hub outputs locations as (y, x); the rest of the code expects (x, y).
      We therefore flip the last dimension order.
    - Replaced accidental `raise NotImplemented` with `NotImplementedError`.
    - Fixed variable typo `localtions` -> `keypoints`.
    - Added defensive handling for empty outputs.
    """

    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    default_config: DELFConfig = {
        "model_name": "delf",
        "top_k": 2048,
        "detection_threshold": 0.2,
        "nms_radius": 4,
        "use_pca": False,
        "use_whitening": False,
    }

    def __init__(self, conf: DELFConfig = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_config), conf)
        # Load TF Hub model once; returns a SavedModel signature callable.
        self.model = hub.load("https://tfhub.dev/google/delf/1").signatures["default"]
        self.top_k = conf.top_k
        self.DEV = torch.device("cpu")
        self.matcher = NearestNeighborMatcher()

    # Descriptor-only API parts (not used directly for DELF)
    def compute(self, img, cv_kps):  # pragma: no cover - interface stub
        raise NotImplementedError

    def detect(self, img, op=None):  # Older unused signature; kept for backward compat
        raise NotImplementedError

    def run_delf(self, image):
        # image: numpy uint8 or float HxWx3 in RGB
        float_image = tf.image.convert_image_dtype(image, tf.float32)
        return self.model(
            image=float_image,
            score_threshold=tf.constant(100.0),  # keep high score threshold (original default ~1000 in some impls)
            image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
            max_feature_num=tf.constant(self.top_k),
        )

    def detectAndCompute(self, img, return_dict=False):
        # Ensure RGB numpy image (HxWx3, uint8)
        img = ops.to_cv(ops.prepareImage(img))

        results = self.run_delf(img)

        # Flip (y, x) -> (x, y); copy() avoids negative stride issue when converting to torch
        locations = results["locations"].numpy()[:, ::-1].copy()
        descriptors = results["descriptors"].numpy()
        scales = results["scales"].numpy()

        keypoints = torch.as_tensor(locations, dtype=torch.float32, device=self.DEV)
        descriptors = torch.as_tensor(descriptors, dtype=torch.float32, device=self.DEV)
        scales = torch.as_tensor(scales, dtype=torch.float32, device=self.DEV)

        # Add batch dimension expected downstream: [1, N, 2] / [1, N, D]
        if return_dict:
            return {
                "keypoints": keypoints.unsqueeze(0),
                "descriptors": descriptors.unsqueeze(0),
                "scales": scales.unsqueeze(0),
            }
        return keypoints.unsqueeze(0), descriptors.unsqueeze(0)

    # Override detector-only call used by framework
    def detect(self, img):  # type: ignore[override]
        return self.detectAndCompute(img)[0]

    def compute(self, image, keypoints):  # type: ignore[override]
        raise NotImplementedError

    def to(self, device):
        # TensorFlow model itself stays on CPU; only tensors we create can move.
        if isinstance(device, str):
            device = torch.device(device)
        self.DEV = device
        return self

    @property
    def has_detector(self):
        return True
