from abc import ABC, abstractmethod
from typing import Optional

import torch


class MethodType:
    """String constants describing the type of feature method.

    - DETECT_DESCRIBE: traditional detector+descriptor operating on a single image
    - DESCRIPTOR_ONLY: descriptor that requires external keypoints
    - END2END_MATCHER: end-to-end matcher operating directly on two images
    """

    DETECT_DESCRIBE = "detect_describe"
    DESCRIPTOR_ONLY = "descriptor_only"
    END2END_MATCHER = "end2end_matcher"


class BaseExtractor(ABC):
    # Optional override in subclasses; if None, inferred from `has_detector`.
    METHOD_TYPE: Optional[str] = None

    @abstractmethod
    def detectAndCompute(self, image, return_dict=False):
        raise NotImplementedError("Every BaseExtractor must implement the detectAndCompute method.")

    @abstractmethod
    def detect(self, image):
        raise NotImplementedError("Every BaseExtractor must implement the detect method.")

    @abstractmethod
    def compute(self, image, keypoints):
        raise NotImplementedError("Every BaseExtractor must implement the compute method.")

    @abstractmethod
    def to(self, device):
        raise NotImplementedError("Every BaseExtractor must implement the to method.")

    @property
    @abstractmethod
    def has_detector(self):
        raise NotImplementedError("Every BaseExtractor must implement the has_detector property.")

    def __call__(self, data):
        """
        data: dict
            {
                'image': image,
            }
        """
        return self.detectAndCompute(data["image"], return_dict=True)

    def addDetector(self, detector):
        def detectAndCompute(image, return_dict=False):
            keypoints = detector.detect(image)
            # Support compute() returning either descriptors OR (keypoints, descriptors)
            out = self.compute(image, keypoints)
            if isinstance(out, tuple) and len(out) == 2:
                keypoints, descriptors = out
            else:
                descriptors = out
            if return_dict:
                return {"keypoints": keypoints, "descriptors": descriptors}
            return keypoints, descriptors

        self.detect = detector.detect
        self.detectAndCompute = detectAndCompute

    # @abstractmethod
    @torch.inference_mode()
    def match(self, image1, image2):
        """Match two images using this method's extractor and matcher.

        Inputs:
        - image1, image2: numpy arrays, torch tensors, or paths. Each will be
          prepared by the concrete extractor's detectAndCompute implementation.

        Returns dict with at least:
        - mkpts0: matched keypoints from image1, shape [M, 2] or [1, M, 2]
        - mkpts1: matched keypoints from image2, shape [M, 2] or [1, M, 2]
        Optionally, may include matcher-specific fields like matches0, matches1, scores.
        """
        # Ensure a matcher exists; lazily fall back to a simple NN matcher.
        if not hasattr(self, "matcher") or self.matcher is None:
            try:
                from ..matching.nearest_neighbor import NearestNeighborMatcher

                self.matcher = NearestNeighborMatcher()
            except Exception as e:
                raise RuntimeError(
                    "No matcher set on extractor and failed to create default NearestNeighborMatcher."
                ) from e
        # Run feature extraction without gradients
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        data = {
            "descriptors0": desc0,
            "descriptors1": desc1,
        }

        response = self.matcher(data)

        m0 = response["matches0"][0]
        valid = m0 > -1
        # Ensure indices/masks live on the same device as keypoints before indexing
        kp_device = kp0.device
        if m0.device != kp_device:
            m0 = m0.to(kp_device)
            valid = valid.to(kp_device)

        mkpts0 = kp0[0, valid]
        mkpts1 = kp1[0, m0[valid]]
        # Ensure keypoints are detached and on CPU
        mkpts0 = mkpts0.detach().cpu()
        mkpts1 = mkpts1.detach().cpu()

        out = {
            "mkpts0": mkpts0,
            "mkpts1": mkpts1,
        }
        # pass-through common optional outputs if present
        for k in ("matches0", "matches1", "matching_scores0", "matching_scores1", "similarity"):
            if k in response:
                out[k] = response[k]
        return out

    @property
    def method_type(self) -> str:
        """Return the standardized method type string.

        Subclasses may override by setting class attribute `METHOD_TYPE` or
        by overriding this property. If not set, we infer from `has_detector`:
        - True  -> detect+describe
        - False -> descriptor-only
        """
        if self.METHOD_TYPE is not None:
            return self.METHOD_TYPE
        return MethodType.DETECT_DESCRIBE if self.has_detector else MethodType.DESCRIPTOR_ONLY
