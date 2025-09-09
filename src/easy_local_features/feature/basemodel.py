from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, overload, Protocol, runtime_checkable

import numpy as np
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


# Public type aliases used across feature modules
ImageLike = Union[str, np.ndarray, torch.Tensor]
KeypointsTensor = torch.Tensor
DescriptorsTensor = torch.Tensor


@runtime_checkable
class DetectorProtocol(Protocol):
    """Minimal protocol for an external detector used by descriptors.

    Any object exposing `detect(image) -> Tensor[[N,2] or [1,N,2]]` is accepted.
    """

    def detect(self, image: ImageLike) -> torch.Tensor:  # pragma: no cover - protocol
        ...


class BaseExtractor(ABC):
    """Base interface for feature extractors.

    Implementations may provide a detector and descriptor (detect+describe) or
    just a descriptor (descriptor-only). Public methods are standardized so
    downstream code can work with any extractor uniformly.

    Key conventions:
    - Images can be file paths, numpy arrays (H×W×C or H×W), or torch tensors
        ([C,H,W] or [B,C,H,W]). Implementations should normalize inputs.
    - Keypoints are in pixel coordinates (x, y). Batched outputs are shaped
        [B, N, 2]; non-batched may be [N, 2] or [1, N, 2].
    - Descriptors are shaped [B, N, D] (or [N, D]).
    """
    # Optional override in subclasses; if None, inferred from `has_detector`.
    METHOD_TYPE: Optional[str] = None

    @abstractmethod
    @overload
    def detectAndCompute(self, image: ImageLike, return_dict: bool) -> Dict[str, torch.Tensor]:
        ...

    @overload
    def detectAndCompute(self, image: ImageLike, return_dict: bool = False) -> Tuple[KeypointsTensor, DescriptorsTensor]:
        ...

    def detectAndCompute(self, image: ImageLike, return_dict: bool = False) -> Union[
        Tuple[KeypointsTensor, DescriptorsTensor], Dict[str, torch.Tensor]
    ]:
        """Run detection and description on an image.

        Args:
            image: Path, numpy array, or tensor. May be batched.
            return_dict: If True, return a dict with keys 'keypoints' and 'descriptors'.

        Returns:
            Tuple[Tensor, Tensor] or dict: keypoints and descriptors. Shapes follow
            the conventions in the class docstring.
        """
        raise NotImplementedError("Every BaseExtractor must implement the detectAndCompute method.")

    @abstractmethod
    def detect(self, image: ImageLike) -> KeypointsTensor:
        """Detect keypoints on an image.

        Args:
            image: Path, numpy array, or tensor. May be batched.

        Returns:
            Tensor: keypoints shaped [B, N, 2] or [N, 2].
        """
        raise NotImplementedError("Every BaseExtractor must implement the detect method.")

    @abstractmethod
    def compute(self, image: ImageLike, keypoints: KeypointsTensor) -> Union[
        DescriptorsTensor, Tuple[KeypointsTensor, DescriptorsTensor]
    ]:
        """Compute descriptors for provided keypoints.

        Args:
            image: Path, numpy array, or tensor. May be batched.
            keypoints: Tensor of keypoints [B, N, 2] or [N, 2].

        Returns:
            Tensor or Tuple[Tensor, Tensor]: descriptors [B, N, D] or optionally
            a pair (keypoints, descriptors) if the method refines keypoints.
        """
        raise NotImplementedError("Every BaseExtractor must implement the compute method.")

    @abstractmethod
    def to(self, device: Union[torch.device, str]) -> "BaseExtractor":
        """Move internal models to the specified device and return self."""
        raise NotImplementedError("Every BaseExtractor must implement the to method.")

    @property
    @abstractmethod
    def has_detector(self) -> bool:
        """Whether this extractor provides its own detector (True) or not (False)."""
        raise NotImplementedError("Every BaseExtractor must implement the has_detector property.")

    def __call__(self, data: Dict[str, ImageLike]) -> Dict[str, torch.Tensor]:
        """Forward entry point compatible with some pipelines.

        Expects a dict with key 'image' and returns the same as detectAndCompute
        with return_dict=True for convenience.
        """
        return self.detectAndCompute(data["image"], return_dict=True)

    def addDetector(self, detector: DetectorProtocol) -> None:
        """Attach an external detector to a descriptor-only extractor.

        Replaces this instance's detect() and detectAndCompute() to use the
        provided detector, while compute() remains from the current extractor.

        Args:
            detector: Object exposing detect(image) -> keypoints.
        """
        def detectAndCompute(image: ImageLike, return_dict: bool = False) -> Union[
            Tuple[KeypointsTensor, DescriptorsTensor], Dict[str, torch.Tensor]
        ]:
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
    def match(self, image1: ImageLike, image2: ImageLike) -> Dict[str, torch.Tensor]:
        """Match two images using this extractor and its matcher.

        Args:
            image1: First image (path/array/tensor).
            image2: Second image (path/array/tensor).

        Returns:
            dict with at least:
                - mkpts0: matched keypoints from image1, [M, 2] or [1, M, 2]
                - mkpts1: matched keypoints from image2, [M, 2] or [1, M, 2]
            and optionally matcher-specific items: matches0, matches1,
            matching_scores0, matching_scores1, similarity.
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

        Subclasses may override by setting class attribute `METHOD_TYPE` or by
        overriding this property. If not set, we infer from `has_detector`:
        - True  -> detect+describe
        - False -> descriptor-only
        """
        if self.METHOD_TYPE is not None:
            return self.METHOD_TYPE
        return MethodType.DETECT_DESCRIBE if self.has_detector else MethodType.DESCRIPTOR_ONLY
    