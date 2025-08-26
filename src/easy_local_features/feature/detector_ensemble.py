from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import torch

from ..utils import ops
from .basemodel import ImageLike


DetectorLike = Union[object]


@dataclass
class EnsembleDetectorConfig:
    """Configuration for the `EnsembleDetector`.

    Args:
        deduplicate: If True, remove duplicate keypoints across detectors by
            rounding coordinates to the nearest pixel and uniquifying.
        sort: If True, sort final keypoints lexicographically for stable order.
        max_keypoints: Optional cap on total keypoints per image. When exceeded,
            a simple deterministic subsample (stride selection) is applied.
    """

    deduplicate: bool = True
    sort: bool = True
    max_keypoints: int | None = None


class EnsembleDetector:
    """Aggregate multiple detectors and expose a single `detect()` API.

    Typical usage:
        - Pass instantiated detectors (any object exposing `detect(image)` that
          returns `[1, N, 2]` or `[N, 2]` keypoints in pixel coordinates).
        - Call `detect(image)` with a single image or a batch (`[B,C,H,W]`, NumPy, or path).
        - Get a tensor `[B, N_total, 2]` (padded across batches) of (x, y) keypoints.

    Notes:
        - Each underlying detector is invoked per image (looping over batch dim)
          to support non-batched detectors (e.g., OpenCV ORB).
        - Deduplication rounds keypoints to integer pixel locations before uniquifying.
    """

    def __init__(self, detectors: Sequence[DetectorLike], config: EnsembleDetectorConfig | None = None):
        assert len(detectors) > 0, "EnsembleDetector requires at least one detector"
        self.detectors: List[DetectorLike] = list(detectors)
        self.config = config or EnsembleDetectorConfig()

    def to(self, device: torch.device | str):
        """Move underlying detectors to a device when supported and return self."""
        for d in self.detectors:
            to_fn = getattr(d, "to", None)
            if callable(to_fn):
                try:
                    to_fn(device)
                except Exception:
                    # Some detectors (e.g., OpenCV) may not support .to(); ignore.
                    pass
        return self

    @torch.inference_mode()
    def detect(self, image: ImageLike) -> torch.Tensor:
        """Detect and merge keypoints from all detectors.

        Args:
            image: Path, numpy array, or torch tensor. Can be batched.

        Returns:
            torch.FloatTensor: `[B, N, 2]` keypoints (x, y) in pixel coords.
            For single images, `B==1`. Across a batch, per-image lists are padded
            to the maximum length in the batch.
        """
        # Normalize input to [B,C,H,W]
        img = ops.prepareImage(image, gray=False, batch=True)
        device = img.device if isinstance(img, torch.Tensor) else torch.device("cpu")

        B = img.shape[0]
        out_kps: List[torch.Tensor] = []

        for b in range(B):
            per_img_kps: List[torch.Tensor] = []
            single = img[b]
            for det in self.detectors:
                kps = det.detect(single)
                if not isinstance(kps, torch.Tensor):
                    kps = torch.as_tensor(kps)
                if kps.ndim == 3 and kps.shape[0] == 1:
                    kps = kps[0]
                assert kps.ndim == 2 and kps.shape[-1] == 2, (
                    f"Detector {type(det).__name__} returned keypoints with shape {tuple(kps.shape)}, expected [N,2] or [1,N,2]."
                )
                per_img_kps.append(kps)

            merged = per_img_kps[0] if len(per_img_kps) == 1 else torch.cat(per_img_kps, dim=0)

            if self.config.deduplicate and merged.numel() > 0:
                rounded = torch.round(merged).to(torch.int64)
                rounded_np = rounded.detach().cpu().numpy()
                _, first_idx = np.unique(rounded_np, axis=0, return_index=True)
                keep = torch.from_numpy(np.sort(first_idx)).to(merged.device)
                merged = merged[keep]

            if self.config.sort and merged.numel() > 0:
                merged_np = merged.detach().cpu().numpy()
                merged_sorted = ops.sort_keypoints(merged_np)
                merged = torch.from_numpy(merged_sorted).to(device=device, dtype=merged.dtype)

            if self.config.max_keypoints is not None and merged.shape[0] > self.config.max_keypoints:
                N = self.config.max_keypoints
                stride = max(1, merged.shape[0] // N)
                merged = merged[::stride][:N]

            out_kps.append(merged.unsqueeze(0))

        max_len = max(k.shape[1] for k in out_kps)
        if max_len == 0:
            return torch.zeros((B, 0, 2), dtype=torch.float32, device=device)

        padded = []
        for k in out_kps:
            if k.shape[1] < max_len:
                pad = torch.zeros((1, max_len - k.shape[1], 2), dtype=k.dtype, device=k.device)
                padded.append(torch.cat([k, pad], dim=1))
            else:
                padded.append(k)
        return torch.cat(padded, dim=0)
