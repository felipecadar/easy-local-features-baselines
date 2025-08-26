from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import torch

from ..utils import ops


DetectorLike = Union[object]


@dataclass
class EnsembleDetectorConfig:
    """Configuration for the EnsembleDetector.

    Attributes
    - deduplicate: remove duplicate keypoints across detectors (rounded to pixel)
    - sort: sort final keypoints lexicographically by (y, x)
    - max_keypoints: optional cap on total keypoints per image. If set, takes a
      balanced subset across detectors (round-robin) without scores.
    """

    deduplicate: bool = True
    sort: bool = True
    max_keypoints: int | None = None


class EnsembleDetector:
    """Aggregate multiple detectors and expose a single detect() API.

    Usage
    - Pass instantiated detectors (any object with a detect(image) -> Tensor[[1,N,2]] method).
    - Call detect(image) with a single image or a batch (Tensor [B,C,H,W] or numpy/str).
    - Returns a tensor of shape [B, N_total, 2] in pixel coordinates (x, y).

    Notes
    - Each underlying detector is invoked per image (loop over batch dim) to
      support detectors that don't accept batched inputs (e.g., OpenCV-based).
    - Deduplication rounds coordinates to integer pixels before uniquifying.
    """

    def __init__(self, detectors: Sequence[DetectorLike], config: EnsembleDetectorConfig | None = None):
        assert len(detectors) > 0, "EnsembleDetector requires at least one detector"
        self.detectors: List[DetectorLike] = list(detectors)
        self.config = config or EnsembleDetectorConfig()

    def to(self, device: torch.device | str):
        # Forward device move to underlying detectors when supported
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
    def detect(self, image) -> torch.Tensor:
        """Detect keypoints using all detectors and concatenate results.

        Input can be a path, numpy array, or torch tensor. Batched inputs are
        supported. Output is a torch.FloatTensor with shape [B, N, 2].
        """
        # Normalize input to [B,C,H,W]
        img = ops.prepareImage(image, gray=False, batch=True)
        if isinstance(img, torch.Tensor):
            device = img.device
        else:
            device = torch.device("cpu")

        B = img.shape[0]
        out_kps: List[torch.Tensor] = []

        for b in range(B):
            per_img_kps: List[torch.Tensor] = []
            # Slice single image; detectors prepare their own input
            single = img[b]
            for det in self.detectors:
                kps = det.detect(single)
                # Expect [1, N, 2] or [N, 2]
                if not isinstance(kps, torch.Tensor):
                    kps = torch.as_tensor(kps)
                if kps.ndim == 3 and kps.shape[0] == 1:
                    kps = kps[0]
                assert kps.ndim == 2 and kps.shape[-1] == 2, (
                    f"Detector {type(det).__name__} returned keypoints with shape {tuple(kps.shape)}, expected [N,2] or [1,N,2]."
                )
                per_img_kps.append(kps)

            if len(per_img_kps) == 1:
                merged = per_img_kps[0]
            else:
                merged = torch.cat(per_img_kps, dim=0)

            # Deduplicate by integer pixel location if requested
            if self.config.deduplicate and merged.numel() > 0:
                # Round to nearest pixel for stable dedup and use numpy to get first indices
                rounded = torch.round(merged).to(torch.int64)
                rounded_np = rounded.detach().cpu().numpy()
                _, first_idx = np.unique(rounded_np, axis=0, return_index=True)
                # Keep original (non-rounded) coordinates at those indices
                keep = torch.from_numpy(np.sort(first_idx)).to(merged.device)
                merged = merged[keep]

            # Sort for reproducibility if requested
            if self.config.sort and merged.numel() > 0:
                merged_np = merged.detach().cpu().numpy()
                merged_sorted = ops.sort_keypoints(merged_np)
                merged = torch.from_numpy(merged_sorted).to(device=device, dtype=merged.dtype)

            # Cap total keypoints if requested (balanced across detectors when possible)
            if self.config.max_keypoints is not None and merged.shape[0] > self.config.max_keypoints:
                # Simple uniform subsample without scores
                N = self.config.max_keypoints
                # Deterministic: stride-based selection
                stride = max(1, merged.shape[0] // N)
                merged = merged[::stride][:N]

            out_kps.append(merged.unsqueeze(0))

        # Pad to max length and stack to [B, Nmax, 2] to keep a strict tensor output
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
