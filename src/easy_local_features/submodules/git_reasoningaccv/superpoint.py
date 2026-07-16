"""
Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork

Adapted by Philipp Lindenberger (Phil26AT)
Vendored from verlab/DescriptorReasoning_ACCV_2024 (reasoning/features/superpoint.py)
"""

import torch
from torch import nn

import math
from typing import List, Optional, Tuple
from omegaconf import OmegaConf

def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])

def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: Tuple[int] = (None, None),
):
    shape = list(x.shape)
    d = x.shape[pad_dim]
    assert d <= length
    if d == length:
        return x
    shape[pad_dim] = length - d

    low, high = bounds

    if mode == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        xn = torch.empty(*shape, device=x.device).uniform_(low, high)
    elif mode == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        xn = torch.cat(
            [
                torch.empty(*shape[:-1], 1, device=x.device).uniform_(
                    x[..., i].min() if d > 0 else low,
                    x[..., i].max() if d > 0 else high,
                )
                for i in range(shape[-1])
            ],
            dim=-1,
        )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)


def pad_and_stack(
    sequences: List[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y

def simple_nms(scores, radius):
    """Perform non maximum suppression on the heatmap using max-pooling.
    This method does not suppress contiguous points that have the same score.
    Args:
        scores: the score heatmap of size `(B, H, W)`.
        radius: an integer scalar, the radius of the NMS window.
    """

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=radius * 2 + 1, stride=1, padding=radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    indices = torch.multinomial(scores, k, replacement=False)
    return keypoints[indices], scores[indices]


def soft_argmax_refinement(keypoints, scores, radius: int):
    width = 2 * radius + 1
    sum_ = torch.nn.functional.avg_pool2d(
        scores[:, None], width, 1, radius, divisor_override=1
    )
    ar = torch.arange(-radius, radius + 1).to(scores)
    kernel_x = ar[None].expand(width, -1)[None, None]
    dx = torch.nn.functional.conv2d(scores[:, None], kernel_x, padding=radius)
    dy = torch.nn.functional.conv2d(
        scores[:, None], kernel_x.transpose(2, 3), padding=radius
    )
    dydx = torch.stack([dy[:, 0], dx[:, 0]], -1) / sum_[:, 0, :, :, None]
    refined_keypoints = []
    for i, kpts in enumerate(keypoints):
        delta = dydx[i][tuple(kpts.t())]
        refined_keypoints.append(kpts.float() + delta)
    return refined_keypoints


# Legacy (broken) sampling of the descriptors
def sample_descriptors(keypoints, descriptors, s):
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


# The original keypoint sampling is incorrect. We patch it here but
# keep the original one above for legacy.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(torch.nn.Module):
    default_conf = {
        "has_detector": True,
        "has_descriptor": True,
        "descriptor_dim": 256,
        # Inference
        "sparse_outputs": True,
        "dense_outputs": False,
        "nms_radius": 4,
        "refinement_radius": 0,
        "detection_threshold": 0.005,
        "max_num_keypoints": -1,
        "max_num_keypoints_val": None,
        "force_num_keypoints": False,
        "randomize_keypoints_training": False,
        "remove_borders": 4,
        "legacy_sampling": False,  # True to use the old broken sampling
    }
    required_data_keys = ["image"]

    checkpoint_url = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth"  # noqa: E501

    def __init__(self, conf):
        super().__init__()
        self.training = False
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        if conf.has_detector:
            self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        if conf.has_descriptor:
            self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
            self.convDb = nn.Conv2d(
                c5, conf.descriptor_dim, kernel_size=1, stride=1, padding=0
            )

        self.load_state_dict(
            torch.hub.load_state_dict_from_url(str(self.checkpoint_url)), strict=False
        )

    def _encode(self, image):
        """Shared encoder. Expects a [B,1,H,W] or [B,3,H,W] image in [0,1]."""
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        return x

    def forward(self, data):
        image = data["image"]
        x = self._encode(image)

        pred = {}
        if self.conf.has_detector:
            # Compute the dense keypoint scores
            cPa = self.relu(self.convPa(x))
            scores = self.convPb(cPa)
            scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
            b, c, h, w = scores.shape
            scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
            scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
            pred["keypoint_scores"] = dense_scores = scores
        if self.conf.has_descriptor:
            # Compute the dense descriptors
            cDa = self.relu(self.convDa(x))
            dense_desc = self.convDb(cDa)
            dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)
            pred["descriptors"] = dense_desc

        if self.conf.sparse_outputs:
            assert self.conf.has_detector and self.conf.has_descriptor

            scores = simple_nms(scores, self.conf.nms_radius)

            # Discard keypoints near the image borders
            if self.conf.remove_borders:
                scores[:, : self.conf.remove_borders] = -1
                scores[:, :, : self.conf.remove_borders] = -1
                if "image_size" in data:
                    for i in range(scores.shape[0]):
                        w, h = data["image_size"][i]
                        scores[i, int(h.item()) - self.conf.remove_borders :] = -1
                        scores[i, :, int(w.item()) - self.conf.remove_borders :] = -1
                else:
                    scores[:, -self.conf.remove_borders :] = -1
                    scores[:, :, -self.conf.remove_borders :] = -1

            # Extract keypoints
            best_kp = torch.where(scores > self.conf.detection_threshold)
            scores = scores[best_kp]

            # Separate into batches
            keypoints = [
                torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)
            ]
            scores = [scores[best_kp[0] == i] for i in range(b)]

            # Keep the k keypoints with highest score
            max_kps = self.conf.max_num_keypoints

            # for val we allow different
            if not self.training and self.conf.max_num_keypoints_val is not None:
                max_kps = self.conf.max_num_keypoints_val

            # Keep the k keypoints with highest score
            if max_kps > 0:
                if self.conf.randomize_keypoints_training and self.training:
                    # instead of selecting top-k, sample k by score weights
                    keypoints, scores = list(
                        zip(
                            *[
                                sample_k_keypoints(k, s, max_kps)
                                for k, s in zip(keypoints, scores)
                            ]
                        )
                    )
                else:
                    keypoints, scores = list(
                        zip(
                            *[
                                top_k_keypoints(k, s, max_kps)
                                for k, s in zip(keypoints, scores)
                            ]
                        )
                    )
                keypoints, scores = list(keypoints), list(scores)

            if self.conf["refinement_radius"] > 0:
                keypoints = soft_argmax_refinement(
                    keypoints, dense_scores, self.conf["refinement_radius"]
                )

            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]

            if self.conf.force_num_keypoints:
                keypoints = pad_and_stack(
                    keypoints,
                    max_kps,
                    -2,
                    mode="random_c",
                    bounds=(
                        0,
                        data.get("image_size", torch.tensor(image.shape[-2:]))
                        .min()
                        .item(),
                    ),
                )
                scores = pad_and_stack(scores, max_kps, -1, mode="zeros")
            else:
                keypoints = torch.stack(keypoints, 0)
                scores = torch.stack(scores, 0)

            # Extract descriptors
            if (len(keypoints) == 1) or self.conf.force_num_keypoints:
                # Batch sampling of the descriptors
                if self.conf.legacy_sampling:
                    desc = sample_descriptors(keypoints, dense_desc, 8)
                else:
                    desc = sample_descriptors_fix_sampling(keypoints, dense_desc, 8)
            else:
                if self.conf.legacy_sampling:
                    desc = [
                        sample_descriptors(k[None], d[None], 8)[0]
                        for k, d in zip(keypoints, dense_desc)
                    ]
                else:
                    desc = [
                        sample_descriptors_fix_sampling(k[None], d[None], 8)[0]
                        for k, d in zip(keypoints, dense_desc)
                    ]

            pred = {
                "keypoints": keypoints + 0.5,
                "keypoint_scores": scores,
                "descriptors": desc.transpose(-1, -2),
            }

            if self.conf.dense_outputs:
                pred["dense_descriptors"] = dense_desc

        return pred

    def forward_desc(self, data, keypoints):
        """Compute descriptors at arbitrary keypoints.

        keypoints: [B, N, 2] (x, y) pixel coordinates in the same convention as the
        keypoints returned by forward() (i.e. including the +0.5 center offset).
        """
        image = data["image"]
        x = self._encode(image)

        cDa = self.relu(self.convDa(x))
        dense_desc = self.convDb(cDa)
        dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)

        # forward() samples descriptors at the raw (pre +0.5) keypoint locations
        kpts = keypoints - 0.5
        if self.conf.legacy_sampling:
            desc = sample_descriptors(kpts, dense_desc, 8)
        else:
            desc = sample_descriptors_fix_sampling(kpts, dense_desc, 8)

        return {
            "keypoints": keypoints,
            "descriptors": desc.transpose(-1, -2),
        }
