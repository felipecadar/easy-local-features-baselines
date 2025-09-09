from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from .basemodel import BaseExtractor, MethodType

"""PyTorch implementation of the SuperPoint model,
   derived from the TensorFlow re-implementation (2018).
   Authors: Rémi Pautrat, Paul-Edouard Sarlin
   https://github.com/rpautrat/SuperPoint
   The implementation of this model and its trained weights are made
   available under the MIT license.
"""


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def batched_nms(scores, nms_radius: int):
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(
            OrderedDict(
                [
                    ("conv", conv),
                    ("activation", activation),
                    ("bn", bn),
                ]
            )
        )


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
    sequences: list[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y


class OpenSPModel(nn.Module):
    checkpoint_url = "https://github.com/rpautrat/SuperPoint/raw/master/weights/superpoint_v6_from_tf.pth"  # noqa: E501

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.stride = 2 ** (len(self.conf.channels) - 2)
        channels = [1, *self.conf.channels[:-1]]

        backbone = []
        for i, c in enumerate(channels[1:], 1):
            layers = [VGGBlock(channels[i - 1], c, 3), VGGBlock(c, c, 3)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            backbone.append(nn.Sequential(*layers))
        self.backbone = nn.Sequential(*backbone)

        c = self.conf.channels[-1]
        self.detector = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.stride**2 + 1, 1, relu=False),
        )
        self.descriptor = nn.Sequential(
            VGGBlock(channels[-1], c, 3),
            VGGBlock(c, self.conf.descriptor_dim, 1, relu=False),
        )

        if conf.weights is not None and Path(conf.weights).exists():
            state_dict = torch.load(conf.weights, map_location="cpu")
        else:
            state_dict = torch.hub.load_state_dict_from_url(self.checkpoint_url)
        self.load_state_dict(state_dict)

    def forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)
        features = self.backbone(image)
        descriptors_dense = torch.nn.functional.normalize(self.descriptor(features), p=2, dim=1)

        # Decode the detection scores
        scores = self.detector(features)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, self.stride, self.stride)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * self.stride, w * self.stride)
        scores = batched_nms(scores, self.conf.nms_radius)

        # Discard keypoints near the image borders
        if self.conf.remove_borders:
            pad = self.conf.remove_borders
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        if b > 1:
            idxs = torch.where(scores > self.conf.detection_threshold)
            mask = idxs[0] == torch.arange(b, device=scores.device)[:, None]
        else:  # Faster shortcut
            scores = scores.squeeze(0)
            idxs = torch.where(scores > self.conf.detection_threshold)

        # Convert (i, j) to (x, y)
        keypoints_all = torch.stack(idxs[-2:], dim=-1).flip(1).float()
        scores_all = scores[idxs]

        keypoints = []
        scores = []
        for i in range(b):
            if b > 1:
                k = keypoints_all[mask[i]]
                s = scores_all[mask[i]]
            else:
                k = keypoints_all
                s = scores_all
            if self.conf.top_k is not None:
                k, s = select_top_k_keypoints(k, s, self.conf.top_k)

            keypoints.append(k)
            scores.append(s)

        if self.conf.force_num_keypoints:
            keypoints = pad_and_stack(
                keypoints,
                self.conf.top_k,
                -2,
                mode="random_c",
                bounds=(
                    0,
                    data.get("image_size", torch.tensor(image.shape[-2:])).min().item(),
                ),
            )
            scores = pad_and_stack(scores, self.conf.top_k, -1, mode="zeros")
        else:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)

        if len(keypoints) == 1 or self.conf.force_num_keypoints:
            # Batch sampling of the descriptors
            desc = sample_descriptors(keypoints, descriptors_dense, self.stride)
        else:
            desc = [sample_descriptors(k[None], d[None], self.stride)[0] for k, d in zip(keypoints, descriptors_dense)]

        pred = {
            "keypoints": keypoints + 0.5,
            "keypoint_scores": scores,
            "descriptors": desc.transpose(-1, -2),
            "descriptors_dense": descriptors_dense,
        }

        return pred


class SuperPoint_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    default_conf = {
        "top_k": 2048,
        "nms_radius": 4,
        "force_num_keypoints": False,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
        "weights": None,  # local path of pretrained weights
    }

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device = torch.device("cpu")
        self.matcher = NearestNeighborMatcher()
        self.model = OpenSPModel(conf)
        self.model.to(self.device)
        self.model.eval()

    def detectAndCompute(self, img, return_dict=None):
        img = ops.prepareImage(img, gray=True).to(self.device)
        response = self.model({"image": img})  # 'keypoints', 'keypoint_scores', 'descriptors'

        if return_dict:
            return response

        return response["keypoints"], response["descriptors"]

    def detect(self, img, op=None):
        return self.detectAndCompute(img, return_dict=True)["keypoints"]

    def compute(self, img, keypoints):
        raise NotImplemented

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    @property
    def has_detector(self):
        return True


if __name__ == "__main__":
    from easy_local_features.utils import io, ops, vis

    method = SuperPoint_baseline()
    # method.to('mps')

    img0 = io.fromPath("test/assets/megadepth0.jpg")
    img1 = io.fromPath("test/assets/megadepth1.jpg")

    nn_matches = method.match(img0, img1)

    vis.plot_pair(img0, img1)
    vis.plot_matches(nn_matches["mkpts0"], nn_matches["mkpts1"])
    vis.add_text("SuperPoint Open")

    vis.show()
