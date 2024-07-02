from ..matching.nearest_neighbor import NearestNeighborMatcher
from omegaconf import OmegaConf
from .basemodel import BaseExtractor
from ..utils.download import downloadModel
from ..utils import ops

from typing import Optional

"""
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

Described in:
    SuperPoint: Self-Supervised Interest Point Detection and Description,
    Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich, CVPRW 2018.

Original code: github.com/MagicLeapResearch/SuperPointPretrainedNetwork

Adapted by Philipp Lindenberger (Phil26AT)
"""

import torch
from torch import nn

def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: tuple[int] = (None, None),
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

class Net(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        descriptor_dim = 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, descriptor_dim, kernel_size=1, stride=1, padding=0
        )
    
    def forward(self, data):
        image = data["image"]
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image * scale).sum(1, keepdim=True)

        # Shared Encoder
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

        pred = {}
        
        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, c, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        pred["keypoint_scores"] = dense_scores = scores

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        dense_desc = self.convDb(cDa)
        dense_desc = torch.nn.functional.normalize(dense_desc, p=2, dim=1)
        pred["descriptors"] = dense_desc

        if self.conf.sparse_outputs:
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
            max_kps = self.conf.top_k

            # Keep the k keypoints with highest score
            if max_kps > 0:
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
    

class SuperPoint_baseline(BaseExtractor):
    default_conf = {
        "top_k": -1,
        "sparse_outputs": True,
        "dense_outputs": False,
        "nms_radius": 4,
        "refinement_radius": 0,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "legacy_sampling": True,  # True to use the old broken sampling
    }

    checkpoint_url = "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth"  # noqa: E501

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.DEV = torch.device('cpu')
        self.matcher = NearestNeighborMatcher()
        self.model = Net(conf)
        self.model.load_state_dict(
            torch.hub.load_state_dict_from_url(str(self.checkpoint_url)), strict=False
        )
        self.model.eval()

    def detectAndCompute(self, img, return_dict=False):
        img = ops.prepareImage(img).to(self.DEV)
        pred = self.model({"image": img})
        
        if return_dict:
            return pred
        
        # if batch size is 1, remove the batch dimension
        if pred['keypoints'].shape[0] == 1:
            pred['keypoints'] = pred['keypoints'][0]
            pred['descriptors'] = pred['descriptors'][0]
        
        return pred['keypoints'], pred['descriptors']
    
    def detect(self, img):
        return self.detectAndCompute(img)[0]
    
    def compute(self, img, keypoints):
        raise NotImplementedError("This method is not implemented.")

    def to(self, device):
        self.model = self.model.to(device)
        self.DEV = device

    def match(self, image1, image2):
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        
        data = {
            "descriptors0": desc0.unsqueeze(0),
            "descriptors1": desc1.unsqueeze(0),
        }
        
        response = self.matcher(data)
        
        m0 = response['matches0'][0]
        valid = m0 > -1
        
        mkpts0 = kp0[valid]
        mkpts1 = kp1[m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }
        
    @property
    def has_detector(self):
        return True