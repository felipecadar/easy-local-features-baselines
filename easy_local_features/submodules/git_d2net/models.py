import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

import torch


def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    elif preprocessing == 'torch':
        image /= 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    else:
        raise ValueError('Unknown preprocessing parameter.')
    return image


def imshow_image(image, preprocessing=None):
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        mean = np.array([103.939, 116.779, 123.68])
        image = image + mean.reshape([3, 1, 1])
        # RGB -> BGR
        image = image[:: -1, :, :]
    elif preprocessing == 'torch':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std.reshape([3, 1, 1]) + mean.reshape([3, 1, 1])
        image *= 255.0
    else:
        raise ValueError('Unknown preprocessing parameter.')
    image = np.transpose(image, [1, 2, 0])
    image = np.round(image).astype(np.uint8)
    return image


def grid_positions(h, w, device, matrix=False):
    lines = torch.arange(
        0, h, device=device
    ).view(-1, 1).float().repeat(1, w)
    columns = torch.arange(
        0, w, device=device
    ).view(1, -1).float().repeat(h, 1)
    if matrix:
        return torch.stack([lines, columns], dim=0)
    else:
        return torch.cat([lines.view(1, -1), columns.view(1, -1)], dim=0)


def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos


def interpolate_dense_features(pos, dense_features, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(1), device=device)

    _, h, w = dense_features.size()

    i = pos[0, :]
    j = pos[1, :]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    if ids.size(0) == 0:
        raise Exception('No valid corners')

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
        w_top_left * dense_features[:, i_top_left, j_top_left] +
        w_top_right * dense_features[:, i_top_right, j_top_right] +
        w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left] +
        w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    pos = pos.cpu()
    descriptors = descriptors.cpu()
    ids = ids.cpu()

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack([
            torch.stack([i_top_left, j_top_left], dim=0),
            torch.stack([i_top_right, j_top_right], dim=0),
            torch.stack([i_bottom_left, j_bottom_left], dim=0),
            torch.stack([i_bottom_right, j_bottom_right], dim=0)
        ], dim=0).cpu()
        return [descriptors, pos, ids, corners]


def savefig(filepath, fig=None, dpi=None):
    # TomNorway - https://stackoverflow.com/a/53516034
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(filepath, pad_inches=0, bbox_inches='tight', dpi=dpi)

def process_multiscale(image, model, scales=[.5, 1, 2]):
    b, _, h_init, w_init = image.size()
    device = image.device
    assert(b == 1)

    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([
        model.dense_feature_extraction.num_channels, 0
    ])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.dense_feature_extraction(current_image)
        del current_image

        _, _, h, w = dense_features.size()

        # Sum the feature maps.
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        # Recover detections.
        detections = model.detection(dense_features)
        if banned is not None:
            banned = F.interpolate(banned.float(), size=[h, w]).bool()
            detections = torch.min(detections, ~banned)
            banned = torch.max(
                torch.max(detections, dim=1)[0].unsqueeze(1), banned
            )
        else:
            banned = torch.max(detections, dim=1)[0].unsqueeze(1)
        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        del detections

        # Recover displacements.
        displacements = model.localization(dense_features)[0].cpu()
        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        del displacements

        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )
        fmap_pos = fmap_pos[:, mask]
        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0)
        del mask, displacements_i, displacements_j

        fmap_keypoints = fmap_pos[1 :, :].float() + valid_displacements
        del valid_displacements

        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),
                dense_features[0]
            )
        except Exception:
            continue
        
        fmap_pos = fmap_pos[:, ids]
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        keypoints = upscale_positions(fmap_keypoints, scaling_steps=2)
        del fmap_keypoints

        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        fmap_pos = fmap_pos.cpu()
        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,
        ], dim=0)

        scores = dense_features[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ].cpu() / (idx + 1)
        del fmap_pos

        all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
        all_scores = torch.cat([all_scores, scores], dim=0)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned

    keypoints = all_keypoints.t()
    del all_keypoints
    scores = all_scores
    del all_scores
    descriptors = all_descriptors.t()
    del all_descriptors
    return keypoints, scores, descriptors

def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
        # RGB -> BGR
        image = image[:: -1, :, :]
        # Zero-center by mean pixel
        mean = np.array([103.939, 116.779, 123.68])
        image = image - mean.reshape([3, 1, 1])
    elif preprocessing == 'torch':
        image /= 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
    else:
        raise ValueError('Unknown preprocessing parameter.')
    return image

class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512

        self.use_relu = use_relu

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output


class D2Net(nn.Module):
    def __init__(self, use_relu=True):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_relu=use_relu
        )

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()


    def forward(self, batch):
        _, _, h, w = batch.size()
        dense_features = self.dense_feature_extraction(batch)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)