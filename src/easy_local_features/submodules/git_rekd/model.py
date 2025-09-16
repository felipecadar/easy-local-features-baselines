import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_multiple_channels(num_channels, sigma):
    r = 2 * sigma
    size = 2 * r + 1
    size = int(math.ceil(size))
    x = torch.arange(0, size, 1, dtype=torch.float)
    y = x.unsqueeze(1)
    x0 = y0 = r

    gaussian = torch.exp(-1 * (((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma**2)))) / (
        (2 * math.pi * (sigma**2)) ** 0.5
    )
    gaussian = gaussian.to(dtype=torch.float32)

    weights = torch.zeros((num_channels, num_channels, size, size), dtype=torch.float32)
    for i in range(num_channels):
        weights[i, i, :, :] = gaussian

    return weights


def apply_nms(score_map, size):
    from scipy.ndimage import maximum_filter

    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))
    return score_map


def remove_borders(images, borders):
    ## input [B,C,H,W]
    shape = images.shape

    if len(shape) == 4:
        for batch_id in range(shape[0]):
            images[batch_id, :, 0:borders, :] = 0
            images[batch_id, :, :, 0:borders] = 0
            images[batch_id, :, shape[2] - borders : shape[2], :] = 0
            images[batch_id, :, :, shape[3] - borders : shape[3]] = 0
    elif len(shape) == 2:
        images[0:borders, :] = 0
        images[:, 0:borders] = 0
        images[shape[0] - borders : shape[0], :] = 0
        images[:, shape[1] - borders : shape[1]] = 0
    else:
        print("Not implemented")
        exit()

    return images


def getAff(x, y, H):
    h11 = H[0, 0]
    h12 = H[0, 1]
    h13 = H[0, 2]
    h21 = H[1, 0]
    h22 = H[1, 1]
    h23 = H[1, 2]
    h31 = H[2, 0]
    h32 = H[2, 1]
    h33 = H[2, 2]
    fxdx = h11 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h31 / (h31 * x + h32 * y + h33) ** 2
    fxdy = h12 / (h31 * x + h32 * y + h33) - (h11 * x + h12 * y + h13) * h32 / (h31 * x + h32 * y + h33) ** 2

    fydx = h21 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h31 / (h31 * x + h32 * y + h33) ** 2
    fydy = h22 / (h31 * x + h32 * y + h33) - (h21 * x + h22 * y + h23) * h32 / (h31 * x + h32 * y + h33) ** 2

    Aff = [[fxdx, fxdy], [fydx, fydy]]

    return np.asarray(Aff)


def apply_homography_to_points(points, h):
    new_points = []

    for point in points:
        new_point = h.dot([point[0], point[1], 1.0])

        tmp = point[2] ** 2 + np.finfo(np.float32).eps

        Mi1 = [[1 / tmp, 0], [0, 1 / tmp]]
        Mi1_inv = np.linalg.inv(Mi1)
        Aff = getAff(point[0], point[1], h)

        BMB = np.linalg.inv(np.dot(Aff, np.dot(Mi1_inv, np.matrix.transpose(Aff))))

        [e, _] = np.linalg.eig(BMB)
        new_radious = 1 / ((e[0] * e[1]) ** 0.5) ** 0.5

        new_point = [
            new_point[0] / new_point[2],
            new_point[1] / new_point[2],
            new_radious,
            point[3],
        ]
        new_points.append(new_point)

    return np.asarray(new_points)


def find_index_higher_scores(map, num_points=1000, threshold=-1):
    # Best n points
    if threshold == -1:
        flatten = map.flatten()
        order_array = np.sort(flatten)

        order_array = np.flip(order_array, axis=0)

        if order_array.shape[0] < num_points:
            num_points = order_array.shape[0]

        threshold = order_array[num_points - 1]

        if threshold <= 0.0:
            ### This is the problem case which derive smaller number of keypoints than the argument "num_points".
            indexes = np.argwhere(order_array > 0.0)

            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes) - 1]]

    indexes = np.argwhere(map >= threshold)

    return indexes[:num_points]


def get_point_coordinates(map, scale_value=1.0, num_points=1000, threshold=-1, order_coord="xysr"):
    ## input numpy array score map : [H, W]
    indexes = find_index_higher_scores(map, num_points=num_points, threshold=threshold)
    new_indexes = []
    for ind in indexes:
        scores = map[ind[0], ind[1]]
        if order_coord == "xysr":
            tmp = [ind[1], ind[0], scale_value, scores]
        elif order_coord == "yxsr":
            tmp = [ind[0], ind[1], scale_value, scores]

        new_indexes.append(tmp)

    indexes = np.asarray(new_indexes)

    return np.asarray(indexes)


class REKD(torch.nn.Module):
    def __init__(self, args, device):
        super(REKD, self).__init__()
        from e2cnn import gspaces, nn

        self.pyramid_levels = 3
        self.factor_scaling = args.factor_scaling_pyramid

        # Smooth Gausian Filter
        num_channels = 1  ## gray scale image
        self.gaussian_avg = gaussian_multiple_channels(num_channels, 1.5)

        r2_act = gspaces.Rot2dOnR2(N=args.group_size)

        self.feat_type_in = nn.FieldType(
            r2_act, num_channels * [r2_act.trivial_repr]
        )  ## input 1 channels (gray scale image)

        feat_type_out1 = nn.FieldType(r2_act, args.dim_first * [r2_act.regular_repr])
        feat_type_out2 = nn.FieldType(r2_act, args.dim_second * [r2_act.regular_repr])
        feat_type_out3 = nn.FieldType(r2_act, args.dim_third * [r2_act.regular_repr])

        feat_type_ori_est = nn.FieldType(r2_act, [r2_act.regular_repr])

        self.block1 = nn.SequentialModule(
            nn.R2Conv(self.feat_type_in, feat_type_out1, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out1),
            nn.ReLU(feat_type_out1, inplace=True),
        )
        self.block2 = nn.SequentialModule(
            nn.R2Conv(feat_type_out1, feat_type_out2, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out2),
            nn.ReLU(feat_type_out2, inplace=True),
        )
        self.block3 = nn.SequentialModule(
            nn.R2Conv(feat_type_out2, feat_type_out3, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(feat_type_out3),
            nn.ReLU(feat_type_out3, inplace=True),
        )

        self.ori_learner = nn.SequentialModule(
            nn.R2Conv(
                feat_type_out3, feat_type_ori_est, kernel_size=1, padding=0, bias=False
            )  ## Channel pooling by 8*G -> 1*G conv.
        )
        self.softmax = torch.nn.Softmax(dim=1)

        self.gpool = nn.GroupPooling(feat_type_out3)
        self.last_layer_learner = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=args.dim_third * self.pyramid_levels),
            torch.nn.Conv2d(
                in_channels=args.dim_third * self.pyramid_levels,
                out_channels=1,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.ReLU(inplace=True),  ## clamp to make the scores positive values.
        )

        self.dim_third = args.dim_third
        self.group_size = args.group_size
        self.exported = False

    def export(self):
        from e2cnn import nn

        for name, module in dict(self.named_modules()).copy().items():
            if isinstance(module, nn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)

        self.exported = True

    def forward(self, input_data):
        features_key, features_o = self.compute_features(input_data)

        return features_key, features_o

    def compute_features(self, input_data):
        B, _, H, W = input_data.shape

        for idx_level in range(self.pyramid_levels):
            with torch.no_grad():
                input_data_resized = self._resize_input_image(input_data, idx_level, H, W)

            if H > 2500 or W > 2500:
                features_t, features_o = self._forwarding_networks_divide_grid(input_data_resized)
            else:
                features_t, features_o = self._forwarding_networks(input_data_resized)

            features_t = F.interpolate(features_t, size=(H, W), align_corners=True, mode="bilinear")
            features_o = F.interpolate(features_o, size=(H, W), align_corners=True, mode="bilinear")

            if idx_level == 0:
                features_key = features_t
                features_ori = features_o
            else:
                features_key = torch.cat([features_key, features_t], axis=1)
                features_ori = torch.add(features_ori, features_o)

        features_key = self.last_layer_learner(features_key)
        features_ori = self.softmax(features_ori)

        return features_key, features_ori

    def _forwarding_networks(self, input_data_resized):
        from e2cnn import nn

        # wrap the input tensor in a GeometricTensor (associate it with the input type)
        features_t = (
            nn.GeometricTensor(input_data_resized, self.feat_type_in) if not self.exported else input_data_resized
        )

        ## Geometric tensor feed forwarding
        features_t = self.block1(features_t)
        features_t = self.block2(features_t)
        features_t = self.block3(features_t)

        ## orientation pooling
        features_o = self.ori_learner(features_t)  ## self.cpool
        features_o = features_o.tensor if not self.exported else features_o

        ## keypoint pooling
        features_t = self.gpool(features_t)
        features_t = features_t.tensor if not self.exported else features_t

        return features_t, features_o

    def _forwarding_networks_divide_grid(self, input_data_resized):
        device = input_data_resized.device
        
        ## for inference time high resolution image. # spatial grid 4
        B, _, H_resized, W_resized = input_data_resized.shape
        features_t = torch.zeros(B, self.dim_third, H_resized, W_resized).to(device)
        features_o = torch.zeros(B, self.group_size, H_resized, W_resized).to(device)
        h_divide = 2
        w_divide = 2
        for idx in range(h_divide):
            for jdx in range(w_divide):
                ## compute the start and end spatial index
                h_start = H_resized // h_divide * idx
                w_start = W_resized // w_divide * jdx
                h_end = H_resized // h_divide * (idx + 1)
                w_end = W_resized // w_divide * (jdx + 1)
                ## crop the input image
                input_data_divided = input_data_resized[:, :, h_start:h_end, w_start:w_end]
                features_t_temp, features_o_temp = self._forwarding_networks(input_data_divided)
                ## take into the values.
                features_t[:, :, h_start:h_end, w_start:w_end] = features_t_temp
                features_o[:, :, h_start:h_end, w_start:w_end] = features_o_temp

        return features_t, features_o

    def _resize_input_image(self, input_data, idx_level, H, W):
        if idx_level == 0:
            input_data_smooth = input_data
        else:
            ## (7,7) size gaussian kernel.
            input_data_smooth = F.conv2d(input_data, self.gaussian_avg.to(input_data.device), padding=[3, 3])

        target_resize = (
            int(H / (self.factor_scaling**idx_level)),
            int(W / (self.factor_scaling**idx_level)),
        )

        input_data_resized = F.interpolate(input_data_smooth, size=target_resize, align_corners=True, mode="bilinear")

        input_data_resized = self.local_norm_image(input_data_resized)

        return input_data_resized

    def local_norm_image(self, x, k_size=65, eps=1e-10):
        pad = int(k_size / 2)

        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        x_mean = F.avg_pool2d(x_pad, kernel_size=[k_size, k_size], stride=[1, 1], padding=0)  ## padding='valid'==0
        x2_mean = F.avg_pool2d(
            torch.pow(x_pad, 2.0),
            kernel_size=[k_size, k_size],
            stride=[1, 1],
            padding=0,
        )

        x_std = torch.sqrt(torch.abs(x2_mean - x_mean * x_mean)) + eps
        x_norm = (x - x_mean) / (1.0 + x_std)

        return x_norm


def load_detector(args, device):
    args.group_size, args.dim_first, args.dim_second, args.dim_third = model_parsing(args)
    model1 = REKD(args, device)
    model1.load_state_dict(torch.load(args.weights, weights_only=True, map_location=device))
    model1.export()
    model1.eval()
    model1.to(device)  ## use GPU

    return model1


## Load our model
def model_parsing(args):
    # get the dir/model.pt
    base_path = "/".join(args.weights.split("/")[-2:])
    
    group_size = base_path.split("_group")[1].split("_")[0]
    dim_first = base_path.split("_f")[1].split("_")[0]
    dim_second = base_path.split("_s")[1].split("_")[0]
    dim_third = base_path.split("_t")[1].split(".log")[0]
    
    return int(group_size), int(dim_first), int(dim_second), int(dim_third)


def upsample_pyramid(image, upsampled_levels, scale_factor_levels):
    ## image np.array([C, H, W]), upsampled_levels int
    up_pyramid = []
    for j in range(upsampled_levels):
        factor = scale_factor_levels ** (upsampled_levels - j)
        up_image = cv2.resize(
            image.transpose(1, 2, 0),
            dsize=(0, 0),
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_LINEAR,
        )
        up_pyramid.append(up_image[np.newaxis])

    return up_pyramid
