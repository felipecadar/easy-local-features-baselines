import argparse
import random
import os
import numpy as np
import torch
from omegaconf import OmegaConf
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.utils import ops, download
from easy_local_features.submodules.git_rekd.model import (
    load_detector,
    apply_homography_to_points,
    get_point_coordinates,
    remove_borders,
    apply_nms,
    upsample_pyramid,
)
import warnings


def fix_randseed(randseed):
    r"""Fix random seed"""
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    # torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = True, False


def get_config(jupyter=False):
    parser = argparse.ArgumentParser(description="Train REKD Architecture")

    ## basic configuration
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../ImageNet2012/ILSVRC2012_img_val",  # default='path-to-ImageNet',
        help="The root path to the data from which the synthetic dataset will be created.",
    )
    parser.add_argument(
        "--synth_dir",
        type=str,
        default="",
        help="The path to save the generated sythetic image pairs.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="trained_models/weights",
        help="The path to save the REKD weights.",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default="",
        help="Set saved model parameters if resume training is desired.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="REKD",
        help="The Rotaton-equivaraiant Keypoint Detection (REKD) experiment name",
    )
    ## network architecture
    parser.add_argument(
        "--factor_scaling_pyramid",
        type=float,
        default=1.2,
        help="The scale factor between the multi-scale pyramid levels in the architecture.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=36,
        help="The number of groups for the group convolution.",
    )
    parser.add_argument(
        "--dim_first",
        type=int,
        default=2,
        help="The number of channels of the first layer",
    )
    parser.add_argument(
        "--dim_second",
        type=int,
        default=2,
        help="The number of channels of the second layer",
    )
    parser.add_argument(
        "--dim_third",
        type=int,
        default=2,
        help="The number of channels of the thrid layer",
    )
    ## network training
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training.")
    ## Loss function
    parser.add_argument(
        "--init_initial_learning_rate",
        type=float,
        default=1e-3,
        help="The init initial learning rate value.",
    )
    parser.add_argument("--MSIP_sizes", type=str, default="8,16,24,32,40", help="MSIP sizes.")
    parser.add_argument(
        "--MSIP_factor_loss",
        type=str,
        default="256.0,64.0,16.0,4.0,1.0",
        help="MSIP loss balancing parameters.",
    )
    parser.add_argument("--ori_loss_balance", type=float, default=100.0, help="")
    ## Dataset generation
    parser.add_argument(
        "--patch_size",
        type=int,
        default=192,
        help="The patch size of the generated dataset.",
    )
    parser.add_argument(
        "--max_angle",
        type=int,
        default=180,
        help="The max angle value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--min_scale",
        type=float,
        default=1.0,
        help="The min scale value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--max_scale",
        type=float,
        default=1.0,
        help="The max scale value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--max_shearing",
        type=float,
        default=0.0,
        help="The max shearing value for generating a synthetic view to train REKD.",
    )
    parser.add_argument(
        "--num_training_data",
        type=int,
        default=9000,
        help="The number of the generated dataset.",
    )
    parser.add_argument(
        "--is_debugging",
        type=bool,
        default=False,
        help="Set variable to True if you desire to train network on a smaller dataset.",
    )
    ## For eval/inference
    parser.add_argument(
        "--num_points",
        type=int,
        default=1500,
        help="the number of points at evaluation time.",
    )
    parser.add_argument("--pyramid_levels", type=int, default=5, help="downsampling pyramid levels.")
    parser.add_argument("--upsampled_levels", type=int, default=2, help="upsampling image levels.")
    parser.add_argument(
        "--nms_size",
        type=int,
        default=15,
        help="The NMS size for computing the validation repeatability.",
    )
    parser.add_argument(
        "--border_size",
        type=int,
        default=15,
        help="The number of pixels to remove from the borders to compute the repeatability.",
    )
    ## For HPatches evaluation
    parser.add_argument(
        "--hpatches_path",
        type=str,
        default="./datasets/hpatches-sequences-release",
        help="dataset ",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="debug",
        help="debug, view, illum, full, debug_view, debug_illum ...",
    )
    parser.add_argument("--descriptor", type=str, default="hardnet", help="hardnet, sosnet, hynet")

    args, weird_args = parser.parse_known_args() if not jupyter else parser.parse_args(args=[])

    fix_randseed(12345)

    if args.synth_dir == "":
        args.synth_dir = "datasets/synth_data"

    args.MSIP_sizes = [int(i) for i in args.MSIP_sizes.split(",")]
    args.MSIP_factor_loss = [float(i) for i in args.MSIP_factor_loss.split(",")]

    return args


WEIGHTS = "https://github.com/felipecadar/easy-local-features-baselines/releases/download/redk/best_model.pt"


class REKD_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECTOR_ONLY
    default_conf = {
        "num_keypoints": 1500,
        "pyramid_levels": 5,
        "upsampled_levels": 2,
        "border_size": 15,
        "nms_size": 15,
        "weights": None,
        "resize": None,
    }

    def __init__(self, conf={}):
        super().__init__()
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

        # configurations
        self.default_num_points = conf.num_keypoints
        self.pyramid_levels = conf.pyramid_levels
        self.upsampled_levels = conf.upsampled_levels
        self.resize = conf.resize
        self.border_size = conf.border_size
        self.nms_size = conf.nms_size
        self.desc_scale_factor = 2.0
        self.scale_factor_levels = np.sqrt(2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create args object for model loading
        args = get_config()
        args.weights = conf.weights
        if args.weights is None or (not os.path.isfile(args.weights)):
            args.weights = str(download.downloadModel("redk", "release_group36_f2_s2_t2.log/best_model.pt", WEIGHTS))
            print(f"Using default REKD weights from {args.weights}")

        args.resize = conf.resize

        self.model = load_detector(args, self.device)
        self.model.eval()
        self.levels = self.pyramid_levels + self.upsampled_levels + 1

        # Intentionally no matcher: REKD is detector-only.

    def to_normalized_coords(self, keypoints, H, W):
        """Convert pixel coordinates to normalized coordinates [-1, 1]"""
        kpts = keypoints.clone()
        kpts[..., 0] = 2 * kpts[..., 0] / (W - 1) - 1  # x
        kpts[..., 1] = 2 * kpts[..., 1] / (H - 1) - 1  # y
        return kpts

    def from_normalized_coords(self, keypoints, H, W):
        """Convert normalized coordinates [-1, 1] to pixel coordinates"""
        kpts = keypoints.clone()
        kpts[..., 0] = (kpts[..., 0] + 1) * (W - 1) / 2  # x
        kpts[..., 1] = (kpts[..., 1] + 1) * (H - 1) / 2  # y
        return kpts

    def detectAndCompute(self, image, return_dict=False):
        raise NotImplementedError("REKD_baseline is detector-only; use detect(image).")

    def detect(self, image):
        """Detect keypoints only"""
        image = ops.prepareImage(image).to(self.device)
        img_np = image[0, 0].cpu().numpy() if len(image.shape) == 4 else image[0].cpu().numpy()
        kps = self._detect_keypoints(img_np)
        return kps

    def compute(self, image, keypoints):
        raise NotImplementedError("REKD_baseline is detector-only; it does not compute descriptors.")

    def match(self, image1, image2):
        raise NotImplementedError("REKD_baseline is detector-only; it does not support matching.")

    def _detect_keypoints(self, image):
        """Internal method to detect keypoints"""
        # Ensure image has the correct shape (C, H, W)
        if len(image.shape) == 2:
            # Add channel dimension for grayscale
            image = image[np.newaxis, ...]

        one, H, W = image.shape
        score_maps, ori_maps = self._compute_score_maps(image)
        im_pts = self._estimate_keypoint_coordinates(score_maps, num_points=self.default_num_points)
        pixel_coords = im_pts[..., :2]
        pixel_coords = torch.tensor(pixel_coords, dtype=torch.float32).to(self.device)
        if pixel_coords.dim() == 2:
            pixel_coords = pixel_coords.unsqueeze(0)  # batch dimension to match other extractors pattern
        return pixel_coords

    def to(self, device):
        """Move model to device"""
        self.model = self.model.to(device)
        self.device = device
        return self

    @property
    def has_detector(self):
        """REKD has a detector"""
        return True

    def _compute_score_maps(self, image):
        from skimage.transform import pyramid_gaussian

        pyramid = pyramid_gaussian(image, max_layer=self.pyramid_levels, downscale=self.scale_factor_levels)
        up_pyramid = upsample_pyramid(
            image,
            upsampled_levels=self.upsampled_levels,
            scale_factor_levels=self.scale_factor_levels,
        )

        score_maps = {}
        ori_maps = {}
        for j, down_image in enumerate(pyramid):  ## Pyramid is downsampling images.
            key_idx = j + 1 + self.upsampled_levels
            score_maps, ori_maps = self._obtain_feature_maps(down_image, key_idx, score_maps, ori_maps)

        if self.upsampled_levels:
            for j, up_image in enumerate(up_pyramid):  ## Upsample levels is for upsampling images.
                key_idx = j + 1
                score_maps, ori_maps = self._obtain_feature_maps(up_image, key_idx, score_maps, ori_maps)

        return score_maps, ori_maps

    def _obtain_feature_maps(self, im, key_idx, score_maps, ori_maps):
        im = torch.tensor(im).unsqueeze(0).to(torch.float32).to(self.device)
        im_scores, ori_map = self.model(im)
        im_scores = remove_borders(im_scores[0, 0, :, :].cpu().detach().numpy(), borders=self.border_size)

        score_maps["map_" + str(key_idx)] = im_scores
        ori_maps["map_" + str(key_idx)] = ori_map

        return score_maps, ori_maps

    def _estimate_keypoint_coordinates(self, score_maps, num_points=None):
        num_points = num_points if num_points is not None else self.default_num_points
        point_level = []
        tmp = 0.0
        factor_points = self.scale_factor_levels**2
        for idx_level in range(self.levels):
            tmp += factor_points ** (-1 * (idx_level - self.upsampled_levels))
            point_level.append(self.default_num_points * factor_points ** (-1 * (idx_level - self.upsampled_levels)))

        point_level = np.asarray(list(map(lambda x: int(x / tmp) + 1, point_level)))

        im_pts = []
        for idx_level in range(self.levels):
            scale_value = self.scale_factor_levels ** (idx_level - self.upsampled_levels)
            scale_factor = 1.0 / scale_value

            h_scale = np.asarray([[scale_factor, 0.0, 0.0], [0.0, scale_factor, 0.0], [0.0, 0.0, 1.0]])
            h_scale_inv = np.linalg.inv(h_scale)
            h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

            num_points_level = point_level[idx_level]
            if idx_level > 0:
                res_points = int(np.asarray([point_level[a] for a in range(0, idx_level + 1)]).sum() - len(im_pts))
                num_points_level = res_points

            ## to make the output score map derive more keypoints
            score_map = score_maps["map_" + str(idx_level + 1)]

            im_scores = apply_nms(score_map, self.nms_size)
            im_pts_tmp = get_point_coordinates(im_scores, num_points=num_points_level)
            im_pts_tmp = apply_homography_to_points(im_pts_tmp, h_scale_inv)

            if not idx_level:
                im_pts = im_pts_tmp
            else:
                im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)

        im_pts = im_pts[(-1 * im_pts[:, 3]).argsort()]
        im_pts = im_pts[:num_points]

        return im_pts


if __name__ == "__main__":
    from easy_local_features.utils import io, vis, ops

    detector = REKD_baseline({"num_keypoints": 512})

    img0 = io.fromPath("tests/assets/megadepth0.jpg")
    img1 = io.fromPath("tests/assets/megadepth1.jpg")

    img0, _ = ops.resize_short_edge(img0, 480)
    img1, _ = ops.resize_short_edge(img1, 480)

    kps0 = detector.detect(img0)
    kps1 = detector.detect(img1)

    vis.plot_pair(img0, img1)
    vis.plot_keypoints(keypoints0=kps0)
    vis.plot_keypoints(keypoints1=kps1)
    vis.show()
