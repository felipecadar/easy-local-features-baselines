import warnings
from typing import TypedDict, Dict, Optional
from pathlib import Path

import torch
import numpy as np
import cv2

from easy_local_features.submodules.git_topicfm.src.models import TopicFM
from easy_local_features.submodules.git_topicfm.src.config.default import get_cfg_defaults
from .basemodel import BaseExtractor, MethodType


class TopicFMConfig(TypedDict):
    """Configuration for TopicFM baseline."""
    config_path: Optional[str]
    checkpoint_path: str
    resize_mode: str  # 'small' (640×480) or 'large' (1200×896)
    coarse_threshold: float


class TopicFM_baseline(BaseExtractor):
    """Simple interface for TopicFM model.

    This class provides a simple interface to load TopicFM weights and make 
    inference to get keypoints, descriptors, and matches between image pairs.

    Usage:
        config = {
            'checkpoint_path': 'pretrained/topicfm_fast.ckpt',
            'config_path': 'configs/megadepth_test_topicfmfast.py',
            'resize_mode': 'small',  # 'small' (640×480) or 'large' (1200×896)
            'coarse_threshold': 0.2
        }
        matcher = TopicFM_baseline(config)
        matcher.to('cuda')

        # For matching two images (end-to-end)
        result = matcher.match(img0, img1)
        mkpts0 = result['mkpts0']  # matched keypoints in image 0
        mkpts1 = result['mkpts1']  # matched keypoints in image 1
        mconf = result['mconf']    # matching confidence scores
    """

    METHOD_TYPE = MethodType.END2END_MATCHER

    # Fixed resize dimensions (both divisible by 8)
    RESIZE_MODES = {
        'small': (640, 480),
        'large': (1200, 896),
    }

    default_conf = TopicFMConfig(
        config_path=None,
        checkpoint_path='pretrained/topicfm_fast.ckpt',
        resize_mode='small',
        coarse_threshold=0.2,
    )

    def __init__(self, conf: Dict = {}):
        """Initialize TopicFM baseline.

        Args:
            conf: Configuration dictionary with keys:
                - checkpoint_path: Path to model checkpoint (.ckpt file)
                - config_path: Optional path to config file (.py file)
                - resize_mode: 'small' (640×480) or 'large' (1200×896)
                - coarse_threshold: Threshold for coarse matching (default: 0.2)
        """
        # Merge default config with user config
        self.conf = {**self.default_conf, **conf}

        # Load configuration
        if self.conf['config_path'] is not None:
            config = get_cfg_defaults()
            config.merge_from_file(self.conf['config_path'])
        else:
            config = get_cfg_defaults()

        # Override threshold if specified
        if 'coarse_threshold' in conf:
            config.MODEL.MATCH_COARSE.THR = conf['coarse_threshold']

        self.config = config

        # Get resize dimensions based on mode
        resize_mode = self.conf.get(
            'resize_mode', self.default_conf['resize_mode'])
        if resize_mode not in self.RESIZE_MODES:
            raise ValueError(
                f"Invalid resize_mode '{resize_mode}'. "
                f"Must be one of {list(self.RESIZE_MODES.keys())}"
            )
        self.resize_dims = self.RESIZE_MODES[resize_mode]  # (width, height)

        # Build model
        from easy_local_features.submodules.git_topicfm.src.utils.misc import lower_config
        _config = lower_config(config)
        self.model = TopicFM(config=_config['model'])

        # Load checkpoint
        checkpoint_path = self.conf['checkpoint_path']
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Load with strict=False to handle potential architecture mismatches
            # This allows loading checkpoints with extra or missing keys
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False)

            if missing_keys:
                warnings.warn(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                warnings.warn(
                    f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys (e.g., {unexpected_keys[:3]}...)")

            print(f"Loaded TopicFM checkpoint from {checkpoint_path}")
        else:
            warnings.warn(
                f"Checkpoint path '{checkpoint_path}' not found. Using random weights.")

        self.device = torch.device('cpu')
        self.model.eval()

    def _prepare_image(self, image, target_size=None):
        """Prepare image for model input.

        Args:
            image: Can be a file path (str), numpy array (H, W) or (H, W, C), 
                   or torch tensor (C, H, W) or (B, C, H, W)
            target_size: Optional (width, height) tuple to resize to.
                        If None, uses self.resize_dims

        Returns:
            torch.Tensor: Image tensor of shape (1, 1, H, W) normalized to [0, 1]
        """
        if target_size is None:
            target_size = self.resize_dims

        target_w, target_h = target_size

        # Load image if it's a path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
        elif isinstance(image, np.ndarray):
            img = image
            # Convert to grayscale if needed
            if img.ndim == 3:
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif img.shape[2] == 1:
                    img = img[:, :, 0]
        elif isinstance(image, torch.Tensor):
            img = image
            # Handle tensor input
            if img.ndim == 4:  # (B, C, H, W)
                if img.shape[1] == 3:
                    # Convert RGB to grayscale
                    img = 0.299 * img[:, 0] + 0.587 * \
                        img[:, 1] + 0.114 * img[:, 2]
                    img = img.unsqueeze(1)
                # Resize to target size
                if img.shape[2:] != (target_h, target_w):
                    img = torch.nn.functional.interpolate(
                        img, size=(target_h, target_w), mode='bilinear', align_corners=False
                    )
                return img.to(self.device)
            elif img.ndim == 3:  # (C, H, W)
                if img.shape[0] == 3:
                    # Convert RGB to grayscale
                    img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
                    img = img.unsqueeze(0).unsqueeze(0)
                else:
                    img = img.unsqueeze(0)
                # Resize to target size
                if img.shape[2:] != (target_h, target_w):
                    img = torch.nn.functional.interpolate(
                        img, size=(target_h, target_w), mode='bilinear', align_corners=False
                    )
                return img.to(self.device)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Convert numpy array to tensor and normalize
        if isinstance(img, np.ndarray):
            # Resize to fixed target dimensions
            h, w = img.shape[:2]
            if (w, h) != (target_w, target_h):
                img = cv2.resize(img, (target_w, target_h))

            img = torch.from_numpy(img).float() / 255.0
            if img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        return img.to(self.device)

    def detect(self, image, num_keypoints: Optional[int] = None):
        """Extract keypoints from a single image.

        This method extracts keypoints by running the TopicFM coarse network on a single image.
        It generates a dense grid of coarse-level features and returns their positions.

        Args:
            image: Image (path, numpy array, or tensor)
            num_keypoints: Optional number of top keypoints to return. If None, returns all coarse grid points.

        Returns:
            torch.Tensor: Keypoints of shape (N, 2) in (x, y) format
        """
        # Prepare image
        img_tensor = self._prepare_image(image)
        h, w = img_tensor.shape[2:]

        # Extract coarse features using the backbone
        with torch.no_grad():
            # Get coarse-level features from backbone
            feats_c, feats_f = self.model.backbone(img_tensor)

            # Get coarse feature map dimensions
            h_c, w_c = feats_c.shape[2:]

            # Generate all coarse grid positions
            scale = h / h_c  # Scale from coarse to original resolution

            # Create grid of all coarse positions
            y_coords = torch.arange(h_c, device=self.device)
            x_coords = torch.arange(w_c, device=self.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Convert to (x, y) format and scale to original image resolution
            keypoints = torch.stack(
                [xx.flatten(), yy.flatten()], dim=1).float()
            keypoints = keypoints * scale

            # Optionally select top keypoints based on feature magnitude
            if num_keypoints is not None and num_keypoints < len(keypoints):
                # Use feature magnitude as confidence
                feat_c_flat = feats_c.squeeze(0).view(
                    feats_c.size(1), -1).t()  # (H*W, C)
                scores = torch.norm(feat_c_flat, dim=1)

                # Select top-k keypoints
                top_indices = torch.topk(
                    scores, k=min(num_keypoints, len(scores)))[1]
                keypoints = keypoints[top_indices]

        return keypoints.cpu().unsqueeze(0)  # [1, N, 2]

    @torch.inference_mode()
    def match(self, image0, image1, return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """Match two images using TopicFM.

        Both images are resized to the same fixed dimensions based on resize_mode.

        Args:
            image0: First image (path, numpy array, or tensor)
            image1: Second image (path, numpy array, or tensor)
            return_dict: If True, return dict (default: True)

        Returns:
            dict with keys:
                - mkpts0: Matched keypoints in image0, shape (N, 2), in original image coordinates
                - mkpts1: Matched keypoints in image1, shape (N, 2), in original image coordinates
                - mconf: Matching confidence scores, shape (N,)
                - topic_matrix: Dict with keys 'img0' and 'img1', containing semantic topic distributions
                    - topic_matrix['img0']: shape (1, H_c*W_c, n_topics) - topics for image0
                    - topic_matrix['img1']: shape (1, H_c*W_c, n_topics) - topics for image1
                - feat_map0: Feature map from backbone for image0, shape (1, 128, H_f, W_f)
                - feat_map1: Feature map from backbone for image1, shape (1, 128, H_f, W_f)
                - keypoints0_coarse: All coarse keypoints in image0 (all grid points), in original coordinates
                - keypoints1_coarse: All coarse keypoints in image1 (all grid points), in original coordinates
                - mkpts0_c: Coarse matches in image0 (before fine refinement), if available
                - mkpts1_c: Coarse matches in image1 (before fine refinement), if available
                - all_mkpts0_c: All coarse matches in image0 (including padding), if available
                - all_mkpts1_c: All coarse matches in image1 (including padding), if available
        """
        # Get original image sizes for coordinate scaling
        if isinstance(image0, (str, Path)):
            img0_orig = cv2.imread(str(image0))
            h0_orig, w0_orig = img0_orig.shape[:2]
        elif isinstance(image0, np.ndarray):
            h0_orig, w0_orig = image0.shape[:2]
        elif isinstance(image0, torch.Tensor):
            if image0.ndim == 4:
                h0_orig, w0_orig = image0.shape[2:]
            elif image0.ndim == 3:
                h0_orig, w0_orig = image0.shape[1:]
            else:
                h0_orig, w0_orig = image0.shape

        if isinstance(image1, (str, Path)):
            img1_orig = cv2.imread(str(image1))
            h1_orig, w1_orig = img1_orig.shape[:2]
        elif isinstance(image1, np.ndarray):
            h1_orig, w1_orig = image1.shape[:2]
        elif isinstance(image1, torch.Tensor):
            if image1.ndim == 4:
                h1_orig, w1_orig = image1.shape[2:]
            elif image1.ndim == 3:
                h1_orig, w1_orig = image1.shape[1:]
            else:
                h1_orig, w1_orig = image1.shape

        # Both images are resized to the same fixed dimensions
        target_w, target_h = self.resize_dims
        img0_tensor = self._prepare_image(image0, target_size=self.resize_dims)
        img1_tensor = self._prepare_image(image1, target_size=self.resize_dims)

        # Prepare batch
        batch = {
            'image0': img0_tensor,
            'image1': img1_tensor,
        }

        # Run model
        self.model(batch)

        # Extract matches
        mkpts0 = batch['mkpts0_f'].cpu()  # (N, 2)
        mkpts1 = batch['mkpts1_f'].cpu()  # (N, 2)
        mconf = batch['mconf'].cpu()      # (N,)

        # Scale keypoints back to original image sizes
        scale_x0 = w0_orig / target_w
        scale_y0 = h0_orig / target_h

        scale_x1 = w1_orig / target_w
        scale_y1 = h1_orig / target_h

        scale_tensor0 = torch.tensor(
            [scale_x0, scale_y0], device=mkpts0.device)
        scale_tensor1 = torch.tensor(
            [scale_x1, scale_y1], device=mkpts1.device)

        mkpts0 = mkpts0 * scale_tensor0
        mkpts1 = mkpts1 * scale_tensor1

        result = {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf,
        }

        # Include topic matrix (semantic topics)
        if 'topic_matrix' in batch:
            result['topic_matrix'] = batch['topic_matrix']

        # Include feature maps
        if 'feat_map0' in batch:
            result['feat_map0'] = batch['feat_map0']
            result['feat_map1'] = batch['feat_map1']

        # Include coarse-level matches (before fine refinement)
        if 'mkpts0_c' in batch:
            mkpts0_c = batch['mkpts0_c'].cpu()
            mkpts1_c = batch['mkpts1_c'].cpu()

            # Scale back to original sizes
            mkpts0_c = mkpts0_c * torch.tensor([scale_x0, scale_y0])
            mkpts1_c = mkpts1_c * torch.tensor([scale_x1, scale_y1])

            result['mkpts0_c'] = mkpts0_c
            result['mkpts1_c'] = mkpts1_c

        # Include all coarse matches (including gt padding during training)
        if 'all_mkpts0_c' in batch:
            all_mkpts0_c = batch['all_mkpts0_c'].cpu()
            all_mkpts1_c = batch['all_mkpts1_c'].cpu()

            # Scale back to original sizes
            all_mkpts0_c = all_mkpts0_c * torch.tensor([scale_x0, scale_y0])
            all_mkpts1_c = all_mkpts1_c * torch.tensor([scale_x1, scale_y1])

            result['all_mkpts0_c'] = all_mkpts0_c
            result['all_mkpts1_c'] = all_mkpts1_c

        # Generate dense coarse keypoints for both images (in original sizes)
        h0_c, w0_c = batch['hw0_c']
        h1_c, w1_c = batch['hw1_c']

        # Scale from coarse resolution to resized image, then to original size
        scale0_base = target_h / h0_c
        scale1_base = target_h / h1_c

        # Create grids for both images
        y0, x0 = torch.meshgrid(torch.arange(
            h0_c), torch.arange(w0_c), indexing='ij')
        keypoints0_coarse = torch.stack(
            [x0.flatten(), y0.flatten()], dim=1).float() * scale0_base

        # Scale to original image size
        keypoints0_coarse = keypoints0_coarse * \
            torch.tensor([scale_x0, scale_y0])

        y1, x1 = torch.meshgrid(torch.arange(
            h1_c), torch.arange(w1_c), indexing='ij')
        keypoints1_coarse = torch.stack(
            [x1.flatten(), y1.flatten()], dim=1).float() * scale1_base

        # Scale to original image size
        keypoints1_coarse = keypoints1_coarse * \
            torch.tensor([scale_x1, scale_y1])

        result['keypoints0_coarse'] = keypoints0_coarse.cpu()
        result['keypoints1_coarse'] = keypoints1_coarse.cpu()

        return result

    def detectAndCompute(self, image, return_dict: bool = False):
        """Detect keypoints and compute descriptors (using coarse features).

        Note: TopicFM is primarily an end-to-end matcher. This method extracts
        coarse keypoints and their feature descriptors for compatibility.

        Args:
            image: Image (path, numpy array, or tensor)
            return_dict: If True, return dict; otherwise return tuple

        Returns:
            If return_dict=True: dict with 'keypoints' and 'descriptors'
            If return_dict=False: tuple of (keypoints, descriptors)
        """
        # Prepare image
        img_tensor = self._prepare_image(image)
        h, w = img_tensor.shape[2:]

        with torch.no_grad():
            # Get features from backbone
            feats_c, feats_f = self.model.backbone(img_tensor)

            # Get coarse feature dimensions
            h_c, w_c = feats_c.shape[2:]
            scale = h / h_c

            # Generate keypoints (grid positions)
            y_coords = torch.arange(h_c, device=self.device)
            x_coords = torch.arange(w_c, device=self.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            keypoints = torch.stack(
                [xx.flatten(), yy.flatten()], dim=1).float() * scale

            # Get descriptors (coarse features)
            descriptors = feats_c.squeeze(0).view(
                feats_c.size(1), -1).t()  # (H*W, C)

        keypoints = keypoints.cpu().unsqueeze(0)  # [1, N, 2]
        descriptors = descriptors.cpu().unsqueeze(0)  # [1, N, C]

        if return_dict:
            return {'keypoints': keypoints, 'descriptors': descriptors}
        return keypoints, descriptors

    def compute(self, image, keypoints):
        """Compute descriptors for given keypoints.

        Note: TopicFM works on a coarse grid. This method samples features
        at the requested keypoint locations using bilinear interpolation.

        Args:
            image: Image (path, numpy array, or tensor)
            keypoints: Keypoints tensor of shape (N, 2) in (x, y) format

        Returns:
            torch.Tensor: Descriptors of shape (N, C)
        """
        # Accept [B,N,2] or [N,2]
        if isinstance(keypoints, torch.Tensor) and keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        # Prepare image
        img_tensor = self._prepare_image(image)
        h, w = img_tensor.shape[2:]

        with torch.no_grad():
            # Get features from backbone
            feats_c, feats_f = self.model.backbone(img_tensor)

            # Normalize keypoint coordinates to [-1, 1] for grid_sample
            # keypoints are in (x, y) format, grid_sample expects (x, y) normalized
            kpts = keypoints.to(self.device).clone()
            kpts[:, 0] = 2.0 * kpts[:, 0] / (w - 1) - 1.0  # x
            kpts[:, 1] = 2.0 * kpts[:, 1] / (h - 1) - 1.0  # y

            # Reshape for grid_sample: (1, N, 1, 2)
            kpts = kpts.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

            # Sample features at keypoint locations
            # feats_c shape: (1, C, H, W)
            descriptors = torch.nn.functional.grid_sample(
                feats_c, kpts, mode='bilinear', align_corners=True
            )  # (1, C, N, 1)

            # Reshape to (N, C)
            descriptors = descriptors.squeeze(3).squeeze(0).t()  # (N, C)

        return descriptors.cpu().unsqueeze(0)  # [1, N, C]

    @property
    def has_detector(self):
        """TopicFM can extract keypoints from the coarse feature grid."""
        return True

    def to(self, device):
        """Move model to specified device.

        Args:
            device: torch.device or string ('cuda', 'cpu', etc.)

        Returns:
            self
        """
        self.device = torch.device(device) if isinstance(
            device, str) else device
        self.model.to(self.device)
        return self


if __name__ == "__main__":
    """Example usage of TopicFM baseline."""
    import matplotlib.pyplot as plt

    # Initialize matcher
    config = {
        'checkpoint_path': 'pretrained/topicfm_fast.ckpt',
        'config_path': 'configs/megadepth_test_topicfmfast.py',
        'resize_mode': 'small',  # or 'large'
        'coarse_threshold': 0.2,
    }

    matcher = TopicFM_baseline(config)
    matcher.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Example with test images (adjust paths as needed)
    img0_path = "assets/scannet_sample_images/scene0711_00_frame-001680.jpg"
    img1_path = "assets/scannet_sample_images/scene0711_00_frame-001995.jpg"

    if Path(img0_path).exists() and Path(img1_path).exists():
        # Match images
        result = matcher.match(img0_path, img1_path)

        print(f"Number of matches: {len(result['mkpts0'])}")
        print(
            f"Match confidence range: [{result['mconf'].min():.3f}, {result['mconf'].max():.3f}]")

        # Visualize matches
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)

        mkpts0 = result['mkpts0'].numpy()
        mkpts1 = result['mkpts1'].numpy()

        # Create visualization
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        # Stack images horizontally
        vis_img = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
        vis_img[:h0, :w0] = img0
        vis_img[:h1, w0:w0+w1] = img1

        # Draw matches
        for i in range(min(100, len(mkpts0))):  # Draw first 100 matches
            pt0 = tuple(mkpts0[i].astype(int))
            pt1 = tuple((mkpts1[i] + np.array([w0, 0])).astype(int))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(vis_img, pt0, 3, color, -1)
            cv2.circle(vis_img, pt1, 3, color, -1)
            cv2.line(vis_img, pt0, pt1, color, 1)

        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"TopicFM Matches: {len(mkpts0)}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('topicfm_matches.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to 'topicfm_matches.png'")
    else:
        print("Test images not found. Please check paths:")
        print(f"  {img0_path}")
        print(f"  {img1_path}")
        print("\nYou can use the matcher with your own images:")
        print("  result = matcher.match('path/to/image0.jpg', 'path/to/image1.jpg')")
