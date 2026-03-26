# I/O and Ops

Utilities for loading, preprocessing, and manipulating images and keypoints.

## Image I/O

### `io.fromPath` &mdash; Load an image

```python
from easy_local_features.utils import io

# Load as color [1, 3, H, W] float tensor (0-1)
img = io.fromPath("path/to/image.jpg")

# Load as grayscale [1, 1, H, W]
img_gray = io.fromPath("path/to/image.jpg", gray=True)

# Without batch dimension [3, H, W]
img_unbatched = io.fromPath("path/to/image.jpg", batch=False)

# With ImageNet normalization
img_inet = io.fromPath("path/to/image.jpg", imagenet=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Path to image file |
| `gray` | `bool` | `False` | Convert to grayscale |
| `batch` | `bool` | `True` | Add batch dimension |
| `imagenet` | `bool` | `False` | Apply ImageNet normalization |

**Returns:** `torch.Tensor` of shape `[1, C, H, W]` (or `[C, H, W]` if `batch=False`), dtype `float32`, range `[0, 1]`.

### Keypoint I/O

```python
from easy_local_features.utils import io

# Save keypoints (list of cv2.KeyPoint)
io.writeKeypoints(keypoints, "keypoints.npz")

# Load keypoints (cached after first read)
keypoints = io.readKeypoints("keypoints.npz")

# Save descriptors (numpy array)
io.writeDescriptors(descriptors, "descriptors.npz")

# Load descriptors (cached after first read)
descriptors = io.readDescriptors("descriptors.npz")
```

## Image Operations

### `ops.prepareImage` &mdash; Normalize any image format

Converts file paths, numpy arrays, or tensors into a standardized tensor format.

```python
from easy_local_features.utils import ops

# From a numpy array
import numpy as np
img_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
tensor = ops.prepareImage(img_np)          # [1, 3, 480, 640] float 0-1
tensor = ops.prepareImage(img_np, gray=True)  # [1, 1, 480, 640]

# From a file path
tensor = ops.prepareImage("image.jpg")

# Without batch dimension
tensor = ops.prepareImage(img_np, batch=False)  # [3, 480, 640]

# With ImageNet normalization
tensor = ops.prepareImage(img_np, imagenet=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `str` / `ndarray` / `Tensor` | required | Input image |
| `gray` | `bool` | `False` | Convert to grayscale |
| `batch` | `bool` | `True` | Add batch dimension |
| `imagenet` | `bool` | `False` | Apply ImageNet normalization (mean/std) |

### `ops.to_cv` &mdash; Convert tensor to OpenCV format

```python
from easy_local_features.utils import ops

# Tensor [1, 3, H, W] -> numpy [H, W, 3] uint8
img_cv = ops.to_cv(tensor)

# Convert RGB to BGR
img_bgr = ops.to_cv(tensor, convert_color=True)

# Convert to grayscale
img_gray = ops.to_cv(tensor, to_gray=True)

# Select a specific batch element
img_batch2 = ops.to_cv(batched_tensor, batch_idx=1)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `torch_image` | `Tensor` / `ndarray` | required | Input image |
| `convert_color` | `bool` | `False` | Convert RGB to BGR |
| `batch_idx` | `int` | `0` | Which batch element to extract |
| `to_gray` | `bool` | `False` | Convert to grayscale |

### `ops.resize_short_edge` &mdash; Resize keeping aspect ratio

```python
from easy_local_features.utils import ops, io

img = io.fromPath("image.jpg")  # e.g., [1, 3, 1200, 800]

resized, scale = ops.resize_short_edge(img, 640)
print(resized.shape)  # [1, 3, 960, 640]
print(scale)          # 0.8

# Use scale to map keypoints back to original resolution
original_kpts = keypoints / scale
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | `Tensor` | Input tensor (2D, 3D, or 4D) |
| `min_size` | `int` | Target size for the short edge |

**Returns:** `(resized_tensor, scale_factor)`.

### `ops.pad_square` &mdash; Pad to square

```python
img = io.fromPath("image.jpg")  # [1, 3, 480, 640]
square = ops.pad_square(img)     # [1, 3, 640, 640] (zero-padded bottom)
```

### `ops.crop_square` &mdash; Crop to square

```python
img = io.fromPath("image.jpg")  # [1, 3, 480, 640]
square = ops.crop_square(img)    # [1, 3, 480, 480] (cropped right)
```

### `ops.crop_patches` &mdash; Extract patches around keypoints

```python
from easy_local_features.utils import ops
import torch

image = torch.rand(2, 3, 256, 256)        # [B, C, H, W]
keypoints = torch.rand(2, 50, 2) * 255    # [B, N, 2]

patches = ops.crop_patches(image, keypoints, patch_size=32)
print(patches.shape)  # [2, 50, 3, 32, 32]  -> [B, N, C, ps, ps]

# With bilinear interpolation
patches = ops.crop_patches(image, keypoints, patch_size=32, mode='bilinear')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `Tensor` | required | Image tensor `[B, C, H, W]` |
| `keypoints` | `Tensor` | required | Keypoint coordinates `[B, N, 2]` (x, y in pixels) |
| `patch_size` | `int` | `32` | Width and height of each patch |
| `mode` | `str` | `'nearest'` | Interpolation mode (`'nearest'`, `'bilinear'`) |

**Returns:** `Tensor` of shape `[B, N, C, patch_size, patch_size]`.

## Keypoint Operations

### `ops.sort_keypoints` &mdash; Lexicographic sort

```python
import numpy as np
from easy_local_features.utils import ops

kpts0 = np.random.rand(100, 2)
kpts1 = np.random.rand(100, 2)

# Sort both arrays by the same order
kpts0_sorted, kpts1_sorted = ops.sort_keypoints(kpts0, kpts1)

# Sort single array
kpts_sorted = ops.sort_keypoints(kpts0)

# Works with batched keypoints [B, N, 2] too
batched = np.random.rand(4, 100, 2)
batched_sorted = ops.sort_keypoints(batched)
```

### `ops.to_homogeneous` / `ops.from_homogeneous` &mdash; Coordinate conversion

```python
import numpy as np
from easy_local_features.utils import ops

kpts = np.random.rand(100, 2)   # [N, 2]

# Convert to homogeneous coordinates
kpts_h = ops.to_homogeneous(kpts)   # [N, 3] with last column = 1

# Convert back
kpts_2d = ops.from_homogeneous(kpts_h)  # [N, 2]

# Works with batched [B, N, 2] and torch tensors too
import torch
kpts_t = torch.rand(4, 100, 2)
kpts_h_t = ops.to_homogeneous(kpts_t)  # [4, 100, 3]
```

### `ops.to_grid` &mdash; Pixel to normalized grid coordinates

Convert pixel coordinates to the `[-1, 1]` range used by `torch.nn.functional.grid_sample`:

```python
import torch
from easy_local_features.utils import ops

kpts = torch.rand(1, 100, 2) * 255  # pixel coords in a 256x256 image
grid_coords = ops.to_grid(kpts, H=256, W=256)  # [-1, 1] range
```

### `ops.warp_kpts` &mdash; Warp keypoints between views

Warp keypoints from image 0 to image 1 using depth maps, camera intrinsics, and relative pose:

```python
import torch
from easy_local_features.utils import ops

# kpts0:   [N, L, 2]  keypoints in image 0
# depth0:  [N, H, W]  depth map for image 0
# depth1:  [N, H, W]  depth map for image 1
# T_0to1:  [N, 3, 4]  relative pose (rotation + translation)
# K0, K1:  [N, 3, 3]  camera intrinsics

valid_mask, warped_kpts = ops.warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1)

# valid_mask: [N, L] boolean mask of successfully warped keypoints
# warped_kpts: [N, L, 2] projected coordinates in image 1
```

## Common patterns

### Load, resize, match, visualize

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis

img0 = ops.resize_short_edge(io.fromPath("img0.jpg"), 640)[0]
img1 = ops.resize_short_edge(io.fromPath("img1.jpg"), 640)[0]

extractor = getExtractor("xfeat", {"top_k": 4096}).to("cpu")
matches = extractor.match(img0, img1)

vis.plot_pair(img0, img1, title="XFeat")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
vis.save("xfeat_matches.png")
```

### Process a directory of images

```python
from pathlib import Path
from easy_local_features.utils import io, ops

images = {}
for p in Path("data/images").glob("*.jpg"):
    img = io.fromPath(str(p))
    img = ops.resize_short_edge(img, 640)[0]
    images[p.stem] = img
```
