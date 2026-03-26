# Features Overview

The library supports four types of feature methods:

| Type | What it does | Example methods |
|------|-------------|-----------------|
| **Detect+Describe** | Finds keypoints and computes descriptors | SuperPoint, ALIKED, DISK, XFeat, ORB |
| **Descriptor-only** | Computes descriptors for given keypoints | SOSNet, TFeat, DINOv2, DINOv3 |
| **Detector-only** | Finds keypoints only (no descriptors) | DAD, REKD |
| **End-to-end matcher** | Directly matches two images | LightGlue, SuperGlue, LoFTR, RoMa |

## Detect+Describe methods

These are the most common. They detect keypoints and compute descriptors in a single call.

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops

img0 = io.fromPath("tests/assets/megadepth0.jpg")
img1 = io.fromPath("tests/assets/megadepth1.jpg")
img0 = ops.resize_short_edge(img0, 640)[0]
img1 = ops.resize_short_edge(img1, 640)[0]

# Any detect+describe method works the same way
extractor = getExtractor("aliked", {"top_k": 2048}).to("cpu")

# Option 1: Match directly
matches = extractor.match(img0, img1)
print(f"Found {len(matches['mkpts0'])} matches")

# Option 2: Detect and describe separately
keypoints, descriptors = extractor.detectAndCompute(img0)
print(f"keypoints: {keypoints.shape}")    # [1, N, 2]
print(f"descriptors: {descriptors.shape}") # [1, N, D]

# Option 3: Detect only
kpts = extractor.detect(img0)  # [1, N, 2]
```

### Available Detect+Describe methods

| Name | Key | Notes |
|------|-----|-------|
| ALIKE | `"alike"` | Lightweight, real-time |
| ALIKED | `"aliked"` | Enhanced ALIKE |
| D2Net | `"d2net"` | Dense feature detection |
| DEAL | `"deal"` | |
| DeDoDe | `"dedode"` | Separate detector + descriptor weights |
| DELF | `"delf"` | TensorFlow-based |
| DISK | `"disk"` | |
| R2D2 | `"r2d2"` | Reliability + repeatability |
| SuperPoint | `"superpoint"` | Classic learned features |
| SuperPoint Open | `"superpoint_open"` | Open-source reimplementation |
| XFeat | `"xfeat"` | Fast and accurate |
| ORB | `"orb"` | Classical (OpenCV) |
| SFD2 | `"sfd2"` | |
| Desc Reasoning | `"desc_reasoning"` | |

## Descriptor-only methods

These methods compute descriptors but **do not detect keypoints**. You must attach an external detector first.

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

# Create the descriptor
desc = getExtractor("sosnet").to("cpu")

# It has no detector on its own
print(desc.has_detector)  # False

# Attach a detector
det = getExtractor("superpoint", {"top_k": 2048, "detection_threshold": 0.005}).to("cpu")
desc.addDetector(det)

# Now it can detect, describe, and match
matches = desc.match(img0, img1)
print(f"Matches: {len(matches['mkpts0'])}")
```

### Compute descriptors for specific keypoints

```python
import torch

desc = getExtractor("sosnet").to("cpu")
img = io.fromPath("tests/assets/megadepth0.jpg")

# Provide your own keypoints [B, N, 2]
my_kpts = torch.rand(1, 100, 2) * 320
descriptors = desc.compute(img, my_kpts)
print(descriptors.shape)  # [1, 100, 128]
```

### Available Descriptor-only methods

| Name | Key | Descriptor dim | Notes |
|------|-----|----------------|-------|
| SOSNet | `"sosnet"` | 128 | Patch-based |
| TFeat | `"tfeat"` | 128 | Patch-based |
| DINOv2 | `"dinov2"` | 384 | Vision transformer features |
| DINOv3 | `"dinov3"` | varies | Updated DINO |
| ResNet | `"resnet"` | varies | CNN backbone features |
| VGG | `"vgg"` | varies | CNN backbone features |
| MuM | `"mum"` | varies | Masked feature learning |
| CroCo | `"croco"` | varies | Cross-view completion |

## Detector-only methods

These only detect keypoints (no descriptors, no matching).

```python
from easy_local_features import getDetector
from easy_local_features.utils import io

img = io.fromPath("tests/assets/megadepth0.jpg")

det = getDetector("rekd", {"num_keypoints": 1500}).to("cpu")
kpts = det.detect(img)  # [1, N, 2]
print(f"Detected {kpts.shape[1]} keypoints")
```

### Use with descriptor-only methods

```python
from easy_local_features import getExtractor, getDetector

det = getDetector("dad", {"num_keypoints": 1024}).to("cpu")
desc = getExtractor("sosnet").to("cpu")
desc.addDetector(det)

matches = desc.match(img0, img1)
```

### Available Detector-only methods

| Name | Key | Config keys |
|------|-----|-------------|
| DAD | `"dad"` | `num_keypoints`, `resize`, `nms_size` |
| REKD | `"rekd"` | `num_keypoints`, `pyramid_levels`, `upsampled_levels`, `border_size`, `nms_size` |

## End-to-end matchers

These methods directly match two images without separate detect/describe steps.

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

# RoMa
roma = getExtractor("romav2", {"top_k": 2000}).to("cpu")
matches = roma.match(img0, img1)

# LightGlue (with variation syntax)
lg = getExtractor("lightglue:superpoint", {"top_k": 2048}).to("cpu")
matches = lg.match(img0, img1)
```

### LightGlue variation syntax

LightGlue supports different feature backends using colon syntax:

```python
# SuperPoint features (default)
lg = getExtractor("lightglue:superpoint", {"top_k": 2048})

# DISK features
lg = getExtractor("lightglue:disk", {"top_k": 2048})

# ALIKED features
lg = getExtractor("lightglue:aliked", {"top_k": 2048})

# DeDoDe features
lg = getExtractor("lightglue:dedode", {"top_k": 2048})
```

### Available End-to-end matchers

| Name | Key | Notes |
|------|-----|-------|
| LightGlue | `"lightglue"` | Supports variation syntax (`:superpoint`, `:disk`, etc.) |
| SuperGlue | `"superglue"` | Classic learned matcher |
| LoFTR | `"loftr"` | Detector-free matcher |
| RoMa | `"roma"` | Dense matching |
| RoMa v2 | `"romav2"` | Improved RoMa |
| TopicFM | `"topicfm"` | Topic-based feature matching |

## Supported image formats

All methods accept flexible image inputs:

```python
# From a file path
kpts, desc = extractor.detectAndCompute("path/to/image.jpg")

# From a numpy array (H, W, 3) uint8
import cv2
img_np = cv2.imread("image.jpg")
kpts, desc = extractor.detectAndCompute(img_np)

# From a torch tensor [C, H, W] or [B, C, H, W]
import torch
img_tensor = torch.rand(1, 3, 480, 640)
kpts, desc = extractor.detectAndCompute(img_tensor)

# Using the io utility
from easy_local_features.utils import io
img = io.fromPath("image.jpg")  # Returns [1, 3, H, W] float tensor
kpts, desc = extractor.detectAndCompute(img)
```

## Configuration

Every method has a `default_conf` dictionary. Override any key by passing a config dict:

```python
extractor = getExtractor("superpoint", {
    "top_k": 1024,
    "detection_threshold": 0.01,
    "nms_radius": 3,
})
```

Inspect defaults without loading the model:

```python
from easy_local_features import describe
info = describe("superpoint")
print(info["defaults"])
print(info["schema"])    # TypedDict annotations
print(info["aliases"])   # Deprecated key mappings
```

Some methods support **config aliases** for backward compatibility:

```python
# These are equivalent for SuperPoint:
getExtractor("superpoint", {"top_k": 1024})
getExtractor("superpoint", {"max_keypoints": 1024})  # alias

getExtractor("superpoint", {"detection_threshold": 0.01})
getExtractor("superpoint", {"keypoint_threshold": 0.01})  # alias
```

See [Extractors Reference](../reference/extractors.md) for the full configuration of every method.
