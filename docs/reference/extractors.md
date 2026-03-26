# Extractors Reference

Complete list of all feature extractors with their configuration keys and default values.

## Detect+Describe

### ALIKE

**Key:** `"alike"`

```python
extractor = getExtractor("alike", {
    "model_name": "alike-t",   # Model variant
    "top_k": -1,               # Max keypoints (-1 = no limit)
    "scores_th": 0.2,          # Detection score threshold
    "n_limit": 2048,           # Hard keypoint limit
    "sub_pixel": True,         # Sub-pixel refinement
    "model_path": None,        # Custom weights path
})
```

### ALIKED

**Key:** `"aliked"`

```python
extractor = getExtractor("aliked", {
    "model_name": "aliked-n16",    # Model variant
    "top_k": -1,                   # Max keypoints (-1 = no limit)
    "detection_threshold": 0.2,     # Score threshold
    "force_num_keypoints": False,   # Force exactly top_k keypoints
    "nms_radius": 2,               # NMS radius in pixels
})
```

### D2-Net

**Key:** `"d2net"`

```python
extractor = getExtractor("d2net", {
    "model_name": "d2net",         # Model variant
    "top_k": 2048,                 # Max keypoints
    "detection_threshold": 0.2,     # Score threshold
    "nms_radius": 4,               # NMS radius
    "use_relu": True,              # Use ReLU activation
})
```

### DEAL

**Key:** `"deal"`

```python
extractor = getExtractor("deal", {
    "model_name": "deal",          # Model variant
    "top_k": 2048,                 # Max keypoints
    "detection_threshold": 0.2,     # Score threshold
    "nms_radius": 4,               # NMS radius
    "force_num_keypoints": False,   # Force exactly top_k keypoints
})
```

### DeDoDe

**Key:** `"dedode"`

```python
extractor = getExtractor("dedode", {
    "model_name": "dedode",            # Model variant
    "top_k": 2048,                     # Max keypoints
    "detection_threshold": 0.2,         # Score threshold
    "nms_radius": 4,                   # NMS radius
    "detector_weights": "L-upright",   # Detector weight variant
    "descriptor_weights": "B-upright", # Descriptor weight variant
    "amp_dtype": "float16",            # AMP data type
})
```

### DELF

**Key:** `"delf"`

```python
extractor = getExtractor("delf", {
    "model_name": "delf",          # Model variant
    "top_k": 2048,                 # Max keypoints
    "detection_threshold": 0.2,     # Score threshold
    "nms_radius": 4,               # NMS radius
    "use_pca": False,              # PCA dimensionality reduction
    "use_whitening": False,        # Whitening normalization
})
```

### DISK

**Key:** `"disk"`

```python
extractor = getExtractor("disk", {
    "window": 8,               # NMS window size
    "desc_dim": 128,           # Descriptor dimension
    "mode": "rng",             # Detection mode
    "top_k": 2048,             # Max keypoints
    "auto_resize": True,       # Auto-resize images for model
})
```

### R2D2

**Key:** `"r2d2"`

```python
extractor = getExtractor("r2d2", {
    "model_name": "r2d2",             # Model variant
    "top_k": 2048,                    # Max keypoints
    "detection_threshold": 0.2,        # Score threshold
    "nms_radius": 4,                  # NMS radius
    "rel_thr": 0.7,                   # Reliability threshold
    "rep_thr": 0.7,                   # Repeatability threshold
    "scale_f": 1.189207115002721,     # Scale factor (2^0.25)
    "min_scale": 0.0,                 # Minimum scale
    "max_scale": 1,                   # Maximum scale
    "min_size": 256,                  # Minimum image size
    "max_size": 1024,                 # Maximum image size
    "pretrained_weigts": "r2d2_WASF_N16",  # Weight variant
    "model_path": None,               # Custom weights path
})
```

### SuperPoint

**Key:** `"superpoint"`

```python
extractor = getExtractor("superpoint", {
    "top_k": -1,                      # Max keypoints (-1 = no limit)
    "sparse_outputs": True,           # Return sparse keypoints
    "dense_outputs": False,           # Also return dense maps
    "nms_radius": 4,                  # NMS radius
    "refinement_radius": 0,           # Keypoint refinement
    "detection_threshold": 0.005,     # Score threshold
    "remove_borders": 4,             # Remove keypoints near border (pixels)
    "legacy_sampling": True,          # Legacy keypoint sampling
    "force_num_keypoints": False,     # Force exactly top_k keypoints
})
```

**Config aliases:**

| Alias | Canonical |
|-------|-----------|
| `keypoint_threshold` | `detection_threshold` |
| `max_keypoints` | `top_k` |

### SuperPoint Open

**Key:** `"superpoint_open"`

```python
extractor = getExtractor("superpoint_open", {
    "top_k": 2048,                   # Max keypoints
    "nms_radius": 4,                 # NMS radius
    "force_num_keypoints": False,    # Force exactly top_k keypoints
    "detection_threshold": 0.005,    # Score threshold
    "remove_borders": 4,            # Remove keypoints near border
    "descriptor_dim": 256,          # Descriptor dimension
    "channels": [64, 64, 128, 128, 256],  # Encoder channels
    "weights": None,                 # Custom weights path
})
```

### XFeat

**Key:** `"xfeat"`

```python
extractor = getExtractor("xfeat", {
    "model_name": "xfeat",          # Model variant
    "top_k": 2048,                  # Max keypoints
    "detection_threshold": 0.2,      # Score threshold
    "nms_radius": 4,                # NMS radius
    "width_confidence": 0.5,        # Width confidence
    "min_corner_score": 0.0,        # Minimum corner score
})
```

### ORB

**Key:** `"orb"`

Classical OpenCV ORB detector+descriptor.

```python
import cv2
extractor = getExtractor("orb", {
    "top_k": 2048,                         # Max keypoints
    "scaleFactor": 1.2,                    # Pyramid scale factor
    "nlevels": 8,                          # Number of pyramid levels
    "edgeThreshold": 31,                   # Edge threshold
    "firstLevel": 0,                       # First pyramid level
    "WTA_K": 2,                            # Points in BRIEF descriptor
    "scoreType": cv2.ORB_HARRIS_SCORE,     # Score type
    "patchSize": 31,                       # BRIEF patch size
    "fastThreshold": 20,                   # FAST threshold
})
```

### SFD2

**Key:** `"sfd2"`

```python
extractor = getExtractor("sfd2", {
    "top_k": 2048,                 # Max keypoints
    "model_name": "ressegnetv2",   # Model variant
    "use_stability": True,         # Use stability score
    "conf_th": 0.001,             # Confidence threshold
    "scales": [1.0],              # Multi-scale factors
})
```

### Desc Reasoning

**Key:** `"desc_reasoning"`

```python
extractor = getExtractor("desc_reasoning", {
    "checkpoint_path": None,           # Custom checkpoint
    "pretrained": "xfeat",            # Pretrained backbone
    "weights_path": None,             # Custom weights
    "device": "cpu",                  # Device
    "cache_namespace": "desc_reasoning",
    "top_k": 2048,                    # Max keypoints
})

# Variation syntax for pretrained backbone
extractor = getExtractor("desc_reasoning:xfeat")
```

---

## Descriptor-only

These methods require an external detector (use `.addDetector()`).

### SOSNet

**Key:** `"sosnet"`

```python
extractor = getExtractor("sosnet", {
    "model_name": "sosnet",         # Model variant
    "top_k": 2048,                  # Max keypoints (for attached detector)
    "detection_threshold": 0.2,      # Detection threshold
    "nms_radius": 4,                # NMS radius
    "desc_dim": 128,                # Descriptor dimension
})
```

### TFeat

**Key:** `"tfeat"`

```python
extractor = getExtractor("tfeat", {
    "model_name": "tfeat",          # Model variant
    "top_k": 2048,                  # Max keypoints
    "detection_threshold": 0.2,      # Detection threshold
    "nms_radius": 4,                # NMS radius
    "desc_dim": 128,                # Descriptor dimension
})
```

### DINOv2

**Key:** `"dinov2"`

```python
extractor = getMethod("dinov2", {
    "weights": "dinov2_vits14",     # Model variant
    "allow_resize": True,           # Auto-resize for patch alignment
    "normalize": "imagenet",        # Normalization method
})
```

### DINOv3

**Key:** `"dinov3"`

```python
extractor = getMethod("dinov3", {
    "weights": "dinov3_vits16",     # Model variant
    "allow_resize": True,           # Auto-resize for patch alignment
    "repo_dir": None,              # Local repo directory
    "weights_path": None,          # Custom weights path
    "source": "hub",               # Weight source ("hub" or "local")
    "normalize": "auto",           # Normalization method
})
```

### ResNet

**Key:** `"resnet"`

```python
extractor = getMethod("resnet", {
    "weights": "resnet18",          # Model variant (resnet18, resnet34, etc.)
    "allow_resize": True,           # Auto-resize
})
```

### VGG

**Key:** `"vgg"`

```python
extractor = getMethod("vgg", {
    "weights": "vgg11",             # Model variant (vgg11, vgg16, etc.)
    "allow_resize": True,           # Auto-resize
})
```

### MuM

**Key:** `"mum"`

```python
extractor = getMethod("mum", {
    "pretrained": True,             # Use pretrained weights
    "resize_size": (256, 256),      # Input resize dimensions
})
```

### CroCo

**Key:** `"croco"`

```python
extractor = getMethod("croco", {
    "weights": "CroCo",            # Model variant
    "allow_resize": True,           # Auto-resize
})
```

---

## Detector-only

These methods only detect keypoints (no descriptors).

### DAD

**Key:** `"dad"`

```python
detector = getDetector("dad", {
    "num_keypoints": 1024,     # Number of keypoints to detect
    "resize": 1024,            # Resize input to this size
    "nms_size": 3,             # NMS window size
})
```

### REKD

**Key:** `"rekd"`

```python
detector = getDetector("rekd", {
    "num_keypoints": 1500,     # Number of keypoints to detect
    "pyramid_levels": 5,       # Scale pyramid levels
    "upsampled_levels": 2,     # Upsampled pyramid levels
    "border_size": 15,         # Border exclusion zone (pixels)
    "nms_size": 15,            # NMS window size
    "weights": None,           # Custom weights path
    "resize": None,            # Resize input (None = no resize)
})
```

---

## End-to-End Matchers

### LightGlue

**Key:** `"lightglue"` (supports variation syntax)

```python
# LightGlue with SuperPoint (default)
extractor = getExtractor("lightglue:superpoint", {"top_k": 2048})

# LightGlue with DISK
extractor = getExtractor("lightglue:disk", {"top_k": 2048})

# LightGlue with ALIKED
extractor = getExtractor("lightglue:aliked", {"top_k": 2048})

# LightGlue with DeDoDe
extractor = getExtractor("lightglue:dedode", {"top_k": 2048})
```

Config:

```python
{
    "features": "superpoint",   # Feature backend
    "top_k": 2048,              # Max keypoints
}
```

### SuperGlue

**Key:** `"superglue"`

```python
extractor = getExtractor("superglue")
```

Constructor parameters: `weights='indoor'`, `sinkhorn_iterations=100`, `match_threshold=0.2`, `descriptor_dim=256`.

### LoFTR

**Key:** `"loftr"`

```python
extractor = getExtractor("loftr")
```

Constructor parameters: `pretrained="outdoor"`.

### RoMa

**Key:** `"roma"`

```python
extractor = getExtractor("roma", {
    "model_name": "roma",          # Model variant
    "top_k": 512,                  # Max matches
    "detection_threshold": 0.2,     # Detection threshold
    "nms_radius": 4,               # NMS radius
    "model": "outdoor",            # "outdoor" or "indoor"
    "upsample_factor": 8,          # Upsample factor for dense matching
})
```

### RoMa v2

**Key:** `"romav2"`

```python
extractor = getExtractor("romav2", {
    "model_name": "romav2",        # Model variant
    "top_k": 5000,                 # Max matches
    "setting": "precise",          # "precise" or "fast"
    "compile": False,              # torch.compile the model
    "device": "cpu",               # Device
    "H_lr": None,                  # Low-res height (None = auto)
    "W_lr": None,                  # Low-res width
    "H_hr": None,                  # High-res height
    "W_hr": None,                  # High-res width
})
```

### TopicFM

**Key:** `"topicfm"`

```python
extractor = getExtractor("topicfm")
```
