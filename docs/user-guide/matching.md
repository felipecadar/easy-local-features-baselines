# Matching

## Default: Nearest-Neighbor Matcher

Every detect+describe and descriptor-only method comes with a built-in `NearestNeighborMatcher`. When you call `.match()`, it runs detection, description, and matching in one step.

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io

img0 = io.fromPath("tests/assets/megadepth0.jpg")
img1 = io.fromPath("tests/assets/megadepth1.jpg")

extractor = getExtractor("aliked", {"top_k": 2048})
matches = extractor.match(img0, img1)
```

### Match output dictionary

| Key | Shape | Description |
|-----|-------|-------------|
| `mkpts0` | `(M, 2)` | Matched keypoint coordinates in image 0 |
| `mkpts1` | `(M, 2)` | Matched keypoint coordinates in image 1 |
| `matches0` | `(N,)` | For each keypoint in img0, index of match in img1 (`-1` = unmatched) |
| `matches1` | `(N,)` | For each keypoint in img1, index of match in img0 (`-1` = unmatched) |
| `matching_scores0` | `(N,)` | Confidence score per keypoint in img0 (0 or 1 for NN matcher) |
| `matching_scores1` | `(N,)` | Confidence score per keypoint in img1 |
| `similarity` | `(N, M)` | Full similarity matrix between descriptors |

## Configuring the NN Matcher

You can customize the matcher by replacing it on the extractor:

```python
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher

extractor = getExtractor("superpoint", {"top_k": 2048})

# Replace the default matcher
extractor.matcher = NearestNeighborMatcher({
    "ratio_thresh": 0.8,       # Lowe's ratio test (None = disabled)
    "distance_thresh": None,    # Absolute distance threshold (None = disabled)
    "mutual_check": True,       # Enforce mutual nearest neighbor
    "normalize": True,          # L2-normalize descriptors before matching
})

matches = extractor.match(img0, img1)
```

### NearestNeighborMatcher config

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ratio_thresh` | `float` or `None` | `None` | Lowe's ratio test threshold. Rejects matches where best/second-best distance ratio exceeds this value. `0.8` is a common choice. |
| `distance_thresh` | `float` or `None` | `None` | Maximum allowed distance. Rejects matches beyond this absolute threshold. |
| `mutual_check` | `bool` | `True` | Only keep matches that are mutual nearest neighbors (A's best match is B, and B's best match is A). |
| `normalize` | `bool` | `True` | L2-normalize descriptors before computing similarity. |

### Example: Strict matching

```python
extractor.matcher = NearestNeighborMatcher({
    "ratio_thresh": 0.7,
    "distance_thresh": 1.0,
    "mutual_check": True,
})
```

### Example: Permissive matching (more matches, lower quality)

```python
extractor.matcher = NearestNeighborMatcher({
    "ratio_thresh": None,       # No ratio test
    "distance_thresh": None,    # No distance filter
    "mutual_check": False,      # Allow one-way matches
})
```

## End-to-End Matchers

For higher quality matching, use end-to-end learned matchers.

### LightGlue

LightGlue wraps a feature extractor and a learned matcher. Use the variation syntax to pick the feature backend:

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 640)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 640)[0]

# LightGlue with SuperPoint features
lg = getExtractor("lightglue:superpoint", {"top_k": 2048}).to("cuda")
matches = lg.match(img0, img1)

vis.plot_pair(img0, img1, title="LightGlue + SuperPoint")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
vis.show()
```

Supported feature backends: `superpoint`, `disk`, `aliked`, `dedode`.

### SuperGlue

```python
sg = getExtractor("superglue").to("cuda")
matches = sg.match(img0, img1)
```

### LoFTR

LoFTR is a **detector-free** matcher &mdash; it finds correspondences without detecting keypoints first.

```python
loftr = getExtractor("loftr").to("cuda")
matches = loftr.match(img0, img1)
```

### RoMa / RoMa v2

Dense matching methods that produce high-quality correspondences.

```python
# RoMa v2 (recommended)
roma = getExtractor("romav2", {"top_k": 5000, "setting": "precise"}).to("cuda")
matches = roma.match(img0, img1)

# Original RoMa
roma_v1 = getExtractor("roma", {"top_k": 512, "model": "outdoor"}).to("cuda")
matches = roma_v1.match(img0, img1)
```

### TopicFM

Topic-based feature matching:

```python
tfm = getExtractor("topicfm").to("cuda")
matches = tfm.match(img0, img1)
```

## Comparing matchers

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

# Method 1: SuperPoint + NN matcher
sp = getExtractor("superpoint", {"top_k": 2048}).to("cpu")
nn_matches = sp.match(img0, img1)

# Method 2: SuperPoint + NN matcher with ratio test
sp.matcher = NearestNeighborMatcher({"ratio_thresh": 0.8})
ratio_matches = sp.match(img0, img1)

# Method 3: LightGlue (end-to-end)
lg = getExtractor("lightglue:superpoint", {"top_k": 2048}).to("cpu")
lg_matches = lg.match(img0, img1)

print(f"NN:        {len(nn_matches['mkpts0'])} matches")
print(f"NN+ratio:  {len(ratio_matches['mkpts0'])} matches")
print(f"LightGlue: {len(lg_matches['mkpts0'])} matches")
```

## Manual matching pipeline

For more control, you can run detection, description, and matching separately:

```python
import torch
from easy_local_features import getExtractor
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from easy_local_features.utils import io

img0 = io.fromPath("tests/assets/megadepth0.jpg")
img1 = io.fromPath("tests/assets/megadepth1.jpg")

# Step 1: Detect and describe
extractor = getExtractor("xfeat", {"top_k": 4096})
kpts0, desc0 = extractor.detectAndCompute(img0)
kpts1, desc1 = extractor.detectAndCompute(img1)

# Step 2: Match descriptors manually
matcher = NearestNeighborMatcher({"ratio_thresh": 0.85, "mutual_check": True})
match_result = matcher({
    "descriptors0": desc0,
    "descriptors1": desc1,
})

# Step 3: Extract matched keypoints
valid0 = match_result["matches0"][0] > -1
mkpts0 = kpts0[0, valid0]
mkpts1 = kpts1[0, match_result["matches0"][0, valid0]]
print(f"Matched {len(mkpts0)} keypoints")
```
