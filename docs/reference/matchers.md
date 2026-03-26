# Matchers Reference

## NearestNeighborMatcher

The default matcher for all detect+describe and descriptor-only methods. Computes cosine similarity between descriptors and finds mutual nearest neighbors.

**Module:** `easy_local_features.matching.nearest_neighbor`

### Configuration

```python
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher

matcher = NearestNeighborMatcher({
    "ratio_thresh": None,       # Lowe's ratio test threshold
    "distance_thresh": None,    # Absolute distance threshold
    "mutual_check": True,       # Enforce mutual nearest neighbors
    "normalize": True,          # L2-normalize descriptors
})
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ratio_thresh` | `float` or `None` | `None` | Lowe's ratio test. Rejects matches where `d_best / d_second > threshold`. Common values: 0.7&ndash;0.9. `None` disables the test. |
| `distance_thresh` | `float` or `None` | `None` | Absolute distance filter. Rejects matches with distance above this value. `None` disables. |
| `mutual_check` | `bool` | `True` | Only keep matches where A's best match is B, and B's best match is A. Strongly recommended. |
| `normalize` | `bool` | `True` | L2-normalize descriptors before computing cosine similarity. |

### Input / Output

**Input dict:**

```python
{
    "descriptors0": Tensor[B, N, D],   # Descriptors from image 0
    "descriptors1": Tensor[B, M, D],   # Descriptors from image 1
}
```

**Output dict:**

```python
{
    "matches0": Tensor[B, N],           # For each kpt in img0: index of match in img1 (-1 = unmatched)
    "matches1": Tensor[B, M],           # For each kpt in img1: index of match in img0 (-1 = unmatched)
    "matching_scores0": Tensor[B, N],   # Binary confidence (0 or 1)
    "matching_scores1": Tensor[B, M],   # Binary confidence (0 or 1)
    "similarity": Tensor[B, N, M],      # Full cosine similarity matrix
}
```

### Example: Manual matching pipeline

```python
from easy_local_features import getExtractor
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from easy_local_features.utils import io

img0 = io.fromPath("tests/assets/megadepth0.jpg")
img1 = io.fromPath("tests/assets/megadepth1.jpg")

# Detect + describe
extractor = getExtractor("superpoint", {"top_k": 2048})
kpts0, desc0 = extractor.detectAndCompute(img0)
kpts1, desc1 = extractor.detectAndCompute(img1)

# Match
matcher = NearestNeighborMatcher({"ratio_thresh": 0.8, "mutual_check": True})
result = matcher({"descriptors0": desc0, "descriptors1": desc1})

# Extract matched keypoints
valid = result["matches0"][0] > -1
mkpts0 = kpts0[0, valid]
mkpts1 = kpts1[0, result["matches0"][0, valid]]
print(f"Found {len(mkpts0)} matches")
```

### Replacing the matcher on an extractor

```python
extractor = getExtractor("aliked", {"top_k": 2048})

# Strict matching
extractor.matcher = NearestNeighborMatcher({
    "ratio_thresh": 0.7,
    "distance_thresh": 1.0,
    "mutual_check": True,
})

matches = extractor.match(img0, img1)
```

---

## LightGlue

Learned feature matcher that uses attention to establish correspondences. Wraps a feature extractor internally.

**Module:** `easy_local_features.matching.baseline_lightglue`

```python
from easy_local_features import getExtractor

# Use variation syntax to select features
lg = getExtractor("lightglue:superpoint", {"top_k": 2048}).to("cuda")
matches = lg.match(img0, img1)
```

**Supported feature backends:** `superpoint`, `disk`, `aliked`, `dedode`

**Config:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `features` | `str` | `"superpoint"` | Feature extractor name |
| `top_k` | `int` | `2048` | Max keypoints per image |

---

## SuperGlue

Learned graph neural network matcher using attention and optimal transport.

**Module:** `easy_local_features.matching.baseline_superglue`

```python
sg = getExtractor("superglue").to("cuda")
matches = sg.match(img0, img1)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `str` | `"indoor"` | Weight variant (`"indoor"` or `"outdoor"`) |
| `sinkhorn_iterations` | `int` | `100` | Sinkhorn normalization iterations |
| `match_threshold` | `float` | `0.2` | Match confidence threshold |
| `descriptor_dim` | `int` | `256` | Descriptor dimension |

---

## LoFTR

Detector-free matcher that finds correspondences using coarse-to-fine matching with transformers.

**Module:** `easy_local_features.matching.baseline_loftr`

```python
loftr = getExtractor("loftr").to("cuda")
matches = loftr.match(img0, img1)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained` | `str` | `"outdoor"` | Weight variant |

---

## TopicFM

Topic-based feature matching using probabilistic topic models.

**Module:** `easy_local_features.matching.baseline_topicfm`

```python
tfm = getExtractor("topicfm").to("cuda")
matches = tfm.match(img0, img1)
```

---

## RoMa / RoMa v2

Dense matching methods that produce pixel-level correspondences.

```python
# RoMa v2 (recommended)
roma = getExtractor("romav2", {
    "top_k": 5000,
    "setting": "precise",
}).to("cuda")

# Original RoMa
roma_v1 = getExtractor("roma", {
    "top_k": 512,
    "model": "outdoor",
}).to("cuda")

matches = roma.match(img0, img1)
```

See [Extractors Reference](extractors.md) for full config details.
