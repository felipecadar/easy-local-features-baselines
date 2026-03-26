# Getting Started

## Installation

Install from PyPI:

```bash
pip install easy-local-features
```

Or install from source (for development):

```bash
git clone https://github.com/felipecadar/easy-local-features-baselines.git
cd easy-local-features-baselines
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch (CPU or CUDA)
- See `pyproject.toml` for the full dependency list

## Your first match

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, vis

# Load images (returns [1, 3, H, W] float tensors)
img0 = io.fromPath("tests/assets/megadepth0.jpg")
img1 = io.fromPath("tests/assets/megadepth1.jpg")

# Create an extractor and match
extractor = getExtractor("aliked", {"top_k": 2048})
matches = extractor.match(img0, img1)

# Visualize
vis.plot_pair(img0, img1, title="ALIKED")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
vis.save("results/aliked.png")
```

The `matches` dictionary contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `mkpts0` | `(M, 2)` | Matched keypoints in image 0 |
| `mkpts1` | `(M, 2)` | Matched keypoints in image 1 |

Additional keys may be present depending on the matcher (see [Matching](user-guide/matching.md)).

## Detect and describe (without matching)

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io

img = io.fromPath("tests/assets/megadepth0.jpg")

extractor = getExtractor("superpoint", {"top_k": 1024})
keypoints, descriptors = extractor.detectAndCompute(img)

print(keypoints.shape)    # [1, N, 2]
print(descriptors.shape)  # [1, N, 256]
```

Or use `return_dict=True`:

```python
result = extractor.detectAndCompute(img, return_dict=True)
# result["keypoints"]    -> [1, N, 2]
# result["descriptors"]  -> [1, N, D]
```

## Detect only

```python
from easy_local_features import getDetector
from easy_local_features.utils import io

img = io.fromPath("tests/assets/megadepth0.jpg")
detector = getDetector("rekd", {"num_keypoints": 1500})
kpts = detector.detect(img)  # [1, N, 2]
```

## Resize images before processing

Large images can be slow. Resize to a manageable size:

```python
from easy_local_features.utils import ops

img0 = io.fromPath("tests/assets/megadepth0.jpg")
img0, scale = ops.resize_short_edge(img0, 640)
# scale tells you how to map keypoints back to the original resolution
```

## Move to GPU

```python
extractor = getExtractor("xfeat", {"top_k": 4096}).to("cuda")
# All subsequent operations run on GPU
matches = extractor.match(img0, img1)
```

Apple Silicon:

```python
extractor = getExtractor("xfeat").to("mps")
```

## Inspect configuration (no model download)

```python
from easy_local_features import describe

info = describe("superpoint")
print(info["defaults"])
# {'top_k': -1, 'sparse_outputs': True, 'dense_outputs': False,
#  'nms_radius': 4, 'detection_threshold': 0.005, ...}
```

Or from the CLI:

```bash
easy-local-features --describe superpoint
```

## List available methods

```python
from easy_local_features import available_extractors, available_detectors, available_methods

print(available_extractors)  # ['alike', 'aliked', 'd2net', 'deal', ...]
print(available_detectors)   # ['dad', 'rekd']
print(available_methods)     # all of the above + dinov2, dinov3, resnet, ...
```

Or from the CLI:

```bash
easy-local-features --list        # extractors only
easy-local-features --list-detectors
easy-local-features --list-all    # everything
```

## What's next?

- [Features overview](user-guide/features.md) &mdash; understand method types and how to use each one
- [Matching guide](user-guide/matching.md) &mdash; configure matchers, tune thresholds
- [Visualization guide](user-guide/visualization.md) &mdash; plot keypoints, matches, depth maps
- [Extractors reference](reference/extractors.md) &mdash; full list with all config keys
