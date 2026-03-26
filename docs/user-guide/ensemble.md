# Detector Ensemble

The `EnsembleDetector` combines multiple detectors into a single detector. It runs each detector on every image, merges the keypoints, and optionally deduplicates and caps the total count.

## Basic usage

```python
from easy_local_features import getExtractor, getDetector
from easy_local_features.feature import EnsembleDetector, EnsembleDetectorConfig
from easy_local_features.utils import io

img = io.fromPath("tests/assets/megadepth0.jpg")

# Create individual detectors
orb = getExtractor("orb", {"top_k": 1024})
sp = getExtractor("superpoint", {"top_k": 1024, "detection_threshold": 0.005})

# Combine them
cfg = EnsembleDetectorConfig(
    deduplicate=True,     # Remove duplicate keypoints
    sort=True,            # Sort lexicographically
    max_keypoints=2048,   # Cap total keypoints
)
detector = EnsembleDetector([orb, sp], cfg)

# Detect
kpts = detector.detect(img)
print(f"Ensemble detected {kpts.shape[1]} keypoints")  # [1, N, 2]
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deduplicate` | `bool` | `True` | Remove duplicate keypoints from different detectors |
| `sort` | `bool` | `True` | Sort keypoints lexicographically by (y, x) |
| `max_keypoints` | `int` or `None` | `None` | Cap total keypoints. `None` = no limit. |

## Using with descriptor-only methods

The primary use case for `EnsembleDetector` is to provide keypoints to descriptor-only methods:

```python
from easy_local_features import getExtractor, getDetector
from easy_local_features.feature import EnsembleDetector, EnsembleDetectorConfig
from easy_local_features.utils import io, ops, vis

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

# Build ensemble detector
orb = getExtractor("orb", {"top_k": 1024})
sp = getExtractor("superpoint", {"top_k": 1024})
ensemble = EnsembleDetector(
    [orb, sp],
    EnsembleDetectorConfig(deduplicate=True, max_keypoints=2048),
)

# Attach to descriptor-only method
desc = getExtractor("sosnet").to("cpu")
desc.addDetector(ensemble)

# Now match using ensemble keypoints + SOSNet descriptors
matches = desc.match(img0, img1)

vis.plot_pair(img0, img1, title="Ensemble + SOSNet")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
vis.show()
```

## Mixing detector-only methods

You can include detector-only methods (DAD, REKD) in the ensemble:

```python
from easy_local_features import getExtractor, getDetector
from easy_local_features.feature import EnsembleDetector, EnsembleDetectorConfig

dad = getDetector("dad", {"num_keypoints": 512})
rekd = getDetector("rekd", {"num_keypoints": 512})
sp = getExtractor("superpoint", {"top_k": 512})

ensemble = EnsembleDetector(
    [dad, rekd, sp],
    EnsembleDetectorConfig(deduplicate=True, max_keypoints=1024),
)

kpts = ensemble.detect(img)
```

## GPU support

Move all underlying detectors to GPU:

```python
ensemble = EnsembleDetector([orb, sp]).to("cuda")
kpts = ensemble.detect(img)
```

Detectors that don't support `.to()` (like OpenCV-based ORB) will remain on CPU automatically.

## DetectorProtocol

Any object that implements the `detect(image) -> Tensor` method can be used in an ensemble. The returned tensor must have shape `(N, 2)` or `(1, N, 2)`:

```python
class MyDetector:
    def detect(self, image):
        # Your custom detection logic
        # Return keypoints as [N, 2] or [1, N, 2]
        return keypoints

ensemble = EnsembleDetector([MyDetector(), sp])
```
