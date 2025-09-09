# Detector ensemble

Combine multiple detectors into one via `EnsembleDetector`. It runs each detector per image, merges keypoints, and behaves like a single detector.

```python
from easy_local_features.feature import EnsembleDetector, EnsembleDetectorConfig
from easy_local_features.feature.baseline_orb import ORB_baseline
from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
from easy_local_features.utils import io

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

orb = ORB_baseline({"top_k": 1024})
sp  = SuperPoint_baseline({"top_k": 1024, "legacy_sampling": False})

cfg = EnsembleDetectorConfig(deduplicate=True, sort=True, max_keypoints=2048)
detector = EnsembleDetector([orb, sp], cfg)

# Single
kps0 = detector.detect(img0)
# Batch
# kpsb = detector.detect(torch.stack([img0, img1]))
```

Attach to descriptor-only extractors via `addDetector`.
