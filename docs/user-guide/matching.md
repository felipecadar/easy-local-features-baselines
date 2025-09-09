# Matching

By default, methods use a simple nearest-neighbor matcher with optional ratio and distance thresholds.

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

ex = getExtractor("aliked", {"top_k": 2048})
ex.matcher = NearestNeighborMatcher({
  "ratio_thresh": 0.8,
  "distance_thresh": None,
  "mutual_check": True,
})
res = ex.match(img0, img1)
```

Advanced matchers like LoFTR, SuperGlue, and LightGlue are available in `matching/`.
