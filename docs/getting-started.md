# Getting started

Install from PyPI:

```bash
pip install easy-local-features
```

Or from source:

```bash
pip install -e .
```

## Minimal example

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, vis

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

extractor = getExtractor("aliked", {"top_k": 1024})
matches = extractor.match(img0, img1)

vis.plot_pair(img0, img1, title="ALIKED")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"]) 
vis.save("results/aliked.png")
```

