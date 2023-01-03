# easy-local-features-baselines

Just some scripts to make things easier for the local features baselines.

# How to use

```python
import sys
sys.path.append('path/to/easy-local-features-baselines')

from baseline_r2d2 import R2D2_baseline

img = cv2.imread("assets/notredame.png")

extractor = R2D2_baseline()

keypoints, descriptors = extractor.detectAndCompute(img)

...
```
# TODO

- [ ] Add a script to download the datasets
- [ ] Add a setup.py to make it a pip package
- [ ] Add a download script for the pretrained models
- [ ] Add more baselines :)