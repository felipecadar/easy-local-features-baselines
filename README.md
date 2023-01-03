# easy-local-features-baselines

Just some scripts to make things easier for the local features baselines.

# Installation

```bash
pip install -r requirements.txt
pip install .
```

# How to use

```python
from easy_local_features_baselines.baseline_r2d2 import R2D2_baseline

img = cv2.imread("assets/notredame.png")

extractor = R2D2_baseline()

keypoints, descriptors = extractor.detectAndCompute(img)

...
```
# TODO

- [x] Add a setup.py to make it a pip package
- [ ] Add a script to download some datasets
- [ ] Add a download script for the pretrained models
- [ ] Add more baselines :)