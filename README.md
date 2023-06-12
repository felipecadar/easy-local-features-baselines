# easy-local-features-baselines

Just some scripts to make things easier for the local features baselines.

# Installation

```bash
pip install -r requirements.txt
pip install .
```

# How to use

```python
from easy_local_features import DEAL
# from easy_local_features import DALF, DISK, R2D2, SuperPoint, SuperGlue

# Load an image
img = cv2.imread("assets/notredame.png")

# Initialize the extractor
extractor = DEAL()

# Return keypoints and descriptors just like OpenCV
keypoints, descriptors = extractor.detectAndCompute(img)

```
# TODO

- [x] Add a setup.py to make it a pip package
- [ ] Add a script to download some datasets
- [ ] Add a download script for the pretrained models
- [ ] Add more baselines :)
  - [ ] ASLFeat
  - [ ] DELF
  - [ ] LoFTR
  - [ ] DKM