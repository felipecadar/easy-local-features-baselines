# easy-local-features-baselines

Just some scripts to make things easier for the local features baselines.

# Installation

```bash
# make sure you have torch installed
# pip install torch torchvision
pip install easy-local-features
```

## Installing from source

You may want to install from source if you want to modify the code or if you want to use the latest version. To do so, you can clone this repository and install the requirements.

I suggest using a conda environment to install the requirements. You can create one using the following command.

```bash
conda create -n elf python=3.9 # the python version is not so critical, but I used 3.9.
conda activate elf
```

Now we can install everything.

```bash
pip install -r requirements.txt
pip install -e .
```

# How to use

```python
# Choose you extractor
from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
# from easy_local_features.feature.baseline_deal import DEAL_baseline
# from easy_local_features.feature.baseline_dalf import DALF_baseline
# from easy_local_features.feature.baseline_aliked import ALIKED_baseline
# from easy_local_features.feature.baseline_alike import ALIKE_baseline
# from easy_local_features.feature.baseline_disk import DISK_baseline
# from easy_local_features.feature.baseline_r2d2 import R2D2_baseline

# also a matcher
from easy_local_features.matching.baseline_lightglue import LightGlue_baseline
# from easy_local_features.matching.baseline_superglue import SuperGlue_baseline
# from easy_local_features.matching.baseline_loftr import LoFTR_baseline
import cv2

# Load an image
img = cv2.imread("assets/notredame.png")

# Initialize the extractor
extractor = SuperPoint_baseline()
matcher = LightGlue_baseline() # works with superpoint and disk

# Return keypoints and descriptors just like OpenCV
keypoints0, descriptors0 = extractor.detectAndCompute(img)
keypoints1, descriptors1 = extractor.detectAndCompute(img)

# Match the descriptors
mkpts0, mkpts1, matches = matcher.match(keypoints0, keypoints1, descriptors0, descriptors1)

img = cv2.drawMatches(img, keypoints0, img, keypoints1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Matched", img)
cv2.waitKey(0)



```
# TODO REFACTOR

  - [x] ALIKE
  - [x] ALIKED
  - [x] DEAL
  - [x] DALF
  - [x] DISK
  - [x] DeDoDe
  - [x] D2Net
  - [ ] SuperPoint
  - [ ] DKM
  - [ ] ASLFeat
  - [ ] R2D2
  - [ ] SuperGlue
  - [ ] LightGlue
  - [ ] LoFTR