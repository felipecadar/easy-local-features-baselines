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
from easy_local_features.feature.baseline_deal import DEAL_baseline
# from easy_local_features.feature.baseline_dalf import DALF_baseline
# from easy_local_features.feature.baseline_aliked import ALIKED_baseline
# from easy_local_features.feature.baseline_disk import DISK_baseline
# from easy_local_features.feature.baseline_r2d2 import R2D2_baseline
# from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
# from easy_local_features.matching.baseline_superglue import SuperGlue_baseline

# Load an image
img = cv2.imread("assets/notredame.png")

# Initialize the extractor
extractor = DEAL_baseline()

# Return keypoints and descriptors just like OpenCV
keypoints, descriptors = extractor.detectAndCompute(img)

```
# TODO

- [x] Add a setup.py to make it a pip package
- [ ] Make a general maching class
  - The idea is to have a class that can match images using any local feature extractor and any matching method.
- [ ] Fix requirements to install automatically with the package (maybe)
- [ ] Add a script to download some datasets
- [ ] Add a download script for the pretrained models
- [ ] Add more baselines :)
  - [x] DEAL
    - [ ] Add LICENSE file
  - [x] DALF
    - [ ] Add LICENSE file
  - [ ] DKM
  - [ ] ASLFeat
  - [x] R2D2
  - [x] DISK
  - [x] SuperPoin
  - [x] SuperGlue
  - [x] LoFTR
  - [ ] ALIKED
    - [x] Add LICENSE file
    - [x] Test on MAC M1 with CPU
    - [ ] Test on Linux with CPU
    - [ ] Test with CUDA