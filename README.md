# easy-local-features-baselines

Just some scripts to make things easier for the local features baselines.

# Installation


I suggest using a conda environment to install the requirements. You can create one using the following command.

```bash
conda create -n elf python=3.9 # the python version is not so critical, but I used 3.9.
conda activate elf
```

Now we can insall the requirements using pip.

```bash
pip install -r requirements.txt
pip install .
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
  - [x] R2D2
  - [x] DISK
  - [x] SuperPoin
  - [x] SuperGlue
  - [x] DALF
    - [ ] Add LICENSE file
  - [ ] ASLFeat
  - [x] LoFTR
    - [ ] Add LICENSE file
  - [ ] DKM
  - [ ] ALIKED
    - NOT WORKING ON MAC M1
    - [X] Add code 
    - [X] Add LICENSE file
    - [X] Test on MAC M1 with CPU
    - [ ] Test on Linux with CPU
    - [x] Test with CUDA