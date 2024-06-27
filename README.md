# easy-local-features-baselines

Just some scripts to make things easier for the local features baselines.

<div style="color:red;">WARNING: PLEASE check the license of your desired baseline before using this code.</div>

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
from easy_local_features.feature.baseline_aliked import ALIKED_baseline
# from easy_local_features.feature.baseline_alike import ALIKE_baseline
# from easy_local_features.feature.baseline_deal import DEAL_baseline
# from easy_local_features.feature.baseline_dalf import DALF_baseline
# from easy_local_features.feature.baseline_disk import DISK_baseline
# from easy_local_features.feature.baseline_dedode import DeDoDe_baseline
# from easy_local_features.feature.baseline_d2net import D2Net_baseline
# from easy_local_features.feature.baseline_delf import DELF_baseline
# from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
# from easy_local_features.feature.baseline_r2d2 import R2D2_baseline
# from easy_local_features.feature.baseline_sosnet import SOSNet_baseline
# from easy_local_features.feature.baseline_tfeat import TFeat_baseline

from easy_local_features.utils import vis, io

# Load an image
image0 = io.fromPath("assets/v_vitro/1.ppm")
image1 = io.fromPath("assets/v_vitro/2.ppm")

# Load the extractor
extractor = ALIKED_baseline({'top_k': 128})

# Macth directly
matches = extractor.match(image0, image1)

# OR

# Extract
# keypoints0, descriptors0 = extractor.detectAndCompute(image0)
# keypoints1, descriptors1 = extractor.detectAndCompute(image1)
# matches = extractor.matcher({
#     'descriptors0': descriptors0,
#     'descriptors1': descriptors1,
#})

# Visualize
vis.plot_pair(image0, image1)
vis.plot_matches(matches['mkpts0'], matches['mkpts1'])
vis.show(f"test/results/{extractor.__name__}.png")

```

# TODO REFACTOR
- [x] ALIKE
- [x] ALIKED
- [x] DEAL
- [x] DALF
- [x] DISK
- [x] DeDoDe
- [x] D2Net
- [x] DELF
- [x] SuperPoint
- [x] R2D2
- [x] LogPolar
- [x] SOSNet
- [x] TFeat
- [ ] DKM
- [ ] ASLFeat
- [ ] SuperGlue
- [ ] LightGlue
- [ ] LoFTR
