# easy-local-features-baselines

Just some scripts to make things easier for the local features baselines.

> ⚠️ **CRITICAL LICENSE DISCLAIMER**
>
> This repository aggregates wrappers around MANY third‑party local feature extractors and matchers. **Each baseline/model keeps its OWN original license (BSD, MIT, Apache 2.0, GPLv3, Non‑Commercial, CC BY‑NC‑SA, custom research licenses, etc.)**. Your rights for a given baseline are governed *only* by that baseline’s upstream license. Some components here are **non‑commercial only** (e.g., SuperGlue, original SuperPoint, R2D2) or **copyleft** (e.g., DISK under GPLv3). Others are permissive (BSD/MIT/Apache 2.0). Before any research publication, internal deployment, redistribution, or commercial/production use, **YOU MUST review and comply with every relevant upstream license, including attribution, notice reproduction, share‑alike, copyleft, and patent clauses.**
>
> The maintainers provide NO warranty, NO guarantee of license correctness, and accept NO liability for misuse. This notice and any summaries are **not legal advice**. If in doubt, consult qualified counsel.
>
> See: [`LICENSES.md`](LICENSES.md) for an overview and links to included full license texts.
> Built with DINOv3.

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

## Run by method type

Below are minimal, copy-paste examples showing how to run each kind of method supported by this package. Methods are grouped into three types:

- Detect+Describe: extract keypoints and descriptors from each image, then match (default: nearest-neighbor with optional ratio/distance checks).
- Descriptor-only: compute descriptors for provided keypoints; you must attach a detector first (e.g., SuperPoint) or pass your own keypoints.
- End-to-end matcher: directly produce correspondences from two images without exposed keypoints/descriptors.

Images can be file paths, NumPy arrays (H×W×C or H×W), or torch tensors. Utilities in `easy_local_features.utils.io` and `easy_local_features.utils.ops` handle loading and formatting.

### 1) Detect + Describe methods

Examples: alike, aliked, d2net, deal, delf, disk, r2d2, superpoint, orb, xfeat

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, vis, ops

# Load images
img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

# Pick any detect+describe extractor; tweak conf as needed
extractor = getExtractor("aliked", {"top_k": 2048, "detection_threshold": 0.2})
# Optional: run on CPU/GPU
# extractor.to("cuda")

# One-shot matching (uses nearest-neighbor matcher by default)
matches = extractor.match(img0, img1)

# Visualize
vis.plot_pair(img0, img1, title="ALIKED")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"]) 
vis.save("results/aliked.png")
```

You can also decouple extraction and matching:

```python
data0 = extractor.detectAndCompute(img0, return_dict=True)
data1 = extractor.detectAndCompute(img1, return_dict=True)

nn = extractor.matcher  # default: NearestNeighborMatcher
res = nn({
	"descriptors0": data0["descriptors"],
	"descriptors1": data1["descriptors"],
})

m0 = res["matches0"][0]
valid = m0 > -1
mkpts0 = data0["keypoints"][0, valid]
mkpts1 = data1["keypoints"][0, m0[valid]]
```

Tip: customize matching thresholds

```python
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
extractor.matcher = NearestNeighborMatcher({
	"ratio_thresh": 0.8,        # Lowe’s ratio (applied in squared-distance space)
	"distance_thresh": None,    # absolute distance gate; set a float to enable
	"mutual_check": True,
})
```

### 2) Descriptor-only methods

Examples: sosnet, tfeat (others exist in the repo but may not be enabled by default)

Descriptor-only extractors require keypoints. Attach any detector (e.g., SuperPoint) via `addDetector`, then proceed as usual.

```python
from easy_local_features import getExtractor
from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
from easy_local_features.utils import io, vis

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

# 1) Create a descriptor-only extractor
desc = getExtractor("sosnet")  # or "tfeat"

# 2) Attach a detector (SuperPoint is a good default)
detector = SuperPoint_baseline({
	"nms_radius": 4,
	"detection_threshold": 0.005,
	"top_k": 2048,
})
desc.addDetector(detector)

# 3) Match
matches = desc.match(img0, img1)

vis.plot_pair(img0, img1, title="SOSNet + SuperPoint detector")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"]) 
vis.save("results/sosnet.png")
```

Alternatively, if you already have keypoints, call `compute(img, keypoints)` and perform matching yourself.

### Detector ensemble (combine multiple detectors)

You can aggregate several detectors into one via `EnsembleDetector`. It runs each detector per image, merges keypoints, optionally deduplicates overlapping points, and behaves like a single detector. This is useful to mix classical and learned detectors.

```python
from easy_local_features.feature import EnsembleDetector, EnsembleDetectorConfig
from easy_local_features.feature.baseline_orb import ORB_baseline
from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
from easy_local_features.utils import io, vis

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

# Build individual detectors (any class exposing detector.detect(image) -> [1,N,2] or [N,2])
orb = ORB_baseline({"top_k": 1024})
sp  = SuperPoint_baseline({"top_k": 1024, "legacy_sampling": False})

# Optional config: deduplicate overlapping kpts (rounded to pixel), sort, cap total
cfg = EnsembleDetectorConfig(deduplicate=True, sort=True, max_keypoints=2048)
detector = EnsembleDetector([orb, sp], cfg)

# Use as a detector in any descriptor-only pipeline
from easy_local_features import getExtractor
desc = getExtractor("sosnet")  # or "tfeat"
desc.addDetector(detector)
out = desc.match(img0, img1)

vis.plot_pair(img0, img1, title="SOSNet + (ORB ⊕ SuperPoint)")
vis.plot_matches(out["mkpts0"], out["mkpts1"]) 
vis.save("results/ensemble_desc.png")

# Or call detector.detect directly (single or batch)
kps0 = detector.detect(img0)  # [1, N, 2]
kpsb = detector.detect(torch.stack([img0, img1]))  # [B, Nmax, 2] padded per-image
```

Notes
- The ensemble loops over the batch and runs each detector per image (safe for OpenCV-based detectors like ORB).
- Deduplication rounds to the nearest pixel before uniquifying across detectors; disable with `deduplicate=False`.
- `max_keypoints` caps the final number per image (simple deterministic subsampling when needed).

### 3) End-to-end matchers

Examples: roma (feature-level API), matchers/LoFTR and matchers/SuperGlue (standalone matchers)

RoMa as a feature-level method:

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, vis

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

roma = getExtractor("roma", {"top_k": 512, "model": "outdoor"})
# roma.to("cuda")  # optional
out = roma.match(img0, img1)

vis.plot_pair(img0, img1, title="RoMa")
vis.plot_matches(out["mkpts0"], out["mkpts1"]) 
vis.save("results/roma.png")
```

LoFTR (standalone matcher):

```python
from easy_local_features.matching.baseline_loftr import LoFTR_baseline
from easy_local_features.utils import io, vis

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

loftr = LoFTR_baseline(pretrained="outdoor")
out = loftr.match(img0, img1)
vis.plot_pair(img0, img1, title="LoFTR")
vis.plot_matches(out["mkpts0"], out["mkpts1"]) 
vis.save("results/loftr.png")
```

LightGlue with built-in feature extraction:

```python
from easy_local_features.matching.baseline_lightglue import LightGlue_baseline
from easy_local_features.utils import io, vis

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

lg = LightGlue_baseline({"features": "superpoint", "top_k": 2048})
out = lg.match(img0, img1)
vis.plot_pair(img0, img1, title="LightGlue+SuperPoint")
vis.plot_matches(out["mkpts0"], out["mkpts1"]) 
vis.save("results/lightglue_sp.png")
```

SuperGlue (requires features you provide):

```python
from easy_local_features import getExtractor
from easy_local_features.matching.baseline_superglue import SuperGlue_baseline
from easy_local_features.utils import io

img0 = io.fromPath("test/assets/megadepth0.jpg")
img1 = io.fromPath("test/assets/megadepth1.jpg")

# Extract features (example: SuperPoint)
sp = getExtractor("superpoint", {"top_k": 2048})
d0 = sp.detectAndCompute(img0, return_dict=True)
d1 = sp.detectAndCompute(img1, return_dict=True)

sg = SuperGlue_baseline(weights="indoor", match_threshold=0.2)
res = sg.match(
	img0, img1,
	kps0=d0["keypoints"][0].cpu().numpy(),
	desc0=d0["descriptors"][0].cpu().numpy(),
	kps1=d1["keypoints"][0].cpu().numpy(),
	desc1=d1["descriptors"][0].cpu().numpy(),
)

# Convert to matched keypoints
import numpy as np, torch
m0 = torch.from_numpy(res["matches0"]).long()
valid = m0 > -1
mkpts0 = d0["keypoints"][0, valid]
mkpts1 = d1["keypoints"][0, m0[valid]]
```

### Available extractors by type (enabled by default)

- Detect+Describe: alike, aliked, d2net, deal, delf, disk, r2d2, superpoint, orb, xfeat
- Descriptor-only: sosnet, tfeat
- End-to-end: roma

Note: Additional baselines exist in the repo but may require extra dependencies or are not included in `available_extractors`. See `src/easy_local_features/feature/` for the full list. First run may download pretrained weights to your cache.

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
