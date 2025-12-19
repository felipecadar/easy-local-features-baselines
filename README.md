# easy-local-features

Unified, minimal wrappers around many local feature extractors and matchers (classical + learned).

> ⚠️ **CRITICAL LICENSE DISCLAIMER**
>
> This repository aggregates wrappers around MANY third‑party local feature extractors and matchers. **Each baseline/model keeps its OWN original license (BSD, MIT, Apache 2.0, GPLv3, Non‑Commercial, CC BY‑NC‑SA, custom research licenses, etc.)**. Your rights for a given baseline are governed *only* by that baseline’s upstream license. Some components here are **non‑commercial only** (e.g., SuperGlue, original SuperPoint, R2D2) or **copyleft** (e.g., DISK under GPLv3). Others are permissive (BSD/MIT/Apache 2.0). Before any research publication, internal deployment, redistribution, or commercial/production use, **YOU MUST review and comply with every relevant upstream license, including attribution, notice reproduction, share‑alike, copyleft, and patent clauses.**
>
> The maintainers provide NO warranty, NO guarantee of license correctness, and accept NO liability for misuse. This notice and any summaries are **not legal advice**. If in doubt, consult qualified counsel.
>
> See: [`LICENSES.md`](LICENSES.md) for an overview and links to included full license texts.
> Built with DINOv3.

## Installation

```bash
pip install easy-local-features
```

## Installing from source

```bash
pip install -e .
```

## Usage

**Stable minimal API**
- `getExtractor(name, conf)` → returns an extractor
- `.to(device)` → `"cpu" | "cuda" | "mps"`
- `.match(img0, img1)` → `{"mkpts0": (M,2), "mkpts1": (M,2), ...}`
- Descriptor-only methods additionally support `.addDetector(detector)`

**Detector-only methods**
- Some methods only implement `detect(image) -> keypoints` (no descriptors, no matching). Use `getDetector(name, conf)` for those.

Example:

```python
from easy_local_features import getDetector
from easy_local_features.utils import io

img = io.fromPath("tests/assets/megadepth0.jpg")
det = getDetector("rekd", {"num_keypoints": 1500}).to("cpu")
kps = det.detect(img)  # [1, N, 2]
```

### Discover available config keys (no model init)

In code:

```python
from easy_local_features import describe
print(describe("superpoint"))
```

On the CLI:

```bash
easy-local-features --describe superpoint
```

### Detect+Describe (one-liner matching)

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis

img0 = io.fromPath("tests/assets/megadepth0.jpg")
img1 = io.fromPath("tests/assets/megadepth1.jpg")
img0 = ops.resize_short_edge(img0, 320)[0]
img1 = ops.resize_short_edge(img1, 320)[0]

extractor = getExtractor("aliked", {"top_k": 2048}).to("cpu")
out = extractor.match(img0, img1)

vis.plot_pair(img0, img1, title="ALIKED")
vis.plot_matches(out["mkpts0"], out["mkpts1"])
vis.save("tests/results/aliked.png")
```

### Descriptor-only (attach a detector)

```python
from easy_local_features import getExtractor
from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
from easy_local_features.utils import io, ops

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

desc = getExtractor("sosnet").to("cpu")
det = SuperPoint_baseline({"top_k": 2048, "detection_threshold": 0.005}).to("cpu")
desc.addDetector(det)

out = desc.match(img0, img1)
```

### End-to-end matchers

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

roma = getExtractor("romav2", {"top_k": 2000}).to("cpu")
out = roma.match(img0, img1)
```

## CLI

```bash
easy-local-features --list
```

```bash
easy-local-features --list-detectors
```
