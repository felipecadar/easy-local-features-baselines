# easy-local-features

Unified, minimal wrappers around many local feature extractors and matchers (classical + learned).

> **License disclaimer** &mdash; This repository wraps many third-party methods. Each baseline keeps its **own** upstream license.
> Review every relevant license before any use beyond personal experiments. See [`LICENSES.md`](https://github.com/felipecadar/easy-local-features-baselines/blob/main/LICENSES.md).

## What is this?

- A **unified interface** over 30+ detectors, descriptors, and matchers.
- Batteries-included **utilities** for loading, resizing, batching, and visualization.
- Simple nearest-neighbor matching by default, with optional advanced matchers (LightGlue, SuperGlue, LoFTR).
- Works on **CPU, CUDA, and MPS** devices.

## Quick example

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, vis

img0 = io.fromPath("tests/assets/megadepth0.jpg")
img1 = io.fromPath("tests/assets/megadepth1.jpg")

extractor = getExtractor("aliked", {"top_k": 2048}).to("cuda")
matches = extractor.match(img0, img1)

vis.plot_pair(img0, img1, title="ALIKED")
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
vis.save("results/aliked.png")
```

## Stable minimal API

| Function | Purpose |
|----------|---------|
| `getExtractor(name, conf)` | Get a feature extractor (detect+describe, descriptor-only, or end-to-end matcher) |
| `getDetector(name, conf)` | Get a detector-only method |
| `getMethod(name, conf)` | Unified factory for any method |
| `describe(name)` | Inspect config schema without loading the model |
| `.to(device)` | Move to `"cpu"`, `"cuda"`, or `"mps"` |
| `.match(img0, img1)` | Match two images &rarr; `{"mkpts0": (M,2), "mkpts1": (M,2), ...}` |
| `.detectAndCompute(img)` | Detect keypoints and compute descriptors |
| `.detect(img)` | Detect keypoints only |
| `.compute(img, kpts)` | Compute descriptors for given keypoints |
| `.addDetector(det)` | Attach an external detector to a descriptor-only method |

## Quick links

- [Getting Started](getting-started.md) &mdash; installation and first example
- **User Guide:** [Features](user-guide/features.md) | [Matching](user-guide/matching.md) | [Ensemble](user-guide/ensemble.md) | [I/O & Ops](user-guide/io-ops.md) | [Visualization](user-guide/visualization.md)
- **How-to:** [Add a new extractor](how-to/add-a-new-extractor.md) | [Run on GPU](how-to/run-on-gpu.md) | [Run benchmarks](how-to/run-benchmarks.md)
- **Reference:** [Extractors](reference/extractors.md) | [Matchers](reference/matchers.md)
- [CLI Reference](reference/cli.md)
- [API (auto-generated)](api/feature.md)
