# Run on GPU

Most extractors and matchers can be moved to GPU for significant speedups.

## CUDA (NVIDIA GPUs)

```python
from easy_local_features import getExtractor

extractor = getExtractor("aliked", {"top_k": 2048}).to("cuda")

# All operations now run on GPU
matches = extractor.match(img0, img1)
```

### Specific GPU

```python
extractor = getExtractor("xfeat").to("cuda:0")  # First GPU
extractor = getExtractor("xfeat").to("cuda:1")  # Second GPU
```

## Apple Silicon (MPS)

```python
extractor = getExtractor("superpoint", {"top_k": 1024}).to("mps")
matches = extractor.match(img0, img1)
```

## CPU fallback

Some methods (like ORB, which uses OpenCV) always run on CPU. Calling `.to("cuda")` on them is safe &mdash; it simply has no effect.

```python
extractor = getExtractor("orb").to("cuda")  # ORB stays on CPU, no error
```

## Ensemble detectors

`EnsembleDetector.to()` moves all underlying detectors that support it:

```python
from easy_local_features.feature import EnsembleDetector

ensemble = EnsembleDetector([orb, superpoint]).to("cuda")
# superpoint moves to GPU, orb stays on CPU
```

## End-to-end matchers

End-to-end matchers also support `.to()`:

```python
lg = getExtractor("lightglue:superpoint", {"top_k": 2048}).to("cuda")
loftr = getExtractor("loftr").to("cuda")
roma = getExtractor("romav2", {"top_k": 5000}).to("cuda")
```

## Checking device

```python
import torch

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

extractor = getExtractor("aliked", {"top_k": 2048}).to(device)
```

## Performance tips

- **Resize images** before processing. Use `ops.resize_short_edge(img, 640)` to keep images manageable.
- **Batch processing** with `plt.close('all')` after each visualization to prevent memory leaks.
- **`top_k`** controls the maximum number of keypoints. Lower values are faster.
- End-to-end matchers (LightGlue, LoFTR) benefit the most from GPU acceleration.
