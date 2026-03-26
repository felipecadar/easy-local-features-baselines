# Run Benchmarks

## Comparing extractors

```python
from easy_local_features import getExtractor, available_extractors
from easy_local_features.utils import io, ops, vis
import time

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 640)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 640)[0]

results = {}
for name in ["superpoint", "aliked", "xfeat", "disk", "orb"]:
    extractor = getExtractor(name, {"top_k": 2048}).to("cpu")

    start = time.time()
    matches = extractor.match(img0, img1)
    elapsed = time.time() - start

    n_matches = len(matches["mkpts0"])
    results[name] = {"matches": n_matches, "time": elapsed}
    print(f"{name:15s} | {n_matches:5d} matches | {elapsed:.3f}s")

    # Save visualization
    vis.plot_pair(img0, img1, title=f"{name} ({n_matches} matches)", figsize=(10, 5))
    vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
    vis.save(f"results/benchmark_{name}.png")
```

## Comparing matchers

```python
from easy_local_features import getExtractor
from easy_local_features.matching.nearest_neighbor import NearestNeighborMatcher
from easy_local_features.utils import io, ops

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

# Same features, different matchers
sp = getExtractor("superpoint", {"top_k": 2048}).to("cpu")

# NN (default)
m1 = sp.match(img0, img1)
print(f"NN default:     {len(m1['mkpts0'])} matches")

# NN + ratio test
sp.matcher = NearestNeighborMatcher({"ratio_thresh": 0.8})
m2 = sp.match(img0, img1)
print(f"NN ratio=0.8:   {len(m2['mkpts0'])} matches")

# NN strict
sp.matcher = NearestNeighborMatcher({"ratio_thresh": 0.7, "distance_thresh": 1.0})
m3 = sp.match(img0, img1)
print(f"NN strict:      {len(m3['mkpts0'])} matches")

# LightGlue
lg = getExtractor("lightglue:superpoint", {"top_k": 2048}).to("cpu")
m4 = lg.match(img0, img1)
print(f"LightGlue:      {len(m4['mkpts0'])} matches")
```

## Pose evaluation

The library includes a pose evaluation pipeline:

```python
from easy_local_features.eval.pose_eval import plot_matches_parallel
```

This function:

1. Loads image pairs
2. Runs RANSAC pose estimation via `poselib`
3. Separates inliers (green) from outliers (red)
4. Saves annotated visualizations

## Saving visualizations for comparison

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

methods = ["superpoint", "aliked", "xfeat", "disk", "d2net", "r2d2"]

for name in methods:
    extractor = getExtractor(name, {"top_k": 2048}).to("cpu")
    matches = extractor.match(img0, img1)

    vis.plot_pair(img0, img1, title=f"{name}", figsize=(10, 5))
    vis.plot_matches(matches["mkpts0"], matches["mkpts1"], linewidth=0.3)
    vis.add_text(f"{len(matches['mkpts0'])} matches")
    vis.save(f"results/{name}.png", dpi=150, bbox_inches='tight')
    vis.plt.close('all')
```

## Tips

- Use `ops.resize_short_edge` to standardize image sizes across methods.
- Save all outputs to `results/` for easy comparison.
- Use `dpi=150` or higher for publication-quality figures.
- Close figures with `vis.plt.close('all')` in batch loops to prevent memory issues.
- For timing benchmarks, run each method multiple times and take the average (first run may include model download).
