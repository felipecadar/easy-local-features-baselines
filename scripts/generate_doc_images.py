"""Generate documentation images for the visualization guide using XFeat."""

import os
import sys
import numpy as np

# Ensure matplotlib uses non-interactive backend
import matplotlib
matplotlib.use("Agg")

from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis

OUT = "docs/assets/images"
os.makedirs(OUT, exist_ok=True)

# --- Load images and extractor ------------------------------------------------

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 480)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 480)[0]

extractor = getExtractor("xfeat", {"top_k": 2048}).to("cuda")
matches = extractor.match(img0, img1)
mkpts0, mkpts1 = matches["mkpts0"], matches["mkpts1"]

# Also get raw keypoints for the keypoints-only demo
kp0, desc0 = extractor.detectAndCompute(img0)
kp1, desc1 = extractor.detectAndCompute(img1)
kp0 = kp0.squeeze(0)  # [1, N, 2] -> [N, 2]
kp1 = kp1.squeeze(0)

print(f"Matched {len(mkpts0)} points, detected {len(kp0)}+{len(kp1)} keypoints")

# --- 1. Basic plot_pair -------------------------------------------------------

vis.plot_pair(img0, img1, title="XFeat — plot_pair", figsize=(10, 5))
vis.save(f"{OUT}/vis_plot_pair.png", dpi=150, bbox_inches="tight")
print("  [1/9] plot_pair")

# --- 2. plot_keypoints (rainbow) ----------------------------------------------

vis.plot_pair(img0, img1, title="XFeat — plot_keypoints", figsize=(10, 5))
vis.plot_keypoints(kp0[:500], kp1[:500], kps_size=3)
vis.save(f"{OUT}/vis_plot_keypoints.png", dpi=150, bbox_inches="tight")
print("  [2/9] plot_keypoints")

# --- 3. plot_matches (rainbow, default) ---------------------------------------

vis.plot_pair(img0, img1, title="XFeat — plot_matches (rainbow)", figsize=(10, 5))
vis.plot_matches(mkpts0, mkpts1, linewidth=0.4, alpha=0.7)
vis.save(f"{OUT}/vis_plot_matches_rainbow.png", dpi=150, bbox_inches="tight")
print("  [3/9] plot_matches rainbow")

# --- 4. plot_matches (single color) ------------------------------------------

vis.plot_pair(img0, img1, title="XFeat — plot_matches (green)", figsize=(10, 5))
vis.plot_matches(mkpts0, mkpts1, color="g", linewidth=0.4, alpha=0.7)
vis.save(f"{OUT}/vis_plot_matches_green.png", dpi=150, bbox_inches="tight")
print("  [4/9] plot_matches green")

# --- 5. Full workflow: keypoints + matches + text -----------------------------

vis.plot_pair(img0, img1, title="XFeat", figsize=(10, 5))
vis.plot_keypoints(mkpts0, mkpts1, kps_size=2, color="b")
vis.plot_matches(mkpts0, mkpts1, linewidth=0.3, alpha=0.5)
vis.add_text(f"Matches: {len(mkpts0)}")
vis.save(f"{OUT}/vis_full_workflow.png", dpi=150, bbox_inches="tight")
print("  [5/9] full workflow")

# --- 6. Inlier / outlier color coding ----------------------------------------

# Simulate inlier mask via RANSAC-style random split (deterministic)
rng = np.random.RandomState(42)
inliers = rng.rand(len(mkpts0)) > 0.3  # ~70% inliers

_mkpts0 = mkpts0.cpu().numpy() if hasattr(mkpts0, "cpu") else np.asarray(mkpts0)
_mkpts1 = mkpts1.cpu().numpy() if hasattr(mkpts1, "cpu") else np.asarray(mkpts1)

vis.plot_pair(img0, img1, title="XFeat — inlier / outlier coloring", figsize=(10, 5))
vis.plot_matches(_mkpts0[~inliers], _mkpts1[~inliers], color="r", alpha=0.3, linewidth=0.4)
vis.plot_matches(_mkpts0[inliers], _mkpts1[inliers], color="g", linewidth=0.4)
vis.add_text(f"Inliers: {inliers.sum()} / {len(inliers)}")
vis.save(f"{OUT}/vis_inliers_outliers.png", dpi=150, bbox_inches="tight")
print("  [6/9] inlier/outlier")

# --- 7. Vertical layout ------------------------------------------------------

vis.plot_pair(img0, img1, title="XFeat — vertical layout", vertical=True, figsize=(6, 10))
vis.plot_matches(mkpts0, mkpts1, linewidth=0.3, alpha=0.6)
vis.save(f"{OUT}/vis_vertical.png", dpi=150, bbox_inches="tight")
print("  [7/9] vertical layout")

# --- 8. Color image (gray=False) ---------------------------------------------

vis.plot_pair(img0, img1, title="XFeat — color mode", figsize=(10, 5), gray=False)
vis.plot_matches(mkpts0, mkpts1, linewidth=0.4, alpha=0.7)
vis.save(f"{OUT}/vis_color_mode.png", dpi=150, bbox_inches="tight")
print("  [8/9] color mode")

# --- 9. Index page hero image ------------------------------------------------

vis.plot_pair(img0, img1, title="XFeat — 2048 keypoints", figsize=(10, 5))
vis.plot_matches(mkpts0, mkpts1, linewidth=0.3, alpha=0.6)
vis.add_text(f"{len(mkpts0)} matches")
vis.save(f"{OUT}/hero_xfeat.png", dpi=150, bbox_inches="tight")
print("  [9/9] hero image")

print(f"\nAll images saved to {OUT}/")
