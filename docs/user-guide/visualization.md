# Visualization

The `vis` module provides a chainable matplotlib-based API for visualizing keypoints, matches, and depth maps.

## Basic workflow

The typical pattern is: **create a figure** &rarr; **add overlays** &rarr; **save or show**.

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

extractor = getExtractor("aliked", {"top_k": 2048}).to("cpu")
matches = extractor.match(img0, img1)

# Step 1: Create a figure with two images
vis.plot_pair(img0, img1, title="ALIKED", figsize=(8, 4))

# Step 2: Add overlays
vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"], kps_size=2)
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
vis.add_text(f"Matches: {len(matches['mkpts0'])}")

# Step 3: Save or show
vis.save("results/aliked.png")
# or: vis.show()
```

## Creating figures

### `plot_pair` &mdash; Side-by-side image pair

```python
# Basic
vis.plot_pair(img0, img1)

# With title and custom size
vis.plot_pair(img0, img1, title="SuperPoint", figsize=(12, 6))

# Vertical layout
vis.plot_pair(img0, img1, vertical=True, figsize=(10, 20))

# Color images (default is grayscale)
vis.plot_pair(img0, img1, gray=False)

# Custom title font size
vis.plot_pair(img0, img1, title="Big Title", title_fontsize=24)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `img0`, `img1` | tensor / ndarray | required | The two images |
| `figsize` | `(w, h)` | `(20, 10)` | Figure size in inches |
| `title` | `str` | `None` | Figure title |
| `title_fontsize` | `int` | `None` | Title font size |
| `vertical` | `bool` | `False` | Stack images vertically |
| `gray` | `bool` | `True` | Convert to grayscale |
| `fig`, `ax` | matplotlib objects | `None` | Reuse existing figure/axes |

**Returns:** `(fig, ax)` tuple.

### `plot_image` &mdash; Single image

```python
vis.plot_image(img0)
vis.plot_image(img0, title="Input", figsize=(6, 6), gray=False)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | tensor / ndarray | required | The image |
| `figsize` | `(w, h)` | `(10, 10)` | Figure size |
| `title` | `str` | `None` | Figure title |
| `title_fontsize` | `int` | `None` | Title font size |
| `gray` | `bool` | `True` | Convert to grayscale |

## Plotting keypoints

### `plot_keypoints` &mdash; Scatter keypoints on a pair

Call after `plot_pair` to overlay keypoints on the existing figure.

```python
vis.plot_pair(img0, img1)

# Rainbow colors (default when color=None)
vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"])

# Single color for all keypoints
vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"], color='g')

# Custom marker size
vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"], kps_size=10)

# Custom marker style and transparency
vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"],
                   marker='x', alpha=0.5, kps_size=8)

# Different colormap
vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"], cmap='viridis')

# Only keypoints on image 0
vis.plot_keypoints(keypoints0=kpts0)

# Only keypoints on image 1
vis.plot_keypoints(keypoints1=kpts1)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keypoints0` | tensor / ndarray | `None` | Keypoints for image 0, shape `(N, 2)` |
| `keypoints1` | tensor / ndarray | `None` | Keypoints for image 1, shape `(N, 2)` |
| `color` | `str` | `None` | Color name (`'g'`, `'r'`, etc.), hex string, or `None` for rainbow |
| `kps_size` | `int` | `5` | Marker size |
| `marker` | `str` | `None` | Matplotlib marker style (`'o'`, `'x'`, `'+'`, `'s'`, etc.) |
| `alpha` | `float` | `1.0` | Transparency (0 = invisible, 1 = opaque) |
| `cmap` | `str` | `'hsv'` | Colormap for rainbow mode (when `color=None`) |
| `all_colors` | list | `None` | Explicit list of colors, one per keypoint |

### `draw_keypoints` &mdash; Burn keypoints into a tensor

Unlike `plot_keypoints`, this modifies image tensors directly (useful for TensorBoard, logging, etc.):

```python
# image: [B, 3, H, W] uint8 tensor
# keypoints: [B, N, 2] tensor
result = vis.draw_keypoints(image, keypoints, color='r', size=3)
# result: [B, 3, H, W] tensor with circles drawn
```

## Plotting matches

### `plot_matches` &mdash; Draw match lines between image pairs

Call after `plot_pair` to draw connection lines between matched keypoints.

```python
vis.plot_pair(img0, img1)

# Rainbow colors (default)
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])

# Single color
vis.plot_matches(matches["mkpts0"], matches["mkpts1"], color='g')

# Custom line width
vis.plot_matches(matches["mkpts0"], matches["mkpts1"], linewidth=1.5)

# Semi-transparent lines
vis.plot_matches(matches["mkpts0"], matches["mkpts1"], alpha=0.3)

# Different colormap
vis.plot_matches(matches["mkpts0"], matches["mkpts1"], cmap='coolwarm')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mkpts0` | tensor / ndarray | required | Matched keypoints in image 0, shape `(M, 2)` |
| `mkpts1` | tensor / ndarray | required | Matched keypoints in image 1, shape `(M, 2)` |
| `color` | `str` / list | `None` | Color or list of per-match colors. `None` = rainbow. |
| `linewidth` | `float` | `0.5` | Line width |
| `alpha` | `float` | `None` | Line transparency |
| `cmap` | `str` | `'hsv'` | Colormap for rainbow mode |

### Color-coding inliers and outliers

A common pattern is to draw outliers in red and inliers in green:

```python
import numpy as np

# After RANSAC or pose estimation
inliers = np.array([True, False, True, ...])  # boolean mask

fig, ax = vis.plot_pair(img0, img1)
vis.plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r', alpha=0.3)
vis.plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')
vis.add_text(f"Inliers: {inliers.sum()} / {len(inliers)}")
vis.save("results/inliers.png")
```

## Text annotations

### `add_text` &mdash; Add text to the figure

```python
vis.plot_pair(img0, img1)
vis.plot_matches(matches["mkpts0"], matches["mkpts1"])

# Basic text (top-left corner)
vis.add_text(f"Matches: {len(matches['mkpts0'])}")

# Custom font size and color
vis.add_text("SuperPoint", fontsize=20, color='white', bg_color='black')

# Custom position
vis.add_text("Bottom", x=100, y=200, ha='center', va='bottom')

# Semi-transparent background
vis.add_text("Score: 0.95", bg_alpha=0.8)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Text to display |
| `fontsize` | `int` | `12` | Font size |
| `color` | `str` | `'black'` | Text color |
| `bg_color` | `str` | `'white'` | Background box color |
| `bg_alpha` | `float` | `0.5` | Background transparency |
| `x`, `y` | `float` | `0, 0` | Position in data coordinates |
| `ha` | `str` | `'left'` | Horizontal alignment (`'left'`, `'center'`, `'right'`) |
| `va` | `str` | `'top'` | Vertical alignment (`'top'`, `'center'`, `'bottom'`) |

## Depth visualization

### `plot_depth` &mdash; Overlay a depth map

```python
vis.plot_image(img)
vis.plot_depth(depth_map)
vis.show()
```

### `plot_depth_pair` &mdash; Depth overlay on an image pair

```python
vis.plot_pair(img0, img1)
vis.plot_depth_pair(depth0, depth1)
vis.show()

# Custom colormap and transparency
vis.plot_pair(img0, img1)
vis.plot_depth_pair(depth0, depth1, cmap='plasma', alpha=0.7)
vis.show()
```

**Parameters for both:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cmap` | `str` | `'jet'` | Matplotlib colormap |
| `alpha` | `float` | `0.5` | Overlay transparency |

## Saving and displaying

### `save` &mdash; Save the current figure

```python
# Basic save
vis.save("results/output.png")

# Custom DPI (higher = larger file, more detail)
vis.save("results/output.png", dpi=300)

# Tight bounding box (removes whitespace)
vis.save("results/output.png", bbox_inches='tight')

# Save without closing the figure (useful for further modifications)
vis.save("results/output.png", close=False)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | required | Output file path. Directories are created automatically. |
| `dpi` | `int` | `None` | Resolution (dots per inch) |
| `bbox_inches` | `str` | `None` | Set to `'tight'` to remove whitespace |
| `close` | `bool` | `True` | Close the figure after saving |

### `show` &mdash; Display interactively

```python
vis.show()
```

## Color system

The `vis` module has a set of predefined color shortcuts:

| Shortcut | Color | Hex |
|----------|-------|-----|
| `'g'` | Green | `#4ade80` |
| `'r'` | Red | `#ef4444` |
| `'b'` | Blue | `#3b82f6` |
| `'y'` | Yellow | `#fbbf24` |
| `'p'` | Purple | `#a855f7` |
| `'o'` | Orange | `#fb923c` |
| `'c'` | Cyan | `#06b6d4` |
| `'m'` | Magenta | `#ec4899` |

You can also use:

- Any matplotlib named color: `'red'`, `'dodgerblue'`, `'darkgreen'`
- Hex strings: `'#ff6600'`
- RGB tuples: `(1.0, 0.5, 0.0)`
- `None` for automatic rainbow coloring

## Batch processing

When processing many image pairs, close figures to prevent memory leaks:

```python
for i, (img0, img1) in enumerate(pairs):
    matches = extractor.match(img0, img1)

    vis.plot_pair(img0, img1)
    vis.plot_matches(matches["mkpts0"], matches["mkpts1"])
    vis.save(f"results/pair_{i:04d}.png")  # close=True by default
    vis.plt.close('all')  # extra safety for batch processing
```

## Using with matplotlib directly

The `plot_pair` and `plot_image` functions return matplotlib `(fig, ax)` objects, so you can add custom annotations:

```python
fig, ax = vis.plot_pair(img0, img1)

# Add custom matplotlib elements
ax[0].set_title("Image 0", fontsize=14)
ax[1].set_title("Image 1", fontsize=14)
ax[0].axhline(y=100, color='red', linestyle='--')

# Add matches on top
vis.plot_matches(matches["mkpts0"], matches["mkpts1"], fig=fig, ax=ax)

vis.show()
```

## Input flexibility

All visualization functions accept both **torch tensors** and **numpy arrays**. Tensors are automatically moved to CPU and converted:

```python
import torch
import numpy as np

# These all work:
vis.plot_pair(torch_tensor, torch_tensor)
vis.plot_pair(numpy_array, numpy_array)
vis.plot_pair(torch_tensor, numpy_array)  # mixed is fine too

vis.plot_keypoints(torch.rand(100, 2), np.random.rand(100, 2))
vis.plot_matches(torch.rand(50, 2), np.random.rand(50, 2))
```

## Complete example: Feature comparison

```python
from easy_local_features import getExtractor
from easy_local_features.utils import io, ops, vis

img0 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth0.jpg"), 320)[0]
img1 = ops.resize_short_edge(io.fromPath("tests/assets/megadepth1.jpg"), 320)[0]

methods = ["superpoint", "aliked", "xfeat", "disk"]

for name in methods:
    extractor = getExtractor(name, {"top_k": 2048}).to("cpu")
    matches = extractor.match(img0, img1)

    vis.plot_pair(img0, img1, title=f"{name} ({len(matches['mkpts0'])} matches)", figsize=(10, 5))
    vis.plot_matches(matches["mkpts0"], matches["mkpts1"], linewidth=0.3, alpha=0.7)
    vis.save(f"results/{name}.png", dpi=150, bbox_inches='tight')
```
