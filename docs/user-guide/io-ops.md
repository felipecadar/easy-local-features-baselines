# I/O and Ops

Utilities for loading and preparing images:

- `io.fromPath(path, gray=False, batch=True)`: load from file path.
- `ops.prepareImage(img, gray=False, batch=True)`: normalize array or tensor into [B,C,H,W].
- `ops.resize_short_edge(img, size)`: resize keeping aspect ratio.
- `ops.crop_patches(img, keypoints, patch_size)`: extract local patches.
