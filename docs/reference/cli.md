# CLI Reference

The `easy-local-features` command provides quick access to method information from the terminal.

## Installation

The CLI is installed automatically when you install the package:

```bash
pip install easy-local-features
```

## Commands

### List available extractors

```bash
easy-local-features --list
```

Output:
```
alike
aliked
d2net
deal
dedode
delf
disk
lightglue
loftr
mum
r2d2
topicfm
sosnet
superglue
superpoint
tfeat
xfeat
roma
romav2
orb
desc_reasoning
```

### List available detectors

```bash
easy-local-features --list-detectors
```

Output:
```
dad
rekd
```

### List all methods

Includes extractors, detectors, and additional methods (DINOv2, DINOv3, ResNet, etc.):

```bash
easy-local-features --list-all
```

### Describe a method

Print the configuration schema and defaults for any method **without downloading or loading the model**:

```bash
easy-local-features --describe superpoint
```

Output:
```json
{
  "aliases": {
    "keypoint_threshold": "detection_threshold",
    "max_keypoints": "top_k"
  },
  "defaults": {
    "dense_outputs": false,
    "detection_threshold": 0.005,
    "force_num_keypoints": false,
    "legacy_sampling": true,
    "nms_radius": 4,
    "refinement_radius": 0,
    "remove_borders": 4,
    "sparse_outputs": true,
    "top_k": -1
  },
  "doc": "...",
  "has_detector": null,
  "method_type": "detect_describe",
  "name": "SuperPoint_baseline",
  "schema": { ... }
}
```

### Describe any method

```bash
easy-local-features --describe aliked
easy-local-features --describe romav2
easy-local-features --describe dad
easy-local-features --describe dinov2
```

### Help

```bash
easy-local-features --help
```

## Usage in scripts

You can also use `describe()` from Python:

```python
from easy_local_features import describe

info = describe("superpoint")
print(info["defaults"])    # Default config
print(info["schema"])      # TypedDict annotations
print(info["aliases"])     # Deprecated key mappings
print(info["method_type"]) # "detect_describe", "descriptor_only", etc.
print(info["doc"])         # First paragraph of class docstring
```
