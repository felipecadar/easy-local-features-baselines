# How to add a new extractor

This guide walks you through the process of adding a new feature extractor to the easy-local-features library. The library supports three types of methods:

- **Detect+Describe**: Methods that detect keypoints and compute descriptors (e.g., SIFT, SuperPoint)
- **Descriptor-only**: Methods that compute descriptors for provided keypoints (e.g., HardNet)
- **End-to-end matcher**: Methods that directly match two images (e.g., LoFTR, SuperGlue)

## Step 1: Create the baseline file

Create a new file `src/easy_local_features/feature/baseline_<name>.py` that subclasses `BaseExtractor`.

```python
from typing import Dict, Any
from omegaconf import OmegaConf
from .basemodel import BaseExtractor, MethodType
from .configs import YourConfigType  # If you add a config type

class YourExtractor_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE  # or DESCRIPTOR_ONLY or END2END_MATCHER
    
    default_conf: YourConfigType = {
        "param1": "value1",
        "param2": 42,
        # ... your default parameters
    }

    def __init__(self, conf: YourConfigType = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        # Initialize your model here
        self.model = YourModel(**conf)
        self.matcher = NearestNeighborMatcher()  # or your custom matcher

    def detectAndCompute(self, img, return_dict=False):
        # Implement detection and description
        # Return keypoints and descriptors
        pass

    def detect(self, img):
        # Implement keypoint detection (if applicable)
        pass

    def compute(self, img, keypoints):
        # Implement descriptor computation
        pass

    def to(self, device):
        # Move model to device
        self.model = self.model.to(device)
        return self

    @property
    def has_detector(self):
        return True  # or False for descriptor-only methods
```

## Step 2: Add configuration type hints (optional but recommended)

Add a TypedDict for your configuration in `src/easy_local_features/feature/configs.py`:

```python
from typing import TypedDict

class YourConfigType(TypedDict):
    param1: str
    param2: int
    # ... add all your parameters
```

## Step 3: Register the extractor

Add your extractor name to `available_extractors` in `src/easy_local_features/__init__.py`:

```python
available_extractors = [
    # ... existing extractors
    'your_extractor_name'
]
```

## Step 4: Handle dependencies and weights

- If your method requires external weights, download them in the `__init__` method
- Add any required dependencies to `pyproject.toml`
- If using submodules, place them in `src/easy_local_features/submodules/`

## Step 5: Add tests

Create or update tests in `tests/test_features.py` to include your new extractor:

```python
@pytest.mark.parametrize("extractor_name", available_extractors)
def test_feature_extractors(extractor_name):
    # Test implementation
    pass
```

## Step 6: Update documentation

- Add your extractor to the API documentation
- Update the README if needed
- Consider adding examples in the docs

## Example: Adding a simple ORB extractor

Here's a minimal example for ORB:

```python
import cv2
from .basemodel import BaseExtractor, MethodType
from .configs import ORBConfig

class ORB_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE
    
    default_conf: ORBConfig = {
        "nfeatures": 500,
        "scaleFactor": 1.2,
        "nlevels": 8,
        "edgeThreshold": 31,
        "firstLevel": 0,
        "WTA_K": 2,
        "scoreType": cv2.ORB_HARRIS_SCORE,
        "patchSize": 31,
        "fastThreshold": 20,
    }

    def __init__(self, conf: ORBConfig = {}):
        self.conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.orb = cv2.ORB_create(**self.conf)

    def detectAndCompute(self, img, return_dict=False):
        img_cv = ops.to_cv(img)
        keypoints, descriptors = self.orb.detectAndCompute(img_cv, None)
        
        if return_dict:
            return {"keypoints": keypoints, "descriptors": descriptors}
        return keypoints, descriptors

    # ... implement other required methods
```

## Tips

- Follow the existing patterns in other baseline files
- Use `ops.prepareImage()` for image preprocessing
- Handle both batched and non-batched inputs
- Ensure your method works with `torch.device` for GPU support
- Test with different image formats (paths, arrays, tensors)
