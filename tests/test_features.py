import importlib
import os
import pkgutil
from pathlib import Path

import pytest
import torch

from easy_local_features import available_extractors, available_methods, getExtractor, getMethod
from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.utils import io, ops, vis

ROOT = Path(__file__).resolve().parent


def get_all_subclasses(cls):
    subclasses = set(cls.__subclasses__())
    for subclass in list(subclasses):
        subclasses.update(get_all_subclasses(subclass))
    return subclasses


def load_all_modules_from_package(package):
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if not is_pkg:
            importlib.import_module(module_name)


@pytest.fixture
def all_subclasses():
    import easy_local_features

    load_all_modules_from_package(easy_local_features.feature)
    return get_all_subclasses(BaseExtractor)


# Methods that are known to be slow or need special handling in CI
SKIP_METHODS = set()


def _load_test_images():
    image0 = io.fromPath(str(ROOT / "assets/megadepth0.jpg"))
    image1 = io.fromPath(str(ROOT / "assets/megadepth1.jpg"))
    image0 = ops.resize_short_edge(image0, 320)[0]
    image1 = ops.resize_short_edge(image1, 320)[0]
    return image0, image1


# ─────────────────────── Original tests (kept unchanged) ───────────────────────

test_variations = [
    "lightglue:superpoint",
    "lightglue:disk",
    "lightglue:aliked",
    "desc_reasoning:xfeat-3_layers",
]

def _run_extractor_test(extractor_name):
    # skip DEAL
    if "deal" in extractor_name:
        print(f"Skipping {extractor_name}")
        return

    image0 = io.fromPath(str(ROOT / "assets/megadepth0.jpg"))
    image1 = io.fromPath(str(ROOT / "assets/megadepth1.jpg"))

    image0 = ops.resize_short_edge(image0, 320)[0]
    image1 = ops.resize_short_edge(image1, 320)[0]

    os.makedirs("tests/results", exist_ok=True)
    extractor = getExtractor(extractor_name, {"top_k": 4096})

    if not extractor.has_detector:
        # from easy_local_features.feature.baseline_alike import ALIKE_baseline
        from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline

        detector = SuperPoint_baseline(
            {
                "nms_radius": 4,
                "keypoint_threshold": 0.005,
                "max_keypoints": 4096,
            }
        )
        extractor.addDetector(detector)

    matches = extractor.match(image0, image1)

    vis_name = extractor_name.replace(":", "_")
    vis.plot_pair(image0, image1, title=extractor_name, figsize=(8, 4))
    vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"], kps_size=2)
    vis.add_text(f"Matches: {len(matches['mkpts0'])}")
    vis.save(f"tests/results/{vis_name}.png")

@pytest.mark.parametrize("extractor_name", available_extractors)
def test_feature_extractors(extractor_name):
    _run_extractor_test(extractor_name)

@pytest.mark.parametrize("extractor_name", test_variations)
def test_feature_variations(extractor_name):
    _run_extractor_test(extractor_name)


def _run_cpu_test(extractor_name):
    if "deal" in extractor_name:
        print(f"Skipping {extractor_name}")
        return

    extractor = getExtractor(extractor_name)
    extractor.to("cpu")

@pytest.mark.parametrize("extractor_name", available_extractors)
def test_cpu(extractor_name):
    _run_cpu_test(extractor_name)

@pytest.mark.parametrize("extractor_name", test_variations)
def test_cpu_variations(extractor_name):
    _run_cpu_test(extractor_name)


# ─────────────────────── New unified tests via getMethod ───────────────────────

@pytest.mark.parametrize("method_name", available_methods)
def test_to_returns_self(method_name):
    """Every method's to() must return self for method chaining."""
    if method_name in SKIP_METHODS:
        pytest.skip(f"Skipping {method_name}")
    method = getMethod(method_name)
    result = method.to("cpu")
    assert result is method, f"{method_name}.to('cpu') did not return self"


@pytest.mark.parametrize("method_name", available_methods)
def test_method_type_is_set(method_name):
    """Every method must have a valid method_type."""
    if method_name in SKIP_METHODS:
        pytest.skip(f"Skipping {method_name}")
    method = getMethod(method_name)
    valid_types = {
        MethodType.DETECT_DESCRIBE,
        MethodType.DESCRIPTOR_ONLY,
        MethodType.DETECTOR_ONLY,
        MethodType.END2END_MATCHER,
    }
    assert method.method_type in valid_types, (
        f"{method_name}.method_type = {method.method_type!r}, expected one of {valid_types}"
    )


@pytest.mark.parametrize("method_name", available_methods)
def test_describe_classmethod(method_name):
    """describe() must return a dict with expected keys without instantiation."""
    from easy_local_features import describe
    if method_name in SKIP_METHODS:
        pytest.skip(f"Skipping {method_name}")
    info = describe(method_name)
    assert isinstance(info, dict)
    for key in ("name", "method_type", "defaults"):
        assert key in info, f"describe('{method_name}') missing key '{key}'"


# ─── DETECT_DESCRIBE methods ───

def _get_detect_describe_methods():
    """Return method names that are DETECT_DESCRIBE type."""
    methods = []
    for name in available_methods:
        if name in SKIP_METHODS:
            continue
        try:
            m = getMethod(name)
            if m.method_type == MethodType.DETECT_DESCRIBE:
                methods.append(name)
        except Exception:
            pass
    return methods


@pytest.fixture(scope="module")
def detect_describe_methods():
    return _get_detect_describe_methods()


@pytest.mark.parametrize("method_name", _get_detect_describe_methods())
def test_detect_describe_detectAndCompute(method_name):
    """DETECT_DESCRIBE methods must implement detectAndCompute with proper output shapes."""
    image0, _ = _load_test_images()
    method = getMethod(method_name, {"top_k": 512}).to("cpu")

    kpts, descs = method.detectAndCompute(image0, return_dict=False)
    assert isinstance(kpts, torch.Tensor), f"{method_name} keypoints not a tensor"
    assert isinstance(descs, torch.Tensor), f"{method_name} descriptors not a tensor"
    assert kpts.ndim in (2, 3), f"{method_name} keypoints shape {kpts.shape} unexpected"
    assert descs.ndim in (2, 3), f"{method_name} descriptors shape {descs.shape} unexpected"

    # Also test return_dict=True
    result = method.detectAndCompute(image0, return_dict=True)
    assert isinstance(result, dict)
    assert "keypoints" in result
    assert "descriptors" in result


@pytest.mark.parametrize("method_name", _get_detect_describe_methods())
def test_detect_describe_detect(method_name):
    """DETECT_DESCRIBE methods must implement detect()."""
    image0, _ = _load_test_images()
    method = getMethod(method_name, {"top_k": 512}).to("cpu")

    kpts = method.detect(image0)
    assert isinstance(kpts, torch.Tensor), f"{method_name} detect() did not return a tensor"


@pytest.mark.parametrize("method_name", _get_detect_describe_methods())
def test_detect_describe_match(method_name):
    """DETECT_DESCRIBE methods must support match() returning mkpts0/mkpts1."""
    image0, image1 = _load_test_images()
    method = getMethod(method_name, {"top_k": 512}).to("cpu")

    result = method.match(image0, image1)
    assert isinstance(result, dict), f"{method_name} match() did not return a dict"
    assert "mkpts0" in result, f"{method_name} match() missing mkpts0"
    assert "mkpts1" in result, f"{method_name} match() missing mkpts1"
    assert isinstance(result["mkpts0"], torch.Tensor)
    assert isinstance(result["mkpts1"], torch.Tensor)
    assert result["mkpts0"].shape[-1] == 2
    assert result["mkpts1"].shape[-1] == 2
    assert result["mkpts0"].shape[0] == result["mkpts1"].shape[0]


# ─── DESCRIPTOR_ONLY methods ───

def _get_descriptor_only_methods():
    methods = []
    for name in available_methods:
        if name in SKIP_METHODS:
            continue
        try:
            m = getMethod(name)
            if m.method_type == MethodType.DESCRIPTOR_ONLY:
                methods.append(name)
        except Exception:
            pass
    return methods


@pytest.mark.parametrize("method_name", _get_descriptor_only_methods())
def test_descriptor_only_compute(method_name):
    """DESCRIPTOR_ONLY methods must implement compute() with external keypoints."""
    image0, _ = _load_test_images()
    method = getMethod(method_name).to("cpu")

    # Get keypoints from SuperPoint
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
    detector = SuperPoint_baseline({"top_k": 256}).to("cpu")
    kpts = detector.detect(image0)

    # Ensure keypoints are batched [B, N, 2]
    if kpts.ndim == 2:
        kpts = kpts.unsqueeze(0)

    result = method.compute(image0, kpts)
    if isinstance(result, tuple):
        kpts_out, descs = result
        assert isinstance(descs, torch.Tensor), f"{method_name} compute() descriptors not a tensor"
    else:
        descs = result
        assert isinstance(descs, torch.Tensor), f"{method_name} compute() did not return a tensor"


@pytest.mark.parametrize("method_name", _get_descriptor_only_methods())
def test_descriptor_only_addDetector_and_match(method_name):
    """DESCRIPTOR_ONLY methods must support addDetector + match flow."""
    image0, image1 = _load_test_images()
    method = getMethod(method_name).to("cpu")

    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
    detector = SuperPoint_baseline({"top_k": 256}).to("cpu")
    method.addDetector(detector)

    result = method.match(image0, image1)
    assert isinstance(result, dict), f"{method_name} match() did not return a dict"
    assert "mkpts0" in result, f"{method_name} match() missing mkpts0"
    assert "mkpts1" in result, f"{method_name} match() missing mkpts1"


# ─── DETECTOR_ONLY methods ───

def _get_detector_only_methods():
    methods = []
    for name in available_methods:
        if name in SKIP_METHODS:
            continue
        try:
            m = getMethod(name)
            if m.method_type == MethodType.DETECTOR_ONLY:
                methods.append(name)
        except Exception:
            pass
    return methods


@pytest.mark.parametrize("method_name", _get_detector_only_methods())
def test_detector_only_detect(method_name):
    """DETECTOR_ONLY methods must implement detect() returning keypoints."""
    image0, _ = _load_test_images()
    method = getMethod(method_name).to("cpu")

    kpts = method.detect(image0)
    assert isinstance(kpts, torch.Tensor), f"{method_name} detect() did not return a tensor"
    assert kpts.shape[-1] == 2, f"{method_name} detect() keypoints last dim should be 2, got {kpts.shape}"


# ─── END2END_MATCHER methods ───

def _get_end2end_matcher_methods():
    methods = []
    for name in available_methods:
        if name in SKIP_METHODS:
            continue
        try:
            m = getMethod(name)
            if m.method_type == MethodType.END2END_MATCHER:
                methods.append(name)
        except Exception:
            pass
    return methods


@pytest.mark.parametrize("method_name", _get_end2end_matcher_methods())
def test_end2end_matcher_match(method_name):
    """END2END_MATCHER methods must implement match() with two images."""
    image0, image1 = _load_test_images()
    method = getMethod(method_name).to("cpu")

    # SuperGlue needs a detector attached since it can't extract features itself
    if method_name == "superglue":
        from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
        detector = SuperPoint_baseline({"top_k": 512}).to("cpu")
        method.addDetector(detector)

    result = method.match(image0, image1)
    assert isinstance(result, dict), f"{method_name} match() did not return a dict"
    assert "mkpts0" in result, f"{method_name} match() missing mkpts0"
    assert "mkpts1" in result, f"{method_name} match() missing mkpts1"


# ─── Batched output shape tests via extract_features ───

@pytest.mark.parametrize("method_name", _get_detect_describe_methods())
def test_extract_features_batched_output(method_name):
    """extract_features() must return [B, N, 2] keypoints and [B, N, D] descriptors."""
    image0, _ = _load_test_images()
    method = getMethod(method_name, {"top_k": 256}).to("cpu")

    result = method.extract_features(image0)
    assert isinstance(result, dict)
    kpts = result["keypoints"]
    descs = result["descriptors"]

    assert kpts.ndim == 3, f"{method_name} extract_features keypoints ndim={kpts.ndim}, expected 3 [B,N,2]"
    assert kpts.shape[0] >= 1, f"{method_name} batch dim should be >= 1"
    assert kpts.shape[2] == 2, f"{method_name} keypoints last dim should be 2"

    assert descs.ndim == 3, f"{method_name} extract_features descriptors ndim={descs.ndim}, expected 3 [B,N,D]"
    assert descs.shape[0] == kpts.shape[0], f"{method_name} batch dim mismatch"
    assert descs.shape[1] == kpts.shape[1], f"{method_name} num keypoints mismatch between kpts and descs"


# ─── getMethod backward compatibility ───

def test_getMethod_is_same_as_getExtractor():
    """getMethod and getExtractor should return equivalent objects."""
    m1 = getMethod("superpoint", {"top_k": 128})
    m2 = getExtractor("superpoint", {"top_k": 128})
    assert type(m1) is type(m2)


def test_getMethod_accepts_detectors():
    """getMethod should accept detector names that getDetector accepts."""
    from easy_local_features import getDetector
    m1 = getMethod("dad")
    m2 = getDetector("dad")
    assert type(m1) is type(m2)


def test_getMethod_accepts_variations():
    """getMethod should support variation syntax."""
    m = getMethod("lightglue:superpoint")
    assert m is not None


def test_getMethod_rejects_unknown():
    """getMethod should raise ValueError for unknown method names."""
    with pytest.raises(ValueError):
        getMethod("nonexistent_method_xyz")


def test_available_methods_superset():
    """available_methods should be a superset of extractors + detectors."""
    from easy_local_features import available_detectors
    for name in available_extractors:
        assert name in available_methods, f"'{name}' in available_extractors but not in available_methods"
    for name in available_detectors:
        assert name in available_methods, f"'{name}' in available_detectors but not in available_methods"


if __name__ == "__main__":
    import argparse

    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", "-m", type=str, default="all", help="Model name to test. Default: all.")
        return parser.parse_args()

    args = parse()

    if args.model == "all":
        _all_subclasses = available_extractors
    else:
        _all_subclasses = [args.model]

    for _model in _all_subclasses:
        print(f"Testing {_model}")
        test_feature_extractors(_model)
