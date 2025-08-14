import importlib
import os
import pkgutil
from pathlib import Path

import pytest

from easy_local_features import available_extractors, getExtractor
from easy_local_features.feature.basemodel import BaseExtractor
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


@pytest.mark.parametrize("extractor_name", available_extractors)
def test_feature_extractors(extractor_name):
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

    vis.plot_pair(image0, image1, title=extractor_name, figsize=(8, 4))
    vis.plot_keypoints(matches["mkpts0"], matches["mkpts1"], kps_size=2)
    vis.add_text(f"Matches: {len(matches['mkpts0'])}")
    vis.save(f"tests/results/{extractor_name}.png")


@pytest.mark.parametrize("extractor_name", available_extractors)
def test_cpu(extractor_name):
    if "deal" in extractor_name:
        print(f"Skipping {extractor_name}")
        return

    extractor = getExtractor(extractor_name)
    extractor.to("cpu")


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
