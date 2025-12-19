import importlib
import os
from .feature.basemodel import BaseExtractor

os.environ["TFHUB_CACHE_DIR"] = os.path.expanduser(
    os.path.join("~", ".cache", "torch", "hub", "checkpoints",
                 "easy_local_features", "tfhub")
)

available_extractors = [
    "alike",
    "aliked",
    "d2net",
    # 'dalf',  # disabled in tests due to formatting issues; re-enable once fixed
    "deal",
    "dedode",
    "delf",
    "disk",
    "mum",
    "r2d2",
    "topicfm",
    "sosnet",
    "superpoint",
    "tfeat",
    "xfeat",
    # 'relf',  # optional dependency mmcv not installed in test env
    "roma",
    "romav2",
    "orb",
    "desc_reasoning",
]

available_detectors = [
    "dad",
    "rekd",
]


def importByName(name):
    package_name = f"easy_local_features.feature.baseline_{name}"
    importlib.import_module(package_name)

    subclasses = BaseExtractor.__subclasses__()
    # find the correct subclass
    for sub in subclasses:
        if f"{name}_baseline" == sub.__name__.lower():
            return sub

    raise ValueError(
        f"Could not find a subclass of BaseExtractor in <{package_name}> that contains <{name}>")


def getExtractor(extractor_name: str, conf={}):
    assert extractor_name in available_extractors, (
        f"Invalid extractor {extractor_name}. Available extractors: {available_extractors}"
    )
    extractor = importByName(extractor_name)
    return extractor(conf)

def getDetector(detector_name: str, conf={}):
    assert detector_name in available_detectors, (
        f"Invalid detector {detector_name}. Available detectors: {available_detectors}"
    )
    det = importByName(detector_name)
    return det(conf)


def main() -> None:
    """Console entrypoint for `easy-local-features`.

    Keeps the CLI intentionally minimal: list available extractors.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="easy-local-features")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available extractors and exit.",
    )
    parser.add_argument(
        "--list-detectors",
        action="store_true",
        help="List available detectors and exit.",
    )
    args = parser.parse_args()

    if args.list:
        for name in available_extractors:
            print(name)
        return
    if args.list_detectors:
        for name in available_detectors:
            print(name)
        return

    parser.print_help()
