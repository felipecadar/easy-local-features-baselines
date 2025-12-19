import importlib
import os
import json
from typing import TYPE_CHECKING, Any, Dict
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


def describe(name: str) -> Dict[str, Any]:
    """Describe an extractor/detector without instantiating it."""
    if name in available_extractors or name in available_detectors:
        cls = importByName(name)
        return cls.describe()
    raise ValueError(
        f"Unknown method '{name}'. Available extractors: {available_extractors}. Available detectors: {available_detectors}."
    )


if TYPE_CHECKING:
    # These imports are ONLY for static typing / IDE autocomplete.
    from typing import overload, Literal

    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline, SuperPointConfig
    from easy_local_features.feature.baseline_xfeat import XFeat_baseline, XFeatConfig
    from easy_local_features.feature.baseline_aliked import ALIKED_baseline, ALIKEDConfig
    from easy_local_features.feature.baseline_orb import ORB_baseline, ORBConfig
    from easy_local_features.feature.baseline_romav2 import RoMaV2_baseline, ROMAV2Config
    from easy_local_features.feature.baseline_roma import RoMa_baseline, ROMAConfig
    from easy_local_features.feature.baseline_dad import DAD_baseline, DadConfig
    from easy_local_features.feature.baseline_rekd import REKD_baseline

    @overload
    def getExtractor(extractor_name: Literal["superpoint"], conf: SuperPointConfig = ...) -> SuperPoint_baseline: ...
    @overload
    def getExtractor(extractor_name: Literal["xfeat"], conf: XFeatConfig = ...) -> XFeat_baseline: ...
    @overload
    def getExtractor(extractor_name: Literal["aliked"], conf: ALIKEDConfig = ...) -> ALIKED_baseline: ...
    @overload
    def getExtractor(extractor_name: Literal["orb"], conf: ORBConfig = ...) -> ORB_baseline: ...
    @overload
    def getExtractor(extractor_name: Literal["romav2"], conf: ROMAV2Config = ...) -> RoMaV2_baseline: ...
    @overload
    def getExtractor(extractor_name: Literal["roma"], conf: ROMAConfig = ...) -> RoMa_baseline: ...

    @overload
    def getDetector(detector_name: Literal["dad"], conf: DadConfig = ...) -> DAD_baseline: ...
    @overload
    def getDetector(detector_name: Literal["rekd"], conf: Dict[str, Any] = ...) -> REKD_baseline: ...


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
    parser.add_argument(
        "--describe",
        type=str,
        default=None,
        help="Print configuration defaults/schema for an extractor or detector (no model init).",
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

    if args.describe is not None:
        info = describe(args.describe)
        print(json.dumps(info, indent=2, sort_keys=True, default=str))
        return

    parser.print_help()
