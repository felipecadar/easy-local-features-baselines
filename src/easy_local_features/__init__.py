import importlib
import os
import json
from typing import TYPE_CHECKING, Any, Dict, List
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
    "lightglue",
    "loftr",
    "mum",
    "r2d2",
    "topicfm",
    "sosnet",
    "superglue",
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

# Unified list: all extractors + detectors + previously unregistered but functional methods
available_methods: List[str] = sorted(set(
    available_extractors + available_detectors + [
        "croco",
        "dinov2",
        "dinov3",
        "resnet",
        "sfd2",
        "superpoint_open",
        "vgg",
    ]
))


def importByName(name):
    package_name_feature = f"easy_local_features.feature.baseline_{name}"
    package_name_matching = f"easy_local_features.matching.baseline_{name}"

    try:
        importlib.import_module(package_name_feature)
    except ModuleNotFoundError:
        try:
            importlib.import_module(package_name_matching)
        except ModuleNotFoundError:
            raise ValueError(f"Could not find module {package_name_feature} or {package_name_matching}")

    subclasses = BaseExtractor.__subclasses__()
    # find the correct subclass
    for sub in subclasses:
        if f"{name}_baseline" == sub.__name__.lower():
            return sub

    raise ValueError(
        f"Could not find a subclass of BaseExtractor that contains <{name}>")


def getMethod(name: str, conf=None) -> BaseExtractor:
    """Unified factory for any method (extractor, detector, or matcher).

    Accepts the same variation syntax as getExtractor (e.g., "lightglue:superpoint").
    """
    if conf is None:
        conf = {}

    variation = None
    if ":" in name:
        name, variation = name.split(":", 1)

    if name not in available_methods:
        raise ValueError(
            f"Unknown method '{name}'. Available methods: {available_methods}"
        )

    if variation is not None:
        if name == "lightglue":
            conf["features"] = variation
        elif name == "desc_reasoning":
            conf["pretrained"] = variation

    cls = importByName(name)
    return cls(conf)


def getExtractor(extractor_name: str, conf=None) -> BaseExtractor:
    """Get a feature extractor by name. Backward-compatible wrapper around getMethod."""
    if conf is None:
        conf = {}

    base_name = extractor_name.split(":")[0] if ":" in extractor_name else extractor_name
    assert base_name in available_extractors, (
        f"Invalid extractor {base_name}. Available extractors: {available_extractors}"
    )
    return getMethod(extractor_name, conf)


def getDetector(detector_name: str, conf=None) -> BaseExtractor:
    """Get a detector by name. Backward-compatible wrapper around getMethod."""
    if conf is None:
        conf = {}

    base_name = detector_name.split(":")[0] if ":" in detector_name else detector_name
    assert base_name in available_detectors, (
        f"Invalid detector {base_name}. Available detectors: {available_detectors}"
    )
    return getMethod(detector_name, conf)


def describe(name: str) -> Dict[str, Any]:
    """Describe an extractor/detector without instantiating it."""
    base_name = name.split(":")[0] if ":" in name else name
    if base_name in available_methods:
        cls = importByName(base_name)
        return cls.describe()
    raise ValueError(
        f"Unknown method '{name}'. Available methods: {available_methods}."
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
    def getMethod(name: Literal["superpoint"], conf: SuperPointConfig = ...) -> SuperPoint_baseline: ...
    @overload
    def getMethod(name: Literal["xfeat"], conf: XFeatConfig = ...) -> XFeat_baseline: ...
    @overload
    def getMethod(name: Literal["aliked"], conf: ALIKEDConfig = ...) -> ALIKED_baseline: ...
    @overload
    def getMethod(name: Literal["orb"], conf: ORBConfig = ...) -> ORB_baseline: ...
    @overload
    def getMethod(name: Literal["romav2"], conf: ROMAV2Config = ...) -> RoMaV2_baseline: ...
    @overload
    def getMethod(name: Literal["roma"], conf: ROMAConfig = ...) -> RoMa_baseline: ...
    @overload
    def getMethod(name: Literal["dad"], conf: DadConfig = ...) -> DAD_baseline: ...
    @overload
    def getMethod(name: Literal["rekd"], conf: Dict[str, Any] = ...) -> REKD_baseline: ...

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

    Keeps the CLI intentionally minimal: list available methods.
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
        "--list-all",
        action="store_true",
        help="List all available methods (extractors, detectors, matchers) and exit.",
    )
    parser.add_argument(
        "--describe",
        type=str,
        default=None,
        help="Print configuration defaults/schema for any method (no model init).",
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
    if args.list_all:
        for name in available_methods:
            print(name)
        return

    if args.describe is not None:
        info = describe(args.describe)
        print(json.dumps(info, indent=2, sort_keys=True, default=str))
        return

    parser.print_help()
