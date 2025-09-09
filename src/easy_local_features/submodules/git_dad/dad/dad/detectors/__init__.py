from .dedode_detector import load_DaD as load_DaD
from .dedode_detector import load_DaDDark as load_DaDDark
from .dedode_detector import load_DaDLight as load_DaDLight
from .dedode_detector import dedode_detector_S as dedode_detector_S
from .dedode_detector import dedode_detector_B as dedode_detector_B
from .dedode_detector import dedode_detector_L as dedode_detector_L
from .dedode_detector import load_dedode_v2 as load_dedode_v2


lg_detectors = ["ALIKED", "ALIKEDROT", "SIFT", "DISK", "SuperPoint", "ReinforcedFP"]
other_detectors = ["HesAff", "HarrisAff", "REKD"]
dedode_detectors = [
    "DeDoDe-v2",
    "DaD",
    "DaDLight",
    "DaDDark",
]
all_detectors = lg_detectors + dedode_detectors + other_detectors


def load_detector_by_name(detector_name, *, resize=1024, weights_path=None):
    if detector_name == "DaD":
        detector = load_DaD(resize=resize, weights_path=weights_path)
    elif detector_name == "DaDLight":
        detector = load_DaDLight(resize=resize, weights_path=weights_path)
    elif detector_name == "DaDDark":
        detector = load_DaDDark(resize=resize, weights_path=weights_path)
    elif detector_name == "DeDoDe-v2":
        detector = load_dedode_v2()
    else:
        raise ValueError(f"Couldn't find detector with detector name {detector_name}")
    return detector
