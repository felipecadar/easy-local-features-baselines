from typing import TypedDict, Optional


class ALIKEDConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    force_num_keypoints: bool
    nms_radius: int


class SuperPointConfig(TypedDict):
    top_k: int
    sparse_outputs: bool
    dense_outputs: bool
    nms_radius: int
    refinement_radius: int
    detection_threshold: float
    remove_borders: int
    legacy_sampling: bool


class DINOv3Config(TypedDict):
    weights: str
    allow_resize: bool
    repo_dir: Optional[str]
    weights_path: Optional[str]
    source: str
    normalize: str


class DEALConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    force_num_keypoints: bool


class DISKConfig(TypedDict):
    window: int
    desc_dim: int
    mode: str
    top_k: int
    auto_resize: bool


class D2NetConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    use_relativeloss: bool
    use_uncertainty: bool


class DELFConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    use_pca: bool
    use_whitening: bool


class R2D2Config(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    reliability_thr: float
    repeatability_thr: float


class SOSNetConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    desc_dim: int


class TFeatConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    desc_dim: int


class XFeatConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    width_confidence: float
    min_corner_score: float


class ROMAConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    upsample_factor: int


class ORBConfig(TypedDict):
    nfeatures: int
    scaleFactor: float
    nlevels: int
    edgeThreshold: int
    firstLevel: int
    WTA_K: int
    scoreType: int
    patchSize: int
    fastThreshold: int


class DeDoDeConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    amp_dtype: str


class DescReasoningConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    desc_dim: int
