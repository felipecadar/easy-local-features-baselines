import zipfile
from pathlib import Path
import torch
from omegaconf import OmegaConf

from .basemodel import BaseExtractor, MethodType
from ..utils import ops
from ..utils.pathutils import CACHE_BASE
from typing import TypedDict, Optional

# Wrap the submodule Reasoning pipeline as a Baseline extractor
from easy_local_features.submodules.git_reasoningaccv.desc_reasoning import (
    ReasoningBase,
    Reasoning,
)


class DescReasoningConfig(TypedDict):
    model_name: str
    top_k: int
    detection_threshold: float
    nms_radius: int
    checkpoint_path: Optional[str]
    pretrained: str
    weights_path: Optional[str]
    device: str
    cache_namespace: str
    desc_dim: int


class Desc_Reasoning_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DETECT_DESCRIBE

    # Minimal configuration: users must provide a checkpoint folder created by the
    # reasoning training code (containing model_config.yaml and weights .pt files).
    default_conf: DescReasoningConfig = {
        # Option A: pass an explicit checkpoint folder with model_config.yaml and *.pt
        "checkpoint_path": None,
        # Option B: specify a pretrained model name to auto-download from the public release
        # See PRETRAINED_URLS keys below.
        "pretrained": "xfeat",  # e.g., "xfeat", "superpoint", "alike", ...
        # Optional: a specific weight filename inside the checkpoint folder
        "weights_path": None,
        # Device selection: if not None, the Reasoning wrapper will auto-pick CUDA if available
        "device": "auto",
        # Cache directory under ~/.cache/torch/hub/checkpoints/easy_local_features/desc_reasoning
        "cache_namespace": "desc_reasoning",

        "top_k": 2048,  # Top-K matches to return; Reasoning uses its own matching logic
    }

    # Public weights from https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/tag/weights
    PRETRAINED_URLS = {
        # Base descriptor backbones
        "xfeat": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat.zip",
        "superpoint": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/superpoint.zip",
        "alike": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/alike.zip",
        "aliked": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/aliked.zip",
        "dedode_B": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/dedode_B.zip",
        "dedode_G": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/dedode_G.zip",
        # Ablations / variants
        "xfeat-12_layers-dino_G": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-12_layers-dino_G.zip",
        "xfeat-12_layers": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-12_layers.zip",
        "xfeat-3_layers": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-3_layers.zip",
        "xfeat-7_layers": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-7_layers.zip",
        "xfeat-9_layers": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-9_layers.zip",
        "xfeat-dino-G": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-dino-G.zip",
        "xfeat-dino_B": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-dino_B.zip",
        "xfeat-dino_L": "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-dino_L.zip",
    }

    def __init__(self, conf: DescReasoningConfig = {}):
        # Merge configs and validate mandatory fields
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

        checkpoint_path = conf.checkpoint_path
        if checkpoint_path is None:
            # Try pretrained auto-download if provided
            if conf.pretrained is None:
                raise ValueError(
                    "Desc_Reasoning_baseline requires either `checkpoint_path` or `pretrained` to be provided."
                )
            checkpoint_path = self._ensure_pretrained(conf.pretrained)

        # Load trained Reasoning model from checkpoint and build the full pipeline
        bundle = ReasoningBase.from_experiment(
            checkpoint_path=str(checkpoint_path),
            weights_path=conf.weights_path,
        )
        self.reasoning_model = bundle["model"]
        self.reasoning_model.conf.extractor.max_num_keypoints = conf.get("top_k", 2048)

        # Build the high-level pipeline (extractor + DinoV2 + reasoning)
        # Using any non-None value triggers auto device selection (cuda if available)
        dev_flag = conf.device if conf.device is not None else None
        self.pipeline = Reasoning(self.reasoning_model, dev=dev_flag)

        # Keep a local device hint (Reasoning manages its own .dev internally)
        self.DEV = getattr(self.pipeline, "dev", None)

        # We override match() to use the model's own matching; no external matcher needed
        self.matcher = None

    @classmethod
    def available_pretrained(cls):
        return list(cls.PRETRAINED_URLS.keys())

    def _ensure_pretrained(self, name: str) -> Path:
        name = str(name)
        if name not in self.PRETRAINED_URLS:
            raise ValueError(
                f"Unknown pretrained '{name}'. Available: {list(self.PRETRAINED_URLS.keys())}"
            )

        url = self.PRETRAINED_URLS[name]
        # Prepare cache paths
        cache_root = Path(CACHE_BASE) / self.conf.cache_namespace
        cache_root.mkdir(parents=True, exist_ok=True)
        zip_path = cache_root / f"{name}.zip"
        extract_dir = cache_root / name

        # If already extracted and valid, return directly
        model_config = extract_dir / "model_config.yaml"
        if model_config.exists():
            return extract_dir

        # Download zip if missing
        if not zip_path.exists():
            torch.hub.download_url_to_file(url, str(zip_path), progress=True)

        # Extract zip
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_root))

        # Some zips may extract into a nested directory; locate folder with model_config.yaml
        candidate = None
        if (extract_dir / "model_config.yaml").exists():
            candidate = extract_dir
        else:
            # search one level deep
            for p in cache_root.iterdir():
                if p.is_dir() and (p / "model_config.yaml").exists():
                    if p.name == name:
                        candidate = p
                        break
                    # fallback to the first valid if names differ
                    if candidate is None:
                        candidate = p

        if candidate is None:
            raise RuntimeError(
                f"Downloaded archive for '{name}' did not contain a checkpoint folder with model_config.yaml."
            )
        return candidate

    def detectAndCompute(self, img, return_dict: bool = False):
        image = ops.prepareImage(img).to(self.DEV) if self.DEV is not None else ops.prepareImage(img)
        with torch.inference_mode():
            keypoints, descriptors = self.pipeline.detectAndCompute(image)

        if return_dict:
            return {"keypoints": keypoints, "descriptors": descriptors}
        return keypoints, descriptors

    def detect(self, img):
        kpts, _ = self.detectAndCompute(img, return_dict=False)
        return kpts

    def compute(self, img, keypoints):
        # Computing descriptors for arbitrary external keypoints isn't exposed by the
        # Reasoning pipeline. This baseline acts as detect+describe.
        raise NotImplementedError("Desc_Reasoning_baseline does not implement compute(keypoints); use detectAndCompute().")

    def to(self, device):
        # Update internal device flag; Reasoning manages its own submodules
        self.DEV = torch.device(device) if not isinstance(device, torch.device) else device
        # Best-effort move of known submodules when available
        for mod_name in ("extractor", "dino", "reasoning_model"):
            mod = getattr(self.pipeline, mod_name, None)
            if hasattr(mod, "to"):
                mod.to(self.DEV)
        # Update pipeline hint
        self.pipeline.dev = self.DEV
        return self

    @property
    def has_detector(self):
        return True

    @torch.inference_mode()
    def match(self, image0, image1):
        # Use the model's dedicated matching that combines reasoning + semantic cues
        im0 = ops.prepareImage(image0).to(self.DEV) if self.DEV is not None else ops.prepareImage(image0)
        im1 = ops.prepareImage(image1).to(self.DEV) if self.DEV is not None else ops.prepareImage(image1)

        out = self.pipeline.match({"image0": im0, "image1": im1})
        # Convert to the standard keys expected by tests/consumers
        mk0 = out.get("matches0")
        mk1 = out.get("matches1")
        # Ensure torch tensors on CPU
        mk0_t = mk0[0] if isinstance(mk0, (list, tuple)) else mk0
        mk1_t = mk1[0] if isinstance(mk1, (list, tuple)) else mk1
        if not isinstance(mk0_t, torch.Tensor):
            mk0_t = torch.as_tensor(mk0_t)
        if not isinstance(mk1_t, torch.Tensor):
            mk1_t = torch.as_tensor(mk1_t)
        mk0_t = mk0_t.detach().cpu()
        mk1_t = mk1_t.detach().cpu()

        return {
            "mkpts0": mk0_t,
            "mkpts1": mk1_t,
            # pass-through raw outputs for debugging/analysis
            **out,
        }


if __name__ == "__main__":
    # Minimal sanity check: auto-download a small pretrained model and run a quick pass
    from easy_local_features.utils import io, ops

    # Prefer the smallest published weights to keep download light
    conf = {
        "pretrained": "xfeat-3_layers",
    }
    method = Desc_Reasoning_baseline(conf)

    # Find example images from the repo
    def _pick(path_options):
        for p in path_options:
            if Path(p).exists():
                return p
        raise FileNotFoundError(f"Could not find any of: {path_options}")

    img0_path = _pick([
        "tests/assets/megadepth0.jpg",
        "test/assets/megadepth0.jpg",
        "assets/megadepth0.jpg",
    ])
    img1_path = _pick([
        "tests/assets/megadepth1.jpg",
        "test/assets/megadepth1.jpg",
        "assets/megadepth1.jpg",
    ])

    img0 = io.fromPath(img0_path)
    img1 = io.fromPath(img1_path)

    # Light resize for speed
    img0 = ops.resize_short_edge(img0, 320)[0]
    img1 = ops.resize_short_edge(img1, 320)[0]

    kpts0, desc0 = method.detectAndCompute(img0)
    print(f"keypoints0: {tuple(kpts0.shape)}, descriptors0: {tuple(desc0.shape)}")

    matches = method.match(img0, img1)
    print(f"matches: {len(matches['mkpts0'])}")