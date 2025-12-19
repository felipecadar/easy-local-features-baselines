from __future__ import annotations

from typing import Literal, Optional, TypedDict

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from romav2 import RoMaV2
import romav2.device as romav2_device
import romav2.romav2 as romav2_mod

from .basemodel import BaseExtractor, MethodType
from ..utils import ops


DeviceName = Literal["auto", "cpu", "cuda", "mps"]
SettingName = Literal["mega1500", "scannet1500", "wxbs", "satast", "turbo", "fast", "base", "precise"]


class ROMAV2Config(TypedDict, total=False):
    model_name: str
    top_k: int
    setting: SettingName
    compile: bool
    device: DeviceName
    # Optional overrides (rarely needed; setting controls these)
    H_lr: Optional[int]
    W_lr: Optional[int]
    H_hr: Optional[int]
    W_hr: Optional[int]


class RoMaV2_baseline(BaseExtractor):
    """RoMa v2 dense matching wrapper.

    Key implementation detail:
    - Upstream `RoMaV2.match()` resizes with `F.interpolate` and then calls `forward()`.
      On MPS/CPU, RoMaV2 weights are often bfloat16, causing dtype mismatches
      (float32 inputs + bfloat16 weights). We therefore:
        1) resize in float32 (always supported),
        2) cast resized images to the model's parameter dtype,
        3) call `forward()` directly (i.e., `self.model(...)`),
        4) build the same `preds` dict expected by `RoMaV2.sample()`.
    """

    METHOD_TYPE = MethodType.END2END_MATCHER

    default_conf: ROMAV2Config = {
        "model_name": "romav2",
        "top_k": 5000,
        "setting": "precise",
        "compile": False,  # torch.compile is often fragile on MPS; enable explicitly if you want it
        "device": "auto",
        "H_lr": None,
        "W_lr": None,
        "H_hr": None,
        "W_hr": None,
    }

    def __init__(self, conf: ROMAV2Config = {}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.num_keypoints = int(conf.top_k)

        # RoMaV2.forward enforces this.
        torch.set_float32_matmul_precision("highest")

        self.device = self._resolve_device(self.conf.device)
        self.model: RoMaV2 = self._build_model(device=self.device)
        self.model.eval()

    def _resolve_device(self, dev: str) -> torch.device:
        if dev == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(dev)

    def _set_romav2_global_device(self, device: torch.device) -> None:
        # romav2 uses a global `device` imported as a module-level symbol.
        # We must update BOTH the source module and the already-imported alias.
        romav2_device.device = device
        romav2_mod.device = device

    def _build_model(self, device: torch.device) -> RoMaV2:
        self._set_romav2_global_device(device)

        cfg = RoMaV2.Cfg(setting=self.conf.setting, compile=bool(self.conf.compile))
        model = RoMaV2(cfg)
        model.to(device)

        # Optional resolution overrides (mostly for debugging/ablation)
        if self.conf.H_lr is not None:
            model.H_lr = int(self.conf.H_lr)
        if self.conf.W_lr is not None:
            model.W_lr = int(self.conf.W_lr)
        if self.conf.H_hr is not None:
            model.H_hr = int(self.conf.H_hr)
        if self.conf.W_hr is not None:
            model.W_hr = int(self.conf.W_hr)

        return model

    def to(self, device):
        self.device = torch.device(device)
        self._set_romav2_global_device(self.device)
        # Recreate model to ensure internal globals + compiled state are consistent.
        self.model = self._build_model(device=self.device)
        self.model.eval()
        return self

    @property
    def has_detector(self):
        return True

    def detectAndCompute(self, image, return_dict=False):
        raise NotImplementedError("RoMaV2 is an end-to-end matcher; use match(image1, image2).")

    def detect(self, image):
        raise NotImplementedError("RoMaV2 is an end-to-end matcher; use match(image1, image2).")

    def compute(self, image, keypoints):
        raise NotImplementedError("RoMaV2 is an end-to-end matcher; use match(image1, image2).")

    def _prepare_image_float32(self, image: torch.Tensor) -> torch.Tensor:
        image = ops.prepareImage(image, batch=True, gray=False)
        # Ensure 3 channels for RoMaV2
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        # Ensure float32 in [0,1]
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        else:
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
        return image.to(device=self.device, dtype=torch.float32)

    def _model_dtype(self) -> torch.dtype:
        # The model might use bf16 on GPU/MPS; match inputs to this dtype AFTER resizing.
        return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def match(self, image1, image2):
        img1 = self._prepare_image_float32(image1)
        img2 = self._prepare_image_float32(image2)

        imA_h, imA_w = img1.shape[2:]
        imB_h, imB_w = img2.shape[2:]

        H_lr, W_lr = int(self.model.H_lr), int(self.model.W_lr)
        img1_lr = F.interpolate(img1, size=(H_lr, W_lr), mode="bicubic", align_corners=False, antialias=True)
        img2_lr = F.interpolate(img2, size=(H_lr, W_lr), mode="bicubic", align_corners=False, antialias=True)

        if getattr(self.model, "H_hr", None) is not None and getattr(self.model, "W_hr", None) is not None:
            H_hr, W_hr = int(self.model.H_hr), int(self.model.W_hr)
            img1_hr = F.interpolate(img1, size=(H_hr, W_hr), mode="bicubic", align_corners=False, antialias=True)
            img2_hr = F.interpolate(img2, size=(H_hr, W_hr), mode="bicubic", align_corners=False, antialias=True)
        else:
            img1_hr = None
            img2_hr = None

        # Cast resized images to model dtype (avoids float32 input + bf16 weights mismatch)
        mdtype = self._model_dtype()
        img1_lr = img1_lr.to(dtype=mdtype)
        img2_lr = img2_lr.to(dtype=mdtype)
        if img1_hr is not None:
            img1_hr = img1_hr.to(dtype=mdtype)
        if img2_hr is not None:
            img2_hr = img2_hr.to(dtype=mdtype)

        # Forward directly (bypasses upstream match() dtype pitfalls)
        fwd = self.model(img1_lr, img2_lr, img_A_hr=img1_hr, img_B_hr=img2_hr)

        warp_AB = fwd["warp_AB"]
        confidence_AB = fwd["confidence_AB"]
        warp_BA = fwd["warp_BA"]
        confidence_BA = fwd["confidence_BA"]

        overlap_AB, precision_AB = romav2_mod._map_confidence(confidence=confidence_AB, threshold=self.model.threshold)
        if getattr(self.model, "bidirectional", False):
            overlap_BA, precision_BA = romav2_mod._map_confidence(confidence=confidence_BA, threshold=self.model.threshold)
        else:
            overlap_BA = None
            precision_BA = None

        preds = {
            "warp_AB": warp_AB,
            "confidence_AB": confidence_AB,
            "overlap_AB": overlap_AB,
            "precision_AB": precision_AB,
            "warp_BA": warp_BA,
            "confidence_BA": confidence_BA,
            "overlap_BA": overlap_BA,
            "precision_BA": precision_BA,
        }

        matches, overlaps, prec_AB_s, prec_BA_s = self.model.sample(preds, int(self.num_keypoints))
        kptsA, kptsB = self.model.to_pixel_coordinates(matches, imA_h, imA_w, imB_h, imB_w)

        out = {
            "mkpts0": (kptsA[0] if kptsA.ndim == 3 and kptsA.shape[0] == 1 else kptsA).detach().cpu(),
            "mkpts1": (kptsB[0] if kptsB.ndim == 3 and kptsB.shape[0] == 1 else kptsB).detach().cpu(),
            "overlaps": overlaps.detach().cpu(),
        }
        if prec_AB_s is not None:
            out["precision_AB"] = prec_AB_s.detach().cpu()
        if prec_BA_s is not None:
            out["precision_BA"] = prec_BA_s.detach().cpu()
        return out

