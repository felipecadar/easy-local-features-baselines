import os

import torch
from omegaconf import OmegaConf

from easy_local_features.submodules.git_relf.descriptor_utils import DescGroupPoolandNorm
from easy_local_features.submodules.git_relf.model import RELF_model

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from ..utils.download import downloadModel
from .basemodel import BaseExtractor, MethodType

try:
    import mmcv
except ImportError:
    raise ImportError(
        "Please install `mmcv` to use RELF. I know, it's a pain, but I'm not putting it in the requirements."
    )

available_models = [
    "re_resnet",
    "E2SFCNN",
    "E2SFCNN_QUOT",
    "EXP",
    "CNN",
    "wrn16_8_stl",
    "e2wrn16_8_stl",
    "e2wrn28_10",
    "e2wrn28_7",
    "e2wrn28_10R",
    "e2wrn28_7R",
    "e2wrn10_8R",
    "e2wrn16_8R_stl",
]

# class ReFTDescriptor:
#     def __init__(self, model:RELF_model):

#         self.model = model
#         self.pool_and_norm = DescGroupPoolandNorm(model.args)

#     def __call__(self, image, kpts):
#         desc = self.model(image, kpts)

#         ## kpts torch.tensor ([B, K, 2]), desc torch.tensor ([B, K, CG])
#         k1, d1 = self.pool_and_norm.desc_pool_and_norm_infer(kpts, desc)

#         return k1, d1


class RELF_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    """'
    Model name:
     're_resnet': Rotation-Equivariant ResNet-18
     'e2wrn16_8R_stl': Rotation-Equivariant Wide-ResNet-16-8
    """

    default_conf = {
        "model": "re_resnet",
        "num_group": 16,
        "channels": 64,
        "candidate": "top1",  # 'topX' or [0,1] float,
        "orientation": 16,
        "top_k": 4096,
    }

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        assert conf.model in available_models, f"Model {conf.model} not available. Choose from {available_models}"

        os.environ["Orientation"] = str(conf.orientation)
        self.max_kps = conf.top_k
        self.device = torch.device("cpu")
        self.net = RELF_model(self.conf)

        if conf.model == "re_resnet":
            url = "https://github.com/felipecadar/easy-local-features-baselines/releases/download/RELF-weights/best_model.pt"
            cache_path = downloadModel("RELF", "re_resnet", url)
        elif conf.model == "e2wrn16_8R_stl":
            url = "https://github.com/felipecadar/easy-local-features-baselines/releases/download/RELF-weights/best_model_wrn.pt"
            cache_path = downloadModel("RELF", "e2wrn16_8R_stl", url)
        else:
            raise NotImplementedError(f"Model {conf.model}: weights not available.")

        weights = torch.load(cache_path, map_location=self.device)
        self.net.train()
        self.net.load_state_dict(weights)

        self.pool_and_norm = DescGroupPoolandNorm(conf)
        self.matcher = NearestNeighborMatcher(
            {
                "ratio_thresh": None,
                "distance_thresh": None,
                "mutual_check": True,
                "loss": None,
                "normalize": False,
            }
        )

        self.net.eval()
        self.net.to(self.device)

    def detectAndCompute(self, image, return_dict=False):
        raise NotImplementedError

    def detect(self, image):
        raise NotImplementedError

    def compute(self, img, kps):
        image = ops.prepareImage(img, imagenet=True).to(self.device)

        with torch.no_grad():
            desc = self.net(image, kps)
            kps, desc = self.pool_and_norm.desc_pool_and_norm_infer(kps, desc)

        return kps, desc

    def to(self, device):
        self.net.to(device)
        self.device = device
        return self

    @property
    def has_detector(self):
        return False


if __name__ == "__main__":
    from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
    from easy_local_features.utils import io

    relf = RELF_baseline()
    sp = SuperPoint_baseline(
        {
            "keypoint_threshold": 0,
            "max_keypoints": 1024,
            "force_num_keypoints": True,
        }
    )

    batch_size = 4
    imgs = io.fromPath("test/assets/megadepth0.jpg").repeat(batch_size, 1, 1, 1)

    kps = sp.detect(imgs)
    kps, desc = relf.compute(imgs, kps)

    print(kps.shape, desc.shape)
