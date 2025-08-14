import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..matching.nearest_neighbor import NearestNeighborMatcher
from ..utils import ops
from .basemodel import BaseExtractor, MethodType

# resnet forward
# def _forward_impl(self, x: Tensor) -> Tensor:
#     # See note [TorchScript super()]
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)

#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)

#     x = self.avgpool(x)
#     x = torch.flatten(x, 1)
#     x = self.fc(x)


class ResNet_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.DESCRIPTOR_ONLY
    available_models = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ]

    available_weights = {
        "resnet18": "ResNet18_Weights.DEFAULT",
        "resnet34": "ResNet34_Weights.DEFAULT",
        "resnet50": "ResNet50_Weights.DEFAULT",
        "resnet101": "ResNet101_Weights.DEFAULT",
        "resnet152": "ResNet152_Weights.DEFAULT",
    }

    default_conf = {"weights": "resnet18", "allow_resize": True}

    def __init__(self, conf={}):
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.device = torch.device("cpu")
        self.matcher = NearestNeighborMatcher()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", conf.weights, weights=self.available_weights[conf.weights]
        )
        self.model.eval()

    def features_forward(self, x, stop=4):
        if stop < 1 or stop > 4:
            raise ValueError("stop must be between 1 and 4")

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        if stop >= 1:
            x = self.model.layer1(x)
        if stop >= 2:
            x = self.model.layer2(x)
        if stop >= 3:
            x = self.model.layer3(x)
        if stop >= 4:
            x = self.model.layer4(x)

        return x

    def sample_features(self, keypoints, features, image_shape, mode="bilinear"):
        b, _, h, w = image_shape
        c = features.shape[1]
        keypoints = keypoints / (keypoints.new_tensor([w, h]))
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        features = torch.nn.functional.grid_sample(
            features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
        )
        features = torch.nn.functional.normalize(features.reshape(b, c, -1), p=2, dim=1)

        features = features.permute(0, 2, 1)
        return features

    def detectAndCompute(self, img, return_dict=None):
        raise NotImplementedError

    def detect(self, img, op=None):
        raise NotImplementedError

    def compute(self, img, keypoints=None, return_dict=False, stop=4):
        img = ops.prepareImage(img, imagenet=True).to(self.device)

        if self.conf.allow_resize:
            img = F.interpolate(img, [224, 224])

        features = self.features_forward(img, stop=stop)

        if keypoints is None:
            return features

        descriptors = self.sample_features(keypoints, features, img.shape)

        return keypoints, descriptors

    def to(self, device):
        self.model.to(device)
        self.device = device

    @property
    def has_detector(self):
        return False


if __name__ == "__main__":
    from easy_local_features.feature.baseline_xfeat import XFeat_baseline
    from easy_local_features.utils import io, ops, vis

    method = ResNet_baseline(
        {
            "weights": "resnet152",
            "allow_resize": False,
        }
    )
    detector = XFeat_baseline(
        {
            "top_k": 512,
        }
    )

    img0 = io.fromPath("test/assets/megadepth0.jpg")
    img1 = io.fromPath("test/assets/megadepth1.jpg")

    # resize to 512x512
    img0 = F.interpolate(img0, [512, 512])
    img1 = F.interpolate(img1, [512, 512])

    kps0 = detector.detect(img0)
    kps1 = detector.detect(img1)

    batched_images = torch.cat([img0, img1], dim=0)
    batched_kps = torch.cat([kps0, kps1], dim=0)

    _, batched_descriptors = method.compute(batched_images, batched_kps)

    descriptors0 = batched_descriptors[0].unsqueeze(0)
    descriptors1 = batched_descriptors[1].unsqueeze(0)

    # _, descriptors0 = method.compute(img0, kps0)
    # _, descriptors1 = method.compute(img1, kps1)

    print(descriptors0.shape)
    print(descriptors1.shape)

    response = method.matcher(
        {
            "descriptors0": descriptors0,
            "descriptors1": descriptors1,
        }
    )

    m0 = response["matches0"][0]
    valid = m0 > -1
    mkpts0 = kps0[valid]
    mkpts1 = kps1[m0[valid]]

    vis.plot_pair(img0, img1)
    vis.plot_matches(mkpts0, mkpts1)
    vis.add_text("ResNet")
    vis.show()
