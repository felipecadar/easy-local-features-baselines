import torch
from easy_local_features.submodules.git_depthanythingv2.dpt import DepthAnythingV2
from omegaconf import OmegaConf

from easy_local_features.utils import ops, vis

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

model_urls = {
    'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true',
    'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true',
    'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true',
    # 'vitg': '' # Comming soon
}

class DepthAnythingV2_baseline(torch.nn.Module):    
    default_conf = {
        'model_name': 'vits',
    }
    
    def __init__(self, conf={}) -> None:
        super(DepthAnythingV2_baseline, self).__init__()
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

        self.model = DepthAnythingV2(**model_configs[conf.model_name])
        self.model.eval()
        weights = torch.hub.load_state_dict_from_url(model_urls[conf.model_name], map_location='cpu')
        self.model.load_state_dict(weights)
        
    def forward(self, image):
        image = ops.prepareImage(image, batch=False)
        image = ops.to_cv(image)
        return self.model.infer_image(image)
    
    def compute(self, img):
        return self.forward(img)
    
if __name__ == "__main__":
    from easy_local_features.utils import io
    img = io.fromPath('./test/assets/megadepth0.jpg')
    model = DepthAnythingV2_baseline({
            'model_name': 'vitl'
        }).to('cpu')

    depth = model.compute(img)
    
    # print(depth.shape)
    vis.plot_image(img)
    vis.plot_depth(depth)
    vis.show()