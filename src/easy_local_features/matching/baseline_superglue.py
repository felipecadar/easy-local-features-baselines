import os

from easy_local_features.feature.basemodel import BaseExtractor, MethodType
from easy_local_features.submodules.git_superglue.models.superglue import SuperGlue

import torch
import numpy as np
import cv2
import wget

class SuperGlue_baseline(BaseExtractor):
    METHOD_TYPE = MethodType.END2END_MATCHER
    
    def __init__(self, conf=None, weights='indoor', sinkhorn_iterations=100, match_threshold=0.2, descriptor_dim=256, device=-1, model_path=None):
        if conf is not None:
            weights = conf.get('weights', weights)
            sinkhorn_iterations = conf.get('sinkhorn_iterations', sinkhorn_iterations)
            match_threshold = conf.get('match_threshold', match_threshold)
            descriptor_dim = conf.get('descriptor_dim', descriptor_dim)
            device = conf.get('device', device)
            model_path = conf.get('model_path', model_path)
            
        self.DEV   = torch.device(f'cuda:{device}' if (torch.cuda.is_available() and device>=0) else 'cpu')
        self.CPU   = torch.device('cpu')
    
        if model_path is None:
            assert weights in ['indoor', 'outdoor'], "weights must be either 'indoor' or 'outdoor'"
            if weights == 'indoor':
                url = 'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth'
            else: # outdoor
                url = 'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth'

            cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'SuperGlue')
            model_name = url.split('/')[-1]
            
            if not os.path.exists(cache_path):
                os.makedirs(cache_path, exist_ok=True)

            cache_path = os.path.join(cache_path, model_name)
            if not os.path.exists(cache_path):
                print(f'Downloading SuperGlue {weights} model...')
                wget.download(url, cache_path)
                print('Done.')

            model_path = cache_path        

        config = {
            'descriptor_dim': descriptor_dim,
            'weights': weights,
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
            'model_path': model_path,
        }

        self.model = SuperGlue(config)
        self.model = self.model.to(self.DEV)
        self.model.eval()
        self.matcher = self # map matcher to self for basemodel generic matching logic

    @property
    def has_detector(self):
        return False

    def detectAndCompute(self, image, return_dict=False):
        if hasattr(self, 'detector_ext'):
            return self.detector_ext.detectAndCompute(image, return_dict=return_dict)
        raise NotImplementedError("SuperGlue is a matcher only. It does not extract features.")
        
    def compute(self, image, keypoints):
        if hasattr(self, 'detector_ext'):
            return self.detector_ext.compute(image, keypoints)
        raise NotImplementedError("SuperGlue is a matcher only. It does not extract features.")

    def addDetector(self, detector):
        self.detector_ext = detector
        
    def to(self, device):
        self.DEV = torch.device(device)
        self.model = self.model.to(self.DEV)
        return self
        
    def __call__(self, data):
        # Compatible with generic __call__ signature expecting data dictionary
        if "image1" in data:
            with torch.no_grad():
                res = self.model(data)
            return res
        return dict()

    def _toTorch(self, img):
        if isinstance(img, torch.Tensor):
            if img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.ndim == 3:
                img = img.unsqueeze(0)
            return img.to(self.DEV)
        
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return torch.from_numpy(img/255.).float()[None, None].to(self.DEV)

    def match(self, img0, img1, *, kps0=None, desc0=None, kps1=None, desc1=None):
        """Unified matcher API.
        - If keypoints/descriptors are provided, uses them.
        - If not provided but a detector is attached via `addDetector`, uses it.
        - Otherwise, raises NotImplementedError since SuperGlue requires features.
        """
        if kps0 is None or kps1 is None or desc0 is None or desc1 is None:
            try:
                kps0, desc0 = self.detectAndCompute(img0)
                kps1, desc1 = self.detectAndCompute(img1)
            except NotImplementedError:
                raise NotImplementedError("SuperGlue requires keypoints and descriptors; provide kps0,kps1,desc0,desc1.")
        return self._match_with_features(img0, kps0, desc0, img1, kps1, desc1)

    def _match_with_features(self, img0, kps0, desc0, img1, kps1, desc1):

        if kps0 is None or kps1 is None:
            return [], []

        if len(kps0) == 0 or len(kps1) == 0:
            return [], []

        if isinstance(kps0[0], cv2.KeyPoint):
            keypoints0 = torch.tensor([kp.pt for kp in kps0]).to(self.DEV).unsqueeze(0).float()
            keypoints1 = torch.tensor([kp.pt for kp in kps1]).to(self.DEV).unsqueeze(0).float()

            scores0 = torch.tensor([kp.response for kp in kps0]).to(self.DEV).unsqueeze(0).float()
            scores1 = torch.tensor([kp.response for kp in kps1]).to(self.DEV).unsqueeze(0).float()

        elif isinstance(kps0, np.ndarray):
            keypoints0 = torch.tensor(kps0).to(self.DEV).unsqueeze(0).float()
            keypoints1 = torch.tensor(kps1).to(self.DEV).unsqueeze(0).float()

            scores0 = torch.ones(len(kps0)).to(self.DEV).unsqueeze(0).float()
            scores1 = torch.ones(len(kps1)).to(self.DEV).unsqueeze(0).float()

        elif isinstance(kps0, torch.Tensor):
            keypoints0 = kps0.to(self.DEV).float()
            if keypoints0.ndim == 2:
                keypoints0 = keypoints0.unsqueeze(0)
            keypoints1 = kps1.to(self.DEV).float()
            if keypoints1.ndim == 2:
                keypoints1 = keypoints1.unsqueeze(0)

            # assign dummy scores if none
            b, n, _ = keypoints0.shape
            scores0 = torch.ones((b, n)).to(self.DEV).float()
            b1, n1, _ = keypoints1.shape
            scores1 = torch.ones((b1, n1)).to(self.DEV).float()

        if isinstance(desc0, np.ndarray):
            descriptors0 = torch.tensor(desc0.T).to(self.DEV).unsqueeze(0).float()
            descriptors1 = torch.tensor(desc1.T).to(self.DEV).unsqueeze(0).float()
        elif isinstance(desc0, torch.Tensor):
            descriptors0 = desc0.to(self.DEV).float()
            if descriptors0.ndim == 2:
                descriptors0 = descriptors0.unsqueeze(0)
            # SuperGlue expects [B, D, N], whereas easy_local_features passes [B, N, D]
            if descriptors0.shape[-1] == self.model.config['descriptor_dim']:
                descriptors0 = descriptors0.transpose(-1, -2)
                
            descriptors1 = desc1.to(self.DEV).float()
            if descriptors1.ndim == 2:
                descriptors1 = descriptors1.unsqueeze(0)
            if descriptors1.shape[-1] == self.model.config['descriptor_dim']:
                descriptors1 = descriptors1.transpose(-1, -2)
    
        inp = {
            'image0': self._toTorch(img0),
            'keypoints0': keypoints0,
            'descriptors0':descriptors0,
            'scores0':scores0,
            'image1': self._toTorch(img1),
            'keypoints1':keypoints1,
            'descriptors1':descriptors1,
            'scores1':scores1,
        }

        with torch.no_grad():
            res = self.model(inp)
            
        b_size = keypoints0.shape[0]
        out_list = []
        
        for b in range(b_size):
            # Move to CPU and detach before converting
            matches0 = res['matches0'][b].detach().cpu()
            matches1 = res['matches1'][b].detach().cpu()
            matching_scores0 = res['matching_scores0'][b].detach().cpu().numpy()
            matching_scores1 = res['matching_scores1'][b].detach().cpu().numpy()
            
            matching_scores0[matching_scores0 == 0] = 1e-9
            matching_scores1[matching_scores1 == 0] = 1e-9

            # Derive matched keypoints (torch tensors detached to CPU then converted to numpy)
            valid = matches0 > -1
            mkpts0 = keypoints0[b, valid].detach().cpu()
            mkpts1 = keypoints1[b, matches0[valid]].detach().cpu()

            # Return raw arrays for downstream conversion; higher-level adapters can build cv2 matches if desired.
            out_dict = {
                "keypoints0": keypoints0[b].detach().cpu(),
                "keypoints1": keypoints1[b].detach().cpu(),
                "descriptors0": descriptors0[b].detach().cpu(),
                "descriptors1": descriptors1[b].detach().cpu(),
                "matches": matches0,
                "matches0": matches0,
                "matches1": matches1,
                "scores": torch.from_numpy(matching_scores0),
                "matching_scores0": matching_scores0,
                "matching_scores1": matching_scores1,
                "mkpts0": mkpts0,
                "mkpts1": mkpts1,
            }
            out_list.append(out_dict)

        if b_size == 1:
            return out_list[0]
        return out_list
