import torch
import numpy as np
import cv2, os

import pyrootutils
root = pyrootutils.find_root()

class DEAL_baseline():
    def __init__(self, max_kps=1024, device_id=-1, model_path=None):

        self.max_kps = max_kps
        self.device_id = device_id

        if device_id >= 0:
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

        # /home/USER/.cache/torch/hub/checkpoints/
        if model_path is None:
            cache_path = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'DEAL')
        else:
            cache_path = model_path

        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)

        self.deal = torch.hub.load('verlab/DEAL_NeurIPS_2021', 'DEAL', True, cache_path)
        self.deal.device = device
        self.deal.net.eval()
        self.sift = cv2.SIFT_create(max_kps)

    def normalize(self, img):
        # if img is tensor, convert to numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))

            if img.max() <= 1.0:
                img = (img * 255)
                
            img = img.astype(np.uint8)

        # if img is color 
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 1:
                img = img[:,:,0]
            else:
                raise ValueError("img shape is wrong")

        return img

    def detectAndCompute(self, img, mask=None):
        gray = self.normalize(img)

        with torch.no_grad():
            kps = self.detect(gray, mask)
            kps, desc = self.compute(gray, kps)

        return kps, desc

    def detect(self, img, mask=None):
        gray = self.normalize(img)

        with torch.no_grad():
            kps = self.sift.detect(gray, None)

        return kps

    def compute(self, img, kps):
        gray = self.normalize(img)

        if not isinstance(kps[0], cv2.KeyPoint):
            kps = [cv2.KeyPoint(kp[0], kp[1], 0) for kp in kps]

        with torch.no_grad():
            desc = self.deal.compute(gray, kps)

        return kps, desc


if __name__ == "__main__":
    img = cv2.imread(str(root / "assets" / "notredame.png"))
    extractor = DEAL_baseline()

    keypoints0, descriptors0 = extractor.detectAndCompute(img)
    
    # Visualize keypoints
    img_kpts = cv2.drawKeypoints(img, keypoints0, None, color=(0, 255, 0), flags=0)

    cv2.imshow("Keypoints", img_kpts)
    cv2.waitKey(0)