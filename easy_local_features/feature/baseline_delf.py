import pyrootutils
root = pyrootutils.find_root()

import numpy as np
import cv2

import tensorflow as tf
import tensorflow_hub as hub

# python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# python3 -c "import torch; print(torch.cuda.is_available())"

class DELF_baseline():
    def __init__(self, top_k=2048, device=-1):
        self.model = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
        self.top_k = top_k

    def compute(self, img, cv_kps):
        raise NotImplemented

    def detect(self, img, op=None):
        raise NotImplemented

    def run_delf(self, image):
        float_image = tf.image.convert_image_dtype(image, tf.float32)

        return self.model(
            image=float_image,
            score_threshold=tf.constant(100.0),
            image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
            max_feature_num=tf.constant(self.top_k))

    def detectAndCompute(self, img, op=None):
        # make sure image is rgb
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = self.run_delf(img)

        locations = results['locations'].numpy()
        descriptors = results['descriptors'].numpy()
        scales = results['scales'].numpy()

        cv2_kps = [cv2.KeyPoint(x=loc[1], y=loc[0], size=scale, angle=0) for (loc, scale) in zip(locations, scales)]

        return cv2_kps, descriptors

if __name__ == "__main__":
    img1 = cv2.imread(str(root / "assets" / "notredame.png"))
    img2 = cv2.imread(str(root / "assets" / "notredame2.jpeg"))
    # img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    # img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)

    sift = cv2.SIFT_create()

    model = DELF_baseline(device=0)

    kps1, desc1 = model.detectAndCompute(img1)
    kps2, desc2 = model.detectAndCompute(img2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(desc1, desc2)

    # ransac
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matches = [m for m,msk in zip(matches, mask) if msk == 1]

    img3 = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, flags=2)

    cv2.imwrite("delf.png", img3)

