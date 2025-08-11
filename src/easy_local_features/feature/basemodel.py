from abc import ABC, abstractmethod
from omegaconf import OmegaConf


class MethodType:
    """String constants describing the type of feature method.

    - DETECT_DESCRIBE: traditional detector+descriptor operating on a single image
    - DESCRIPTOR_ONLY: descriptor that requires external keypoints
    - END2END_MATCHER: end-to-end matcher operating directly on two images
    """
    DETECT_DESCRIBE = "detect_describe"
    DESCRIPTOR_ONLY = "descriptor_only"
    END2END_MATCHER = "end2end_matcher"

class BaseExtractor(ABC):
    # Optional override in subclasses; if None, inferred from `has_detector`.
    METHOD_TYPE: str | None = None

    @abstractmethod
    def detectAndCompute(self, image, return_dict=False):
        raise NotImplementedError("Every BaseExtractor must implement the detectAndCompute method.")
    
    @abstractmethod
    def detect(self, image):
        raise NotImplementedError("Every BaseExtractor must implement the detect method.")
    
    @abstractmethod
    def compute(self, image, keypoints):
        raise NotImplementedError("Every BaseExtractor must implement the compute method.")    
    
    @abstractmethod
    def to(self, device):
        raise NotImplementedError("Every BaseExtractor must implement the to method.")
    
    @property
    @abstractmethod
    def has_detector(self):
        raise NotImplementedError("Every BaseExtractor must implement the has_detector property.")
    
    
    def __call__(self, data):
        '''
        data: dict
            {
                'image': image,
            }
        '''
        return self.detectAndCompute(data['image'], return_dict=True)
    
    def addDetector(self, detector):
        
        def detectAndCompute(image, return_dict=False):
            keypoints = detector.detect(image)
            # Support compute() returning either descriptors OR (keypoints, descriptors)
            out = self.compute(image, keypoints)
            if isinstance(out, tuple) and len(out) == 2:
                keypoints, descriptors = out
            else:
                descriptors = out
            if return_dict:
                return {
                    'keypoints': keypoints,
                    'descriptors': descriptors
                }
            return keypoints, descriptors
        
        self.detect = detector.detect
        self.detectAndCompute = detectAndCompute
    
    # @abstractmethod
    def match(self, image1, image2):
        
        kp0, desc0 = self.detectAndCompute(image1)
        kp1, desc1 = self.detectAndCompute(image2)
        
        data = {
            "descriptors0": desc0,
            "descriptors1": desc1,
        }
        
        response = self.matcher(data)
        
        m0 = response['matches0'][0]
        valid = m0 > -1
        
        mkpts0 = kp0[0, valid]
        mkpts1 = kp1[0, m0[valid]]
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
        }    

    @property
    def method_type(self) -> str:
        """Return the standardized method type string.

        Subclasses may override by setting class attribute `METHOD_TYPE` or
        by overriding this property. If not set, we infer from `has_detector`:
        - True  -> detect+describe
        - False -> descriptor-only
        """
        if self.METHOD_TYPE is not None:
            return self.METHOD_TYPE
        return MethodType.DETECT_DESCRIBE if self.has_detector else MethodType.DESCRIPTOR_ONLY