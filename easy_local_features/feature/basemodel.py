from abc import ABC, abstractmethod
from omegaconf import OmegaConf

class BaseExtractor(ABC):
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
    
    @abstractmethod
    def match(self, image0, image1):
        raise NotImplementedError("Every BaseExtractor must implement the match method.")
    
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
        detector = detector(self.conf)
        self.detect = detector.detect
        
        def detectAndCompute(image, return_dict=False):
            keypoints = detector.detect(image)
            keypoints, descriptors = self.compute(image, keypoints)
            if return_dict:
                return {
                    'keypoints': keypoints,
                    'descriptors': descriptors
                }
            return keypoints, descriptors
        
        self.detectAndCompute = detectAndCompute