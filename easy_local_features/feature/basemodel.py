from abc import ABC, abstractmethod
from omegaconf import OmegaConf

class BaseExtractor(ABC):
    @abstractmethod
    def detectAndCompute(self, image):
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
    