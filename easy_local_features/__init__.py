import importlib
import os
from .feature.basemodel import BaseExtractor
os.environ["TFHUB_CACHE_DIR"] = os.path.expanduser(os.path.join('~', '.cache', 'torch', 'hub', 'checkpoints', 'easy_local_features', 'tfhub'))

available_extractors = [
    'alike', 'aliked', 'd2net',
    'dalf', 'deal', 'dedode',
    'delf', 'disk', 'r2d2',
    'sosnet', 'superpoint', 'tfeat',
]

def importByName(name):
    
    
    package_name = f"easy_local_features.feature.baseline_{name}"
    importlib.import_module(package_name)
    
    subclasses = BaseExtractor.__subclasses__()
    # find the correct subclass
    for sub in subclasses:
        if f"{name}_baseline" == sub.__name__.lower():
            return sub
    
    raise ValueError(f"Could not find a subclass of BaseExtractor in <{package_name}> that contains <{name}>")

def getExtractor(extractor_name:str, conf={}):
    assert extractor_name in available_extractors, f"Invalid extractor {extractor_name}. Available extractors: {available_extractors}"
    extractor = importByName(extractor_name)    
    return extractor(conf)