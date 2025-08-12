import pytest
import os
from easy_local_features.feature.basemodel import BaseExtractor
from easy_local_features.utils import io, vis, ops
import numpy as np
import pkgutil
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def get_all_subclasses(cls):
    subclasses = set(cls.__subclasses__())
    for subclass in list(subclasses):
        subclasses.update(get_all_subclasses(subclass))
    return subclasses

def load_all_modules_from_package(package):
    for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        if not is_pkg:
            importlib.import_module(module_name)

@pytest.fixture
def all_subclasses():
    import easy_local_features
    load_all_modules_from_package(easy_local_features.feature)
    return  get_all_subclasses(BaseExtractor)

@pytest.mark.skip
def test_feature_extractors(all_subclasses: list[BaseExtractor]):
    image0 = io.fromPath(str(ROOT / "assets/notredame.png"))
    image1 = io.fromPath(str(ROOT / "assets/notredame2.jpeg"))
    
    image0 = ops.crop_square(ops.resize_short_edge(image0, 320)[0])
    image1 = ops.crop_square(ops.resize_short_edge(image1, 320)[0])
    
    os.makedirs("test/results", exist_ok=True)
    
    for subclass in all_subclasses:
        if hasattr(subclass, "variations"):
            variations = subclass.variations
            for key, values in variations.items():
                for value in values:
                    try:
                        extractor = subclass({key: value, 'top_k': 128})
                        matches = extractor.match(image0, image1)
                        
                        vis.plot_pair(image0, image1, title=subclass.__name__ + " " + value, figsize=(8, 4))
                        vis.plot_matches(matches['mkpts0'], matches['mkpts1'])
                        vis.save(f"test/results/{subclass.__name__}_{value}.png")
                    except:
                        print(f"Error in {subclass.__name__} with {key}={value}")
        else:
            extractor = subclass({'top_k': 128})
            matches = extractor.match(image0, image1)
            
            vis.plot_pair(image0, image1, title=subclass.__name__, figsize=(8, 4))
            vis.plot_matches(matches['mkpts0'], matches['mkpts1'])
            vis.save(f"test/results/{subclass.__name__}.png")

if __name__ == "__main__":
    import argparse
    def parse():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model",  "-m", type=str, default="all", help="Model name to test. Default: all.")
        return parser.parse_args()
    args = parse()
    
    import easy_local_features.matching
    load_all_modules_from_package(easy_local_features.matching)
    _all_subclasses = get_all_subclasses(BaseExtractor)
    
    # print all subclasses
    for subclass in _all_subclasses:
        print(subclass.__name__)
    
    if args.model != "all":
        _all_subclasses = [subclass for subclass in _all_subclasses if args.model in subclass.__name__]
    
    test_feature_extractors(_all_subclasses)

        
            
        