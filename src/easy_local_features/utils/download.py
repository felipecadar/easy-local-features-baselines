import torch
import os

from .pathutils import CACHE_BASE

def getCache(namespace):
    cache_path = CACHE_BASE / namespace
    if not cache_path.exists():
        os.makedirs(str(cache_path), exist_ok=True)
    return cache_path

def downloadModel(namespace, model_name, url, force=False):
    if "pth" not in model_name and "pt" not in model_name:
        model_name += ".pth"

    cache_path = getCache(namespace)
    cache_path = cache_path / model_name
    
    dirpath = cache_path.parent
    if not dirpath.exists():
        os.makedirs(str(dirpath), exist_ok=True)
    
    if not cache_path.exists() or force:
        print(f"Downloading {model_name} model... from {url}")
        torch.hub.download_url_to_file(url, str(cache_path))
    return cache_path