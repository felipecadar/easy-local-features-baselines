import torch
import os

from .pathutils import CACHE_BASE

def getCache(namespace):
    cache_path = CACHE_BASE / namespace
    if not cache_path.exists():
        os.makedirs(str(cache_path), exist_ok=True)
    return cache_path

def downloadModel(namespace, model_name, url):
    cache_path = getCache(namespace)
    file_model_name = url.split('/')[-1]
    cache_path = cache_path / (model_name + '.pth')
    if not cache_path.exists():
        torch.hub.download_url_to_file(url, str(cache_path))
    return cache_path