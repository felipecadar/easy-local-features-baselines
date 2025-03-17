## Spair71K dataset
import torch

SPAIR_URL = "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"

class Spair71K:
    
    def __init__(self, root_dir):
        pass
    
    def download(self, save_dir):
        torch.hub.download_url_to_file(SPAIR_URL, save_dir)
        pass