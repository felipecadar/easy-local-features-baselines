# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   r2d2 -> extract_localization
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   13/07/2022 09:59
=================================================='''
import os
import os.path as osp
import h5py
import numpy as np
import cv2
import torch.utils.data as Data
from tqdm import tqdm
from types import SimpleNamespace
import logging
import pprint
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf

from easy_local_features.submodules.git_sfd2.nets import ResSegNet, ResSegNetV2
from easy_local_features.submodules.git_sfd2.utils import extract_resnet_return

confs = {
    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n3000-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n3000-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 3000,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n2000-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n2000-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 2000,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n1000-r1600': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n1000-r1600',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 1000,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
        'mask': False,
    },

    'ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1024': {
        'output': 'feats-ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1024',
        'model': {
            'name': 'ressegnetv2',
            'use_stability': True,
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': False,
            'scales': [1.0],
            'model_fn': osp.join(os.getcwd(),
                                 "weights/20220810_ressegnetv2_wapv2_ce_sd2mfsf_uspg.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
        'mask': False,
    },
}

class ImageDataset(Data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
    }

    def __init__(self, root, conf, image_list=None,
                 mask_root=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        if image_list is None:
            for g in conf.globs:
                self.paths += list(Path(root).glob('**/' + g))
            if len(self.paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            self.paths = [i.relative_to(root) for i in self.paths]
        else:
            with open(image_list, "r") as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    self.paths.append(Path(l))

        logging.info(f'Found {len(self.paths)} images in root {root}.')

        if mask_root is not None:
            self.mask_root = mask_root
        else:
            self.mask_root = None

        print("mask_root: ", self.mask_root)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(w, h) > self.conf.resize_max):
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_CUBIC)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': str(path),
            'image': image,
            'original_size': np.array(size),
        }

        if self.mask_root is not None:
            mask_path = Path(str(path).replace("jpg", "png"))
            if osp.exists(mask_path):
                mask = cv2.imread(str(self.mask_root / mask_path))
                mask = cv2.resize(mask, dsize=(image.shape[2], image.shape[1]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = np.zeros(shape=(image.shape[1], image.shape[2], 3), dtype=np.uint8)

            data['mask'] = mask

        return data

    def __len__(self):
        return len(self.paths)


def get_model(model_name, use_stability=False):
    if model_name == 'ressegnet':
        model = ResSegNet(outdim=128, require_stability=use_stability).eval()
        # model.load_state_dict(torch.load(weight_path)['model'], strict=True)
        extractor = extract_resnet_return
    if model_name == 'ressegnetv2':
        model = ResSegNetV2(outdim=128, require_stability=use_stability).eval()
        # model.load_state_dict(torch.load(weight_path)['model'], strict=False)
        extractor = extract_resnet_return

    return model, extractor

class SFD2(nn.Module):
    def __init__(self, 
            config_name="ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1600",
            use_stability=False,
            top_k=4096
        ):
        super().__init__()
        model_name = config_name.split("-")[0]
        self.model_config = confs[config_name]
        self.model, self.extractor = get_model(model_name, use_stability)
        self.top_k = top_k
        self.model.eval()
        
    def forward(self, data):
        pred = self.extractor(self.model, img=data["image"],
                    topK=self.top_k,
                    mask=None,
                    conf_th=self.model_config["model"]["conf_th"],
                    scales=self.model_config["model"]["scales"],
                    )

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred.keys():
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5
            
        return pred
