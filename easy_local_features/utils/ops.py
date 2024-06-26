import torch
import torchvision
import numpy as np
import cv2

from .io import fromPath

def prepareImage(image, gray=False, batch=True, imagenet=False):

    if isinstance(image, str):
        return fromPath(image, gray, batch, imagenet)
    
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image)

    if len(image.shape) == 2:
        image = image.unsqueeze(0)
    
    if len(image.shape) == 3 and image.shape[0] not in [1, 3]:
        image = image.permute(2, 0, 1)
        
    if len(image.shape) == 4 and image.shape[1] not in [1, 3]:
        image = image.permute(0, 3, 1, 2)
        
    if gray and image.shape[-3] == 3:
        image = image.mean(-3, keepdimage=True)
        
    if imagenet:
        image = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)
        
    if batch and len(image.shape) == 3:
        image = image.unsqueeze(0)
        
    if not batch and len(image.shape) == 4:
        image = image[0]
        
    return image


def to_cv(torch_image, convert_color=True, batch_idx=0, to_gray=False):
    '''Converts a torch tensor image to a numpy array'''
    if isinstance(torch_image, torch.Tensor):
        if len(torch_image.shape) == 2:
            torch_image = torch_image.unsqueeze(0)
        if len(torch_image.shape) == 4 and torch_image.shape[0] == 1:
            torch_image = torch_image[0]
        if len(torch_image.shape) == 4 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx]
        if len(torch_image.shape) == 3 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx].unsqueeze(0)
            
        if torch_image.max() > 1:
            torch_image = torch_image / torch_image.max()
        
        img = (torch_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    else:
        img = torch_image

    if convert_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def resize_short_edge(image, min_size):
    h, w = image.shape[-2:]
    if h > w:
        new_w = min_size
        new_h = int(h * min_size / w)
    else:
        new_h = min_size
        new_w = int(w * min_size / h)
    
    scale = new_h / h

    if image.dim() == 2:
        return torch.nn.functional.interpolate(image.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='nearest').squeeze(0).squeeze(0), scale
    
    if image.dim() == 3:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0), scale
    
    return torch.nn.functional.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False), scale

def pad_square(image):
    # pad bottom or right
    h, w = image.shape[-2:]
    if h > w:
        pad = h - w
        image = torch.nn.functional.pad(image, (0, pad, 0, 0), mode='constant', value=0)
    else:
        pad = w - h
        image = torch.nn.functional.pad(image, (0, 0, 0, pad), mode='constant', value=0)
    return image

def crop_square(image):
    # crop bottom or right
    h, w = image.shape[-2:]
    if h == w:
        return image
    if h > w:
        crop = h - w
        image = image[...,:-crop, :]
    else:
        crop = w - h
        image = image[..., :-crop]
    return image