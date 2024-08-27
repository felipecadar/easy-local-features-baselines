import torch
import torchvision
import numpy as np
import cv2
from torch.nn import functional as F

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
        image = image.mean(-3, keepdim=True)
        
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


def to_cv(torch_image, convert_color=False, batch_idx=0, to_gray=False):
    '''Converts a torch tensor image to a numpy array'''
    
    if isinstance(torch_image, torch.Tensor):
        if len(torch_image.shape) == 2:
            torch_image = torch_image.unsqueeze(0)
        elif len(torch_image.shape) == 4:
            torch_image = torch_image[batch_idx]
            
        if torch_image.max() > 1:
            torch_image = torch_image / torch_image.max()
        
        img = (torch_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    else:
        img = torch_image

    if convert_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    if to_gray:
        if len(img.shape) == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:
            img = img[None]
        
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

def crop_patches(image, keypoints, patch_size = 32, mode='nearest'):
    B, C, H, W = image.shape
    N = keypoints.shape[1]

    # Ensure the keypoints are in the correct range
    x_coords = keypoints[:, :, 0]
    y_coords = keypoints[:, :, 1]
    
    # Calculate the left and top edges of the patches
    left = x_coords - patch_size / 2
    top = y_coords - patch_size / 2

    # Normalize to [-1, 1] for grid_sample
    left_norm = 2 * left / (W - 1) - 1
    top_norm = 2 * top / (H - 1) - 1

    # Create the sampling grid
    # grid_x = torch.linspace(-1, 1, patch_size).to(image.device)
    # grid_y = torch.linspace(-1, 1, patch_size).to(image.device)
    # this is not from -1 to 1, bt from the left to the right in normalized coordinates
    patch_size_normalized_x = patch_size / W
    patch_size_normalized_y = patch_size / H
    grid_x = torch.linspace(-patch_size_normalized_x, patch_size_normalized_x, patch_size).to(image.device)
    grid_y = torch.linspace(-patch_size_normalized_y, patch_size_normalized_y, patch_size).to(image.device)
    
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)  # 1xpatch_sizexpatch_sizex2

    grid = grid.repeat(B, N, 1, 1, 1)  # BxNxpatch_sizexpatch_sizex2

    # Adjust the grid according to keypoints
    grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] + left_norm.unsqueeze(2).unsqueeze(3))
    grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] + top_norm.unsqueeze(2).unsqueeze(3))
    
    # the patch is anchored at the bottom right corner of the keypoint
    # let's move it to the center of the patch
    grid[:, :, :, :, 0] += patch_size_normalized_x
    grid[:, :, :, :, 1] += patch_size_normalized_y
    
    # Reshape for grid_sample
    grid = grid.view(B * N, patch_size, patch_size, 2)

    # Repeat images and apply grid_sample
    image_repeated = image.unsqueeze(1).repeat(1, N, 1, 1, 1).view(B * N, C, H, W)
    patches = F.grid_sample(image_repeated, grid, mode=mode, align_corners=True)

    patches = patches.view(B, N, C, patch_size, patch_size)
    # flip the patches to have the same orientation as the other methods
    # transpose the last two dimensions
    patches = patches.transpose(-1, -2)
    
    return patches


def sort_keypoints(mkpts0, mkpts1=None):
    # get a lexical index of the keypoints based on the x and y coordinates
    if len(mkpts0.shape) == 3:
        # batched keypoints
        B, N, _ = mkpts0.shape
        for b in range(B):
            idxs = np.lexsort((mkpts0[b, :, 0], mkpts0[b, :, 1]))
            mkpts0[b] = mkpts0[b, idxs]
            if mkpts1 is not None:
                mkpts1[b] = mkpts1[b, idxs]

        if mkpts1 is not None:
            return mkpts0, mkpts1
        return mkpts0
    else:
        idxs = np.lexsort((mkpts0[:, 0], mkpts0[:, 1]))
        mkpts0 = mkpts0[idxs]
        if mkpts1 is not None:
            mkpts1 = mkpts1[idxs]
            return mkpts0, mkpts1

        return mkpts0    
    
def to_homogeneous(kpts):
    if isinstance(kpts, np.ndarray):
        if len(kpts.shape) == 3:
            B, N, _ = kpts.shape
            return np.concatenate((kpts, np.ones((B, N, 1))), axis=-1)
        else:
            return np.concatenate((kpts, np.ones((kpts.shape[0], 1))), axis=-1)
    elif isinstance(kpts, torch.Tensor):
        if len(kpts.shape) == 3:
            B, N, _ = kpts.shape
            return torch.cat((kpts, torch.ones(B, N, 1, device=kpts.device)), dim=-1)
        else:
            return torch.cat((kpts, torch.ones(kpts.shape[0], 1, device=kpts.device)), dim=-1)
        
def to_grid(kpts, H, W):
    # kpts => in pixel coordinates
    # H, W => image dimensions
    # returns the grid coordinates (-1, 1) for grid_sample
    
    if isinstance(kpts, np.ndarray):
        if len(kpts.shape) == 3:
            B, N, _ = kpts.shape
            return 2 * kpts / np.array([W, H]) - 1
        else:
            return 2 * kpts / np.array([W, H]) - 1
        
    elif isinstance(kpts, torch.Tensor):
        if len(kpts.shape) == 3:
            B, N, _ = kpts.shape
            return 2 * kpts / torch.tensor([W, H], device=kpts.device) - 1
        else:
            return 2 * kpts / torch.tensor([W, H], device=kpts.device) - 1
        
def from_homogeneous(kpts):
    if isinstance(kpts, np.ndarray):
        if len(kpts.shape) == 3:
            B, N, _ = kpts.shape
            return kpts[:, :, :2] / kpts[:, :, 2:]
        else:
            return kpts[:, :2] / kpts[:, 2:]
    elif isinstance(kpts, torch.Tensor):
        if len(kpts.shape) == 3:
            B, N, _ = kpts.shape
            return kpts[:, :, :2] / kpts[:, :, 2:]
        else:
            return kpts[:, :2] / kpts[:, 2:]