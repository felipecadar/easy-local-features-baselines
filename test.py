from easy_local_features.utils import ops, io, vis
import torch

from einops import rearrange

import torch
import torch.nn.functional as F

def crop_patches_gpt(image, keypoints, patch_size, mode='nearest'):
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


img = io.fromPath("assets/v_vitro/1.ppm")

keypoints = torch.tensor([
    [100, 200],
    [200, 300],
    [300, 400]
]).unsqueeze(0)

# patches1 = crop_patches_gemini(img, keypoints, 32)
# print(patches1.shape)

img = vis.draw_keypoints(img, keypoints, color='r')
vis.plot_pair(img, img, figsize=(8,4), gray=False)

patches3 = crop_patches_gpt(img, keypoints, 32, 'nearest')
print(patches3.shape)

patches3 = rearrange(patches3, 'B N C H W -> B C H (N W)')

vis.plot_pair(patches3, patches3, vertical=True, figsize=(8, 4), gray=False)
vis.show()