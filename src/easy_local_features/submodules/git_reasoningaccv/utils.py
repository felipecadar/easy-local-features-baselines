import torch
import torchvision
import cv2
import numpy as np
import h5py
import torch

def load_image(path):
    '''Loads an image from a file path and returns it as a torch tensor
    Output shape: (3, H, W) float32 tensor with values in the range [0, 1]
    '''
    image = torchvision.io.read_image(str(path)).float() / 255
    return image

def to_cv(torch_image, convert_color=True, batch_idx=0, to_gray=False):
    '''Converts a torch tensor image to a numpy array'''
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
    if convert_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def batch_to_device(batch, device):
    '''Moves a batch of tensors to a device'''
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, str):
        return batch
    # is None
    elif batch is None:
        return None

# --- DATA IO ---

def imread_gray(path, augment_fn=None):
    image = cv2.imread(str(path), 1)
    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new

def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask

def resize_long_edge(image, max_size):
    h, w = image.shape[-2:]
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    scale = new_h / h
    
    if image.dim() == 3:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0), scale
    
    return torch.nn.functional.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False), scale

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

def save_in_case_of_error(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            # save args in pickle
            import pickle, datetime, os
            os.makedirs('error_logs', exist_ok=True)
            time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            with open(f"error_logs/error_{func.__name__}_{time}.pkl", 'wb') as f:
                pickle.dump(args, f)
            # save traceback in txt
            import traceback
            with open(f"error_logs/error_{func.__name__}_{time}.txt", 'w') as f:
                f.write(traceback.format_exc())
            raise e
    return wrapper

    
@save_in_case_of_error
@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, do_on_cpu=False):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    if do_on_cpu:
        og_device = kpts0.device
        kpts0 = kpts0.cpu()
        depth0 = depth0.cpu()
        depth1 = depth1.cpu()
        T_0to1 = T_0to1.cpu()
        K0 = K0.cpu()
        K1 = K1.cpu()
    
    kpts0_long = (kpts0 - 0.5).round().long().clip(0, 2000-1)

    depth0[:, 0, :] = 0 ; depth1[:, 0, :] = 0 
    depth0[:, :, 0] = 0 ; depth1[:, :, 0] = 0 

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth > 0

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :] 

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth
    
    kpts1_depth = torch.zeros_like(kpts0_depth)
    
    w_kpts0_long = w_kpts0.round().long()
    for i in range(w_kpts0.shape[0]):
        valid = (w_kpts0_long[i, :, 0] >= 0) & (w_kpts0_long[i, :, 0] < depth1.shape[2]) & (w_kpts0_long[i, :, 1] >= 0) & (w_kpts0_long[i, :, 1] < depth1.shape[1])
        kpts1_depth[i, valid] = depth1[i, w_kpts0_long[i, valid, 1], w_kpts0_long[i, valid, 0]]

    depth_diff = torch.abs(w_kpts0_depth_computed - kpts1_depth)
    depth_diff_mask = depth_diff < 0.2
    
    inside_the_image_mask = (w_kpts0[..., 0] > 0) & (w_kpts0[..., 0] < depth1.shape[2]) & (w_kpts0[..., 1] > 0) & (w_kpts0[..., 1] < depth1.shape[1])
    
    valid_mask = nonzero_mask * depth_diff_mask * inside_the_image_mask
    
    if do_on_cpu:
        valid_mask = valid_mask.to(og_device)
        w_kpts0 = w_kpts0.to(og_device)
        kps0 = kpts0.to(og_device)
        depth0 = depth0.to(og_device)
        depth1 = depth1.to(og_device)
        T_0to1 = T_0to1.to(og_device)
        K0 = K0.to(og_device)
        K1 = K1.to(og_device)

    return valid_mask, w_kpts0

def get_correspondences_0to1(kpts0, data, idx, filter=True):
    valid, w_kpts = warp_kpts(kpts0, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    w_kpts = torch.where(valid[..., None], w_kpts, torch.tensor(-1.0))
    
    if idx >= 0:
        valid = valid[idx]
        w_kpts = w_kpts[idx]
        kpts0 = kpts0[idx]
        if filter:
            kpts0 = kpts0[valid]
            w_kpts = w_kpts[valid]

    joint = torch.cat([kpts0, w_kpts], dim=-1)
    return joint
    
def get_correspondences_1to0(kpts1, data, idx, filter=True):
    valid, w_kpts = warp_kpts(kpts1, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_kpts = torch.where(valid[..., None], w_kpts, torch.tensor(-1.0))
    if idx >= 0:
        valid = valid[idx]
        w_kpts = w_kpts[idx]
        kpts1 = kpts1[idx]
        if filter:
            kpts1 = kpts1[valid]
            w_kpts = w_kpts[valid]
    joint = torch.cat([w_kpts, kpts1], dim=-1)
    return joint

def add_batch_dim(dict_obj):
    for k, v in dict_obj.items():
        if isinstance(v, torch.Tensor):
            dict_obj[k] = v.unsqueeze(0)
    return dict_obj

def get_relative_transform(pose0, pose1):
    R0 = pose0[..., :3, :3] # Bx3x3
    t0 = pose0[..., :3, [3]] # Bx3x1

    R1 = pose1[..., :3, :3] # Bx3x3
    t1 = pose1[..., :3, [3]] # Bx3x1
    
    R_0to1 = R1.transpose(-1, -2) @ R0 # Bx3x3
    t_0to1 = R1.transpose(-1, -2) @ (t0 - t1) # Bx3x1
    T_0to1 = torch.cat([R_0to1, t_0to1], dim=-1) # Bx3x4

    return T_0to1