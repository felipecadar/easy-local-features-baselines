try:
    import math
    import torch
    from torch import Tensor
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise Exception('CUDA not available')

    from pathlib import Path
    file_path = Path(__file__)
    for f in file_path.parent.glob('get_patches*.so'):
        torch.ops.load_library(f)
    
    class get_patches(torch.autograd.Function):

        @staticmethod
        def forward(ctx, fmap, points, kernel_size):
            fmap = fmap.contiguous()
            points = points.contiguous()
            patches = torch.ops.custom_ops.get_patches_forward(fmap, points, kernel_size)

            ctx.save_for_backward(points, torch.tensor(fmap.shape))

            return patches

        @staticmethod
        def backward(ctx, d_patches):
            points, shape = ctx.saved_tensors
            H = shape[1].cpu().item()
            W = shape[2].cpu().item()
            d_fmap = torch.ops.custom_ops.get_patches_backward(d_patches.contiguous(), points, H, W)
            return d_fmap, None, None
    
    def get_patches_torch(fmap: Tensor, points: Tensor, K: int):
        # fmap: CxHxW
        # points: Nx2
        # pad the fmap
        N = points.shape[0]
        C = fmap.shape[0]
        radius = (K - 1.0) / 2.0
        pad_left_top = math.floor(radius)
        pad_right_bottom = math.ceil(radius)
        # K=2, radius=0.5, pad_left_top=0, pad_right_bottom=1
        # K=3, radius=1.0, pad_left_top=1, pad_right_bottom=1
        # K=4, radius=1.5, pad_left_top=1, pad_right_bottom=2
        # K=5, radius=2.0, pad_left_top=2, pad_right_bottom=2        
        # Cx(H+K-1)x(W+K-1)
        map_pad = F.pad(fmap.unsqueeze(0),(pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom)).squeeze(0)
        patches_left = (points[:,1] - pad_left_top).long()
        patches_top = (points[:,0] - pad_left_top).long()
        patches_right = patches_left + K
        patches_bottom = patches_top + K

        patches = map_pad[:, patches_top:patches_bottom, patches_left:patches_right]

        return patches

except:
    def get_patches_forward_cpu(map, points, kernel_size):
        N = points.size(0)
        C = map.size(0)
        radius = (kernel_size - 1.0) / 2.0
        pad_left_top = math.floor(radius)
        pad_right_bottom = math.ceil(radius)

        # pad map
        map_pad = F.pad(map.unsqueeze(0), (pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom), mode='constant').squeeze(0)  # Cx(H+2*radius)x(W+2*radius)

        # get patches
        patches = torch.zeros(N, C, kernel_size, kernel_size, dtype=map.dtype, device=map.device)
        for in_idx in range(N):
            w_start = points[in_idx][0]
            h_start = points[in_idx][1]

            # copy data using slicing and indexing
            patches[in_idx] = map_pad[:, h_start:h_start+kernel_size, w_start:w_start+kernel_size]

        return patches

    def get_patches_backward_cpu(d_patches, points, H, W):
        N = d_patches.size(0)
        C = d_patches.size(1)
        kernel_size = d_patches.size(2)
        radius = (kernel_size - 1.0) / 2.0
        pad_left_top = math.floor(radius)
        pad_right_bottom = math.ceil(radius)

        d_map_pad = torch.zeros(C, H + int(2 * radius), W + int(2 * radius), dtype=d_patches.dtype, device=d_patches.device)

        for in_idx in range(N):
            w_start = points[in_idx][0].item()  # Convert to scalar using item()
            h_start = points[in_idx][1].item()  # Convert to scalar using item()

            # copy data using slicing and indexing
            d_map_pad[:, h_start:h_start+kernel_size, w_start:w_start+kernel_size] = d_patches[in_idx]

        d_map = d_map_pad[:, pad_left_top:-pad_right_bottom, pad_left_top:-pad_right_bottom]

        return d_map


    class get_patches(torch.autograd.Function):

        @staticmethod
        def forward(ctx, fmap, points, kernel_size):
            fmap = fmap.contiguous()
            points = points.contiguous()
            patches = get_patches_forward_cpu(fmap, points, kernel_size)

            ctx.save_for_backward(points, torch.tensor(fmap.shape))

            return patches

        @staticmethod
        def backward(ctx, d_patches):
            points, shape = ctx.saved_tensors
            H = shape[1].cpu().item()
            W = shape[2].cpu().item()
            d_fmap = get_patches_backward_cpu(d_patches.contiguous(), points, H, W)
            return d_fmap, None, None
    
    def get_patches_torch(fmap: Tensor, points: Tensor, K: int):
        # fmap: CxHxW
        # points: Nx2
        # pad the fmap
        N = points.shape[0]
        C = fmap.shape[0]
        radius = (K - 1.0) / 2.0
        pad_left_top = math.floor(radius)
        pad_right_bottom = math.ceil(radius)
        # K=2, radius=0.5, pad_left_top=0, pad_right_bottom=1
        # K=3, radius=1.0, pad_left_top=1, pad_right_bottom=1
        # K=4, radius=1.5, pad_left_top=1, pad_right_bottom=2
        # K=5, radius=2.0, pad_left_top=2, pad_right_bottom=2        
        # Cx(H+K-1)x(W+K-1)
        map_pad = F.pad(fmap.unsqueeze(0),(pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom)).squeeze(0)
        patches_left = (points[:,1] - pad_left_top).long()
        patches_top = (points[:,0] - pad_left_top).long()
        patches_right = patches_left + K
        patches_bottom = patches_top + K

        patches = map_pad[:, patches_top:patches_bottom, patches_left:patches_right]

        return patches