import torch
from e2cnn.nn import *
import torch.nn.functional as F

def normalize_coordinates(coords, h, w):
    h=h-1
    w=w-1
    coords=coords.clone().detach()
    coords[:, :, 0]-= w / 2
    coords[:, :, 1]-= h / 2
    coords[:, :, 0]/= w / 2
    coords[:, :, 1]/= h / 2
    return coords

def interpolate_feats(img,pts,feats):
    """
    img : B, 3, H, W
    pts : B, N, 2
    feats : B, C, H', W'
    """
    # compute location on the feature map (due to pooling)
    _, _, h, w = feats.shape
    pool_num = img.shape[-1] / feats.shape[-1]
    pts_warp= pts / pool_num
    pts_norm=normalize_coordinates(pts_warp,h,w)
    pts_norm=torch.unsqueeze(pts_norm, 1)  # b,1,n,2

    # interpolation
    pfeats=F.grid_sample(feats, pts_norm, 'bilinear', align_corners=True)[:, :, 0, :]  # b,f,n
    pfeats=pfeats.permute(0,2,1) # b,n,f

    return pfeats

class RELF_model(torch.nn.Module):
    def __init__(self, args):
        super(RELF_model, self).__init__()
        if args.model == 're_resnet':
            from .reresnet_model.model import ReResNet_wrapper
            self.model = ReResNet_wrapper(args.num_group, depth=18, channels=64)
        else:
            from .rewrn_model.model import E2CNN_wrapper         
            args.num_group = 8   
            self.model = E2CNN_wrapper(args)

        self.in_type = self.model.get_in_type()
        self.max_size_ap = 1024  ## 'max size of adaptive image pyramid.'
        
        self.mode = 'test'

    def forward(self, image, kpts):
        """ multi-scale inference of image pyramid 
            image: [B, 3, H, W], kpts: [B, K, 2]
        """
        images, scales = self.get_scale_space_images(image)

        pfeats_ms = []        
        for sc, image_resize in zip(scales, images):
            kpts_resize = sc * kpts 
            pfeats = self.forward_single_scale(image_resize, kpts_resize)
            pfeats_ms.append(pfeats)
            
        ## We collapse the scale-equivariant features using simple max operation. (scale pooling)
        pfeats_ms = torch.max(torch.stack(pfeats_ms), dim=0)[0]

        return pfeats_ms

    def forward_single_scale(self, image, kpts):
        image = GeometricTensor(image, self.in_type)
        feats = self.model(image)
        pfeats = interpolate_feats(image, kpts, feats)
        return pfeats
    
    def get_scale_space_images(self, image):
        if self.mode == 'train':
            images, scales = ([image], [1.0])
        else:
            images, scales = adaptive_image_pyramid(image, max_size=self.max_size_ap)
        # images, scales = ([image], [1.0])

        return images, scales

    def get_in_type(self):
        return self.in_type

    def switch_mode(self, mode='test'):
        self.mode = mode

## adaptive image pyramid
def adaptive_image_pyramid(img, min_scale=0.0, max_scale=1, min_size=256, max_size=1536, scale_f=2**0.25, verbose=False):
    
    B, _, H, W = img.shape

    ## upsample the input to bigger size.
    s = 1.0
    if max(H, W) < max_size:
        s = max_size / max(H, W)
        max_scale = s
        nh, nw = round(H*s), round(W*s)
        # if verbose:  print(f"extracting at highest scale x{s:.02f} = {nw:4d}x{nh:3d}")
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)
        
    ## downsample the scale pyramid
    output = []
    scales = []
    while s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]

            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            output.append(img)
            scales.append(s)
        # print(f"passing the loop x{s:.02f} = {nw:4d}x{nh:3d}")        

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)
    
    return output, scales



if __name__ == "__main__":
    import torch
    model = RELF_model()

    image = torch.randn(2,3,400,400)
    kpts = torch.abs(torch.randn(2,300,2))
    kpts = kpts / torch.max(kpts) * 399
    print(kpts.min(), kpts.max())
    
    ret = model(image, kpts)
    print(ret.shape)
    
