import torch
from easy_local_features.utils import io, ops
from easy_local_features import getExtractor
from pathlib import Path

ROOT = Path("/Users/cadar/Documents/Github/PhD/easy-local-features-baselines/tests")

image0 = ops.crop_square(ops.resize_short_edge(io.fromPath(str(ROOT / "assets/notredame.png")), 320)[0])
image1 = ops.crop_square(ops.resize_short_edge(io.fromPath(str(ROOT / "assets/notredame2.jpeg")), 320)[0])

batched_image0 = torch.cat([image0, image0], dim=0)
batched_image1 = torch.cat([image1, image1], dim=0)

matchers = ["lightglue", "superglue"]

for matcher_name in matchers:
    try:
        print(f"Testing {matcher_name}...")
        matcher = getExtractor(matcher_name)
        
        if matcher_name == "superglue":
            sp = getExtractor("superpoint")
            sp.to("cuda" if torch.cuda.is_available() else "cpu")
            kp0_single, desc0_single = sp.detectAndCompute(image0)
            kp1_single, desc1_single = sp.detectAndCompute(image1)
            print(f"B=1: {matcher_name} feeding in: img0: {image0.shape}, kp0: {kp0_single.shape}, desc0: {desc0_single.shape}")
            out_single = matcher.match(image0, image1, kps0=kp0_single, desc0=desc0_single, kps1=kp1_single, desc1=desc1_single)
            
            kp0_batch, desc0_batch = sp.detectAndCompute(batched_image0)
            kp1_batch, desc1_batch = sp.detectAndCompute(batched_image1)
            print(f"B=2: {matcher_name} feeding in: img0: {batched_image0.shape}, kp0: {kp0_batch.shape}, desc0: {desc0_batch.shape}")
            out_batch = matcher.match(batched_image0, batched_image1, kps0=kp0_batch, desc0=desc0_batch, kps1=kp1_batch, desc1=desc1_batch)
        else:
            print(f"B=1: {matcher_name} feeding in: img0: {image0.shape}")
            out_single = matcher.match(image0, image1)
            print(f"B=2: {matcher_name} feeding in: img0: {batched_image0.shape}")
            out_batch = matcher.match(batched_image0, batched_image1)
        assert isinstance(out_batch, list), f"{matcher_name} batched should return a list"
        assert len(out_batch) == 2, f"{matcher_name} batched should return list of length 2"
        
        for k, v in out_batch[0].items():
            if v is not None and isinstance(v, torch.Tensor):
                print(f"B=2[0]: {k} shape -> {v.shape}")

        assert "mkpts0" in out_batch[0], f"{matcher_name} B=2 missing mkpts0"
        assert len(out_batch[0]["mkpts0"].shape) == 2, f"{matcher_name} B=2 mkpts0 has wrong dims: {out_batch[0]['mkpts0'].shape} (expected 2)"
        print(f"{matcher_name} passed ✅\n", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{matcher_name} failed ❌: {e}\n", flush=True)
