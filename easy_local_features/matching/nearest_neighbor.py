import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

@torch.no_grad()
def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2) * dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    return matches

def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    inds1 = torch.arange(m1.shape[-1], device=m1.device)
    loop0 = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    loop1 = torch.gather(m0, -1, torch.where(m1 > -1, m1, m1.new_tensor(0)))
    m0_new = torch.where((m0 > -1) & (inds0 == loop0), m0, m0.new_tensor(-1))
    m1_new = torch.where((m1 > -1) & (inds1 == loop1), m1, m1.new_tensor(-1))
    return m0_new, m1_new

def nn_matcher(desc1, desc2, ratio_thresh=None, do_distance_thresh=None, do_mutual_check=True):
    sim = torch.einsum("bnd,bmd->bnm", desc1, desc2)
    matches0 = find_nn(sim, ratio_thresh, do_distance_thresh)
    matches1 = find_nn(sim.transpose(1, 2), ratio_thresh, do_distance_thresh)
    if do_mutual_check:
        matches0, matches1 = mutual_check(matches0, matches1)
    mscores0 = (matches0 > -1).float()
    mscores1 = (matches1 > -1).float()
    return {
        "matches0": matches0,
        "matches1": matches1,
        "matching_scores0": mscores0,
        "matching_scores1": mscores1,
        "similarity": sim,
    }

class NearestNeighborMatcher(torch.nn.Module):
    default_conf = {
        "ratio_thresh": None,
        "distance_thresh": None,
        "mutual_check": True,
        "loss": None,
        "normalize": True,
    }
    required_data_keys = ["descriptors0", "descriptors1"]

    def __init__(self, conf={}):
        super().__init__()
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

    def forward(self, data):
        if self.conf.normalize:
            data["descriptors0"] = F.normalize(data["descriptors0"], p=2, dim=-1)
            data["descriptors1"] = F.normalize(data["descriptors1"], p=2, dim=-1)
        
        sim = torch.einsum("bnd,bmd->bnm", data["descriptors0"], data["descriptors1"])
        matches0 = find_nn(sim, self.conf.ratio_thresh, self.conf.distance_thresh)
        matches1 = find_nn(
            sim.transpose(1, 2), self.conf.ratio_thresh, self.conf.distance_thresh
        )
        if self.conf.mutual_check:
            matches0, matches1 = mutual_check(matches0, matches1)
        b, m, n = sim.shape
        mscores0 = (matches0 > -1).float()
        mscores1 = (matches1 > -1).float()
        
        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "similarity": sim,
        }
        
        
class MultipleNearestNeighborMatcher(torch.nn.Module):
    default_conf = {
        "ratio_thresh": None,
        "distance_thresh": None,
        "mutual_check": True,
        "matching_keys": ["semantic_features", "texture_features"],
        "fusion": "product", # "product" or "sum" or "concat"
        "sum_weights": [1, 1],
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)

    @torch.no_grad()
    def forward(self, data):
        sims = []
        # normalize all
        for key in self.conf.matching_keys:
            data[key + "0"] = F.normalize(data[key + "0"], p=2, dim=-1)
            data[key + "1"] = F.normalize(data[key + "1"], p=2, dim=-1)
        
        if self.conf.fusion in ["product", "sum"]:
            for key in self.conf.matching_keys:
                sim = torch.einsum("bnd,bmd->bnm", data[key + "0"], data[key + "1"])
                sims.append(sim)
            if self.conf.fusion == "sum":
                sims = [s * w for s, w in zip(sims, self.conf.sum_weights)]
                sim = torch.sum(torch.stack(sims), dim=0)
            else:
                sim = torch.prod(torch.stack(sims), dim=0)
        elif self.conf.fusion == "concat":
            d0 = torch.cat([data[key + "0"] for key in self.conf.matching_keys], dim=-1)
            d1 = torch.cat([data[key + "1"] for key in self.conf.matching_keys], dim=-1)
            sim = torch.einsum("bnd,bmd->bnm", d0, d1)
        
        matches0 = find_nn(sim, self.conf.ratio_thresh, self.conf.distance_thresh)
        matches1 = find_nn(
            sim.transpose(1, 2), self.conf.ratio_thresh, self.conf.distance_thresh
        )
        if self.conf.mutual_check:
            matches0, matches1 = mutual_check(matches0, matches1)
        b, m, n = sim.shape
        # mscores0 = (matches0 > -1).float()
        # mscores1 = (matches1 > -1).float()
        
        
        # pull from similiarity matrix
        mscores0 = sim[torch.arange(b)[:, None], torch.arange(m)[None, :], matches0]
        mscores1 = sim[torch.arange(b)[:, None], matches1, torch.arange(n)[None, :]]
        
        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "similarity": sim,
        }