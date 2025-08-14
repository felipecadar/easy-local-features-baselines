import warnings
import torch
from omegaconf import OmegaConf
import torch.nn.functional as F
import os 
from easy_local_features.submodules.git_reasoningaccv.utils import resize_long_edge
from easy_local_features.utils import ops
import time

def sum_dicts(d1, d2):
    set_keys = set(d1.keys()).intersection(set(d2.keys()))
    return {k: d1[k] + d2[k] for k in set_keys}

def load_reasoning_from_checkpoint(checkpoint_path, weights_path=None, load_extractor=False, load_dino=False):   
    config = OmegaConf.load(checkpoint_path + "/model_config.yaml")
    if weights_path is None:
        # get the last checkpoint by time
        all_files = os.listdir(checkpoint_path)
        all_files = [f for f in all_files if f.endswith(".pt")]
        all_files = sorted(all_files, key=lambda x: os.path.getmtime(checkpoint_path + "/" + x))
        weights_path = all_files[-1]
        print(f"Loading weights from {weights_path}")
        weights_path = checkpoint_path + "/" + weights_path
    else:
        weights_path = checkpoint_path + "/" + weights_path
        
    model = ReasoningBase(config)
    weights = torch.load(weights_path, map_location="cpu")
    # remove "dino." and "extractor." and "semantic_reconstruction." from the dict 
    if 'state_dict' in weights:
        weights = weights['state_dict']

    if load_extractor:
        weights = {k: v for k, v in weights.items() if k.startswith("extractor.")}
        return weights

    if load_dino:
        weights = {k: v for k, v in weights.items() if k.startswith("dino.")}
        return weights
        
    weights = {k: v for k, v in weights.items() if not k.startswith("dino.")}
    weights = {k: v for k, v in weights.items() if not k.startswith("extractor.")}
    weights = {k: v for k, v in weights.items() if not k.startswith("semantic_reconstruction.")}
    
    model.load_state_dict(weights)
    model.eval()
    return {
        'model': model,
        'config': config,
        'weights_path': weights_path,
    }

def crop_patches(tensor, coords, size = 7):
    '''
        Crop [size x size] patches around 2D coordinates from a tensor.
    '''
    B, C, H, W = tensor.shape
    coords = coords.long()

    x, y = coords[:, 0], coords[:, 1]
    y = y.view(-1, 1, 1)
    x = x.view(-1, 1, 1)
    halfsize = size // 2
    # Create meshgrid for indexing
    x_offset, y_offset = torch.meshgrid(torch.arange(-halfsize, halfsize+1), torch.arange(-halfsize, halfsize+1), indexing='xy')
    y_offset = y_offset.to(tensor.device)
    x_offset = x_offset.to(tensor.device)

    # Compute indices around each coordinate
    y_indices = (y + y_offset.view(1, size, size)).squeeze(0) + halfsize
    x_indices = (x + x_offset.view(1, size, size)).squeeze(0) + halfsize

    # Handle out-of-boundary indices with padding
    tensor_padded = torch.nn.functional.pad(tensor, (halfsize, halfsize, halfsize, halfsize), mode='constant')

    # Index tensor to get patches
    patches = tensor_padded[:, :, y_indices, x_indices] # [B, C, N, H, W]
    return patches
 
class Transformer(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.n_heads = conf.num_heads
        
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=conf.features_dim,
            num_heads=conf.num_heads,
            batch_first=True,
        )
        
        self.fc = torch.nn.Linear(conf.features_dim, conf.features_dim)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(conf.features_dim * 2, conf.features_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(conf.features_dim * 2, conf.features_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(conf.features_dim * 2, conf.features_dim),
        )
        
    def forward(self, q, k, v):
        attn_output, attn_output_weights = self.attention(q, k, v)
        attn_output = self.fc(attn_output)
        v = v + self.mlp(torch.cat([attn_output, v], dim=-1))
        return v, attn_output_weights
        

def sample_features_dino(keypoints, features, s=14, mode="bicubic"):
    b, c, h, w = features.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    features = torch.nn.functional.grid_sample(
        features, keypoints.view(b, 1, -1, 2), mode=mode, align_corners=False
    )
    features = torch.nn.functional.normalize(
        features.reshape(b, c, -1), p=2, dim=1
    )
    
    features = features.permute(0, 2, 1)
    return features
        
class ReasoningBase(torch.nn.Module):
    default_conf = {
        
        "dino":{
            "weights": "dinov2_vits14", # dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
            "allow_resize": True,
        },
        
        "extractor":{
            "model_name": "aliked-n16",
            "max_num_keypoints": 2048,
            "detection_threshold": 0.0,
            "force_num_keypoints": True,
            "pretrained": True,
            "nms_radius": 2,
        },
        
        "reasoning":{
            "extractor_features_dim": None,
            "semantic_features_dim": None,
            "features_dim": 256,
            "n_attention_layers": 14,
            "num_heads": 4, 
            "attention": "full", # full, linear
        },
        "semantic_interpolation_mode": "bicubic",
        "activate_timers": False,
        "attention_progression": "alternating", # alternating, semantic, visual
        "deep_supervision": True,
        "semantic_conditioning": False,
        "fix_dino_size": -1,
    }
    
    required_data_keys = ["image"]

    def __init__(self, conf={}):
        super().__init__()
        self.conf = conf = OmegaConf.merge(OmegaConf.create(self.default_conf), conf)
        self.dino_reprojection = torch.nn.Linear(conf.reasoning.semantic_features_dim, conf.reasoning.features_dim)
        self.extractor_reprojection = torch.nn.Linear(conf.reasoning.extractor_features_dim, conf.reasoning.features_dim)
        
        self.texture_attentions = torch.nn.ModuleList([
            Transformer(conf.reasoning)
            for _ in range(conf.reasoning.n_attention_layers)
        ])

        self.semantic_attentions = torch.nn.ModuleList([
            Transformer(conf.reasoning)
            for _ in range(conf.reasoning.n_attention_layers)
        ])
        
        self.joint_features = torch.nn.Linear(conf.reasoning.features_dim * 2, conf.reasoning.features_dim)


    @classmethod
    def from_experiment(self, checkpoint_path, weights_path=None):
        return load_reasoning_from_checkpoint(checkpoint_path, weights_path)
        
    def forward(self, data, keypoints=None):
        '''data = {
            'image': torch.Tensor,
            'keypoints': torch.Tensor,
            'descriptors': torch.Tensor,
            'semantic_features': torch.Tensor,
        }
        '''
            
        dino_features = data['semantic_features']
        extractor_descriptors = data['descriptors']
        keypoints = data['keypoints']
        
        # reproject features to common space
        proj_dino_features = self.dino_reprojection(dino_features)
        proj_extractor_descriptors = self.extractor_reprojection(extractor_descriptors)
        
        # all_reasoning_descriptors, all_layers_attn_output_weights = self.reasoning(proj_dino_features, proj_extractor_descriptors)
        all_reasoning_descriptors, all_layers_attn_output_weights = self.reasoning(proj_extractor_descriptors, proj_dino_features, self.texture_attentions, 'alternating')
        all_semantic_descriptors, all_semantic_attn_output_weights  = self.reasoning(proj_dino_features, proj_dino_features, self.semantic_attentions, 'semantic')

        return_data =  {
            # final features
            "keypoints": keypoints,
            "reasoning_features": all_reasoning_descriptors[-1],
            "semantic_features": all_semantic_descriptors[-1],
            # intermediate features
            "all_reasoning_features": all_reasoning_descriptors,
            "all_semantic_features": all_semantic_descriptors,
            # base features                
            "dino_features": dino_features,
            "extractor_features": extractor_descriptors,
        }
            
        return return_data
    
    def reasoning(self, start_feature, query_feature, attention_layers, attention_progression):
        reasoning_descriptors = start_feature.clone()
        all_attention_weights = []
        all_reasoning_descriptors = []
        if attention_progression == "alternating":
            for layer_idx in range(len(attention_layers)):
                if layer_idx % 2 == 0:
                    reasoning_descriptors, attn_output_weights = attention_layers[layer_idx](
                        q=query_feature, 
                        k=reasoning_descriptors, 
                        v=reasoning_descriptors,
                    )

                else:
                    reasoning_descriptors, attn_output_weights = attention_layers[layer_idx](
                        q=start_feature, 
                        k=reasoning_descriptors, 
                        v=reasoning_descriptors,
                    )
                
                all_attention_weights.append(attn_output_weights.clone())
                all_reasoning_descriptors.append(reasoning_descriptors.clone())
        elif attention_progression == "semantic":
            for layer_idx in range(len(attention_layers)):
                reasoning_descriptors, attn_output_weights = attention_layers[layer_idx](
                    q=query_feature,
                    k=reasoning_descriptors, 
                    v=reasoning_descriptors,
                )
                
                all_attention_weights.append(attn_output_weights.clone())
                all_reasoning_descriptors.append(reasoning_descriptors.clone())
        else:                
            raise ValueError(f"Unknown attention progression {self.conf.attention_progression}")

        all_reasoning_descriptors = torch.stack(all_reasoning_descriptors, dim=0)
        return all_reasoning_descriptors, all_attention_weights

    def loss(self, data, pred, args):
        loss = 0
        log_dict = {}

        softmax_loss = self.dual_softmax_loss(pred, data)
        log_dict['0_train/loss_softmax'] =  softmax_loss
        loss += softmax_loss['matchable'] * 0.25
        loss += softmax_loss['unmatchable'] * 0.1
            
        return loss, log_dict

    def match(self, data, pred=None):
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
        
        
        if pred is None:
            with torch.inference_mode():        
                pred0 = self({"image": data['image0']})
                pred1 = self({"image": data['image1']})

            pred = {}
            pred.update({k+"0": v for k, v in pred0.items()})
            pred.update({k+"1": v for k, v in pred1.items()})
            
        f0 = pred['reasoning_features0']
        f1 = pred['reasoning_features1']
        s0 = pred['semantic_features0']
        s1 = pred['semantic_features1']
        
        # normalize the features
        f0 = F.normalize(f0, dim=-1)
        f1 = F.normalize(f1, dim=-1)
        s0 = F.normalize(s0, dim=-1)
        s1 = F.normalize(s1, dim=-1)
        
        reasoning_sim = torch.einsum("bnd,bmd->bnm", f0, f1)
        semantic_sim = torch.einsum("bnd,bmd->bnm", s0, s1)
        if self.conf.semantic_conditioning:
            sim = reasoning_sim * semantic_sim
        else:
            sim = reasoning_sim
        
        matches0 = find_nn(sim, None, None)
        matches1 = find_nn(sim.transpose(1, 2), None, None)
        matches0, matches1 = mutual_check(matches0, matches1)
        
        return_dict = {
            'matches0': [],
            'matches1': [],
        }
        
        b_size = sim.shape[0]
        for b in range(b_size):
            keypoints0 = pred['keypoints0'][b]
            keypoints1 = pred['keypoints1'][b]      
            m0 = matches0[b]  

            valid = m0 > -1
            
            if valid.sum() == 0:
                print(f"No matches found for image {b}")
            
            mkpts0 = keypoints0[valid]
            mkpts1 = keypoints1[m0[valid]]
            
            return_dict['matches0'].append(mkpts0)
            return_dict['matches1'].append(mkpts1)
            
        return return_dict

class Reasoning(torch.nn.Module):
    def __init__(self, reasoning_model, dev=None):
        super().__init__()
        self.conf = conf = reasoning_model.conf
        if dev is not None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.dev = dev
        print(f'Looking for "{conf.extractor.model_name}"')

        with warnings.catch_warnings():
            from easy_local_features.submodules.git_reasoningaccv.dinov2 import DinoV2
            self.dino = DinoV2(conf.dino).eval()
        if 'aliked-n' in conf.extractor.model_name:
            from easy_local_features.submodules.git_reasoningaccv.aliked import ALIKED
            assert conf.extractor.model_name in ALIKED.cfgs, f"Model {conf.extractor.model_name} not found in ALIKED.cfgs"
            self.extractor = ALIKED(conf.extractor).eval()     
        elif 'alike-n' in conf.extractor.model_name:
            from easy_local_features.feature.baseline_alike import ALIKE_baseline
            self.extractor = ALIKE_baseline(
                {
                    "model_name":'alike-n',
                    "top_k":self.conf.extractor.max_num_keypoints,
                    "n_limit":self.conf.extractor.max_num_keypoints,
                }
            )
            self.extractor.to(self.dev)
        elif 'dedode' in conf.extractor.model_name:
            from kornia.feature import DeDoDe
            assert conf.extractor.model_name in ["dedode-G", "dedode-B"], f"Model {conf.extractor.model_name} not found in DeDoDe"
            descriptor_type = conf.extractor.model_name.split("-")[1]
            self.extractor = DeDoDe.from_pretrained(detector_weights="L-upright", descriptor_weights=f"{descriptor_type}-upright")
        elif 'superpoint' in conf.extractor.model_name:
            from easy_local_features.submodules.git_reasoningaccv.superpoint import SuperPoint
            self.extractor = SuperPoint(conf.extractor).eval()
                
        elif 'xfeat' in conf.extractor.model_name:
            self.extractor = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = conf.extractor.max_num_keypoints, detection_threshold=0.)
       
        elif 'relf' in conf.extractor.model_name:
            from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
            from easy_local_features.feature.baseline_relf import RELF_baseline
            self.detector = SuperPoint_baseline({
                "top_k":conf.extractor.max_num_keypoints,
                "legacy_sampling": False,
            })
            self.extractor = RELF_baseline({
                'model': 're_resnet'
            })
            self.extractor.to(self.dev)
            self.detector.to(self.dev)
        else:
            raise ValueError(f"Model {conf.extractor.model_name} not found")
        
        self.reasoning_model = reasoning_model
        
    def forward(self, data):
        start_extract = time.time()
        if 'xfeat' in self.conf.extractor.model_name:
            response = self.extractor.detectAndCompute(data['image'], top_k = self.conf.extractor.max_num_keypoints)
            extractor_pred = {
                'keypoints': response[0]['keypoints'].unsqueeze(0),
                'descriptors': response[0]['descriptors'].unsqueeze(0),
            }
        elif 'dedode' in self.conf.extractor.model_name:
            keypoints, scores, features = self.extractor(data['image'], n=self.conf.extractor.max_num_keypoints, apply_imagenet_normalization=True)
            extractor_pred = {"keypoints": keypoints, "descriptors": features}                        
        elif self.conf.extractor.model_name in ['superpoint', 'aliked-n']:
            response = self.extractor(data)
            extractor_pred = response
        elif 'alike-n' in self.conf.extractor.model_name:
            keypoints, descriptors = self.extractor.detectAndCompute(data['image'])
            extractor_pred = {
                'keypoints': keypoints.unsqueeze(0),
                'descriptors': descriptors.unsqueeze(0),
            }
        elif 'relf' in self.conf.extractor.model_name:
            images = data['image'].to(self.dev)
            keypoints = self.detector.detect(images)
            keypoints, descriptors = self.extractor.compute(images, keypoints)
            extractor_pred = {
                'keypoints': keypoints,
                'descriptors': descriptors,
            }
        else: 
            raise ValueError(f"Model {self.conf.extractor.model_name} not found")
        end_extract = time.time()
        
        start_dino = time.time()
        resized, scale = resize_long_edge(data['image'], 896)
        if self.conf.fix_dino_size > 0:
            resized, scale = resize_long_edge(data['image'], self.conf.fix_dino_size)

        semantic_features = self.dino({'image': resized})
        dino_features = sample_features_dino(extractor_pred['keypoints']*scale, semantic_features['features'], mode=self.conf.semantic_interpolation_mode)
        end_dino = time.time()
        
        start_reasoning = time.time()        
        reasoning_features = self.reasoning_model({
            'image': data['image'],
            'keypoints': extractor_pred['keypoints'],
            'descriptors': extractor_pred['descriptors'],
            'semantic_features': dino_features,
        }) 
        end_reasoning = time.time()
        
        time_dict = {
            'extract': end_extract - start_extract,
            'dino': end_dino - start_dino,
            'reasoning': end_reasoning - start_reasoning,
        }
        reasoning_features['time'] = time_dict
        reasoning_features['dino_map'] = semantic_features['features']
        
        return reasoning_features
    
    @torch.inference_mode()
    def detectAndCompute(self, image):
        image = ops.prepareImage(image).to(self.dev)
        response = self.forward({'image': image})
        
        keypoints = response['keypoints'] # [B, N, 2]
        descritexture_features = response['reasoning_features'] # [B, N, D]
        semantic_features = response['semantic_features'] # [B, N, D]
        descriptors = torch.stack([descritexture_features, semantic_features], dim=-1) # [B, N, D, 2]
        
        return keypoints, descriptors
        
    @torch.inference_mode()
    def match(self, data, pred={}):
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

        if pred == {}:
            pred0 = self({
                "image": data['image0']
            })
            pred1 = self({
                "image": data['image1']
            })

            pred = {}
            pred.update({k+"0": v for k, v in pred0.items()})
            pred.update({k+"1": v for k, v in pred1.items()})
            
        f0 = pred['reasoning_features0']
        f1 = pred['reasoning_features1']
        s0 = pred['semantic_features0']
        s1 = pred['semantic_features1']
        
        # # normalize the features
        f0 = F.normalize(f0, dim=-1)
        f1 = F.normalize(f1, dim=-1)
        s0 = F.normalize(s0, dim=-1)
        s1 = F.normalize(s1, dim=-1)
        
        reasoning_sim = torch.einsum("bnd,bmd->bnm", f0, f1)
        semantic_sim = torch.einsum("bnd,bmd->bnm", s0, s1)

        if self.conf.semantic_conditioning:
            sim = reasoning_sim * semantic_sim
        else:
            sim = reasoning_sim
        
        matches0 = find_nn(sim, None, None)
        matches1 = find_nn(sim.transpose(1, 2), None, None)
        matches0, matches1 = mutual_check(matches0, matches1)
        
        return_dict = {
            'matches0': [],
            'matches1': [],
            'pred': pred,
        }
        
        b_size = sim.shape[0]
        for b in range(b_size):
            keypoints0 = pred['keypoints0'][b]
            keypoints1 = pred['keypoints1'][b]      
            m0 = matches0[b]  
                
            valid = m0 > -1
            
            if valid.sum() == 0:
                print(f"No matches found for image {b}")
            
            mkpts0 = keypoints0[valid]
            mkpts1 = keypoints1[m0[valid]]
            
            return_dict['matches0'].append(mkpts0)
            return_dict['matches1'].append(mkpts1)
            
        return return_dict
    