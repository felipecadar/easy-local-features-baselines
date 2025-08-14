import numpy as np
import os
import cv2
from tqdm import tqdm
import json
import multiprocessing as mp
import torch

from easy_local_features.utils import vis, io
import argparse

try:
    import poselib
except:
    print("Please install poselib library with `pip install poselib`")

def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }

# 
def plot_matches_parallel(args_plot):
    pair_idx, pair, mkpts0, mkpts1, out_folder = args_plot
    out_path = os.path.join(out_folder, f'{pair_idx}.png')
    image0 = cv2.imread(pair['image0'])
    image1 = cv2.imread(pair['image1'])
    if isinstance(mkpts0, torch.Tensor):
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()
        
    if len(mkpts0) == 0 or len(mkpts1) == 0:
        fig, ax = vis.plot_pair(image0, image1)
        vis.save(out_path)
        vis.plt.close('all')
        return True

    (pose,details) = poselib.estimate_relative_pose(
            mkpts0.tolist(), 
            mkpts1.tolist(),
            intrinsics_to_camera(pair['K0']),
            intrinsics_to_camera(pair['K1']),
            ransac_opt={
                'max_iterations': 10000, # default 100000
                'success_prob': 0.99999, # default 0.99999
                'max_epipolar_error': 6.0,
            })
    inliers = np.array(details['inliers'])

    fig, ax = vis.plot_pair(image0, image1)
    vis.plot_matches(mkpts0[~inliers], mkpts1[~inliers], color='r')
    vis.plot_matches(mkpts0[inliers], mkpts1[inliers], color='g')
    vis.add_text(f'Inliers: {inliers.sum()} / {len(inliers)}')
    vis.save(out_path)

    vis.plt.close('all')
    return True

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))

def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999, pose_estimator='poselib'):
    if len(kpts0) < 5:
        return None
    if pose_estimator == 'poselib':
        import poselib
        (pose,details) = poselib.estimate_relative_pose(
            kpts0.tolist(), 
            kpts1.tolist(),
            intrinsics_to_camera(K0),
            intrinsics_to_camera(K1),
            ransac_opt={
                'max_iterations': 10000, # default 100000
                # 'min_iterations': 1000,
                # 'dyn_num_trials_mult': 3.0,
                'success_prob': conf, # default 0.99999
                # 'max_reproj_error': 12.0,
                'max_epipolar_error': thresh, # default 1.0
                # 'seed': 0,
                # 'progressive_sampling': False,
                # 'max_prosac_iterations': 100000
            },
            bundle_opt={  # all defaults
                # 'max_iterations': 100,
                # 'loss_scale': 1.0,
                # 'loss_type': 'CAUCHY',
                # 'gradient_tol': 1e-10,
                # 'step_tol': 1e-08,
                # 'initial_lambda': 0.001,
                # 'min_lambda': 1e-10,
                # 'max_lambda': 10000000000.0,
                # 'verbose': False
                },
            )
        ret = (pose.R, pose.t, details['inliers'])

    elif pose_estimator == 'opencv':
        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        norm_thresh = thresh / f_mean

        kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
        kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

        E, mask = cv2.findEssentialMat(
            kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
            method=cv2.RANSAC)

        assert E is not None

        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], mask.ravel() > 0)
    else:
        raise NotImplementedError

    return ret

def estimate_pose_parallel(args):
    return estimate_pose(*args)

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs

def pose_accuracy(errors, thresholds):
    return [np.mean(errors < t) * 100 for t in thresholds]

def resize_long_edge(image, max_size):
    h, w = image.shape[-2:]
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    scale = new_h / h
    
    return torch.nn.functional.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False), scale

class PoseEval:
    default_config = {
        'data_path': "",
        'pairs_path': "",
        'pose_estimator': 'poselib', # poselib, opencv
        'cache_images': False,
        'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        'pose_thresholds': [5, 10, 20],
        'max_pairs': -1,
        'output': './output/pose_eval/',
        'n_workers': 8,
        'resize': None,
        'detector_only': False,
    }

    def __init__(self, config={}) -> None:
        self.config = {**self.default_config, **config}
        for k,v in self.config.items():
            print(k, v)
        
        self.pairs = self.read_gt()
        print(f'Loaded {len(self.pairs)} pairs')

        os.makedirs(self.config['output'], exist_ok=True)

        if self.config['n_workers'] == -1:
            self.config['n_workers'] = mp.cpu_count()

        self.image_cache = {}
        if self.config['cache_images']:
            self.load_images()

    def load_images(self):
        for pair in tqdm(self.pairs, desc='Caching images'):
            if pair['image0'] not in self.image_cache:
                self.image_cache[pair['image0']] = io.fromPath(pair['image0'])
            if pair['image1'] not in self.image_cache:
                self.image_cache[pair['image1']] = io.fromPath(pair['image1'])

    def read_image(self, path):
        if self.config['cache_images']:
            return self.image_cache[path]
        else:
            return io.fromPath(path)

    def read_gt(self):
        pairs = []
        with open(self.config['pairs_path'], 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').replace('  ', '')
                all_info = line.split(' ')
                image0 = all_info[0]
                image1 = all_info[1]

                # 2 3x3 matrices and one 4x4 matrix
                if len(all_info) == 38:
                    # scannet format
                    K0 = np.array(all_info[4:13]).astype(float).reshape(3, 3)
                    K1 = np.array(all_info[13:22]).astype(float).reshape(3, 3)
                    T_0to1 = np.array(all_info[22:38]).astype(float).reshape(4, 4)
                    
                    image0 = os.path.join(self.config['data_path'], image0)
                    image1 = os.path.join(self.config['data_path'], image1)
                    depth0 = image0.replace('color', 'depth').replace('jpg', 'png')
                    depth1 = image1.replace('color', 'depth').replace('jpg', 'png')

                    pairs.append({
                        'image0': image0,
                        'image1': image1,
                        'depth0': depth0,
                        'depth1': depth1,
                        'K0': K0,
                        'K1': K1,
                        'T_0to1': T_0to1,
                    })

                # 2 3x3 matrices and one 4x4 matrix
                if len(all_info) == 36:
                    # scannet format
                    K0 = np.array(all_info[2:11]).astype(float).reshape(3, 3)
                    K1 = np.array(all_info[11:20]).astype(float).reshape(3, 3)
                    T_0to1 = np.array(all_info[20:36]).astype(float).reshape(4, 4)
                    
                    image0 = os.path.join(self.config['data_path'], image0)
                    image1 = os.path.join(self.config['data_path'], image1)
                    depth0 = image0.replace('color', 'depth').replace('jpg', 'png')
                    depth1 = image1.replace('color', 'depth').replace('jpg', 'png')

                    pairs.append({
                        'image0': image0,
                        'image1': image1,
                        'depth0': depth0,
                        'depth1': depth1,
                        'K0': K0,
                        'K1': K1,
                        'T_0to1': T_0to1,
                    })
                    
                elif len(all_info) == 32:
                    # megadepth format
                    K0 = np.array(all_info[2:11]).astype(float).reshape(3, 3)
                    K1 = np.array(all_info[11:20]).astype(float).reshape(3, 3)
                    
                    pose_elems = all_info[20:32]
                    R, t = pose_elems[:9], pose_elems[9:12]
                    R = np.array([float(x) for x in R]).reshape(3, 3).astype(np.float32)
                    t = np.array([float(x) for x in t]).astype(np.float32)
                    T_0to1 = np.eye(4)
                    T_0to1[:3, :3] = R
                    T_0to1[:3, 3] = t

                    image0 = os.path.join(self.config['data_path'], image0)
                    image1 = os.path.join(self.config['data_path'], image1)
                    
                    pairs.append({
                        'image0': image0,
                        'image1': image1,
                        'K0': K0,
                        'K1': K1,
                        'T_0to1': T_0to1,
                    })
                else:
                    print(f'Unknown format for pair with {len(all_info)} elements: {line}')
                    raise ValueError(f'Unknown format for pair {line}')

            if self.config['max_pairs'] > 0:
                pairs = pairs[:self.config['max_pairs']]
        return pairs


    def extract_and_save_matches(self, matcher_fn, name='', force=False):
        all_matches = []
        if name == '':
            name = matcher_fn.__name__

        os.makedirs(os.path.join(self.config['output'], name), exist_ok=True)
        fname = os.path.join(self.config['output'], f'{name}/matches.npz')
        
        if not force and os.path.exists(fname):
            return np.load(fname, allow_pickle=True)['all_matches']

        for pair in tqdm(self.pairs, desc='Extracting matches'):
            image0 = self.read_image(pair['image0'])
            image1 = self.read_image(pair['image1'])
            
            if self.config['resize'] is not None:
                image0, scale0 = resize_long_edge(image0, self.config['resize'])
                image1, scale1 = resize_long_edge(image1, self.config['resize'])

            res = matcher_fn(image0, image1)
            mkpts0 = res['mkpts0']
            mkpts1 = res['mkpts1']
            if isinstance(mkpts0, torch.Tensor):
                mkpts0 = mkpts0.cpu().numpy()
                mkpts1 = mkpts1.cpu().numpy()
            
            if self.config['resize'] is not None:
                mkpts0 = mkpts0 / scale0
                mkpts1 = mkpts1 / scale1

            if self.config['detector_only']:
                mkpts0, mkpts1 = self.oracle_matches_scannet(pair, mkpts0, mkpts1)

            all_matches.append({
                'image0': pair['image0'],
                'image1': pair['image1'],
                'mkpts0': mkpts0,
                'mkpts1': mkpts1,
            })
        
        np.savez(fname, all_matches=all_matches)
        
        return all_matches
    
    def oracle_matches_scannet(self, pair, kpts0, kpts1):
        from easy_local_features.utils.ops import warp_kpts
        
        # project points to the other image and get gt matches based on distance
        K0, K1 = pair['K0'], pair['K1']
        T_0to1 = pair['T_0to1']
        depth0 = cv2.imread(pair['depth0'], cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000
        depth1 = cv2.imread(pair['depth1'], cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000
        image0 = cv2.imread(pair['image0'])
        image1 = cv2.imread(pair['image1'])
        
        depth0 = cv2.resize(depth0, (image0.shape[1], image0.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth1 = cv2.resize(depth1, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        if not isinstance(kpts0, torch.Tensor):
            kpts0 = torch.tensor(kpts0).float()
            kpts1 = torch.tensor(kpts1).float()
            K0, K1 = torch.tensor(K0).float(), torch.tensor(K1).float()
            T_0to1 = torch.tensor(T_0to1).float()
            depth0 = torch.tensor(depth0).float()
            depth1 = torch.tensor(depth1).float()
        
        valid0, proj1 = warp_kpts(kpts0[None], depth0[None], depth1[None], T_0to1[None], K0[None], K1[None], do_on_cpu=True)
        proj1 = torch.where(valid0[..., None], proj1, torch.tensor(-1.0))[0]
        valid0 = valid0[0]
        
        # cdist between projected points and kpts1
        dists = torch.cdist(proj1, kpts1)
        dist, idx = dists.min(dim=1)
        valid = dist < 1.0
        
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[idx[valid]]
        
        return mkpts0.numpy(), mkpts1.numpy()
        

    def run_benchmark(self, matcher_fn, name='', force=False):
        if name == '':
            name = matcher_fn.__name__

        all_matches = self.extract_and_save_matches(matcher_fn, name=name, force=force)
        
        # plot matches in parallel
        out_folder = os.path.join(self.config['output'], name, 'matches')
        os.makedirs(out_folder, exist_ok=True)
        plot_args = [ (pair_idx, pair, all_matches[pair_idx]['mkpts0'], all_matches[pair_idx]['mkpts1'], out_folder) for pair_idx, pair in enumerate(self.pairs) ]
        # select one every 10 
        plot_args = plot_args[::10]
        pool = mp.Pool(self.config['n_workers'])
        r = list(tqdm(pool.imap(plot_matches_parallel, plot_args), total=len(plot_args), desc=f'Plotting matches for {name}', leave=False))
        pool.close()
        # for args in tqdm(plot_args, desc=f'Plotting matches for {name}'):
            # vis.plot_matches_parallel(args)
            
        aucs_by_thresh = {}
        accs_by_thresh = {}
        for ransac_thresh in self.config['ransac_thresholds']:

            fname = os.path.join(self.config['output'], f'{name}/{self.config["pose_estimator"]}_{ransac_thresh}.txt')
            # check if exists and has the right number of lines
            if not force and os.path.exists(fname) and len(open(fname, 'r').readlines()) == len(self.pairs) and False:
                errors = []
                with open(fname, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.replace('\n', '')
                        err_t, err_R = line.split(' ')
                        errors.append([float(err_t), float(err_R)])
            # redo the benchmark
            else:
                errors = []
                pairs = self.pairs
                errors_file = open(fname, 'w')

                # do the benchmark in parallel
                if self.config['n_workers'] != 1:
                    

                    pool = mp.Pool(self.config['n_workers'])
                    pool_args = [ (all_matches[pair_idx]['mkpts0'], all_matches[pair_idx]['mkpts1'], pair['K0'], pair['K1'], ransac_thresh, 0.99999, self.config['pose_estimator']) for pair_idx, pair in enumerate(pairs) ]
                    results = list(tqdm(pool.imap(estimate_pose_parallel, pool_args), total=len(pool_args), desc=f'Running benchmark for th={ransac_thresh}', leave=False))
                    pool.close()

                    for pair_idx, ret in enumerate(results):
                        if ret is None:
                            err_t, err_R = np.inf, np.inf
                        else:
                            R, t, inliers = ret
                            pair = pairs[pair_idx]
                            err_t, err_R = compute_pose_error(pair['T_0to1'], R, t)
                        errors_file.write(f'{err_t} {err_R}\n')
                        errors.append([err_t, err_R])
                # do the benchmark in serial
                else:
                    for pair_idx, pair in tqdm(enumerate(pairs), desc=f'Running benchmark for th={ransac_thresh}', leave=False, total=len(pairs)):
                        mkpts0 = all_matches[pair_idx]['mkpts0']
                        mkpts1 = all_matches[pair_idx]['mkpts1']
                        ret = estimate_pose(mkpts0, mkpts1, pair['K0'], pair['K1'], ransac_thresh, 0.99999, self.config['pose_estimator'])
                        if ret is None:
                            err_t, err_R = np.inf, np.inf
                        else:
                            R, t, inliers = ret
                            err_t, err_R = compute_pose_error(pair['T_0to1'], R, t)
                        errors_file.write(f'{err_t} {err_R}\n')
                        errors_file.flush()
                        errors.append([err_t, err_R])

                errors_file.close()

            # compute AUCs
            errors = np.array(errors)
            errors = errors.max(axis=1) 
            aucs = pose_auc(errors, self.config['pose_thresholds'])
            aucs = {k: v*100 for k, v in zip(self.config['pose_thresholds'], aucs)}
            aucs_by_thresh[ransac_thresh] = aucs
            
            accs = pose_accuracy(errors, self.config['pose_thresholds'])
            accs_by_thresh[ransac_thresh] = accs
            

            # dump summary for this method
            summary = {
                'name': name,
                'aucs_by_thresh': aucs_by_thresh,
                'accs_by_thresh': accs_by_thresh,
            } 
            json.dump(summary, open(os.path.join(self.config['output'], f'{name}/{self.config["pose_estimator"]}_summary.json'), 'w'), indent=2)

        return aucs_by_thresh

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",  "-m", help="Method to eval")
    parser.add_argument("--benchmark", "-b", help="eval on scannet1500 or megadepth1500")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    
    pose_eval =    this_dir = os.path.dirname(os.path.abspath(__file__))
    from easy_local_features import getExtractor
    method = getExtractor(args.method, {'top_k': 4096})
    megadepth = PoseEval({
        'data_path': '/Users/cadar/Documents/Datasets/scannet_test_1500/',
        'pairs_path': '/Users/cadar/Documents/Datasets/scannet_test_1500/pairs_calibrated.txt',
        'output': './output/pose_eval/scannet/',
        'max_pairs': 100,
        'pose_estimator': 'poselib',
        'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        'n_workers': 36,
        # 'resize': -1,
        # 'detector_only': True,
    })
    
    megadepth.run_benchmark(method.match, name=f'xfeat_full', force=True) 
    
    for size in range(400, 2000, 100):
        megadepth = PoseEval({
            'data_path': '/Users/cadar/Documents/Datasets/scannet_test_1500/',
            'pairs_path': '/Users/cadar/Documents/Datasets/scannet_test_1500/pairs_calibrated.txt',
            'output': './output/pose_eval/scannet/',
            'max_pairs': 100,
            'pose_estimator': 'poselib',
            'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            'n_workers': 36,
            'resize': size,
            # 'detector_only': True,
        })
        
        megadepth.run_benchmark(method.match, name=f'xfeat_full_{size}', force=True) 
