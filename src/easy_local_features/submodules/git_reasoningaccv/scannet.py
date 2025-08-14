
import os, struct
import numpy as np
import zlib
import imageio
import cv2
import png
from tqdm import tqdm

from omegaconf import OmegaConf
import torch
import argparse
import h5py
from reasoning.modules.utils import load_image, load_depth
from reasoning.datasets.utils import *
from reasoning.features.desc_reasoning import sample_features_dino
from glob import glob

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

import logging
class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))


    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise


    def decompress_depth_zlib(self):
          return zlib.decompress(self.depth_data)


    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise


    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)

class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            tmp = struct.unpack('c'*strlen, f.read(strlen))
            self.sensor_name = ''.join(map(lambda x: x.decode('utf-8'), tmp))
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height =  struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height =  struct.unpack('I', f.read(4))[0]
            self.depth_shift =  struct.unpack('f', f.read(4))[0]
            num_frames =  struct.unpack('Q', f.read(8))[0]
            self.frames = []
            for i in tqdm(range(num_frames), desc='Loading frames'):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):os.makedirs(output_path)
        print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip), desc='Exporting depth images'):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
            if image_size is not None:depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        #imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
            with open(os.path.join(output_path, str(f) + '.png'), 'wb') as f: # write 16-bit
                writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
                depth = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(f, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):os.makedirs(output_path)
        print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip), desc='Exporting color images'):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)


    def save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')


    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):os.makedirs(output_path)
        print('exporting', len(self.frames)//frame_skip, 'camera poses to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip), desc='Exporting poses'):
            self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))


    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):os.makedirs(output_path)
        print('exporting camera intrinsics to', output_path)
        self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
        self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
        self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
        self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))

class ScannetImages(torch.utils.data.Dataset):
    defalut_config = {
        "data_path": "./datasets/scannet/scans/scene0000_00",
    }
    
    def __init__(self, config={}):
        self.config = config = OmegaConf.merge(OmegaConf.create(self.defalut_config), config)
        self.root = config.data_path
        # lexical sort
        self.all_color_files = sorted(os.listdir(os.path.join(self.root, 'color')), key=lambda x: int(x.split('.')[0]))
         
        self.intrincics_color = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsics', 'intrinsic_color.txt'))).float()
        self.intrincics_depth = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsics', 'intrinsic_depth.txt'))).float()
        self.extrincics_color = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsics', 'extrinsic_color.txt'))).float()
        self.extrincics_depth = torch.from_numpy(np.loadtxt(os.path.join(self.root, 'intrinsics', 'extrinsic_depth.txt'))).float()
        
        self.intrincics_color = self.intrincics_color[:3, :3]
        self.intrincics_depth = self.intrincics_depth[:3, :3]
        
        self.idx_to_id = {i: int(f.split('.')[0]) for i, f in enumerate(self.all_color_files)}
        
        
    def __len__(self):
        return len(self.all_color_files)
        
    def __getitem__(self, idx):
        idx = self.idx_to_id[idx]
        
        image = load_image(os.path.join(self.root, 'color', str(idx) + '.jpg'))
        depth = load_depth(os.path.join(self.root, 'depth', str(idx) + '.png')) / 1000
        pose =  torch.from_numpy(np.loadtxt(os.path.join(self.root, 'poses', str(idx) + '.txt'))).float()
        
        image_key = os.path.join(self.root, 'color', str(idx) + '.jpg')
        image_key = os.path.join(*image_key.split('/')[-4:])
        
        depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=(image.shape[1], image.shape[2]), mode='nearest').squeeze(0).squeeze(0)
        
        return {
            'image': image, # 3x968x1296
            'depth': depth, # 1x480x640 in meters
            'pose': pose, # 3x4
            'K': self.intrincics_color, # 3x3
            'image_id': idx,
            'image_key': image_key,
        }
        
class ScannetPairsScene(torch.utils.data.Dataset):
    defalut_config = {
        "data_path": "./datasets/scannet/scans/scene0000_00",
        "min_overlap_score": 0.2,
        "max_overlap_score": 0.8,
        "mode": "train",
        "resize": 512, #-1
    }
    
    def __init__(self, config={}):
        self.config = config = OmegaConf.merge(OmegaConf.create(self.defalut_config), config)
        self.covis_file = os.path.join(config.data_path, 'covisibility.txt')
        self.covis_pairs = np.loadtxt(self.covis_file, delimiter=',')
        self.covis_pairs = self.covis_pairs[(self.covis_pairs[:, 2] > config.min_overlap_score) & (self.covis_pairs[:, 2] < config.max_overlap_score)]
        
        self.intrincics_color = torch.from_numpy(np.loadtxt(os.path.join(config.data_path, 'intrinsics', 'intrinsic_color.txt'))).float()[0:3, 0:3]
        
    def __len__(self):
        return self.covis_pairs.shape[0]
        
    def __getitem__(self, idx):
        idx0, idx1, covis = self.covis_pairs[idx]
        idx0 = int(idx0)
        idx1 = int(idx1)
        
        img0_key = os.path.join(self.config.data_path, 'color', str(idx0) + '.jpg')
        img1_key = os.path.join(self.config.data_path, 'color', str(idx1) + '.jpg')
        img0_key = os.path.join(*img0_key.split('/')[-4:])
        img1_key = os.path.join(*img1_key.split('/')[-4:])
        
        image0 = load_image(os.path.join(self.config.data_path, 'color', str(idx0) + '.jpg'))
        image1 = load_image(os.path.join(self.config.data_path, 'color', str(idx1) + '.jpg'))

        depth0 = load_depth(os.path.join(self.config.data_path, 'depth', str(idx0) + '.png')) / 1000
        depth1 = load_depth(os.path.join(self.config.data_path, 'depth', str(idx1) + '.png')) / 1000

        # resize depth to image size
        depth0 = torch.nn.functional.interpolate(depth0.unsqueeze(0), size=(image0.shape[1], image0.shape[2]), mode='nearest').squeeze(0).squeeze(0)
        depth1 = torch.nn.functional.interpolate(depth1.unsqueeze(0), size=(image1.shape[1], image1.shape[2]), mode='nearest').squeeze(0).squeeze(0)

        K = self.intrincics_color
        
        if self.config.resize > 0:
            # rescale the images
            image0, scale = resize_short_edge(image0, self.config.resize)
            image1, _ = resize_short_edge(image1, self.config.resize)
            depth0, _ = resize_short_edge(depth0, self.config.resize)
            depth1, _ = resize_short_edge(depth1, self.config.resize)
            
            # crop the images to square            
            image0 = crop_square(image0)
            image1 = crop_square(image1)
            depth0 = crop_square(depth0)
            depth1 = crop_square(depth1)

            # rescale the intrinsics
            K = K * scale
            K[2, 2] = 1
            
        pose0 =  torch.from_numpy(np.loadtxt(os.path.join(self.config.data_path, 'poses', str(idx0) + '.txt'))).float()
        pose1 =  torch.from_numpy(np.loadtxt(os.path.join(self.config.data_path, 'poses', str(idx1) + '.txt'))).float()

        T_0to1 = get_relative_transform(pose0, pose1)
        T_1to0 = get_relative_transform(pose1, pose0)

        return {
            'image0': image0, # 3x968x1296
            'image1': image1, # 3x968x1296
            'depth0': depth0, # 1x480x640
            'depth1': depth1, # 1x480x640
            'pose0': pose0, # 4x4
            'pose1': pose1, # 4x4
            'T_0to1': T_0to1, # 3x4
            'T_1to0': T_1to0, # 3x4
            'K': K, # 3x3
            'K0': K, # 3x3
            'K1': K, # 3x3
            'covis': covis,
            'dataset_name': 'scannet',
            'image0_key': img0_key,
            'image1_key': img1_key,
        }

class ScannetSingleH5(torch.utils.data.Dataset):
    def __init__(self, h5_file, only_images=False, features_cache=None, dino_cache=None):
        super().__init__()

        self.h5_file = h5_file
        self.only_images = only_images
        self.features_cache = features_cache
        self.dino_cache = dino_cache
        if only_images:
            cached_keys_fname = h5_file + '.keys.images'
            if os.path.exists(cached_keys_fname):
                with open(cached_keys_fname, 'r') as f:
                    self.keys = f.readlines()
                    self.keys = [k.strip() for k in self.keys]
            else:
                with h5py.File(h5_file, 'r') as h5:
                    self.keys = list(h5['images'].keys())
                with open(cached_keys_fname, 'w') as f:
                    for key in self.keys:
                        f.write(f"{key}\n")
        else:
            cached_keys_fname = h5_file + '.keys.pairs'
            if os.path.exists(cached_keys_fname):
                with open(cached_keys_fname, 'r') as f:
                    self.keys = f.readlines()
                    self.keys = [k.strip() for k in self.keys]
            else:
                with h5py.File(h5_file, 'r') as h5:
                    self.keys = list(h5['T_0to1'].keys())
                with open(cached_keys_fname, 'w') as f:
                    for key in self.keys:
                        f.write(f"{key}\n")
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        if self.only_images:
            with h5py.File(self.h5_file, 'r') as h5:
                image = h5['images'][self.keys[idx]][...]
                return {
                    'image': torch.tensor(image),
                    'image_key': self.keys[idx],
                }
        
        key0, key1 = self.keys[idx].split(':')

        with h5py.File(self.h5_file, 'r') as h5:
            T_0to1 = h5['T_0to1'][self.keys[idx]][...]
            T_1to0 = h5['T_1to0'][self.keys[idx]][...]
            covis = h5['covis'][self.keys[idx]][...]

            image0 = h5['images'][key0][...]
            image1 = h5['images'][key1][...]
            depth0 = h5['depth'][key0][...]
            depth1 = h5['depth'][key1][...]

            K0 = h5['K'][key0][...]
            K1 = h5['K'][key1][...]
            
        return_data = {
            'image0': torch.tensor(image0),
            'image1': torch.tensor(image1),
            'depth0': torch.tensor(depth0),
            'depth1': torch.tensor(depth1),
            'T_0to1': torch.tensor(T_0to1),
            'T_1to0': torch.tensor(T_1to0),
            'K0': torch.tensor(K0),
            'K1': torch.tensor(K1),
            'covis': torch.tensor(covis),
            'dataset': 'scannet',
        }
        
        if self.features_cache is not None:
            with h5py.File(self.features_cache, 'r') as h5:
                for cache_key in ['keypoints', 'descriptors']:
                    return_data[f'{cache_key}0'] = torch.tensor(h5[cache_key][key0][...]).float()
                    return_data[f'{cache_key}1'] = torch.tensor(h5[cache_key][key1][...]).float()
                    
        if self.dino_cache is not None:
            with h5py.File(self.dino_cache, 'r') as h5:
                dino_map0 = torch.tensor(h5['features'][key0][...])
                dino_map1 = torch.tensor(h5['features'][key1][...])
                keypoints0 = return_data['keypoints0']
                keypoints1 = return_data['keypoints1']
                
                # sample_features_dino
                semanitc_features0 = sample_features_dino(keypoints0.unsqueeze(0), dino_map0.unsqueeze(0))[0]
                semanitc_features1 = sample_features_dino(keypoints1.unsqueeze(0), dino_map1.unsqueeze(0))[0]
                
                return_data['semantic_features0'] = semanitc_features0
                return_data['semantic_features1'] = semanitc_features1
                
        # normalize size to be fixed at 32 chars
        image0_key = key0.ljust(40, '#')
        image1_key = key1.ljust(40, '#')
        return_data['image0_key'] = image0_key
        return_data['image1_key'] = image1_key
        
        return return_data

class ScannetH5(torch.utils.data.Dataset):
    def __init__(self, root, split, max_samples=-1, only_images=False, features_cache=None, dino_cache=None):
        super().__init__()
        assert split in ['train', 'val', 'test']
        assert os.path.exists(root)
        self.root = root
        self.split = split
        self.max_samples = max_samples
        self.h5_files = sorted(glob(os.path.join(root, split, '*.h5')))
        assert len(self.h5_files) > 0, f"No h5 files found in {root}/{split}"
        
        if features_cache is not None:
            h5_files_features_cache = sorted(glob(os.path.join(root, 'features', features_cache, split, '*.h5')))
            if len(h5_files_features_cache) == 0:
                raise ValueError(f"No features cache found in {root}/features/{features_cache}/{split}")
            if len(h5_files_features_cache) != len(self.h5_files):
                raise ValueError(f"Features cache and h5 files do not match in length")
        else:
            h5_files_features_cache = [None] * len(self.h5_files)
            
        if dino_cache is not None:
            h5_file_dino_cache = sorted(glob(os.path.join(root, 'features', dino_cache, split, '*.h5')))
            if len(h5_file_dino_cache) == 0:
                raise ValueError(f"No dino cache found in {root}/features/{dino_cache}/{split}")
            if len(h5_file_dino_cache) != len(self.h5_files):
                raise ValueError(f"Dino cache and h5 files do not match in length")
        else:
            h5_file_dino_cache = [None] * len(self.h5_files)
        
        self.dataset = torch.utils.data.ConcatDataset([
            ScannetSingleH5(h5_file, only_images=only_images, features_cache=features_cache, dino_cache=dino_cache)
            for (h5_file, features_cache, dino_cache) 
            in tqdm(zip(self.h5_files, h5_files_features_cache, h5_file_dino_cache), total=len(self.h5_files), desc=f"Loading {split} h5 files")
        ])

    def __len__(self):
        if self.max_samples > 0:
            return min(len(self.dataset), self.max_samples)
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def makeScannetDataset(scannet_scenes_folder, min_overlap_score, max_overlap_score, mode, max_scenes=-1, concat=True):
    assert mode in ['train', 'val', 'test']
    this_folder = os.path.dirname(os.path.abspath(__file__))
    split = os.path.join(this_folder, 'scannetv1_' + mode + '.txt')
    with open(split, 'r') as f:
        scenes = f.readlines()
    scenes = sorted([s.strip() for s in scenes])
    if max_scenes > 0:
        scenes = scenes[:max_scenes]
    if max_scenes == 1:
        scenes = ['scene0000_00']
    
    if concat:
        concat_dataset = torch.utils.data.ConcatDataset([
            ScannetPairsScene({
                "data_path": os.path.join(scannet_scenes_folder, s),
                "min_overlap_score": min_overlap_score,
                "max_overlap_score": max_overlap_score,
                "mode": mode,
            }) for s in scenes
        ])
    else:
        concat_dataset = {
            s: ScannetPairsScene({
                "data_path": os.path.join(scannet_scenes_folder, s),
                "min_overlap_score": min_overlap_score,
                "max_overlap_score": max_overlap_score,
                "mode": mode,
            }) for s in scenes
        }
        # scene0011_00, scene0011_01, scene0011_02 should be in the same dataset
        joint = {}
        for s, dataset in concat_dataset.items():
            if s.split('_')[0] not in joint:
                joint[s.split('_')[0]] = []
            joint[s.split('_')[0]].append(dataset)

        for s, datasets in joint.items():
            joint[s] = torch.utils.data.ConcatDataset(datasets)
            
        concat_dataset = joint
        
    return concat_dataset     

def calculate_covisibility(scene_folder):

    scannet_images = ScannetImages({
        'data_path': scene_folder,
    })
    
    covis_file = open( os.path.join(scannet_images.root, 'covisibility.txt'), 'w')

    norm_grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32)), dim=-1).reshape(-1, 2)

    for source_idx in tqdm(range(len(scannet_images)), position=0, leave=True):
        data_source = add_batch_dim(scannet_images[source_idx])
        
        for target_idx in range(source_idx + 1, len(scannet_images)):            
            data_target = add_batch_dim(scannet_images[target_idx])
            kps0 = (norm_grid * torch.tensor([data_source['image'].shape[3]-1, data_source['image'].shape[2]-1]).float()).unsqueeze(0) # BxNx2
            T_0to1 = get_relative_transform(data_source['pose'], data_target['pose'])
            valid_mask, w_kpts0 = warp_kpts(kps0, data_source['depth'], data_target['depth'], T_0to1, data_source['K'], data_target['K'])
            covis = valid_mask.sum().item() / valid_mask.shape[1]
            covis_file.write(f"{data_source['image_id']},{data_target['image_id']},{covis}\n")
            covis_file.flush()

    covis_file.close()
    