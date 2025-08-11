import os, sys
import glob
import json
import numpy as np

import fnmatch
import os

import tqdm

DEFAULT_DATA_PATH = '/srv/storage/datasets/cadar/simulator/_dataset_test2/' 
OUTPUT_PATH = '/srv/storage/datasets/cadar/semantic-local-features/metrics_output/simulation/brand/'
SIFT_PATH = '/srv/storage/datasets/cadar/semantic-local-features/metrics_output/simulation/sift/'
BIN = '/srv/storage/datasets/cadar/easy-local-features-baselines/easy_local_features/submodules/git_brand/build/desc_brand'

if __name__ == "__main__":
    # images = glob.glob(DEFAULT_DATA_PATH + '/**/rgba*.png', recursive=True)
    # desc_brand fx fy cx cy n <rgb_image depth_image keypoints_csv output_file>

    images = []
    for root, dirnames, filenames in os.walk(DEFAULT_DATA_PATH):
        for filename in fnmatch.filter(filenames, 'rgba*.png'):
            images.append(os.path.join(root, filename))

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    metadata_file = images[0].split('rgba')[0] + 'metadata.json'
    metadata = json.load(open(metadata_file, 'r'))
    resolution = metadata['metadata']['resolution']
    focal = metadata['camera']['focal_length']
    sensor_width = metadata['camera']['sensor_width']
    field_of_view = metadata['camera']['field_of_view']

    # print(camera_matrix)

    fx = focal * resolution[0] / sensor_width
    fy = focal * resolution[1] / sensor_width
    cx = resolution[0] / 2
    cy = resolution[1] / 2

    for image_path in tqdm.tqdm(images):
        cmd = "{} {} {} {} {} {} ".format(BIN, fx, fy, cx, cy, 1)
    
        depth_path = image_path.replace('rgba', 'depth').replace('png', 'tiff')
        npz_keypoints_path = image_path.replace(DEFAULT_DATA_PATH, SIFT_PATH) + ".kpts.npz"
        csv_keypoints_path = image_path.replace('rgba', 'sift').replace('.png', '.csv').replace(DEFAULT_DATA_PATH, OUTPUT_PATH)
        output_path = image_path.replace('rgba', 'brand').replace('.png', '.csv').replace(DEFAULT_DATA_PATH, OUTPUT_PATH)

        npz_output_path = image_path.replace(DEFAULT_DATA_PATH, OUTPUT_PATH) + '.desc'

        out_dir = os.path.dirname(output_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # read and turn to csv if not already
        if not os.path.exists(csv_keypoints_path):
            kps = np.load(npz_keypoints_path)['keypoints']
            # save csv keypoints with ", "
            np.savetxt(csv_keypoints_path, kps, delimiter=", ", fmt='%.3f')

            # cp npz_keypoints_path to csv_keypoints_path folder
            dst_folder = os.path.dirname(csv_keypoints_path)
            os.system("cp {} {}".format(npz_keypoints_path, dst_folder))

        cmd += "{} {} {} {}".format(image_path, depth_path, csv_keypoints_path, output_path)

        ret = os.system(cmd)
        if ret != 0:
            print("Error in {}".format(image_path))
            continue

        #read output_path
        descs = np.loadtxt(output_path, delimiter=",", skiprows=1)
        # save as npz
        np.savez_compressed(npz_output_path, descriptors=descs)

        # save the 




