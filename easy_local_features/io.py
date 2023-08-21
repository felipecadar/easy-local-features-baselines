import numpy as np
import cv2
import functools

def writeKeypoints(keypoints, filename):
    """
    Write keypoints to a file using numpy compressed format
    """

    total_keypoints = len(keypoints)
    np_keypoints = np.zeros((total_keypoints, 4))
    for i, kp in enumerate(keypoints):
        np_keypoints[i, :] = [kp.pt[0], kp.pt[1], kp.size, kp.angle]

    np.savez_compressed(filename, keypoints=np_keypoints)

@functools.lru_cache(maxsize=128)
def readKeypoints(filename):
    """
    Read keypoints from a file
    """
    
    npzfile = np.load(filename)
    keypoints = []
    for row in npzfile['keypoints']:
        kp = cv2.KeyPoint(x=row[0], y=row[1], size=row[2], angle=row[3])
        keypoints.append(kp)

    return keypoints

def writeDescriptors(descriptors, filename):
    """
    Write descriptors to a file using numpy compressed format
    """

    np.savez_compressed(filename, descriptors=descriptors)

@functools.lru_cache(maxsize=128)
def readDescriptors(filename):
    """
    Read descriptors from a file
    """
    
    npzfile = np.load(filename)
    return npzfile['descriptors']


