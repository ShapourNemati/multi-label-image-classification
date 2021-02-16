import cv2 as cv
import sys
import numpy as np


def dense_sampling_points(img):
    """Compute the keypoints sampling at with a spacing of 4 pixel and patch sizes of 8, 12, 16, 24, and 30."""
    keypoints = []
    sizes = [8, 12, 16, 24, 30]
    for z in sizes:
        xs = np.arange(0, img.shape[1], 4.0)
        ys = np.arange(0, img.shape[0], 4.0)
        for x in xs:
            for y in ys:
                    keypoints.append(cv.KeyPoint(x, y, z))
    return keypoints
