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

def compute_descriptors(img_list):
    """Compute the descriptors of all the images provided."""
    sift = cv.SIFT_create()
    descriptors = []

    for img in img_list:
        keypoints = dense_sampling_points(img)
        _, des = sift.compute(img, keypoints)
        descriptors.append(des)
    
    return descriptors

def compute_codebook(descriptors):
    """Compute the codebook for BoW using K-means and a fixed cluster number of 256."""
    bow_trainer = cv.BOWKMeansTrainer(256)
    return bow_trainer.cluster(descriptors)