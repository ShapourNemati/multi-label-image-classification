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

def compute_histogram(descriptor, vocabulary):
    """Compute the histogram of the descriptor with the given vocabulary."""
    histogram = np.zeros(256)
    for d in descriptor:
        min_val = sys.float_info.max
        min_index = 0
        i = 0
        for w in vocabulary:
            dist = np.linalg.norm(d-w)
            if (dist < min_val):
                min_val = dist
                min_index = i
            i = i + 1
        histogram[min_index] = histogram[min_index] + 1
    return histogram

def classify(histogram, model):
    """Compute the class of the input histogram with respect to the given model."""
    min_distance = sys.float_info.max
    min_index = 0
    distances = []
    index = 1
    for m in model:
        d = np.linalg.norm(m - histogram)
        if (d < min_distance):
            min_distance = d
            min_index = index
        index = index + 1
    return min_index

def divide_in_grids(image):
    """
    Divides the image in 50 patches.
    The grids are: 4x2, 2x4, 3x3, and 5x5. 
    """
    patches = []
    xns = (5, 3, 4, 6)
    yns = (3, 5, 4, 6)
    for i in range(0, 4):    
        get_grid_division(image, xns[i], yns[i], patches)
    return patches

def get_grid_division(image, xn, yn, patches):
    xs = np.linspace(0, image.shape[1], num = xn)
    ys = np.linspace(0, image.shape[0], num = yn)
    for i in range(0, xn - 1):
        for j in range(0, yn - 1):
            patch = image[int(ys[j]):int(ys[j+1]), int(xs[i]):int(xs[i+1])]
            patches.append(patch)
