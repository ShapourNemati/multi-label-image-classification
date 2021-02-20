import cv2 as cv
import numpy as np

def descriptors_matrix(img):
    """Compute the descriptors and puts them in a matrix with the pixel indexes."""
    step = 16.0
    xs = np.arange(0, img.shape[1], step)
    ys = np.arange(0, img.shape[0], step)
    descriptors = np.empty((int(img.shape[1]/step) + 1, int(img.shape[0]/step) + 1, 128))
    sift = cv.SIFT_create()
    for x in xs:
        for y in ys:
                keypoint = cv.KeyPoint(x, y, 4)
                _, d = sift.compute(img, [keypoint])
                descriptors[int(x/step)][int(y/step)] = d
    return descriptors

def subregion_descriptors(descriptors_matrix, x, y, width, height):
    step = 16
    descriptors = []
    for i in range(x, width, step):
        for j in range (y, height, step):
            descriptors.append(descriptors_matrix[int(i/step)][int(j/step)])
    return descriptors
