import sys

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
    for i in range(x, width + x, step):
        for j in range (y, height + y, step):
            descriptors.append(descriptors_matrix[int(j/step)][int(i/step)])
    return descriptors

def sliding_window_detection(template, img):
    step = 16
    template_width, template_height, _ = template.shape
    img_width, img_height, _ = img.shape
    desc_mat = descriptors_matrix(img)
    min_dist = sys.float_info.max
    min_upper_corner = (0, 0)
    template_descriptors = subregion_descriptors(descriptors_matrix(template), 0, 0, template_width, template_height)
    for i in range(0, img_height - template_height, step):
        for j in range(0, img_width - template_width, step):
            sub_desc = subregion_descriptors(desc_mat, j, i, template_width, template_height)
            dist = 0
            for z in range(len(sub_desc)):
                dist = dist + np.linalg.norm(sub_desc[z] - template_descriptors[z])
            mean_dist = dist / len(sub_desc)
            if mean_dist < min_dist:
                min_dist = mean_dist
                min_upper_corner = (i, j)
    return (min_dist, min_upper_corner)

def multi_scale_template_matching(template, img, sizes):
    min_dist = sys.float_info.max
    x, y, x2, y2 = (0, 0, 0, 0)
    for size in sizes:
        template_width, template_height, img_width, img_height = size
        resized_template = cv.resize(template, (template_width, template_height))
        resized_img = cv.resize(img, (img_width, img_height))
        dist, upper_corner = sliding_window_detection(resized_template, resized_img)
        if (dist < min_dist):
            min_dist = dist
            x, y = upper_corner
            x = x / (img.shape[1] / img_width)
            y = y / (img.shape[0] / img_height)
            x2 = x + template_width
            y2 = y + template_height
    return (min_dist, (x, y), (x2, y2))