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

def sliding_window_detection(template, img, desc_mat):
    step = 16
    candidates = []
    template_width, template_height, _ = template.shape
    img_width, img_height, _ = img.shape
    template_descriptors = subregion_descriptors(descriptors_matrix(template), 0, 0, template_width, template_height)
    for i in range(0, img_height - template_height, step):
        for j in range(0, img_width - template_width, step):
            sub_desc = subregion_descriptors(desc_mat, j, i, template_width, template_height)
            dist = 0
            for z in range(len(sub_desc)):
                dist = dist + np.linalg.norm(sub_desc[z] - template_descriptors[z])
            mean_dist = dist / len(sub_desc)
            candidates.append((mean_dist, (i, j), (i + template_width, j + template_height)))
    return non_maximum_suppression(candidates)

def intersection_over_union(c1, w1, h1, c2, w2, h2):

    dx = min(c1[0] + w1, c2[0] + w2) - max(c1[0], c2[0])
    dy = min(c1[1] + w1, c2[1] + w2) - max(c1[1], c2[1])
    if dx > 0 and dy > 0:
        intersection = dx * dy
    else:
        intersection = 0

    area_1 = w1 * h1
    area_2 = w2 * h2

    union = area_1 + area_2 - intersection

    return intersection/union

def non_maximum_suppression(candidates):
    result = []
    while len(candidates) > 0:
        min_dist = sys.float_info.max
        min_index = 0
        for i in range(len(candidates)):
            dist, p1, p2 = candidates[i]
            if dist < min_dist:
                min_dist = dist
                min_index = i
        dist, p1, p2 = candidates[min_index]
        to_remove = []
        for i in range(len(candidates)):
            dist_2, q1, q2 = candidates[i]
            if intersection_over_union_2(p1, p2, q1, q2) > 0.15:
                to_remove.append(candidates[i])
        for x in to_remove:
            candidates.remove(x)
        result.append((dist, p1, p2))
    return result

def intersection_over_union_2(p1, p2, q1, q2):
    c1 = p1
    c2 = q1

    w1 = p2[0] - p1[0]
    h1 = p2[1] - p1[1]

    w2 = q2[0] - q1[0]
    h2 = q2[1] - q1[1]

    return intersection_over_union(c1, w1, h1, c2, w2, h2)


def multi_scale_template_matching(template_base, img, template_sizes, img_sizes):
    matches = []
    for img_size in img_sizes:
        img_width, img_height = img_size
        resized_img = cv.resize(img, (img_width, img_height))
        desc_mat = descriptors_matrix(img)
        template_matches = []
        for rot in range(0, 4):
            template = np.rot90(template_base, rot)
            for template_size in template_sizes:
                template_width, template_height = template_size
                resized_template = cv.resize(template, (template_width, template_height))
                bounding_boxes = sliding_window_detection(resized_template, resized_img, desc_mat)
                for bounding_box in bounding_boxes:
                    template_matches.append(bounding_box)
    return non_maximum_suppression(template_matches)