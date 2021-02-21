import time

import cv2 as cv
import numpy as np

from multi_class_ranking import dense_sampling_points, compute_descriptors, compute_codebook, compute_histogram, divide_in_grids, classify
from file_utilities import all_images_from_folder, save_descriptors
from detection import descriptors_matrix, subregion_descriptors, sliding_window_detection, multi_scale_template_matching

vocabulary = np.load("vocabulary.npy")
model = np.load("model.npy")

classes_votes = []
test = cv.imread("test/Shelf_1_frame_174.jpg")
grids = divide_in_grids(test)
for grid in grids:
    descriptors = compute_descriptors(grid)
    histogram = compute_histogram(descriptors, vocabulary)
    classes_votes.append(classify(histogram, model))

classes_votes.sort()
classes = list(dict.fromkeys(classes_votes))
print(classes)

metadata = np.load("test/Shelf_1_frame_109.npy")
print(metadata)