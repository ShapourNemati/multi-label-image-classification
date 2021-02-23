import time
import glob

import cv2 as cv
import numpy as np

from multi_class_ranking import dense_sampling_points, compute_descriptors, compute_codebook, compute_histogram, divide_in_grids, classify
from file_utilities import all_images_from_folder, save_descriptors
from detection import descriptors_matrix, subregion_descriptors, sliding_window_detection, multi_scale_template_matching

folder_name = "test\\"
jpg_filenames = glob.glob(folder_name + "*.jpg")

vocabulary = np.load("vocabulary.npy")
model = np.load("model.npy")

for i in range(0, len(jpg_filenames), 15):
    image_file = jpg_filenames[i]

    classes_votes = []
    test = cv.imread(image_file)
    grids = divide_in_grids(test)
    for grid in grids:
        descriptors = compute_descriptors(grid)
        histogram = compute_histogram(descriptors, vocabulary)
        classes_votes.append(classify(histogram, model))

    sorted_list = sorted(classes_votes, key = classes_votes.count, reverse = True)
    ranking = list(dict.fromkeys(sorted_list))
    
    save_descriptors(ranking, image_file[:-4] + "_ranking")

    print ("saved " + str(i))
