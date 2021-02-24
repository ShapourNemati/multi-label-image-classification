import time
import glob

import cv2 as cv
import numpy as np

from multi_class_ranking import dense_sampling_points, compute_descriptors, compute_codebook, compute_histogram, divide_in_grids, classify
from file_utilities import all_images_from_folder, save_descriptors
from detection import descriptors_matrix, subregion_descriptors, sliding_window_detection, multi_scale_template_matching

folder_name = "test\\"
jpg_filenames = glob.glob(folder_name + "*.jpg")
metadata_filenames = [f for f in glob.glob(folder_name + "*.npy") if "_ranking" not in f]

#STEP 1 - Compute and save multi-class-ranking

# vocabulary = np.load("vocabulary.npy")
# model = np.load("model.npy")

# for i in range(0, len(jpg_filenames), 15):
#     image_file = jpg_filenames[i]

#     classes_votes = []
#     test = cv.imread(image_file)
#     grids = divide_in_grids(test)
#     for grid in grids:
#         descriptors = compute_descriptors(grid)
#         histogram = compute_histogram(descriptors, vocabulary)
#         classes_votes.append(classify(histogram, model))

#     sorted_list = sorted(classes_votes, key = classes_votes.count, reverse = True)
#     ranking = list(dict.fromkeys(sorted_list))
    
#     save_descriptors(ranking, image_file[:-4] + "_ranking")

#     print ("saved " + str(i))

#STEP 2 - Calculate mAP

average_precision = []
average_recall = []
max_k = 50

for i in range(0, len(metadata_filenames), 15):
    metadata_file = metadata_filenames[i]
    ranking_file = metadata_file[:-4] + "_ranking.npy"

    metadata = np.load(metadata_file)
    ranking = np.load(ranking_file)

    relevant_items = (metadata[:, 5]).flatten()
    relevant_items = set(relevant_items)

    precision = []
    recall = []

    for k in range(1, max_k + 1):
        if k < len(ranking):
            considered_votes = ranking[:k].flatten()
            true_positives = list(set(considered_votes) & relevant_items)
            precision.append(len(true_positives) / len(considered_votes))
            recall.append(len(true_positives) / len(relevant_items))

    average_precision.append(sum(precision) / len(precision))
    average_recall.append(sum(recall) / len(recall))

print(sum(average_precision) / len(average_precision))
print(sum(average_recall) / len(average_recall))