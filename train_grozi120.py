import numpy as np
import cv2 as cv

from file_utilities import save_descriptors
from multi_class_ranking import compute_descriptors, compute_codebook, compute_histogram

### STEP 1

# i = 0
# images = []
# descriptor_base_name = "descriptor"
# file_index = 0
# for x in range(1, 121):
#     i = i + 1
#     filename = "E:\\UNI\\VISIONE\\inVitro\\" + str(x) + "\\web\\JPEG\\web1.jpg"
#     images.append(cv.imread(filename))
#     if (i >= 10):
#         print(filename)
#         descriptor_name = descriptor_base_name + str(file_index)
#         d = compute_descriptors(images)
#         save_descriptors(d, descriptor_name)
#         print(descriptor_name)
#         images = []
#         file_index = file_index + 1
#         i = 0

### STEP 2

# descriptors = []
# for i in range (0, 12):
#     descriptors.append(np.load("descriptor" + str(i) + ".npy"))
# print ("descriptors loaded")
# descriptors = np.vstack(descriptors)
# vocab = compute_codebook(descriptors)
# save_descriptors(vocab, "vocabulary")

### STEP 3

# v = np.load("vocabulary.npy")
# model = []
# images = []
# for x in range(1, 121):
#     filename = "E:\\UNI\\VISIONE\\inVitro\\" + str(x) + "\\web\\JPEG\\web1.jpg"
#     img = cv.imread(filename)
#     print(filename)
#     d = compute_descriptors(img)
#     h = compute_histogram(d, v)
#     model.append(h)

# save_descriptors(model, "model")
