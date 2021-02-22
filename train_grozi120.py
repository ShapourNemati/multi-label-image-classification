import numpy as np
import cv2 as cv
import glob

from file_utilities import save_descriptors
from multi_class_ranking import compute_descriptors, compute_codebook, compute_histogram

### STEP 1

# i = 0
# image_files = []
# descriptor_base_name = "descriptor"
# descriptor_index = 0
# for x in range(1, 121):
#     i = i + 1
#     folder = "E:\\UNI\\VISIONE\\inVitro\\" + str(x) + "\\\web\\JPEG\\"
#     files = glob.glob(folder + "*.jpg")
#     print(files)
#     image_files.append(files)
#     if (i >= 10):
#         images = []
#         for class_files in image_files:
#             print(class_files)
#             for filename in class_files:
#                 print(filename)
#                 images.append(cv.imread(filename))
#         descriptor_name = descriptor_base_name + str(descriptor_index)
#         d = compute_descriptors(images)
#         save_descriptors(d, descriptor_name)
#         print(descriptor_name)
#         descriptor_index = descriptor_index + 1
#         i = 0
#         image_files = []

### STEP 2

# descriptors = []
# for i in range (0, 12):
#     descriptors.append(np.load("descriptor" + str(i) + ".npy"))
# print ("descriptors loaded")
# descriptors = np.vstack(descriptors)
# vocab = compute_codebook(descriptors)
# save_descriptors(vocab, "vocabulary")

### STEP 3

# i = 0
# vocabulary = np.load("vocabulary.npy")
# model = []
# model_base_name = "model\\model"
# for x in range(1, 121):
#     i = i + 1
#     folder = "E:\\UNI\\VISIONE\\inVitro\\" + str(x) + "\\\web\\JPEG\\"
#     class_files = glob.glob(folder + "*.jpg")
#     for filename in class_files:
#         img = cv.imread(filename)
#         d = compute_descriptors(img)
#         h = compute_histogram(d, vocabulary)
#         h = np.hstack((h, i))
#         model.append(h)
#     model_name = model_base_name + str(i)
#     print(model_name)
#     save_descriptors(model, model_name)
#     model = []

model = []
for x in range(1, 121):
    filename = "model\\model" + str(x) + ".npy"
    model.append(np.load(filename))

save_descriptors(model, "model")
