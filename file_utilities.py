import glob

import cv2 as cv
import numpy as np

def all_images_from_folder(folder_name):
    """Create and return all images contained in the input directory."""
    images = []
    filenames = glob.glob(folder_name + "*.jpg")
    for file in filenames:
        images.append(cv.imread(file))
    return images

def save_descriptors(descriptors, filename):
    """Save input descriptor in the specified file."""
    descriptors = np.vstack(descriptors)
    np.save(filename, descriptors)