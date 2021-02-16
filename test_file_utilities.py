import cv2 as cv
import numpy as np

from file_utilities import all_images_from_folder, save_descriptors
from multi_class_ranking import compute_descriptors

def test_all_images_from_folder():
    img1 = cv.imread("images/8159.JPG")
    img2 = cv.imread("images/9039.JPG")
    img3, img4 = all_images_from_folder("images/")
    assert (img1.shape, img2.shape) == (img3.shape, img4.shape)

def test_save_descriptors():
    img1 = cv.imread("images/8159.JPG")
    img2 = cv.imread("images/9039.JPG")
    des = compute_descriptors([img1, img2])
    save_descriptors(des, "des")
    des2 = np.load("des.npy")
    assert np.array_equal(des, des2) 