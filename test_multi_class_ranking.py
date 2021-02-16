import cv2 as cv
import numpy as np

from multi_class_ranking import dense_sampling_points, compute_and_save_descriptors

def test_dense_sampling_points():
    img = cv.imread("8159.JPG")
    dense_sampling_points(img)

def test_compute_and_save_descriptors():
    img1 = cv.imread("8159.JPG")
    img2 = cv.imread("8159.JPG")
    filename = "prova"
    des = compute_and_save_descriptors([img1, img2], filename)
    des2 = np.load(filename + ".npy")
    assert np.array_equal(des, des2) 