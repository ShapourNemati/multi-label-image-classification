import cv2 as cv

from multi_class_ranking import dense_sampling_points

def test_dense_sampling_points():
    img = cv.imread("8159.JPG")
    dense_sampling_points(img)

test_dense_sampling_points()