import cv2 as cv
import numpy as np

from multi_class_ranking import dense_sampling_points, compute_descriptors, compute_codebook, compute_histogram, classify

def test_dense_sampling_points():
    img = cv.imread("8159.JPG")
    dense_sampling_points(img)

def test_compute_descriptors():
    img1 = cv.imread("8159.JPG")
    img2 = cv.imread("8159.JPG")
    des = compute_descriptors([img1, img2])

def test_compute_codebook():
    descriptors = np.load("sample_descriptors.npy")
    vocabulary = compute_codebook(descriptors)

def test_classify():
    vocabulary = np.load("vocabulary.npy")
    model = np.load("model.npy")
    img = cv.imread("images\\web1.jpg")
    descriptors = compute_descriptors(img)
    histogram = compute_histogram(descriptors, vocabulary)
    assert classify(histogram, model) == 53