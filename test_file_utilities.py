import cv2 as cv

from file_utilities import all_images_from_folder

def test_all_images_from_folder():
    img1 = cv.imread("images/8159.JPG")
    img2 = cv.imread("images/9039.JPG")
    img3, img4 = all_images_from_folder("images/")
    assert (img1.shape, img2.shape) == (img3.shape, img4.shape)
