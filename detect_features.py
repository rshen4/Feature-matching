import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from nonmaxsuppts import *

def detect_features(image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    Returns:
        pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
    """
    pixel_coords = list()

    (m, n) = image.shape
    Ix = np.zeros((m, n))
    Iy = np.zeros((m, n))
    sigma = 3
    filters.gaussian_filter(image, (sigma, sigma), (0, 1), Ix)
    filters.gaussian_filter(image, (sigma, sigma), (1, 0), Iy)
    A = Ix * Ix
    B = Ix * Iy
    C = Iy * Iy
    R = np.zeros((m, n))
    r = 3
    k = 0.05
    for j in xrange(r, n - r - 1):
        for i in xrange(r, m - r - 1):
            a = np.sum(A[i - r:i + r + 1, j - r:j + r + 1])
            b = np.sum(B[i - r:i + r + 1, j - r:j + r + 1])
            c = np.sum(C[i - r:i + r + 1, j - r:j + r + 1])
            R[i][j] = a * c - b * b - k * (a + c) ** 2
    min_R = np.min(R)
    R = (R - min_R) * 10.0 / (np.max(R) - min_R)
    radius = 5
    thres = np.mean(R) * 1.2
    pixel_coords = nonmaxsuppts(R, radius, thres)
    return pixel_coords
