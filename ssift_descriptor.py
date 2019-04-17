import cv2
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import filters
def ssift_descriptor(feature_coords,image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords (list of tuples): list of (row,col) tuple feature coordinates from image
        image (numpy.ndarray): The input image to compute ssift descriptors on. Note: this is NOT the image name or image path.
    Returns:
        descriptors (dictionary{(row,col): 128 dimensional list}): the keys are the feature coordinates (row,col) tuple and
                                                                   the values are the 128 dimensional ssift feature descriptors.
    """
	
    descriptors = dict()
    (m, n) = image.shape
    Ix = np.zeros((m, n))
    Iy = np.zeros((m, n))
    filters.gaussian_filter(image, (2, 2), (0, 1), Ix)
    filters.gaussian_filter(image, (2, 2), (1, 0), Iy)

    orientations = np.arctan2(Iy, Ix)
    magnitudes = np.sqrt(Ix ** 2 + Iy ** 2)
    hog_range = [i * 2 * np.pi / 8.0 for i in xrange(8)]
    for (i, j) in feature_coords:
        if i < 20 or i > m - 21 or j < 20 or j > n - 21:
            continue
        orient_win = orientations[i - 20:i + 21, j - 20:j + 21]
        mag_win = magnitudes[i - 20:i + 21, j - 20:j + 21]
        grange = [k * 10 for k in xrange(4)]
        ssift = list()
        for g1 in grange:
            for g2 in grange:
                orients = orient_win[g1:g1 + 10, g2:g2 + 10]
                mags = mag_win[g1:g1 + 10, g2:g2 + 10]
                hogs = [0] * 8
                for y in xrange(10):
                    for x in xrange(10):
                        orient = orients[x][y]
                        mag = mags[x][y]
                    ind = np.argmin(abs(hog_range - orient))
                    hogs[ind] += mag
                if len(ssift) == 0:
                    ssift = hogs
                else:
                    ssift += hogs
        descriptors[(i, j)] = ssift
    for coord, ssift in descriptors.items():
        unit_ssift = np.array(ssift) / sum(ssift)
        thres_ssift = [(s if s < 0.2 else 0.2) for s in unit_ssift]
        new_ssift = np.array(thres_ssift) / sum(thres_ssift)
        descriptors[coord] = new_ssift
    return descriptors
