import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def compute_proj_xform(matches,features1,features2,image1,image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        proj_xform (numpy.ndarray): a 3x3 Projective transformation matrix between the two images, computed using the matches.
    """
    proj_xform = np.zeros((3,3))
    s = 5
    match_inds = [i for i in xrange(len(matches))]
    matched = []
    N = 30000
    max_in = -1
    for n in xrange(N):
        random.shuffle(match_inds)
        inds = match_inds[:s]
        if inds in matched:
            n -= 1
            continue
        matched.append(inds)
        A = list()
        b = list()
        for i in inds:
            (f1, f2) = matches[i]
            (y1, x1) = features1[f1]
            (y2, x2) = features2[f2]
            A.append([x1, y1, 1, 0, 0, 0, 0, 0, 0])
            A.append([0, 0, 0, x1, y1, 1, 0, 0, 0])
            A.append([0, 0, 0, 0, 0, 0, x1, y1, 1])
            b.append(x2)
            b.append(y2)
            b.append(1)
        h = np.linalg.lstsq(A, b)[0]
        h = np.resize(h, (3, 3))
        ex = list()
        ey = list()
        for j in match_inds:
            if j in inds:
                continue
            (f1, f2) = matches[j]
            (y1, x1) = features1[f1]
            (y2, x2) = features2[f2]
            [y, x, one] = np.dot(h, [y1, x1, 1])
            ex.append((x - x2) ** 2)
            ey.append((y - y2) ** 2)
        [a, b] = np.polyfit(ex, ey, 1)
        num_in = 0
        err_thres = 30
        for x, y in zip(ex, ey):
            d = abs(a * x - y + b) / np.sqrt(a * a + 1)
            if d <= err_thres:
                num_in += 1
        if num_in > max_in:
            max_in = num_in
            proj_xform = h
    return proj_xform
