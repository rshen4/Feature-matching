import cv2
import matplotlib.pyplot as plt
from detect_features import *
from match_features import *
from ssift_descriptor import *
from match_ssifts import *
from compute_affine_xform import *
from compute_proj_xform import *


def main():
    path1 = 'bikes1.png'
    path2 = 'bikes2.png'
    image1 = cv2.imread(path1, 0)
    image2 = cv2.imread(path2, 0)
    (m, n) = image1.shape
    if image2.shape != image1.shape:
        image2 = cv2.resize(image2, (n, m))
    features1 = detect_features(image1)
    features2 = detect_features(image2)
    ssift1 = ssift_descriptor(features1, image1)
    ssift2 = ssift_descriptor(features2, image2)
    matches = match_ssifts(features1, features2, ssift1, ssift2)
    img1 = plt.imread(path1)
    img2 = plt.imread(path2)
    if img2.shape != img1.shape:
        img2 = cv2.resize(img2, (n, m))
    SbS = np.concatenate((img1, img2), axis=1)
    for (f1, f2) in matches:
        (y1, x1) = features1[f1]
        (y2, x2) = features2[f2]
        x2 += n
        plt.plot(x1, y1, 'ob')
        plt.plot(x2, y2, 'ob')
        cv2.line(SbS, (x1, y1), (x2, y2), (0, 255, 0))
    plt.axis('off')
    plt.imshow(SbS)
    plt.show()
    affine_xform = compute_affine_xform(matches, features1, features2, image1, image2)
    pad_width = 60
    npad = ((pad_width, pad_width), (pad_width, pad_width))
    img1 = np.pad(image1, pad_width=npad, mode='constant', constant_values=0)
    img2 = np.pad(image2, pad_width=npad, mode='constant', constant_values=0)
    (m, n) = img1.shape
    warp_img1 = cv2.warpAffine(img1, affine_xform[:2], (n, m))
    img = cv2.addWeighted(warp_img1, 0.5, img2, 0.5, 0)
    plt.axis('off')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
if __name__ == '__main__':
    main()