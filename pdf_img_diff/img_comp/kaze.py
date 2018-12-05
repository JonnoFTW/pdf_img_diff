import cv2
import numpy as np
import scipy.spatial

memo = {}


def extract_features(img, im_name, vector_size=16):
    if im_name in memo:
        return memo[im_name]
    alg = cv2.KAZE_create()
    kps = alg.detect(img)
    kps = sorted(kps, key=lambda x: x.response)[:vector_size]
    kps, dsc = alg.compute(img, kps)
    dsc = dsc.flatten()
    needed_size = (vector_size * 64)
    if dsc.size < needed_size:
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    memo[im_name] = dsc
    return dsc


def compare_images_kaze(a_img, b_img, a_name, b_name, vector_size=16, **kwargs):
    """
    Compare images using KAZE technique
    :param a_img:
    :param b_img:
    :param vector_size:
    :return:
    """
    a_features, b_features = (extract_features(x, y, vector_size) for x, y in ((a_img, a_name), (b_img, b_name)))
    distance = scipy.spatial.distance.cosine(a_features, b_features)
    return 1 - distance
