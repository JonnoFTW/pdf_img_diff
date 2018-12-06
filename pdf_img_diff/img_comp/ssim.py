import cv2
from skimage.measure import compare_ssim


def compare_images_ssim(**kwargs):
    """
    Compare 2 images using patented SSIM technique:
    https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    :param a:
    :param b:
    :return:
    """
    a = kwargs['a_gray']
    b = kwargs['b_gray']
    if a.size > b.size:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    (score, diff) = compare_ssim(a, b, full=True)
    diff = (diff * 255).astype("uint8")
    return score
