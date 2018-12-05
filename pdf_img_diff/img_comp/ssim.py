# from skimage.measure import compare_ssim


def compare_images_ssim(**kwargs):
    """
    Compare 2 images using patented SSIM technique:
    https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
    :param a:
    :param b:
    :return:
    """
    a = kwargs['a_name']
    b = kwargs['b_name']
    (score, diff) = compare_ssim(a, b, full=True)
    diff = (diff * 255).astype("uint8")
    return score
