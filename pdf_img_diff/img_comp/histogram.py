import cv2


def compare_images_hist(a_img, b_img, **kwargs):
    """
    Compare 2 images for similarity
    by comparing their histograms

    :param a_img: first image
    :param b_img: second image
    :return: similarity of the two images as a percentage
    """
    hists = [cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256]) for im in (a_img, b_img)]
    return cv2.compareHist(*hists, method=kwargs.get('method', cv2.HISTCMP_CORREL))
