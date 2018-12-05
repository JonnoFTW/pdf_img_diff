import cv2


def compare_images_hist(a_img, b_img, **kwargs):
    """
    Compare 2 images for similarity
    Returns an int between 0 and 100
    100 is an identical copy, 0 is completely different

    Should not be susceptible to the following:
        * Random noise
        * Cropping
        * Rotation
        * Colour space reduction
    :param a_img: first image
    :param b_img: second image
    :return: similarity of the two images as a percentage
    """
    hists = [cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256]) for im in (a_img, b_img)]
    return cv2.compareHist(*hists, method=kwargs['method'])
