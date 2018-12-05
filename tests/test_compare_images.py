from collections import defaultdict
from math import ceil
from pdf_img_diff.main import compare_images
from glob import glob
import cv2
import matplotlib.pyplot as plt


def test_compare_images():
    # load all the images
    images = {}
    histograms = {}
    for fname in glob('./data/images/*.jpg'):
        im = cv2.imread(fname)
        images[fname] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        histograms[fname] = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    test_fname = list(iter(images))[-1]

    results = defaultdict(dict)
    OPENCV_METHODS = (
        ("Correlation", cv2.HISTCMP_CORREL),
        ("Chi-Squared", cv2.HISTCMP_CHISQR),
        ("Chi-Squared Alt", cv2.HISTCMP_CHISQR_ALT),
        ("Intersection", cv2.HISTCMP_INTERSECT),
        ("Hellinger", cv2.HISTCMP_HELLINGER),
        ('KL-Div', cv2.HISTCMP_KL_DIV)
    )
    for f, im in images.items():
        for method in OPENCV_METHODS:
            results[method[0]][(test_fname, f)] = cv2.compareHist(histograms[test_fname], histograms[f], method[1])

    fig = plt.figure("Query")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images[test_fname])
    plt.axis("off")
    for method in OPENCV_METHODS:
        # initialize the results figure
        fig = plt.figure("Results: "+method[0])
        fig.suptitle(method[0], fontsize=20)

        # loop over the results
        cols = 4
        rows = ceil(len(images) / float(cols))
        for i, (files, comp_score) in enumerate(results[method[0]].items()):
            # show the result
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title("%s:\n %.2f" % (files[1].split('/')[-1], comp_score))
            plt.imshow(images[files[1]])
            plt.axis("off")

        # show the OpenCV methods
    plt.show()
