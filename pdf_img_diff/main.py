#!/usr/bin/env python3
"""
    Program to detect when images are used multiple times between
    different documents in a folder of PDF and docx files

    Should not be susceptible to the following:
        * Random noise
        * Cropping
        * Rotation
        * Colour space reduction
"""
from collections import defaultdict

import numpy as np
import cv2
from glob import glob
import argparse
from itertools import product, combinations
import fitz
from multiprocessing import Pool
import zipfile
from PIL import Image
import io
from matplotlib import pyplot as plt
from tqdm import tqdm
from pdf_img_diff.img_comp import compare_images_ssim, compare_images_hist, compare_images_kaze


def bytes_to_cv2img(b):
    """
    Converts a collection of bytes into an opencv image
    Trims surrounding whitespace using this technique https://stackoverflow.com/a/48399309/150851
    :param b:  the bytes to convert
    :return: an image in the RGB space (usually)
    """
    im = np.array(Image.open(io.BytesIO(b)))
    return im


def show_implot(arr):
    plt.figure()
    plt.imshow(arr)


def show_images(im, im2, im_name, im2_name, s, fa, fb, method):
    """

    :param im: first image
    :param im2:  second image
    :param im_name:  first image name
    :param im2_name:  second image name
    :param s: the matching score
    :param fa: the name of the file the first image came from
    :param fb:  the name of the file the second image came from
    :param method: the name of the method used to compare the image
    """
    f, ax = plt.subplots(1, 2)
    f.suptitle("{}: {}".format(method, s))
    ax[0].imshow(im)
    ax[0].set_title(fa.split('_')[0] + ' ' + im_name)
    ax[1].imshow(im2)
    ax[1].set_title(fb.split('_')[0] + ' ' + im2_name)


def get_images(files, verbose=False):
    """
    Compare images between files in files
    :param method:
    :param verbose:
    :param files:
    :return:
    """
    documents = defaultdict(dict)

    for fname in sorted(files):
        short_name = fname.split('/')[-1]
        if verbose:
            print("\nExtracting", short_name)
        if fname.endswith('.pdf'):
            document = fitz.open(fname)
            for page_num in range(len(document)):
                for im_num, im in enumerate(document.getPageImageList(page_num)):
                    x_ref = im[0]
                    pix = fitz.Pixmap(document, x_ref)
                    im_name = "{}_{}".format(page_num, im_num)
                    if verbose:
                        print(im_name, end=', ')
                    documents[short_name][im_name] = bytes_to_cv2img(pix.getPNGData())
            document.close()
        elif fname.endswith('.docx'):
            with zipfile.ZipFile(fname, 'r') as zf:
                for f in zf.namelist():
                    if any(f.endswith(ext) for ext in ('.png', '.jpg')):
                        if verbose:
                            print("\t", f)
                        documents[short_name][f] = bytes_to_cv2img(zf.read(f))
    return documents


def _do_comp(f, f2, a_img, b_img, a_name, b_name, func):
    a_gray = cv2.cvtColor(a_img, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b_img, cv2.COLOR_RGB2GRAY)
    if any(cv2.countNonZero(i) < 10 for i in (a_gray, b_gray)):
        return None

    args = {
        'a_img': a_img,
        'b_img': b_img,
        'a_gray': a_gray,
        'b_gray': b_gray,
        'a_name': f + a_name,
        'b_name': f2 + b_name,
    }
    score = func(**args)
    return f, f2, a_name, b_name, score


def compare_all(documents, method):
    """

    :param documents: a dictionary mapping file names to a dictionary of images
    :param method: the method used to compare images
    :return:
    """
    report = defaultdict(list)  # a dict of tuples, (f,f2) and image match scores
    func = {
        'hist': compare_images_hist,
        'ssim': compare_images_ssim,
        'kaze': compare_images_kaze
    }[method]
    combs = combinations(documents.keys(), 2)

    prog = tqdm(total=100, desc='Files: ')

    def comp_finished(arg):
        prog.update()
        if arg is None:
            return
        f, f2, a_name, b_name, score = arg
        report[(f, f2)].append([a_name, b_name, score])

    pool = Pool(processes=4)
    total = 0
    for (f, f2) in combs:
        images = documents[f]
        images2 = documents[f2]
        # compare all images with all images2
        for a_name, b_name in product(images, images2):
            total += 1
            a_img = documents[f][a_name]
            b_img = documents[f2][b_name]
            pool.apply_async(_do_comp, args=(f, f2, a_img, b_img, a_name, b_name, func),
                             callback=comp_finished)
    prog.total = total
    pool.close()
    pool.join()
    return report


def show_report(report, verbose, threshold, method, documents):
    """

    :param method:
    :param documents:
    :param threshold:
    :param verbose:
    :param report:
    :return:
    """
    print("\n\nFinal Report\n")
    for (fa, fb), matches in report.items():
        matches_above_threshold = [x for x in matches if x[-1] > threshold]
        if matches_above_threshold:
            print("Matches between:\n  {}\n  {}:".format(fa, fb))
            for m in matches_above_threshold:
                print("\t", m)
                if verbose:
                    show_images(documents[fa][m[0]], documents[fb][m[1]], m[0], m[1], m[2], fa, fb, method)
    if verbose:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Make a report about a folder of PDF or docx files and any suspiciously similar images or'
                    ' blocks of text that are images')
    # parser.
    parser.add_argument('folder', default='./', nargs='?', type=str, help='The folder of images to use')
    parser.add_argument('method', default='ssim', nargs='?', choices=['hist', 'kaze', 'ssim'],
                        help='Comparison method to use')
    parser.add_argument('threshold', default=None, nargs='?', type=float,
                        help='Similarity between images must exceed this value to count as a match')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    verbose = args.verbose
    _files = []
    if args.threshold is None:
        threshold = {
            'hist': 0.9999,
            'kaze': 0.8,
            'ssim': 0.95
        }[args.method]
    else:
        threshold = args.threshold
    for ftype in ('*.pdf', '*.docx'):
        _files.extend(glob(args.folder + '/' + ftype))
    images = get_images(_files, verbose=verbose)
    report = compare_all(images, method=args.method)
    show_report(report, verbose=verbose, threshold=threshold, method=args.method, documents=images)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
