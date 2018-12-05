import numpy as np
import cv2
from glob import glob
import argparse
from itertools import product
import fitz
import zipfile
from PIL import Image
import io
from matplotlib import pyplot as plt
from pdf_img_diff.img_comp import compare_images_ssim, compare_images_hist, compare_images_kaze


def bytes_to_cv2img(b):
    return np.array(Image.open(io.BytesIO(b)))


def show_images(im, im2, im_name, im2_name, s, fa, fb, method):
    f, ax = plt.subplots(1, 2)
    f.suptitle("{}: {}".format(method, s))
    ax[0].imshow(im)
    ax[0].set_title(fa.split('_')[0] + ' ' + im_name)
    ax[1].imshow(im2)
    ax[1].set_title(fb.split('_')[0] + ' ' + im2_name)


def do_report(files, method='hist', verbose=False, threshold=0.9):
    """
    Compare images between files in files
    :param verbose:
    :param files:
    :return:
    """
    documents = {fname.split('/')[-1]: {} for fname in files}

    for fname in sorted(files):
        short_name = fname.split('/')[-1]
        if verbose:
            print("Extracting", short_name)
        if fname.endswith('.pdf'):
            document = fitz.open(fname)
            for page_num in range(len(document)):
                if verbose:
                    print("\tPage:", page_num, end=': ')
                for im_num, im in enumerate(document.getPageImageList(page_num)):
                    x_ref = im[0]
                    pix = fitz.Pixmap(document, x_ref)
                    im_name = "{}_{}".format(page_num, im_num)
                    if verbose:
                        print(im_num, end=', ')
                    documents[short_name][im_name] = bytes_to_cv2img(pix.getPNGData())
                if verbose:
                    print()
            document.close()
        elif fname.endswith('.docx'):
            with zipfile.ZipFile(fname, 'r') as zf:
                for f in zf.namelist():
                    if any(f.endswith(ext) for ext in ('.png', '.jpg')):
                        if verbose:
                            print("\t", f)
                        documents[short_name][f] = bytes_to_cv2img(zf.read(f))
    report = {}  # a dict of tuples, (f,f2) and potential image matches
    for f in documents.keys():
        for f2 in documents.keys():
            f, f2 = sorted([f, f2])
            images = documents[f]
            images2 = documents[f2]
            if f != f2:
                t = (f, f2)
                if t not in report:
                    # compare all images with all images2
                    matches = []
                    for a_name, b_name in product(images, images2):
                        a_img = documents[f][a_name]
                        b_img = documents[f2][b_name]
                        a_gray = cv2.cvtColor(a_img, cv2.COLOR_RGB2GRAY)
                        b_gray = cv2.cvtColor(b_img, cv2.COLOR_RGB2GRAY)
                        if any(cv2.countNonZero(i) < 10 for i in (a_gray, b_gray)):
                            continue
                        func = {
                            'hist': compare_images_hist,
                            'ssim': compare_images_ssim,
                            'kaze': compare_images_kaze
                        }[method]
                        args = {
                            'a_img': a_img,
                            'b_img': b_img,
                            'a_name': f + a_name,
                            'b_name': f2 + b_name,
                        }
                        score = func(**args)
                        if score > threshold:
                            matches.append([a_name, b_name, score])

                    report[t] = matches
    for (fa, fb), matches in report.items():
        if matches:
            print("Matches between: \n{}\n{}:".format(fa, fb))
            for m in matches:
                print("\t", m)
                if verbose:
                    show_images(documents[fa][m[0]], documents[fb][m[1]], m[0], m[1], m[2], fa, fb, method)
    if verbose:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Make a report about a folder of PDF or docx files and any suspiciously similar images or blocks of text that are images')
    # parser.
    parser.add_argument('folder', default='./', nargs='?', type=str, help='The folder of images to use')
    parser.add_argument('method', default='kaze', nargs='?', choices=['hist', 'kaze', 'ssim'],
                        help='Comparison method to use')
    parser.add_argument('threshold', default=0.8, nargs='?', type=float,
                        help='Similarity between images must exceed this value to count as a match')
    parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    _files = []
    for ftype in ('*.pdf', '*.docx'):
        _files.extend(glob(args.folder + '/' + ftype))
    do_report(_files, method=args.method, verbose=args.verbose, threshold=args.threshold)


if __name__ == "__main__":
    main()
