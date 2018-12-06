# PDF Image Diff

A simple tool that checks a folder of PDF files to see if there are similar or matching images occuring between documents.
### Installation
```bash
pip install git+https://github.com/JonnoFTW/pdf_img_diff.git
```

### Usage
```
pdf-img-diff /path/to/folder/ -v 
```

Will output a report saying which files have matching images between them.

If you want you can also specify a method


## Methods

Currently these methods are supported:

* SSIM: Structural similarity index. This is the default and my recommendation
* [KAZE](https://www.doc.ic.ac.uk/~ajd/Publications/alcantarilla_etal_eccv2012.pdf)  
* Histogram comparison: doesn't work very well

## Future work

Look into comparisons of feature points using:

* ORB
* SIFT
* SURF
* BRIEF

This might be useful https://www.researchgate.net/publication/323561586_A_comparative_analysis_of_SIFT_SURF_KAZE_AKAZE_ORB_and_BRISK