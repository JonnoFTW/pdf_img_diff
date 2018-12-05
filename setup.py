from setuptools import setup, find_packages

setup(
    name='PDF_Img_Diff',
    version='1.0',
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='Compare and report on duplicated images between pdf and docx files',
    classifiers=[
        "Programming Language :: Python",
    ],
    author="Jonathan Mackenzie",
    url="https://github.com/jonnoftw/pdf_img_diff",
    keywords=["opencv pdf docx "],
    long_description=open('README.md').read(),
    install_requires=[x.strip() for x in open('requirements.txt').readlines()],
    entry_points={
        'console_scripts': ['pdf-img-diff=pdf_img_diff:main'],
    }
)
