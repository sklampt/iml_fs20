# USAGE
# python compare.py --dataset images

# import the necessary packages
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import glob
import cv2
import progressbar

NUM_ROWS=1000
# NUM_ROWS=59543

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
bar = progressbar.ProgressBar(max_value=NUM_ROWS)


data = pd.read_csv(
    'data/train_triplets.txt',
    dtype=str,
    sep=" ",
    header=None,
    names=['A','B','C'],
    nrows=NUM_ROWS,
)

methods = [
    cv2.HISTCMP_CORREL,
    cv2.HISTCMP_CHISQR,
    cv2.HISTCMP_INTERSECT,
    cv2.HISTCMP_BHATTACHARYYA,
    cv2.HISTCMP_CHISQR_ALT,
    cv2.HISTCMP_KL_DIV
]

correct_count = [
    0,
    0,
    0,
    0,
    0,
    0
]
# loop over the image paths
for index, row in data.iterrows():
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary
    compare = {}
    for label, value in row.items():
        filename = "data/food/" + value + ".jpg"
        # print(filename)
        image = cv2.imread(filename, 1)

        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        compare[label] = hist

    for method in methods:
        dB = cv2.compareHist(compare['A'], compare['B'], method)
        dC = cv2.compareHist(compare['A'], compare['C'], method)

        # print(dB, dC)

        if (dB < dC):
            # print(1)
            correct_count[method] += 1
        else:
            # print(0)
            pass
    
    bar.update(index)

print()
print("Results:")
for method in methods:
    print(method, "accuray:", correct_count[method] / NUM_ROWS)