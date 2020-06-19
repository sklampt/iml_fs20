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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())

# initialize the index dictionary to store the image name
# and corresponding histograms and the images dictionary
# to store the images themselves
index = {}
images = {}
counter = 0

data = pd.read_csv(
    'data/test_triplets.txt',
    dtype=str,
    sep=" ",
    header=None,
    names=['A','B','C'],
    # nrows=1000,
)

# loop over the image paths
for index, row in data.iterrows():
    # extract the image filename (assumed to be unique) and
    # load the image, updating the images dictionary
    compare = {}
    for label, value in row.items():
        filename = "data/food/" + value + ".jpg"
        # print(filename)
        image = cv2.imread(filename, 1)
        images[filename] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update
        # the index
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        compare[label] = hist
    
    dB = cv2.compareHist(compare['A'], compare['B'], cv2.HISTCMP_BHATTACHARYYA)
    dC = cv2.compareHist(compare['A'], compare['C'], cv2.HISTCMP_BHATTACHARYYA)

    # print(dB, dC)

    if (dB < dC):
        print(1)
        counter += 1
    else:
        print(0)

# print(counter / 100)