#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math, sys

def distanceBetweenPoints(x1, x2):
    return np.sqrt(np.sum( (x1 - x2)**2 ))


def main():
    file1 = sys.argv[1]
    dataset = pd.read_csv(file1, sep=" ", header=None)
    print(dataset)


if __name__ == '__main__':
    main()


class KNearest:
    def __init__(self, k=3):
        self.k = k

    def predictClass(self,point):
        predicted_labels = [self._predict(x) for x in X]

    def _predict(self):
        # compute distances

        # get k KNearest samples and labels

        # majority vote for most common class label

    def fit(self, point, y):
        self.pointTrain = point
        self.yTrain = y
