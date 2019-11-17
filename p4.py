#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math, sys
from collections import Counter

def distanceBetweenPoints(x1, x2):
    return np.sqrt(np.sum( (x1 - x2)**2 ))


def main():
    random_seed = int(sys.argv[1])
    num_clusters = int(sys.argv[2])
    training_data = sys.argv[3] # string
    testing_data = sys.argv[4] # string

    train_list = []  ; test_list = []
    train_label = []  ; test_label = []
    # The program should read in the training data file (one training example per line, see below)
    train_lines = np.loadtxt(training_data)
    for line in train_lines:
        # strip off class label
        label = line[-1]
        line = line[:-1]
        train_list.append(line)
        train_label.append(label)

    #print(train_label)

    # The program should read in the testing data file (one testing example per line, see below)
    test_lines = np.loadtxt(testing_data)
    for line in test_lines:
        label = line[-1]
        line = line[:-1]
        test_list.append(line)
        test_label.append(label)


    # The program should perform a K-means clustering by first initializing K centroid vectors (no class labels included) using K random examples from the training data
    initializing_kNearest = KNearest(num_clusters)
    initializing_kNearest.fit(train_list, train_label)
    predictions = initializing_kNearest.predictClass(test_list)

    acc = np.sum(predictions == test_label) / len(predictions)
    print(acc)

    # The program should then determine the closest vector to each training example (using Euclidean distance), and create a new set of vectors by averaging the feature vectors of the closest training examples.
    # The program should repeat the previous step until the centroid vectors no longer change (i.e. until all training examples are assigned to the same vector on two consequtive iterations)
    # Once the mean cluster vectors have been calculated, a class label will be assigned to each vector by taking a majority vote amongst it's assigned examples from the training set (ties will be broken by preferring the smallest integer class label).
    # Finally, the program will calculate the closest vector to each testing example and determine if the cluster label and the testing example label match (a correct classification).
    # The program should then output the number of testing examples classified correctly by the K-means clustering
class KNearest:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.yTrain = y


    def predictClass(self,X):
        # fro each sample we call helper pethod
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        distance = [distanceBetweenPoints(x, x_train) for x_train in self.X_train]
        # compute distances
        k_indices = np.argsort(distance)[:self.k]
        k_nearLabels =  [self.yTrain[i] for i in k_indices]
        # get k KNearest samples and labels

        # majority vote for most common class label
        most_common = Counter(k_nearLabels).most_common(1)
        return most_common[0][0]




if __name__ == '__main__':
    main()
