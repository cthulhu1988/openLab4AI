#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math, sys
import random as rd
from collections import Counter
centroid_list = []

def pick_centroids(data, num_clusters):
    num_lines = len(data.index)
    for x in range(num_clusters):
        number_list = []
        random_line = rd.randint(0,num_lines)
        info = (data.iloc[random_line,:-1])
        for val in info:
            number_list.append(val)
        number_list = np.array(number_list)
        new_center = Centroid(number_list)
        new_center.name = x
        centroid_list.append(new_center)
    return centroid_list

def first_pass(centroids, data):
    num_lines = len(data.index)
    for y in range(num_lines):
        num_list = []
        info = (data.iloc[y,:-1])
        for v in info:
            num_list.append(v)
        numLn = np.array(num_list)
        # assign lines of data to each centroid based on how close they are .
        min_distance = 1000000
        for i, center in enumerate(centroids):
            delta = (distanceBetweenPoints(center.loc, numLn))
            if delta < min_distance:
                min_distance = delta
                idx = i
                lne = numLn
        centroids[idx].pointList.append(lne)
        idx, lne = 0,0 ; min_distance = 1000000

def distanceBetweenPoints(x1, x2):
    #return np.sqrt(np.sum( (x1 - x2)**2 ))
    return np.linalg.norm(x2 - x1)

def main():
    random_seed = int(sys.argv[1])
    num_clusters = int(sys.argv[2])
    training_data = sys.argv[3] # string
    testing_data = sys.argv[4] # string

    # The program should read in the training data file (one training example per line, see below)
    training_data_df = pd.read_csv(training_data, sep=" ", header=None)
    num_train_rows = len(training_data_df.index)
    num_train_attributes = len(training_data_df.columns)-1

    # The program should read in the testing data file (one testing example per line, see below)
    testing_data_df = pd.read_csv(testing_data, sep=" ", header=None)


    # The program should perform a K-means clustering by first initializing K centroid vectors (no class labels included) using K random examples from the training data
    centroid_vector = pick_centroids(training_data_df, num_clusters)

    # first pass sets up the inital clustering.
    first_pass(centroid_vector, training_data_df)
    for i in centroid_vector:
        print("BEFORE PASS ",i.loc)

    for j in centroid_vector:
        print("After Pass ")
        j.update_position(num_train_attributes)

    for x in centroid_vector:
        print("AFTER PASS ",x.loc)

    done = False
    max_iter = 0
    while not done:
        max_iter +=1
        first_pass(centroid_vector, training_data_df)
        print()
        for i in centroid_vector:
            print("BEFORE PASS ",i.loc)
        for ii in centroid_vector:
            j.update_position(num_train_attributes)
        if centroid_vector[0].done == True or max_iter > 300:
            done = True



    # The program should then determine the closest vector to each training example (using Euclidean distance), and create a new set of vectors by averaging the feature vectors of the closest training examples.
    # The program should repeat the previous step until the centroid vectors no longer change (i.e. until all training examples are assigned to the same vector on two consequtive iterations)
    # Once the mean cluster vectors have been calculated, a class label will be assigned to each vector by taking a majority vote amongst it's assigned examples from the training set (ties will be broken by preferring the smallest integer class label).
    # Finally, the program will calculate the closest vector to each testing example and determine if the cluster label and the testing example label match (a correct classification).
    # The program should then output the number of testing examples classified correctly by the K-means clustering

class Centroid:
    def __init__(self, loc):
        self.loc = loc
        self.pointList= []
        self.name = ""
        self.done = False
        self.numCols = 0

    def update_position(self, num):
        self.numCols = num
        if len(self.pointList) > 0:
            df = pd.DataFrame(self.pointList)
            for j in range(num):
                new_loc = (df[j].mean())
                self.check_done(self.loc, new_loc)
                self.loc[j] = new_loc

    def check_done(self, loc, new_loc):
            sum_total = np.linalg.norm(loc-new_loc)
            if sum_total < 0.01:
                self.done = True

    def print_points(self):
        print("NAME ",self.name)
        for i in self.pointList:
            print("Line {}".format(i))



if __name__ == '__main__':
    main()
