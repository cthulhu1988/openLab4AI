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
        centroid_list.append(new_center)
    return centroid_list

def pass_through(centroids, data):
    for cent in centroids:
        cent.pointList = []
        cent.class_labels = []
    # iterate through rows
    for y in range(len(data.index)):
        num_list = [] ; class_label_list= []
        # strip off class label, store it in class_labels
        info = (data.iloc[y,:-1]) ; class_label = data.iloc[y,-1:]

        for l in class_label:
            class_label_list.append(int(l))
        # append each val to a list and covert to np array 1d
        for v in info:
            num_list.append(v)
        numLn = np.array(num_list)
        ##########################################
        # assign lines of data to each centroid based on how close they are .
        min_distance = 1000000
        ## for each centroid, find distance between centroid location and line
        for i, center in enumerate(centroids):
            delta = (distanceBetweenPoints(center.loc, numLn))
            if delta < min_distance:
                min_distance = delta
                idx = i
                lne = numLn
                cls = class_label_list
        centroids[idx].pointList.append(lne)
        centroids[idx].class_labels.append(cls)
        idx, lne, cls = 0,0, -1 ; min_distance = 1000000

def distanceBetweenPoints(x1, x2):
    # nifty numpy function found in the literatrure
    return np.linalg.norm(x2 - x1)

def centroids_all_true(centroid_vector):
    is_done = True
    for counter, center in enumerate(centroid_vector):
        if (center.done == False):
            is_done = False
    return is_done

def main():
    #################SET UP DATA #############################
    random_seed = int(sys.argv[1])
    num_clusters = int(sys.argv[2])
    training_data = sys.argv[3] # string
    testing_data = sys.argv[4] # string

    training_data_df = pd.read_csv(training_data, sep=" ", header=None)
    num_train_rows = len(training_data_df.index)
    num_train_attributes = len(training_data_df.columns)-1
    testing_data_df = pd.read_csv(testing_data, sep=" ", header=None)

    ################# PICK CENTROIDS ######################################################
    centroid_vector = pick_centroids(training_data_df, num_clusters)

    #################### LOOP THROUGH AND ADJUST CENTROIDS #######################################
    done = False
    max_iter = 0
    while not done:
        max_iter +=1
        if centroids_all_true(centroid_vector) or max_iter > 300:
            done = True
        else:
            pass_through(centroid_vector, training_data_df)
            for ii in centroid_vector:
                ii.update_position(num_train_attributes)
    ########### MAJORITY VOTE CLASS LABEL #####################
    for i in centroid_vector:
        i.name = i.majority_cls()



    # Finally, the program will calculate the closest vector to each testing example and determine if the cluster label and the testing example label match (a correct classification).
    # The program should then output the number of testing examples classified correctly by the K-means clustering

class Centroid:
    def __init__(self, loc):
        self.loc = loc
        self.pointList= []
        self.name = ""
        self.done = False
        self.numCols = 0
        self.class_labels = []

    def update_position(self, num):
        # number of attributes
        self.numCols = num
        # make sure pointList is not empty, temp save old location
        temp_old = self.loc
        if len(self.pointList) > 0:
            # turn all points into a DataFrame to get mean
            df = pd.DataFrame(self.pointList)
            # for range of attributes
            new_loc_list = []
            for x in range(self.numCols):
                new_loc_list.append(df[x].mean())
            new_loc_np = np.array(new_loc_list)
            self.check_done(self.loc, new_loc_np)
            self.loc = new_loc_np

    def check_done(self, old_loc, new_loc):
            sum_total = np.linalg.norm(old_loc-new_loc)
            #print("sum total {}".format(sum_total))
            if sum_total < 0.01:
                self.done = True

    def print_points(self):
        print("NAME  {} and LENG {}".format(self.name, len(self.pointList)))
        for i in self.pointList:
            print("Line {}".format(i))

    def majority_cls(self):
        max = 0
        maj = -1
        a = np.array(self.class_labels)
        unique, counts = np.unique(a, return_counts=True)
        values = dict(zip(unique, counts))
        for k,v in values.items():
            if v > max:
                max = v
                maj = k
        return maj


if __name__ == '__main__':
    main()
