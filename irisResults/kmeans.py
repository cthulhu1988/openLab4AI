#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
import random
centroid_list = []

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
    centroid_vector = pick_centroids(training_data_df, num_clusters, random_seed)
    #################### LOOP THROUGH AND ADJUST CENTROIDS #######################################
    done = False
    max_iter = 0
    while not done:
        max_iter +=1
        if centroids_all_true(centroid_vector) or max_iter > 20:
            done = True
        else:
            pass_through(centroid_vector, training_data_df)
            for centroid_obj in centroid_vector:
                centroid_obj.update_position(num_train_attributes)
    ########### MAJORITY VOTE CLASS LABEL #####################
    for i in centroid_vector:
        i.name = i.majority_cls()

    ############CLASSIFY DATA #################################
    total_correct, acc = classify_test_data(testing_data_df, centroid_vector)
    #print("accuracy {} and correct {}".format(acc, total_correct))
    print(total_correct)
######################################################
################### FUNCTIONS ########################
######################################################
def classify_test_data(data, centroid_vector):
    total = 0 ; counter = 0 ; min = 1000000
    # get two datasets, one with data and one with class labels.
    info = (data.iloc[:,:-1]) ; class_label = data.iloc[:,-1:]
    # convert them to np arrays for funsies.
    np_info = np.array(info) ; np_cls = np.array(class_label)
    #iterate through each line
    for idx, line in enumerate(np_info):
        total +=1
        classified_center = -1
        for j, center in enumerate(centroid_vector):
            delta = distanceBetweenPoints(center.loc, line)
            #print("delta {}".format(delta))
            if delta < min:
                min = delta
                classified_center = j
        min = 1000000
        predicted_label = int(centroid_vector[classified_center].name)
        actual_label = int((np_cls[idx]))
        if(predicted_label == actual_label):
            counter +=1
    return counter, (counter / total)


def pick_centroids(data, num_clusters, random_seed):
    num_lines = len(data.index)
    for x in range(num_clusters):
        # get random line number
        #random.seed = int(random_seed)
        random_line_idx = np.random.randint(1,num_lines-1)
        #random_line_idx = random.randint(1,num_lines-1)
        info = np.array(data)
        info = info[random_line_idx,:-1]
        # the pandas way....
        #info = (data.iloc[random_line_idx,:-1])
        number_list = np.array([val for val in info ])
        new_center = Centroid(number_list)
        centroid_list.append(new_center)
    return centroid_list

def pass_through(centroids, data):
    for cent in centroids:
        cent.pointList = []
        cent.class_labels = []
    # iterate through rows
    for y in range(len(data.index)):
        # strip off class label, store it in class_labels
        info = (data.iloc[y,:-1]) ; class_label = data.iloc[y,-1:]
        # list comprehensions save a bit of space
        class_label_list = [int(l) for l in class_label]
        num_list = [v for v in info]
        # convert to np for eash of manipulation
        numLn = np.array(num_list)
        ##########################################
        # assign lines of data to each centroid based on how close they are .
        min_distance = 1000000
        ## for each centroid, find distance between centroid location and line
        for i, center in enumerate(centroids):
            delta = (distanceBetweenPoints(center.loc, numLn))
            if delta < min_distance:
                min_distance = delta
                idx = i ; lne = numLn ; cls = class_label_list
        centroids[idx].pointList.append(lne)
        centroids[idx].class_labels.append(cls)
        idx, lne, cls = 0,0,-1 ; min_distance = 1000000

def distanceBetweenPoints(x1, x2):
    # nifty numpy function found in the literature
    return np.linalg.norm(x2 - x1)

def centroids_all_true(centroid_vector):
    is_done = True
    for counter, center in enumerate(centroid_vector):
        if (center.done == False):
            is_done = False
    return is_done

######################################################
################### CLASSES ##########################
######################################################
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
            new_loc_list = [df[x].mean() for x in range(self.numCols)]
            new_loc_np = np.array(new_loc_list)
            self.check_done(self.loc, new_loc_np)
            self.loc = new_loc_np

    def check_done(self, old_loc, new_loc):
            sum_total = np.linalg.norm(old_loc-new_loc)
            if sum_total < 0.01:
                self.done = True

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
