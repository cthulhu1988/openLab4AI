#!/usr/bin/env python3
import numpy as np
import sys
from sklearn.preprocessing import normalize


def main():

    file1 = sys.argv[1]
    data = np.loadtxt(file1)

    mean = np.mean(data)
    mean_data = data-mean
    #print(mean_data)
    normalized = mean_data / np.sqrt(np.sum(mean_data**2))
    np.savetxt("normalized.txt",normalized,delimiter=" ")

if __name__ == '__main__':
    main()
