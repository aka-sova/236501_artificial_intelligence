import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import math
import pickle
import copy
import os

from DT_total import *

if __name__ == '__main__':

    # setting this parameter to 1 will force all trees and classifiers to be created from zero
    # setting to 0 will allow to use classifiers that were already generated and saved in the 'db' folder

    FORCE_NEW = 0



    # q6 

    tree_classifier = TreeClassifier()
    tree_classifier.load_data("train.csv", "train")
    depth_parameters = [9]
    accuracy_depth = []

    for d in depth_parameters:
        # print(f"Start for depth = {d}")
        tree_classifier.clean_tree()

        if  os.path.isfile(f"db/q6_dict_tree_depth_{d}.pkl") == False or FORCE_NEW:
            tree_classifier.build_epsilon_tree(depth = d)
            tree_classifier.save_tree(f"db/q6_dict_tree_depth_{d}.pkl")
        else:
            tree_classifier.load_tree(f"db/q6_dict_tree_depth_{d}.pkl")


        accuracy_depth.append(tree_classifier.classify("test.csv", use_epsilon = True, use_KNN = False, K = None))
        print(f"Epsilon tree accuracy(max depth = {d}) = {accuracy_depth[-1]}")