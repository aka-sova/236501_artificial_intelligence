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


    # q2 
    tree_classifier = TreeClassifier()
    tree_classifier.load_data("train.csv", "train")
    tree_classifier.clean_tree()

    if  os.path.isfile("db/q2_dict_tree.pkl") == False or FORCE_NEW:
        tree_classifier.build_tree()
        tree_classifier.save_tree("db/q2_dict_tree.pkl")
    else:
        tree_classifier.load_tree("db/q2_dict_tree.pkl")


    accuracy = tree_classifier.classify("test.csv", False, False, None)
    print(f"Regular accuracy = {accuracy}")


    # q3

    tree_classifier = TreeClassifier()
    tree_classifier.load_data("train.csv", "train")
    pruning_parameters = [3, 9, 27]
    accuracy_pruning = []

    for x in pruning_parameters:
        # print(f"Start for pruning p = {x}")
        tree_classifier.clean_tree()

        if  os.path.isfile(f"db/q3_dict_tree_pruned_{x}.pkl") == False or FORCE_NEW:
            tree_classifier.build_pruned_tree(pruning_p = x)
            tree_classifier.save_tree(f"db/q3_dict_tree_pruned_{x}.pkl")
        else:
            tree_classifier.load_tree(f"db/q3_dict_tree_pruned_{x}.pkl")


        accuracy_pruning.append(tree_classifier.classify("test.csv", False, False, None))
        print(f"Pruned tree accuracy(P = {x}) = {accuracy_pruning[-1]}")



    plt.plot(pruning_parameters, accuracy_pruning)
    plt.xlabel('Pruning parameter')
    plt.ylabel('Accuracy')
    plt.savefig('db/q3.png')