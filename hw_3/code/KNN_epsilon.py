import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import math
import pickle
import copy
import os

from DT_total import *


def check_tree_valid(node):
    """ Each leaf either has label or nodes """
    

    if node.is_leaf: 
        assert node.label is not None

    if node.is_branch:
        assert node.attribute is not None
        assert node.thresholds is not None
        assert node.nodes != {}

        for child_node in node.nodes:
            check_tree_valid(node.nodes[child_node])

if __name__ == '__main__':

    # setting this parameter to 1 will force all trees and classifiers to be created from zero
    # setting to 0 will allow to use classifiers that were already generated and saved in the 'db' folder

    FORCE_NEW = 1

    # q11 

    depth = 9
    k = 9
    tree_classifier = TreeClassifier()
    tree_classifier.load_data("train.csv", "train")


    if  os.path.isfile(f"db/q11_dict_tree_KNN.pkl") == False or FORCE_NEW:

        tree_classifier.clean_tree()
        tree_classifier.normalize_train_data()
        tree_classifier.build_epsilon_tree(depth = depth)
        tree_classifier.save_tree(f"db/q11_dict_tree_KNN.pkl")
    else:
        tree_classifier.normalize_train_data()
        tree_classifier.load_tree(f"db/q11_dict_tree_KNN.pkl")

    
    # check_tree_valid(tree_classifier.root_node)

    accuracy = tree_classifier.classify("test.csv", use_epsilon = True, use_KNN = True, K = 9)
    print(f"Epsilon tree accuracy with KNN (max depth = {depth}) = {accuracy}")