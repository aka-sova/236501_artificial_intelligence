

from decorators import time_it

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import math
import pickle
import copy


class Node():
    def __init__(self):
        self.is_branch = False
        self.attribute = None
        self.thresholds = None
        self.nodes = {}             # dict of type  


        self.label = None
        self.is_leaf = False

class TreeClassifier():
    def __init__(self):
        self.train_data_T_pandas = None

        self.train_data_T = None
        self.train_data_pandas = None
        self.target_attribute = None
        self.attributes = None

        self.root_node = None

    def clean_tree(self):
        self.root_node = None

    def save_tree(self, location):
        with open(location, 'wb') as f: 
            pickle.dump(self.root_node, f)

    def load_tree(self, location):
    
        with open(location, 'rb') as f:
            ID3_loaded_tree = pickle.load(f)

        self.root_node = ID3_loaded_tree

    def load_data(self, csv_file_location, type_data):


        if type_data == "train":

            self.train_data_pandas = pd.read_csv(csv_file_location)

            # change to dict to make faster
            self.train_data_T_pandas = self.train_data_pandas.to_dict()

            self.train_data_T = {}
            for key in self.train_data_T_pandas.keys():
                self.train_data_T[key] = self.train_data_T_pandas[key]

            # self.data_train_M = np.empty((len(self.train_data_T[self.target_attribute]),len(self.train_data_T.keys())))

            # for att_num, att in enumerate(self.train_data_T.keys()):
            #     self.data_train_M[:, att_num] = list(self.train_data_T[att].values())

            self.all_attributes_list = self.train_data_pandas.columns

        elif type_data == "test":

            self.test_data_pandas = pd.read_csv(csv_file_location)

            # change to dict to make faster
            self.test_data_T_pandas = self.test_data_pandas.to_dict()

            self.test_data_T = {}
            for key in self.test_data_T_pandas.keys():
                self.test_data_T[key] = self.test_data_T_pandas[key]

            self.all_attributes_list = self.test_data_pandas.columns

         
        self.target_attribute = self.all_attributes_list[0]
        self.attributes = set(self.all_attributes_list[1:])




    def build_tree(self):
        """ Build tree using some algorithm (ID3 here) """

        self.root_node = ID3(self.train_data_T, self.attributes, self.target_attribute, None)

    def build_pruned_tree(self, pruning_p):
        """ Build tree using some algorithm (ID3 here) """

        self.root_node = ID3(self.train_data_T, self.attributes, self.target_attribute, pruning_p)

    def classify(self, csv_file_location: str):
        """ Classify the set and return the accuracy """

        self.load_data(csv_file_location, "test")


        assert self.root_node is not None

        total_data_amount = len(self.test_data_T[self.target_attribute])
        true_predictions_amount = 0
        sample = {}
        

        for sample_idx in range(total_data_amount):
            
            # collect the sample data from attributes
            for att in self.all_attributes_list:
                sample[att] = self.test_data_T[att][sample_idx]

            class_prediction = self.get_class_prediction(self.root_node, sample)
            class_true = int(sample[self.target_attribute])

            if class_true == class_prediction:
                true_predictions_amount += 1


        return true_predictions_amount/total_data_amount

    def get_class_prediction(self, node, sample):

        if node.label is not None:
            prediction =  node.label
        else:
            threshold = node.thresholds


            if sample[node.attribute] > threshold:
                prediction = self.get_class_prediction(node.nodes["above"], sample)
            else:
                prediction = self.get_class_prediction(node.nodes["below"], sample)
            

        return prediction


def find_entropy(dataset, target_attribute):
    """ Find the entropy to the whole dataset in the node """

    # if no examples in this dataset

    if len(dataset[target_attribute]) == 0:
        return 0
    
    # calculate the probability for each output class
    dataset_size = len(dataset[target_attribute])
    unique_values = set(list(dataset[target_attribute].values()))

    total_entropy = 0

    for unique_value in unique_values:
        data_amount = list(dataset[target_attribute].values()).count(int(unique_value))
        # data_amount = data_counts[unique_value]
        prob = data_amount / dataset_size
        total_entropy += prob * math.log(prob, 2)

    total_entropy *= -1
    return total_entropy

def get_dataset_partition(dataset, attribute, threshold):
    

    indexes_below = []
    indexes_above = []

    for item in dataset[attribute].items():
        if item[1] > threshold:
            indexes_above.append(item[0])
        else:
            indexes_below.append(item[0])


    # create new partitions of datasets using those indexes
    att_list = dataset.keys()

    dataset_below = {}
    dataset_above = {}

    for idx_norm, index_above in enumerate(indexes_above):
        for att in att_list:
            if att not in dataset_above.keys():
                dataset_above[att] = {}
            dataset_above[att][idx_norm] = dataset[att][index_above]

    for idx_norm, index_below in enumerate(indexes_below):
        for att in att_list:
            if att not in dataset_below.keys():
                dataset_below[att] = {}
            dataset_below[att][idx_norm] = dataset[att][index_below]

    return dataset_below, dataset_above


def find_attribute_threshold_entropy(dataset, attribute, threshold, target_attribute):
    """ Find the entropy for the dataset using a given attribute and threshold  """

    # 1. partition the dataset into the part below and part above the threshold

    # find indexes for the dict where the threshold is above/below


    dataset_below, dataset_above = get_dataset_partition(dataset, attribute, threshold)

    dataset_below_size = len(dataset_below[target_attribute])
    dataset_above_size = len(dataset_above[target_attribute])

    dataset_size = len(dataset[target_attribute])


    entropy_below_normalized = find_entropy(dataset_below, target_attribute) * (dataset_below_size / dataset_size)
    entropy_above_normalized = find_entropy(dataset_above, target_attribute) * (dataset_above_size / dataset_size)

    entropy_sum = entropy_below_normalized + entropy_above_normalized

    return entropy_sum

# @time_it
def get_best_attribute(dataset, attributes, target_attribute) -> tuple:

    # for each attribute, we try every possible threshold (which is between the values for this attribute)
    # currently we cut only with 1 threshold
    # another way to try different cuts would be with K-means to try multiple regions

    max_information_gain = float('-inf')
    max_entropy_cut = None, None
    parent_entropy = find_entropy(dataset, target_attribute)



    for _, attribute in enumerate(attributes):

        print(f"\tAnalyzing attribute: {attribute}")

        # get all the values for this attribute and order them increasingly
        attribute_values = list(dataset[attribute].values())
        attribute_values.sort()
        # attribute_dict = {x : y for x, y in enumerate(attribute_values)}

        # attribute_dict_sorted = {k: v for k, v in sorted(attribute_dict.items(), key=lambda item: item[1])}
        # attribute_values = list(attribute_dict_sorted.values())

        thresholds_list = [(attribute_values[idx] + attribute_values[idx-1])/2 for idx in range(len(attribute_values)) \
            if (idx > 0 and (attribute_values[idx] != attribute_values[idx-1]))]
        thresholds_list_unique = list(set(thresholds_list))
        thresholds_list_unique.sort()

        for threshold in thresholds_list_unique:

            threshold_entropy_normalized = find_attribute_threshold_entropy(dataset, attribute, threshold, target_attribute)

            # calculate the Information Gain IG

            information_gain = parent_entropy - threshold_entropy_normalized

            if information_gain > max_information_gain:
                # print(f"\tInformation Gain improved : {information_gain}")
                max_entropy_cut = attribute, threshold
                max_information_gain = information_gain
        
    if max_information_gain == float("-inf"):
        raise Exception ("Negative max info gain")


    return max_entropy_cut

def get_majority_class(dataset, target_attribute) -> str:
    """ Get the class for which the majority of samples suit """
    unique_vals = set(list(dataset[target_attribute].values()))
    max_occurences = float("-inf")
    res = None

    for val in unique_vals:
        occurences = list(dataset[target_attribute].values()).count(int(val))
        if occurences > max_occurences:
            max_occurences = occurences
            res = val

    assert res is not None
    return res


def get_class_labels(dataset, target_attribute) -> list :
    """ Get all the possible unique values for the target attribute """


    unique_values = set(list(dataset[target_attribute].values()))

    return unique_values


def ID3(dataset, attributes : set, target_attribute : str, pruning_parameter : int) -> Node:
    """ Recursive function which builds the ID3 tree with consistent dataset"""

    node = Node()

    # print(f"Attributes number : {len(attributes)}")


    # 0. If all of the dataset accounts for 1 single class,
    #     create a leaf node with the label of this class
    labels = get_class_labels(dataset, target_attribute)
    if len(labels) == 1:
        node.is_leaf = True
        node.label = int(labels.pop())
        return node



    # 1.1 If pruning parameter exists, check if the number of samples is below this parameter
    #       If true, create a leaf node with majority of cases

    if pruning_parameter is not None:
        if len(dataset[target_attribute]) <= pruning_parameter:
            majority_class = get_majority_class(dataset, target_attribute)
            node.is_leaf = True
            node.label = majority_class
            return node

    # 2. Pick the attribute which maximizes the IG, and the threshold 
    #       accept multiple threshold values, and partition the dataset according to this

    best_attribute, threshold = get_best_attribute(dataset, attributes, target_attribute)
    node.attribute = best_attribute
    node.thresholds = threshold
    node.is_branch = True

    # 3. remove this attribute from attributes list
    # attributes.remove(best_attribute)

    dataset_below, dataset_above = get_dataset_partition(dataset, best_attribute, threshold)

    node.nodes['below'] = ID3(dataset_below, attributes, target_attribute, pruning_parameter)
    node.nodes['above'] = ID3(dataset_above, attributes, target_attribute, pruning_parameter)

    return node




def check_tree_valid(node):
    """ Each leaf either has label or nodes """

    

    if node.is_leaf: 
        assert node.label is not None
    if node.is_branch:
        assert node.attribute is not None
        assert node.threshold is not None
        assert node.nodes != {}

        for child_node in node.nodes:
            check_tree_valid(node.nodes[child_node])


if __name__ == '__main__':

    CREATE_NEW_TREES = 0
    
    pruning_parameters = [3, 9, 27]
    accuracy_pruning = []

    tree_classifier = TreeClassifier()



    if CREATE_NEW_TREES : 

        tree_classifier.load_data("train.csv", "train")

        # q2 
        tree_classifier.clean_tree()
        tree_classifier.build_tree()
        tree_classifier.save_tree("db/q2_dict_tree.pkl")
        accuracy = tree_classifier.classify("test.csv")
        print(f"Regular accuracy = {accuracy}")


        # q3

        for x in pruning_parameters:
            print(f"Start for pruning p = {x}")
            tree_classifier.clean_tree()
            tree_classifier.build_pruned_tree(pruning_p = x)
            tree_classifier.save_tree(f"db/q3_dict_tree_pruned_{x}.pkl")
            accuracy_pruning.append(tree_classifier.classify("test.csv"))
            print(f"Pruned tree accuracy(P = {x}) = {accuracy_pruning[-1]}")

    else:

        # q2

        tree_classifier.clean_tree()
        tree_classifier.load_tree("db/q2_dict_tree.pkl")
        accuracy = tree_classifier.classify("test.csv")
        print(f"Regular accuracy = {accuracy}")

        for x in pruning_parameters:
            tree_classifier.clean_tree()
            tree_classifier.load_tree(f"db/q3_dict_tree_pruned_{x}.pkl")
            accuracy_pruning.append(tree_classifier.classify("test.csv"))
            print(f"Pruned tree accuracy(P = {x}) = {accuracy_pruning[-1]}")



    plt.plot(pruning_parameters, accuracy_pruning)
    plt.xlabel('Pruning parameter')
    plt.ylabel('Accuracy')
    plt.savefig('db/q3.png')




