

# from decorators import time_it

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import math
import pickle
import copy
import os

class Node():
    def __init__(self):
        self.is_branch = False
        self.attribute = None
        self.thresholds = None
        self.nodes = {}             # dict of type  
        self.epsilon = None


        self.label = None
        self.is_leaf = False
        self.samples = None

class TreeClassifier():
    def __init__(self):

        self.train_data_T_pandas = None
        self.train_data_T = None
        self.train_data_pandas = None

        self.test_data_T_pandas = None
        self.test_data_T = None
        self.test_data_pandas = None

        self.all_attributes_list = None
        self.target_attribute = None
        self.attributes = None

        self.normalization_params = {}

        self.root_node = None

    def clean_tree(self):
        """ For case where same Tree object used, clean before """

        self.root_node = None

    def save_tree(self, location):
        """ Save generated tree """

        with open(location, 'wb') as f: 
            pickle.dump(self.root_node, f)

    def load_tree(self, location):
        """ Load generated tree """

        with open(location, 'rb') as f:
            ID3_loaded_tree = pickle.load(f)

        self.root_node = ID3_loaded_tree

    def load_data(self, csv_file_location, type_data):

        """ Load data into train or test variables """

        if type_data == "train":

            self.train_data_pandas = pd.read_csv(csv_file_location)

            # change to dict to make faster
            self.train_data_T_pandas = self.train_data_pandas.to_dict()

            self.train_data_T = {}
            for key in self.train_data_T_pandas.keys():
                self.train_data_T[key] = self.train_data_T_pandas[key]


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

    def normalize_train_data(self):

        """ Normalize all the training data """

        # 1. iterate over every param except of the target attribute, fill the normalization params
        #       and normalize all the values for that attribute
            

        for attribute in self.train_data_T.keys():

            # print(f"Normalizing {attribute}")
            if attribute == self.target_attribute:
                continue

            min_attribute = min(self.train_data_T[attribute].values())
            max_attribute = max(self.train_data_T[attribute].values())
            difference = max_attribute - min_attribute

            self.normalization_params[attribute] = (min_attribute, max_attribute, difference)


            new_values_list = [(value - min_attribute) / difference for value in self.train_data_T[attribute].values()]
            new_values_dict = dict(enumerate(new_values_list))

            self.train_data_T[attribute] = new_values_dict



    def build_tree(self):
        """ Build tree using some algorithm (ID3 here) """

        self.root_node = ID3(self.train_data_T, self.attributes, self.target_attribute, None, False, False, float('inf'))

    def build_pruned_tree(self, pruning_p):
        """ Build tree using ID3 with pruning """

        self.root_node = ID3(self.train_data_T, self.attributes, self.target_attribute, pruning_p, False, False, float('inf'))

    def build_epsilon_tree(self, depth):
        """ Build tree using ID3 with Epsilon and depth limit """
        self.root_node = ID3(self.train_data_T, self.attributes, self.target_attribute, None, True, True, depth)



    def classify(self, csv_file_location: str, use_epsilon : bool, use_KNN : bool, K : int):
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

            if use_epsilon == False:
                # as in q2, q3 - regular classification
                class_prediction = self.get_class_prediction(self.root_node, sample)
            else:
                if use_KNN == False:
                    # q6 - use the epsilon value, obtain list of leaves reached, and choose most common
                    class_prediction_list = self.get_class_prediction_epsilon(self.root_node, sample, [])
                    class_prediction = most_frequent(class_prediction_list)
                else:
                    # q11 - Use the epsilon to obtain list of samples that were present at the nodes reached
                    #       and use the KNN classifier with K neighbors with all those samples

                    assert K is not None
                    sample = self.normalize_sample(sample)
                    relevant_datasets_list = self.get_datasets_at_leaf_nodes(self.root_node, sample, [])

                    # unite the list of datasets - samples - into 1 dictionary
                    relevant_dataset = self.get_dataset_from_lists(relevant_datasets_list)

                    # create KNN classifier and classify using those nodes
                    class_prediction = self.get_KNN_classification(relevant_dataset, sample, K)

            class_true = int(sample[self.target_attribute])

            if int(class_true) == int(class_prediction):
                true_predictions_amount += 1


        return true_predictions_amount/total_data_amount

    def get_class_prediction(self, node, sample):
        """ go down the tree until leaf node reached """
        if node.label is not None:
            prediction =  node.label
        else:
            threshold = node.thresholds


            if sample[node.attribute] > threshold:
                prediction = self.get_class_prediction(node.nodes["above"], sample)
            else:
                prediction = self.get_class_prediction(node.nodes["below"], sample)
            

        return prediction


    def get_class_prediction_epsilon(self, node, sample, predictions_list):
        """ go down the tree until leaf node reached, collect all reached nodes in a list """
        if node.label is not None:
            predictions_list.append(node.label)
        else:

            # find to which threshold index it suits (for case with multiple thresholds)
            threshold = node.thresholds
            next_nodes_list = []

            if sample[node.attribute] < threshold + node.epsilon:
                next_nodes_list.append(node.nodes["below"])

            if sample[node.attribute] > threshold - node.epsilon:
                next_nodes_list.append(node.nodes["above"])


            for node in next_nodes_list:
                predictions_list = self.get_class_prediction_epsilon(node, sample, predictions_list)
            

        return predictions_list

    def get_datasets_at_leaf_nodes(self, node, sample, datasets_list):
        """ go down the tree until leaf node reached, collect the samples of all reached nodes in a list """
        if node.label is not None:
            datasets_list.append(node.samples)
        else:

            # find to which threshold index it suits (for case with multiple thresholds)
            threshold = node.thresholds
            next_nodes_list = []

            if sample[node.attribute] < threshold + node.epsilon:
                next_nodes_list.append(node.nodes["below"])

            if sample[node.attribute] > threshold - node.epsilon:
                next_nodes_list.append(node.nodes["above"])

            assert next_nodes_list != []

            for node in next_nodes_list:
                datasets_list = self.get_datasets_at_leaf_nodes(node, sample, datasets_list)
            

        return datasets_list

    def normalize_sample(self, sample):
        """ Normalize sample using the parameters calculated from train data """
        for attribute in sample.keys():
            if attribute == self.target_attribute:
                continue
            # value - min / (max - min)
            sample[attribute] = (sample[attribute] - self.normalization_params[attribute][0]) / self.normalization_params[attribute][2]

        return sample
        

    def get_KNN_classification(self, dataset, sample, K):
        
        """ Similar to code in KNN.py """

        # 2. Find the Euclidean distance to EVERY item in the training set
        euclidean_dict = {}  # of type euclidean_dict[0] = (distance, classification) 

        test_data_values = list(sample.values())
        test_data_values = test_data_values[1:]


        for train_example_idx in range(len(dataset[self.target_attribute])):
            classification = dataset[self.target_attribute][train_example_idx]

            train_data_values = []
            for attribute in sample.keys():
                train_data_values.append(dataset[attribute][train_example_idx])

            train_data_values = train_data_values[1:]

            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(train_data_values, test_data_values)]))
            euclidean_dict[train_example_idx] = (distance, classification)



        # 3. Sort the distances, take K first, find the majority of classificaion, return

        euclidean_list_sorted = sorted(euclidean_dict.items(),key=lambda x: x[1][0])

        K = min(len(euclidean_list_sorted), K)
        euclidean_list_truncated = euclidean_list_sorted[0:K]
        
        # take the majority class
        class_1_len = len(list(distance_inst[1][1] for distance_inst in euclidean_list_truncated if distance_inst[1][1] == 1))
        class_0_len = len(list(distance_inst[1][1] for distance_inst in euclidean_list_truncated if distance_inst[1][1] == 0))

        if class_0_len > class_1_len:
            return 0
        return 1


    def get_dataset_from_lists(self, dataset_lists):

        dataset_dict = {}
        dataset_idx = 0


        for dataset in dataset_lists:

            for input_dataset_idx in dataset[self.target_attribute].keys():

                for att in self.all_attributes_list:

                    if att not in dataset_dict.keys():
                        dataset_dict[att] = {}

                    dataset_dict[att][dataset_idx] = dataset[att][input_dataset_idx]

                dataset_idx += 1

        return dataset_dict

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

def most_frequent(input_list): 
    """ Find the most frequent value. If equal, set to preference """

    occurences_dict = {}
    preference = 1  # 0/1 in case of tie

    for val in input_list:
        if val not in occurences_dict.keys():
            occurences_dict[val] =1
        else:
            occurences_dict[val] += 1

    max_occurences_key = max(occurences_dict, key=occurences_dict.get)

    for key in occurences_dict.keys():
        if occurences_dict[key] >= occurences_dict[max_occurences_key] and key == preference:
            max_occurences_key = preference

    return int(max_occurences_key)

def get_dataset_partition(dataset, attribute, threshold, use_epsilon : bool, epsilon_value):
    

    indexes_below = []
    indexes_above = []

    if use_epsilon == True:
        threshold_low = threshold - epsilon_value
        threshold_high = threshold + epsilon_value


        for item in dataset[attribute].items():
            if item[1] > threshold_low:
                indexes_above.append(item[0])

            if item[1] < threshold_high:
                indexes_below.append(item[0])

    else:
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
    dataset_below, dataset_above = get_dataset_partition(dataset, attribute, threshold, False, None)

    # for normalization
    dataset_below_size = len(dataset_below[target_attribute])
    dataset_above_size = len(dataset_above[target_attribute])
    dataset_size = len(dataset[target_attribute])


    entropy_below_normalized = find_entropy(dataset_below, target_attribute) * (dataset_below_size / dataset_size)
    entropy_above_normalized = find_entropy(dataset_above, target_attribute) * (dataset_above_size / dataset_size)

    entropy_sum = entropy_below_normalized + entropy_above_normalized

    return entropy_sum


def get_best_attribute(dataset, attributes, target_attribute) -> tuple:

    # for each attribute, we try every possible threshold (which is between the values for this attribute)

    max_information_gain = float('-inf')
    max_entropy_cut = None, None
    parent_entropy = find_entropy(dataset, target_attribute) # to calculate the IG


    for _, attribute in enumerate(attributes):

        # print(f"\tAnalyzing attribute: {attribute}")

        # get all the values for this attribute and order them increasingly
        attribute_values = list(dataset[attribute].values())
        attribute_values.sort()

        # create the list of thresholds which length is 1 less than length of attribute_values
        # and has values all values as average between consecutive values in the attribute_values
        thresholds_list = [(attribute_values[idx] + attribute_values[idx-1])/2 for idx in range(len(attribute_values)) \
            if (idx > 0 and (attribute_values[idx] != attribute_values[idx-1]))]

        # save only unique threshold values
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
        raise Exception ("Invalid max info gain")


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


def ID3(dataset, attributes : set, target_attribute : str, pruning_parameter : int, \
    use_epsilon : bool, use_depth_limit : bool, current_depth : int) -> Node:
    """ Recursive function which builds the ID3 tree with consistent dataset"""

    node = Node()

    #print(f"Attributes number : {len(attributes)}")
    #print(f"Dataset size : {len(dataset[target_attribute])} depth: {current_depth}")

    # 0. If all of the dataset accounts for 1 single class,
    #     create a leaf node with the label of this class
    labels = get_class_labels(dataset, target_attribute)
    if len(labels) == 1:
        node.is_leaf = True
        node.label = int(labels.pop())      # only 1 label exists
        node.samples = dataset              # save the node samples for q11, where we use KNN on all samples of every leaf
        return node


    if use_depth_limit == True:
        if current_depth == 0:
            majority_class = get_majority_class(dataset, target_attribute)
            node.is_leaf = True
            node.samples = dataset
            node.label = majority_class
            return node



    # 1.1 If pruning parameter exists, check if the number of samples is below this parameter
    #       If true, create a leaf node with majority of cases

    if pruning_parameter is not None:
        if len(dataset[target_attribute]) <= pruning_parameter:
            majority_class = get_majority_class(dataset, target_attribute)
            node.is_leaf = True
            node.samples = dataset
            node.label = majority_class
            return node

    # 2. Pick the attribute which maximizes the IG, and the threshold 

    best_attribute, threshold = get_best_attribute(dataset, attributes, target_attribute)
    node.attribute = best_attribute
    node.thresholds = threshold
    node.is_branch = True

    # 2.5 Calculate the epsilon for q6
    if use_epsilon:
        node.epsilon = 0.1 * np.std(list(dataset[best_attribute].values()))

    # divide the dataset into dataset below and above the threshold
    # if epsilon is used, use the epsilon value as well
    dataset_below, dataset_above = get_dataset_partition(dataset, best_attribute, threshold, use_epsilon, node.epsilon)

    # create new children nodes for samples below and above the threshold
    # decrease depth. If no depth used, 'inf' depth remains 'inf' anyway.
    node.nodes['below'] = ID3(dataset_below, attributes, target_attribute, pruning_parameter, use_epsilon, use_depth_limit, current_depth-1)
    node.nodes['above'] = ID3(dataset_above, attributes, target_attribute, pruning_parameter, use_epsilon, use_depth_limit, current_depth-1)

    return node


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

    accuracy = tree_classifier.classify("test.csv", use_epsilon = True, use_KNN = True, K = 9)
    print(f"Epsilon tree accuracy with KNN (max depth = {depth}) = {accuracy}")