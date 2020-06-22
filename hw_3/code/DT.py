

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



def find_entropy(dataset, target_attribute):
    """ Find the entropy to the whole dataset in the node """

    if dataset.empty == True:
        return 0
    
    # calculate the probability for each output class
    dataset_size = dataset.shape[0]
    unique_values = list(dataset[target_attribute].unique())
    data_counts = dataset[target_attribute].value_counts()

    total_entropy = 0

    for unique_value in unique_values:
        data_amount = data_counts[unique_value]
        prob = data_amount / dataset_size
        total_entropy += prob * math.log(prob, 2)

    total_entropy *= -1
    return total_entropy


def find_attribute_threshold_entropy(dataset, attribute, threshold, target_attribute):
    """ Find the entropy for the dataset using a given attribute and threshold  """

    # 1. partition the dataset into the part below and part above the threshold

    # there should be no value equal to the threshold

    dataset_below = dataset[(dataset[attribute] < threshold)]
    dataset_above = dataset[(dataset[attribute] > threshold)]

    # dataset_below = dataset[(dataset[attribute].between(float("-inf"), threshold))]



    entropy_below_normalized = find_entropy(dataset_below, target_attribute) * (dataset_below.shape[0] / dataset.shape[0])
    entropy_above_normalized = find_entropy(dataset_above, target_attribute) * (dataset_above.shape[0] / dataset.shape[0])

    entropy_sum = entropy_below_normalized + entropy_above_normalized



    return entropy_sum

@time_it
def get_best_attribute(dataset, attributes, target_attribute) -> tuple:

    # for each attribute, we try every possible threshold (which is between the values for this attribute)
    # currently we cut only with 1 threshold
    # another way to try different cuts would be with K-means to try multiple regions

    max_information_gain = float('-inf')
    max_entropy_cut = None, None
    parent_entropy = find_entropy(dataset, target_attribute)

    for _, attribute in enumerate(attributes):

        # print(f"\tAnalyzing attribute: {attribute}")

        # get all the values for this attribute and order them increasingly
        attribute_values = dataset[attribute].tolist()
        attribute_values.sort()

        first = True

        for threshold_idx in range(len(attribute_values)+1):

            last = threshold_idx == len(attribute_values)

            if first == True:
                # first threshold
                threshold_value = float('-inf')
                first = False
            elif last == True:
                threshold_value = float('inf')
            else:
                threshold_value = (attribute_values[threshold_idx] + attribute_values[threshold_idx - 1]) / 2 

            threshold_entropy_normalized = find_attribute_threshold_entropy(dataset, attribute, threshold_value, target_attribute)

            # calculate the Information Gain IG

            information_gain = parent_entropy - threshold_entropy_normalized

            

            if information_gain > max_information_gain:
                # print(f"\tInformation Gain improved : {information_gain}")
                max_entropy_cut = attribute, [threshold_value]  # make threshold_value a list
                max_information_gain = information_gain
        
    if max_information_gain == float("-inf"):
        raise Exception ("Negative max info gain")


    return max_entropy_cut

def get_majority_class(dataset, target_attribute) -> str:
    """ Get the class for which the majority of samples suit """
    majority_class = str(dataset[target_attribute].value_counts().argmax())

    return majority_class

def get_class_labels(dataset, target_attribute) -> list :
    """ Get all the possible unique values for the target attribute """

    assert target_attribute in dataset.columns

    unique_values = list(dataset[target_attribute].unique())

    return unique_values


def ID3(dataset : pd.core.frame.DataFrame , attributes : set, target_attribute : str, pruning_parameter : int) -> Node:
    """ Recursive function which builds the ID3 tree with consistent dataset"""

    node = Node()

    print(f"Attributes number : {len(attributes)}")


    # 0. If all of the dataset accounts for 1 single class,
    #     create a leaf node with the label of this class
    labels = get_class_labels(dataset, target_attribute)
    if len(labels) == 1:
        node.is_leaf = True
        node.label = labels[0]
        return node
    

    # 1. If no attributes left, create the LEAF out of this node
    #    put the label as the majority of the examples in dataset
    if len(attributes) == 0:
        majority_class = get_majority_class(dataset, target_attribute)
        node.is_leaf = True
        node.label = majority_class
        return node

    # 1.1 If pruning parameter exists, check if the number of samples is below this parameter
    #       If true, create a leaf node with majority of cases

    if pruning_parameter is not None:
        if dataset.shape[0] <= pruning_parameter:
            majority_class = get_majority_class(dataset, target_attribute)
            node.is_leaf = True
            node.label = majority_class
            return node

    # 2. Pick the attribute which maximizes the IG, and the threshold 
    #       accept multiple threshold values, and partition the dataset according to this

    best_attribute, thresholds = get_best_attribute(dataset, attributes, target_attribute)
    node.attribute = best_attribute
    node.thresholds = thresholds
    node.is_branch = True

    # 3. remove this attribute from attributes list
    attributes.remove(best_attribute)

    
    # 4. iterate over all of the group which are created by partitioning through this best_attribute
    for threshold_idx, threshold in enumerate(thresholds):
        
        # 4.1  select all the dataset for this region
        if threshold_idx == 0:
            partition = dataset[(dataset[best_attribute] < thresholds[threshold_idx])]
        else:
            partition = dataset[(dataset[best_attribute] < thresholds[threshold_idx] and dataset[best_attribute] > thresholds[threshold_idx-1])]
        
        if partition.empty == False:
            node.nodes[threshold_idx] = ID3(partition, copy.deepcopy(attributes), target_attribute, pruning_parameter)


    # 4.2 Get the last partition and create a subtree
    partition = dataset[(dataset[best_attribute] > thresholds[-1])]
    if partition.empty == False:
        node.nodes[len(thresholds)] = ID3(partition, copy.deepcopy(attributes), target_attribute, pruning_parameter)

    return node


def measure_tree_accuracy(root_node, dataset, target_attribute):
    """ Measure and return the accuracy of the predictions on the dataset """
    total_data_amount = dataset.shape[0]
    true_predictions_amount = 0
    
    # for 
    # 
    for sample_idx in range(dataset.shape[0]):
        sample = dataset.iloc[sample_idx]
        class_prediction = get_class_prediction(root_node, sample)
        class_true = int(sample[target_attribute])

        if class_true == class_prediction:
            true_predictions_amount += 1


    return true_predictions_amount/total_data_amount


def get_class_prediction(node, sample):

    if node.label is not None:
        prediction =  node.label
    else:
        # get the value for the attribute
        value_attribute = sample[node.attribute]

        # find to which threshold index it suits (for case with multiple thresholds)
        current_threshold_idx = 0

        for thr_idx, threshold in enumerate(node.thresholds):
            if value_attribute > threshold:
                current_threshold_idx = thr_idx + 1
                break

        prediction = get_class_prediction(node.nodes[current_threshold_idx], sample)
        

    return prediction

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

    train_dataset = pd.read_csv("train.csv")
    test_dataset = pd.read_csv("test.csv")

    CREATE_NEW_TREE = 0
    CREATE_NEW_PRUNED_TREES = 0


    # get the attributes list 
    all_attributes_list = train_dataset.columns 

    target_attribute = all_attributes_list[0]
    attributes = set(all_attributes_list[1:])

    if CREATE_NEW_TREE:

        ID3_tree = ID3(train_dataset, copy.deepcopy(attributes), target_attribute, None)

        # save into file
        with open('db/id3_tree_2.pkl', 'wb') as f: 
            pickle.dump(ID3_tree, f)
    else:
        with open('db/id3_tree_2.pkl', 'rb') as f:
            ID3_tree = pickle.load(f)


    check_tree_valid(ID3_tree)
    accuracy = measure_tree_accuracy(ID3_tree, test_dataset, target_attribute)

    print(f"Pruning: N/A accuracy: {round(accuracy,5)}")


    pruning_parameters = [3, 9, 27]
    accuracy_pruning = []
    pruned_trees = []

    if CREATE_NEW_PRUNED_TREES:
        

        for x in pruning_parameters:
            print(f"Initialize parameter {x}")
            pruned_tree = ID3(train_dataset, copy.deepcopy(attributes), target_attribute, x)
            pruned_trees.append(pruned_tree)
            check_tree_valid(pruned_tree)

            # save into file
            with open(f"db/id3_pruned_{x}_tree_3.pkl", 'wb') as f: 
                pickle.dump(pruned_tree, f)
    else:
        for x in pruning_parameters:

            with open(f"db/id3_pruned_{x}_tree_3.pkl", 'rb') as f:
                ID3_loaded_tree = pickle.load(f)

            check_tree_valid(ID3_loaded_tree)
            pruned_trees.append(ID3_loaded_tree)

    for idx, x in enumerate(pruning_parameters):

        current_tree = pruned_trees[idx]
        accuracy_pruning.append(measure_tree_accuracy(current_tree, test_dataset, target_attribute))

        print(f"Pruning: {x} accuracy: {round(accuracy_pruning[idx],5)}")


    plt.plot(pruning_parameters, accuracy_pruning)
    plt.xlabel('Pruning parameter')
    plt.ylabel('Accuracy')
    plt.show()




