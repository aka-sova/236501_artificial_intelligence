


from decorators import time_it

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import math
import pickle
import copy


class KNN_classifier():
    def __init__(self):

        self.train_data = None
        self.train_data_T = None
        self.train_data_pandas = None
        self.target_attribute = None
        self.attributes = None
        self.normalization_params = {}



    def normalize(self):
        """ for each attribute, normalize it based on it's min/max values """

        # 1. iterate over every param except of the target attribute, fill the normalization params
        #       and normalize all the values for that attribute

        for attribute in self.train_data["columns"]:

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

        self.train_data_pandas = pd.DataFrame.from_dict(self.train_data_T)
        self.train_data = self.train_data_pandas.to_dict("split")


        

    def load_train_data(self, csv_file_location: str):
        """ Load the data into the structure """

        self.train_data_pandas = pd.read_csv(csv_file_location)
        all_attributes_list = self.train_data_pandas.columns 

        # change to dict to make faster
        self.train_data = self.train_data_pandas.to_dict("split")
        self.train_data_T = self.train_data_pandas.to_dict()
        self.target_attribute = all_attributes_list[0]
        self.attributes = set(all_attributes_list[1:])


    def classify(self, csv_file_location: str, K):
        """ classify the data using K neighbors"""

        test_data_pandas = pd.read_csv(csv_file_location)
        test_data_dict = test_data_pandas.to_dict("split")

        correct_predictions = 0
        total_test_samples = len(test_data_dict["index"])

        attributes_names = test_data_dict["columns"]
        attributes_names = attributes_names[1:]

        for test_item in test_data_dict["index"]:
            # print(f"Analyzing: {test_item}")

            values = test_data_dict["data"][test_item]

            true_result = values[0]
            attribute_values = values[1:]

            attributes_dict = {att_name : att_value for att_name, att_value in zip(attributes_names, attribute_values)}

            predicted_result = self.evaluate(attributes_dict, K)

            if predicted_result == true_result:
                correct_predictions += 1

        return correct_predictions / total_test_samples



    def evaluate(self, attributes_dict, K : int):
        """ evaluate a given example using K nearest neighbors """

        # 1. Normalize 
        for attribute in attributes_dict.keys():
            # value - min / (max - min)
            attributes_dict[attribute] = (attributes_dict[attribute] - self.normalization_params[attribute][0]) / self.normalization_params[attribute][2]


        # 2. Find the Euclidean distance to EVERY item in the training set
        euclidean_dict = {}  # of type euclidean_dict[0] = (distance, classification) 
        test_data_values = attributes_dict.values()

        for train_data_idx, train_data_example in zip(self.train_data["index"], self.train_data["data"]):
            classification = train_data_example[0]
            train_data_values = train_data_example[1:]

            distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(train_data_values, test_data_values)]))
            euclidean_dict[train_data_idx] = (distance, classification)

        # 3. Sort the distances, take K first, find the majority of classificaion, return


        euclidean_list_sorted = sorted(euclidean_dict.items(),key=lambda x: x[1][0])
        if K > len(self.train_data["index"]):
            K = len(self.train_data["index"])
        euclidean_list_truncated = euclidean_list_sorted[0:K]
        
        # take the majority class
        class_1_len = len(list(distance_inst[1][1] for distance_inst in euclidean_list_truncated if distance_inst[1][1] == 1))
        class_0_len = len(list(distance_inst[1][1] for distance_inst in euclidean_list_truncated if distance_inst[1][1] == 0))

        if class_0_len > class_1_len:
            return 0
        return 1


if __name__ == '__main__':


    # load the train data

    
    knn_classifier = KNN_classifier()
    knn_classifier.load_train_data("train.csv")
    knn_classifier.normalize()

    K_options = [1, 3, 9, 27]
    # K_options = list(range(1, 20, 3)) + list(range(25, 100, 5)) + list(range(120, 400, 20))

    accuracy = []

    for K in K_options:
        print(f"Init classification for K= {K}")

        accuracy.append(knn_classifier.classify("test.csv", K = K))

        print(f"\tAccuracy: {accuracy[-1]}")

    plt.plot(K_options, accuracy)
    plt.xlabel('K parameter in KNN')
    plt.ylabel('Accuracy')
    plt.savefig('db/q9.png')
