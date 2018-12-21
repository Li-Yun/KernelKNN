import numpy as np
import math

class KernelKNN():
    def __init__(self, k_neighbor):
        self.k = k_neighbor
    def training(self, X, y):
        self.X_train = X
        self.y_train = y
    def majority_vote(self, label_list):
        dic = {}
        dic[label_list[0]] = 1
        for index in range(1, len(label_list)):
            if label_list[index] in dic:
                dic[label_list[index]] += 1
            else:
                dic[label_list[index]] = 1
        
        return max(dic, key = dic.get)
    def choice_kernel(self, vector1, vector2, kernel_param):
        if kernel_param == 'linear':
            return np.dot(vector1, vector2)
        elif kernel_param == 'polynomial':
            degree_valu = 3
            return math.pow(1 + np.dot(vector1, vector2), degree_valu)
        elif kernel_param == 'gaussian':
            sigma_val = 0.5
            return math.exp(-np.sum(np.square(np.subtract(vector1, vector2))) / 2.0 * math.pow(sigma_val, 2))
    def distance_computation(self, X, kernel_name):
        distance_list = []
        test_num = X.shape[0]

        for index in range(test_num):
            tmp_array = np.zeros(self.X_train.shape[0])
            for training_index in range(self.X_train.shape[0]):
                tmp_array[training_index] = self.choice_kernel(X[index, :], X[index, :], kernel_name) + self.choice_kernel(self.X_train[training_index, :], self.X_train[training_index, :], kernel_name) - (2 * self.choice_kernel(X[index, :], self.X_train[training_index, :], kernel_name))
            distance_list.append(tmp_array)
        return distance_list
    def prediction(self, testing_data, input_kernel):
        num_test = testing_data.shape[0]
        y_pred = np.zeros(num_test)
        distance_collection = self.distance_computation(testing_data, input_kernel)
        
        for test_index in range(num_test):
            k_neighbors = np.argsort(distance_collection[test_index])[:self.k]
            k_label_list = []
            for k_index in range(k_neighbors.shape[0]):
                k_label_list.append(self.y_train[k_neighbors[k_index]])
            # perform majority vote
            y_pred[test_index] = self.majority_vote(k_label_list)
        return y_pred
    def evaluation_prediction(self, prediction, actual_label):
        correct_num = 0
        for index in range(prediction.shape[0]):
            if prediction[index] == actual_label[index]:
                correct_num += 1
        print('Classification Accuracy for kernel k-NN:', (correct_num / prediction.shape[0]) * 100.0, '%')
        print('====================================')
