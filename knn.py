import numpy as np

class KNN():
    def __init__(self, k_neighbor):
        self.k = k_neighbor
    def training(self, X, y):
        self.X_train = X
        self.y_train = y
    def distance_computation(self, X):
        distance_list = []
        test_num= X.shape[0]
        
        for index in range(test_num):
            distance_list.append( np.sum(np.square(np.subtract(self.X_train, X[index, :][None, :])), axis = 1) )
        return distance_list
    def majority_vote(self, label_list):
        dic = {}
        dic[label_list[0]] = 1
        for index in range(1, len(label_list)):
            if label_list[index] in dic:
                dic[label_list[index]] += 1
            else:
                dic[label_list[index]] = 1
        
        return max(dic, key = dic.get)
    def prediction(self, testing_data):
        num_test = testing_data.shape[0]
        y_pred = np.zeros(num_test)
        distance_collection = self.distance_computation(testing_data)
        
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
        print('Classification Accuracy for k-NN:', (correct_num / prediction.shape[0]) * 100.0, '%')
        print('====================================')
        
