import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from knn import KNN
from kernel_knn import KernelKNN
import pickle
import pandas as pd
from copy import copy, deepcopy

def simple_data_generation():
    print('Generate a simple dataset')
    # create multiple training instances
    pos_point_num = 40
    neg_point_num = 40
    testing_point_num = 3
    x_pos = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], pos_point_num)
    x_neg = np.random.multivariate_normal([-1, -1], [[2, 0], [0, 2]], neg_point_num)
    y = ([1] * pos_point_num) + ([-1] * neg_point_num)       
    # create testing instances
    testing_X = np.random.multivariate_normal([1, -1], [[2, 0], [0, 2]], testing_point_num)
    
    return x_pos, x_neg, y, testing_X
def import_dataset():
    print('import an existing dataset.')
    with open('data_batch_1', 'rb') as fo:
        dict_out = pickle.load(fo, encoding='bytes')
    training_data = dict_out[b'data']
    train_labels = np.asarray(dict_out[b'labels'])
    for i in range(2, 6):
        with open('data_batch_' + str(i), 'rb') as fo:
            dict_out = pickle.load(fo, encoding='bytes')
        training_data = np.concatenate((training_data, dict_out[b'data']), axis = 0)
        train_labels = np.concatenate((train_labels, np.asarray(dict_out[b'labels'])))
    with open('test_batch', 'rb') as test_file:
        test_dict = pickle.load(test_file, encoding='bytes')
    test_labels = np.asarray(test_dict[b'labels'])
    test_data = test_dict[b'data'] / 255.0
     
    return training_data / 255.0, train_labels, test_data[:1000, :], test_labels[:1000]
def import_mnist_dataset(file_name, n):
    # read a csv file and convert it to a numpy array
    csv_data = pd.read_csv(file_name).values
    
    return csv_data[:n, 1:] / 255.0, csv_data[:n, 0]
def import_iris_dataset(file_name):
    # read a csv file and convert it to a numpy array
    data = pd.read_csv(file_name).values
    np.random.shuffle(data)

    return data[:100, :4], data[:100, 4], data[100:, :4], data[100:, 4]
def simple_data_simulation(k_value):
    # data generation
    X_pos, X_neg, training_labels, testing_data = simple_data_generation()
    X = np.concatenate((X_pos, X_neg), axis = 0)
    testing_data_copy = deepcopy(testing_data)
    
    # run Vanilla k-NN classifier on the simple dataset
    k_NN = KNN(k_value)
    k_NN.training(X, training_labels)
    prediction_result = k_NN.prediction(testing_data)
    
    # plot all training examples and predictions of K-NN
    f1 = plt.figure(1)
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color = 'red', label='positive (training)')
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color = 'blue', label='negative (training)')
    for index in range(len(prediction_result)):
        if prediction_result[index] == 1:
            plt.scatter(testing_data[index, 0], testing_data[index, 1], marker = '+',color= 'green', label='positive (testing)', s = 60)
        elif prediction_result[index] == -1:
            plt.scatter(testing_data[index, 0], testing_data[index, 1], marker = '^',color= 'green', label='negative (testing)', s = 60)
    plt.legend()

    # run kernelized k-NN classifier on the simple dataset
    kernel_kNN = KernelKNN(k_value)
    kernel_kNN.training(X, training_labels)
    predicted = kernel_kNN.prediction(testing_data_copy, 'polynomial')
    
    # plot all training examples and predictions of K-NN
    f2 = plt.figure(2)
    plt.scatter(X_pos[:, 0], X_pos[:, 1], color = 'red', label='positive (training)')
    plt.scatter(X_neg[:, 0], X_neg[:, 1], color = 'blue', label='negative (training)')
    for index in range(len(prediction_result)):
        if predicted[index] == 1:
            plt.scatter(testing_data_copy[index, 0], testing_data_copy[index, 1], marker = '<',color= 'black', label='positive (testing)', s = 60)
        elif predicted[index] == -1:
            plt.scatter(testing_data_copy[index, 0], testing_data_copy[index, 1], marker = '^',color= 'black', label='negative (testing)', s = 60)
    plt.legend()
    
    plt.xlabel('x1 Feature', fontsize=16)
    plt.ylabel('x2 Feature', fontsize=16)
    plt.show()
    
def cifar10_simulation(k_value):
    # import CIFAR10 dataset
    X_train, y_train, X_test, y_test = import_dataset()

    # run Vanilla k-NN classifier on CIFAR10
    k_NN_cifar = KNN(k_value)
    k_NN_cifar.training(X_train, y_train)
    cifar_prediction = k_NN_cifar.prediction(X_test)
    k_NN_cifar.evaluation_prediction(cifar_prediction, y_test)

    # run kernelized k-NN classifier on CIFAR10
    kernel_k_NN_cifar = KernelKNN(k_value)
    kernel_k_NN_cifar.training(X_train, y_train)
    predicted_cifar = kernel_k_NN_cifar.prediction(X_test, 'gaussian')
    kernel_k_NN_cifar.evaluation_prediction(predicted_cifar, y_test)
def mnist_simulation(k_value):
    # import MNIST dataset
    mnist_train, mnist_train_label = import_mnist_dataset('mnist_train.csv', 60000)
    mnist_test, mnist_test_label = import_mnist_dataset('mnist_test.csv', 1000)

    # run Vanilla k-NN classifier on MNIST
    k_NN_mnist = KNN(k_value)
    k_NN_mnist.training(mnist_train, mnist_train_label)
    mnist_prediction = k_NN_mnist.prediction(mnist_test)
    k_NN_mnist.evaluation_prediction(mnist_prediction, mnist_test_label)

    # run kernel k-NN classifier on MNIST
    kernel_k_NN_mnist = KernelKNN(k_value)    
    kernel_k_NN_mnist.training(mnist_train, mnist_train_label)
    predicted_mnist = kernel_k_NN_mnist.prediction(mnist_test, 'gaussian')
    kernel_k_NN_mnist.evaluation_prediction(predicted_mnist, mnist_test_label)
def iris_data_simulation(k_value):
    # read iris dataset
    X_train, y_train, X_test, y_test = import_iris_dataset('new_iris.csv')

    # run Vanilla k-NN classifier on IRIS dataset
    k_NN_iris = KNN(k_value)
    k_NN_iris.training(X_train, y_train)
    iris_prediction = k_NN_iris.prediction(X_test)
    k_NN_iris.evaluation_prediction(iris_prediction, y_test)    
    
    # run kernel k-NN classifier on IRIS dataset
    kernel_k_NN_iris = KernelKNN(k_value)
    kernel_k_NN_iris.training(X_train, y_train)
    predicted_iris = kernel_k_NN_iris.prediction(X_test, 'gaussian')
    kernel_k_NN_iris.evaluation_prediction(predicted_iris, y_test)
def main():
    # declare variables
    k_value = 3
    
    # simple data simulation
    simple_data_simulation(k_value)
    # IRIS dataset simulation
    iris_data_simulation(k_value)
    # CIFAR-10 simulation
    cifar10_simulation(k_value)
    # MNIST simulation
    mnist_simulation(k_value)

if __name__ == "__main__":
    main()
