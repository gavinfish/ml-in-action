import numpy as np
import matplotlib.pyplot as plt
from kNN import normalize
from kNN import classify_knn

FILE_NAME = "data/datingTestSet2.txt"
K = 3


def load_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        n = len(lines)
        data = np.zeros((n, 3))
        labels_list = []
        index = 0
        for line in lines:
            line_list = line.strip().split('\t')
            data[index, :] = line_list[:3]
            labels_list.append(int(line_list[-1]))
            index += 1
        labels = np.array(labels_list)
        return data, labels


def plot_data(dimension1=0, dimension2=1):
    '''Plot data according different dimensions.

    :param dimension1: Show in x axis, choose from 0-2
    :param dimension2: Show in y axis, choose form 0-2
    :return: Plot
    '''
    data, labels = load_file(FILE_NAME)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, dimension1], data[:, dimension2], 15.0 * labels, 15.0 * labels)
    plt.show()


def test_dating(ratio=0.10):
    data, labels = load_file(FILE_NAME)
    normalization_data, min_vals, ranges = normalize(data)
    n = normalization_data.shape[0]
    test_count = int(n * ratio)
    error_count = 0.0
    for i in range(test_count):
        result = classify_knn(normalization_data[test_count:, :], labels[test_count:], K, normalization_data[i, :])
        if result != labels[i]:
            error_count += 1
            print("test for id: %d doesn't match, expected: %d, real: %d" % (n - test_count + i, labels[i], result))
    print("the total error rate is: %f" % (error_count / test_count))


def classify_person():
    result_list = ["not at all", "in small doses", "in large doses"]
    flier_miles = float(input("frequent flier miles earned per year?\n"))
    game_percentage = float(input("percentage of time spent playing video games?\n"))
    ice_cream = float(input("liters of ice ice cream consumed per year?\n"))
    data, labels = load_file(FILE_NAME)
    normalization_data, min_vals, ranges = normalize(data)
    input_data = np.array([flier_miles, game_percentage, ice_cream])
    result = classify_knn(normalization_data, labels, K, (input_data - min_vals) / ranges)
    print("You will probably like this person: ", result_list[result - 1])
