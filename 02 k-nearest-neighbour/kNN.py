import numpy as np


def get_basic_test_data():
    '''Generate basic test data

    The data is coordinates with their label in a two dimension board, these coordinates
    are fixed.

    :return: Data set and label:ndarray,ndarray
    '''
    coordinate = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                           [10, 10], [10, 9], [9, 9], [9, 10]])
    labels = np.array(['A'] * 4 + ['B'] * 4)
    return coordinate, labels


def normalize(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    ranges = max_vals - min_vals
    normalization_data = np.zeros(data.shape)
    n = data.shape[0]
    normalization_data = data - np.tile(min_vals, (n, 1))
    normalization_data = normalization_data / np.tile(ranges, (n, 1))
    return normalization_data, min_vals, ranges


def classify_knn(train_data, labels, k, data):
    '''Core method for k nearest neighbour algorithm

    According training data set to decide the label of input data

    :param train_data: Training data:ndarray
    :param labels: Labels for training data:ndarray
    :param k: Threshold:int
    :param data: Input data:ndarray
    :return: Label for input data
    '''
    n = train_data.shape[0]
    diff_matrix = np.tile(data, (n, 1)) - train_data
    sq_diff_matrix = np.square(diff_matrix)
    sq_distances = sq_diff_matrix.sum(axis=1)
    distances = np.sqrt(sq_distances)
    # sort distances between input data and train data
    sorted_index = distances.argsort()
    labels_count_map = {}
    # classify the k closest train data
    for i in range(k):
        label = labels[sorted_index[i]]
        labels_count_map[label] = labels_count_map.get(label, 0) + 1
    # find the most label for the k elements
    target = None
    max_count = 0
    for label, count in labels_count_map.items():
        if count > max_count:
            max_count, target = count, label
    return target


# test cases
def test_basic_data(x, y):
    coordinates, labels = get_basic_test_data()
    target = classify_knn(coordinates, labels, 3, [x, y])
    print("the data is belong to label: ", target)
