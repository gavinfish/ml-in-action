from collections import defaultdict
import zipfile
import numpy as np
import matplotlib.pyplot as plt

SEP = "\t"


# ----------------------------------------------------------------
#   K Nearest Neighbors
# ----------------------------------------------------------------

def create_basic_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return group, labels


def plot_basic_data_set(group, labels):
    for point, label in zip(group, labels):
        plt.scatter(point[0], point[1])
        plt.text(point[0] + 0.01, point[1] + 0.01, label)
    plt.show()


def classify_knn(data_train, labels, k, data_test):
    n = data_train.shape[0]
    diff_matrix = np.tile(data_test, (n, 1)) - data_train
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


def test_basic_data_set(x, y):
    coordinates, labels = create_basic_data_set()
    target = classify_knn(coordinates, labels, 3, [x, y])
    print("the data is belong to label: ", target)


# ----------------------------------------------------------------
#   Improving matches from a dating site with kNN
# ----------------------------------------------------------------

def dating_load_data(filename):
    with open(filename) as f:
        data_list, labels_list = [], []
        for line in f.readlines():
            line_list = line.strip().split(SEP)
            data_list.append(line_list[:3])
            labels_list.append(int(line_list[-1]))
        data, labels = np.array(data_list).astype(float), np.array(labels_list)
        return data, labels


def dating_plot(data, labels, dimension1=1, dimension2=2):
    data_dict = defaultdict(list)
    for d, label in zip(data, labels):
        data_dict[label].append(d)
    colors = ["red", "green", "blue"]
    legends = ["Did Not Like", "Liked in Small Doses", "Liked in Large Doses"]
    for k, v in data_dict.items():
        v = np.array(v)
        index = k - 1
        plt.scatter(v[:, dimension1], v[:, dimension2], c=colors[index], label=legends[index])
    label_texts = ["Frequent Flyer Miles Earned Per Year",
                   "Percentage of Time Spent Playing Video Games",
                   "Liters of Ice Cream Consumed Per Week"]
    plt.xlabel(label_texts[dimension1])
    plt.ylabel(label_texts[dimension2])
    plt.legend()
    plt.show()


def normalize(data):
    min_values = data.min(0)
    max_values = data.max(0)
    ranges = max_values - min_values
    n = data.shape[0]
    normalization_data = (data - np.tile(min_values, (n, 1))) / np.tile(ranges, (n, 1))
    return normalization_data


def dating_test(data, labels, k=3, ratio=0.10):
    normalization_data = normalize(data)
    n = normalization_data.shape[0]
    test_count = int(n * ratio)
    error_count = 0.0
    for i in range(test_count):
        result = classify_knn(normalization_data[test_count:, :], labels[test_count:], k, normalization_data[i, :])
        if result != labels[i]:
            error_count += 1
            print("test for id: %d doesn't match, expected: %d, real: %d" % (n - test_count + i, labels[i], result))
    print("the total error rate is: %f" % (error_count / test_count))


def classify_person(data, labels, k=3):
    result_list = ["not at all", "in small doses", "in large doses"]
    game_percentage = float(input("percentage of time spent playing video games?\n"))
    flier_miles = float(input("frequent flier miles earned per year?\n"))
    ice_cream = float(input("liters of ice ice cream consumed per year?\n"))
    normalization_data = normalize(data)
    input_data = np.array([flier_miles, game_percentage, ice_cream])
    min_values = data.min(0)
    ranges = data.max(0) - min_values
    result = classify_knn(normalization_data, labels, k, (input_data - min_values) / ranges)
    index = result - 1
    print("You will probably like this person: ", result_list[index])


# ----------------------------------------------------------------
#   Handwriting Recognition System
# ----------------------------------------------------------------

def __img2vector(img_file):
    vector = np.zeros((1, 1024))
    img_width = 32
    img_height = 32
    for i in range(img_height):
        line = img_file[i]
        for j in range(img_width):
            vector[0, img_width * i + j] = int(line[j])
    return vector


def handwriting_test(training_zip_path, test_zip_path, k=3):
    training_zf = zipfile.ZipFile(training_zip_path, "r")
    n = len(training_zf.namelist())
    training_data = np.zeros((n, 1024))
    labels = []
    for i in range(n):
        filename = training_zf.namelist()[i]
        # get label
        training_label = int(filename.split("_")[0])
        labels.append(training_label)
        # get training data
        f = training_zf.open(filename)
        img_file = f.readlines()
        training_data[i, :] = __img2vector(img_file)
        f.close()

    error_count = 0.0
    test_zf = zipfile.ZipFile(test_zip_path, "r")
    m = len(test_zf.namelist())
    for i in range(m):
        filename = test_zf.namelist()[i]
        test_label = int(filename.split("_")[0])
        f = test_zf.open(filename)
        img_file = f.readlines()
        input_data = __img2vector(img_file)
        # k nearest neighbour algorithm
        result = classify_knn(training_data, labels, k, input_data)
        if result != test_label:
            print("test file %s gets wrong answer, expected: %d, result: %d" % (filename, test_label, result))
            error_count += 1
    print("total number of errors is %d" % error_count)
    print("error rate is %f" % (error_count / m))
