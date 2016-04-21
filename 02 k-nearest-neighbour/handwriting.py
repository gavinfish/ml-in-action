import numpy as np
import zipfile
from kNN import classify_knn

TRAINING_DIGITS_FILE = "data/trainingDigits.zip"
TEST_DIGITS_FILE = "data/testDigits.zip"
K = 3


def img2vector(img_file):
    vector = np.zeros((1, 1024))
    img_width = 32
    img_height = 32
    for i in range(img_height):
        line = img_file[i]
        for j in range(img_width):
            vector[0, img_width * i + j] = int(line[j])
    return vector


def test_handwriting():
    training_zf = zipfile.ZipFile(TRAINING_DIGITS_FILE, "r")
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
        training_data[i, :] = img2vector(img_file)

    error_count = 0.0
    test_zf = zipfile.ZipFile(TEST_DIGITS_FILE, "r")
    m = len(test_zf.namelist())
    for i in range(m):
        filename = test_zf.namelist()[i]
        test_label = int(filename.split("_")[0])
        f = test_zf.open(filename)
        img_file = f.readlines()
        input_data = img2vector(img_file)
        # k nearest neighbour algorithm
        result = classify_knn(training_data, labels, K, input_data)
        if result != test_label:
            print("test file %s gets wrong answer, expected: %d, result: %d" % (filename, test_label, result))
            error_count += 1
    print("error rate is %f" % (error_count / n))
