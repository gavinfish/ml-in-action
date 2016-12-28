import matplotlib.pyplot as plt
import numpy as np

SEP = "\t"


# ----------------------------------------------------------------
#   Standard Linear Regression
# ----------------------------------------------------------------

def load_data_and_label(file_name):
    data, label = [], []
    with open(file_name) as f:
        for line in f.readlines():
            items = list(map(lambda x: float(x), line.split(SEP)))
            data.append(items[:-1])
            label.append(items[-1])
    data, label = np.array(data), np.array(label)
    return data, label


def standard_regression(data_train, label_train):
    data_train, label_train = np.array(data_train), np.array(label_train).T
    xtx_matrix = np.dot(data_train.T, data_train)
    if np.linalg.det(xtx_matrix) == 0.0:
        print("This is matrix is singular, cannot do inverse")
        return
    weights = np.dot(np.dot(np.linalg.inv(xtx_matrix), data_train.T), label_train)
    return weights


def standard_regression_plot(title, data, label, predict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    __regression_plot(ax, data, label, predict)
    plt.title(title)
    plt.show()


def __regression_plot(_plt, data, label, predict):
    _plt.scatter(data[:, 1], label)
    sort_index = data[:, 1].argsort()
    _plt.plot(data[:, 1][sort_index], predict[sort_index], color="red", linewidth=3)


# ----------------------------------------------------------------
#   Locally Weighted Linear Regression
# ----------------------------------------------------------------

def lwlr(test_point, data_train, label_train, k=0.1):
    m = data_train.shape[0]
    weights = np.eye(m)
    for j in range(m):
        diff = test_point - data_train[j, :]
        weights[j, j] = np.exp(np.dot(diff, diff.T) / (-2.0 * k ** 2))
    xtx_matrix = np.dot(data_train.T, np.dot(weights, data_train))
    if np.linalg.det(xtx_matrix) == 0.0:
        print()
        return
    weights = np.dot(np.linalg.inv(xtx_matrix), np.dot(data_train.T, np.dot(weights, label_train)))
    return np.dot(test_point, weights)


def lwlr_test(test_data, data_train, label_train, k=0.1):
    predict = [lwlr(data, data_train, label_train, k) for data in test_data]
    predict = np.array(predict)
    return predict


def lwlr_plot(data, label, ks):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    predicts = [lwlr_test(data, data, label, k) for k in ks]
    for p, k, predict in zip([ax1, ax2, ax3], ks, predicts):
        __regression_plot(p, data, label, predict)
        p.set_title("Locally Weighted Linear Regression k = %2f" % k)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# ----------------------------------------------------------------
#   Abalone Test with LWLR
# ----------------------------------------------------------------

def rss_error(label_test, predict):
    return np.sum(np.square(label_test - predict))


def abalone_test():
    data, label = load_data_and_label('data/abalone.txt')
    data_train, label_train = data[0:99], label[0:99]
    data_test, label_test = data[100:199], label[100:199]
    ks = [0.1, 1, 10]
    predicts_train = [lwlr_test(data_train, data_train, label_train, k) for k in ks]
    for predict, k in zip(predicts_train, ks):
        rss = rss_error(label_train, predict)
        print("Residual sum of squares in train data with k = %f is %f" % (k, rss))

    # TODO check why when k=0.1 the rss value is not stable
    predicts_test = [lwlr_test(data_test, data_train, label_train, k) for k in ks]
    for predict, k in zip(predicts_test, ks):
        rss = rss_error(label_test, predict)
        print("Residual sum of squares in test data with k = %f is %f" % (k, rss))

    w = standard_regression(data_train, label_train)
    predict = np.dot(data_test, w)
    rss = rss_error(label_test, predict)
    print("Standard regression's residual sum of square in test data: %f" % rss)


# ----------------------------------------------------------------
#   Ridge Regression
# ----------------------------------------------------------------

def ridge_regression(data_train, label_train, lam=0.2):
    xtx_matrix = np.dot(data_train.T, data_train)
    denom = xtx_matrix + np.eye(data_train.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This is matrix is singular, cannot do inverse")
        return
    w = np.dot(np.linalg.inv(denom), np.dot(data_train.T, label_train))
    return w


def regularize(data, label=np.array([])):
    data_mean = data.mean(0)
    data_var = data.var(0)
    data_norm = (data - data_mean) / data_var
    label_mean = label.mean(0)
    label_norm = label - label_mean
    return data_norm, label_norm


def ridge_test(data_train, label_train):
    # normalization
    data_norm, label_norm = regularize(data_train, label_train)

    lam_count = 30
    weights = np.zeros((lam_count, data_train.shape[1]))
    for i in range(lam_count):
        w = ridge_regression(data_norm, label_norm, np.exp(i - 10))
        weights[i, :] = w.T
    return weights


def ridge_plot(weights):
    plt.plot(weights)
    plt.xlabel("log(lambda)")
    plt.ylabel("w(i)")
    plt.show()


# ----------------------------------------------------------------
#   Forward Stepwise Regression
# ----------------------------------------------------------------

def stage_wise(data_train, label_train, eps=0.01, num=100):
    data_norm, label_norm = regularize(data_train, label_train)
    m, n = data_norm.shape
    weights_record = np.zeros((num, n))
    weights = np.zeros((n, 1))
    weights_test, weights_max = weights.copy(), weights.copy()
    for i in range(num):
        print(weights.T)
        lowest_error = np.inf
        for j in range(n):
            for sign in [-1, 1]:
                weights_test = weights.copy()
                weights_test[j] += eps * sign
                label_test = np.dot(data_norm, weights_test)
                rss = rss_error(label_norm, label_test.T)
                if rss < lowest_error:
                    lowest_error = rss
                    weights_max = weights_test
        weights = weights_max.copy()
        weights_record[i, :] = weights.T
    return weights_record


def abalone_standard_regression_test(data_train, label_train):
    data_norm, label_norm = regularize(data_train, label_train)
    weights = standard_regression(data_norm, label_norm)
    print(weights.T)


def stage_wise_plot(data_train, label_train, eps=0.005, num=1000):
    weights = stage_wise(data_train, label_train, eps, num)
    plt.plot(weights)
    plt.show()

    # ----------------------------------------------------------------
    #   LEGO
    # ----------------------------------------------------------------

    # TODO: google API has changed
