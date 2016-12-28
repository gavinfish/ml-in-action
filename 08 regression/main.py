from regression import *

# --------------------------------------------------
# train data set ex0.txt with standard regression
# --------------------------------------------------

data, label = load_data_and_label("data/ex0.txt")
w = standard_regression(data, label)
print("Coefficients: ", w)
predict = np.dot(data, w)
standard_regression_plot("Standard Regression for data set ex0.txt", data, label, predict)
print("Correlation coefficient: ")
print(np.corrcoef(predict.T, label))

# --------------------------------------------------
# train data set ex0.txt with locally weighted linear regression
# --------------------------------------------------

lwlr_plot(data, label, [1, 0.01, 0.003])

# --------------------------------------------------
# train data set abalone.txt with LWLR
# --------------------------------------------------

abalone_test()

# --------------------------------------------------
#   train data set abalone.txt with ridge regression
# --------------------------------------------------

data_abalone, label_abalone = load_data_and_label("data/abalone.txt")
ridge_weights = ridge_test(data_abalone, label_abalone)
ridge_plot(ridge_weights)

# --------------------------------------------------
#   train data set abalone.txt with forward stage wise linear regression
# --------------------------------------------------

stage_wise(data_abalone, label_abalone, 0.01, 200)
stage_wise(data_abalone, label_abalone, 0.001, 5000)
abalone_standard_regression_test(data_abalone, label_abalone)
stage_wise_plot(data_abalone, label_abalone)
