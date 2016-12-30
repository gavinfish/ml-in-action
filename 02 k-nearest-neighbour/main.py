from kNN import *

# --------------------------------------------------
#   test k nearest neighbours with basic data set
# --------------------------------------------------

group, label = create_basic_data_set()
plot_basic_data_set(group, label)
test_basic_data_set(0, 0)

# --------------------------------------------------
#   dating site
# --------------------------------------------------

data, labels = dating_load_data("data/datingTestSet2.txt")
print(data)
print(labels)
dating_plot(data, labels)
dating_plot(data, labels, 0, 1)

# test normalize function
data_norm = normalize(data)
print(data_norm)

dating_test(data, labels, 3, 0.5)

# classify_person(data,labels)

# --------------------------------------------------
#   handwriting
# --------------------------------------------------

handwriting_test("data/trainingDigits.zip", "data/testDigits.zip")
