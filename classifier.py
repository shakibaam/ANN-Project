import numpy as np
import random
import pickle


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# loading training set features
f = open("Datasets/train_set_features.pkl", "rb")
train_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=train_set_features2, axis=0)
train_set_features = train_set_features2[:, features_STDs > 52.3]

# changing the range of data between 0 and 1
train_set_features = np.divide(train_set_features, train_set_features.max())

# loading training set labels
f = open("Datasets/train_set_labels.pkl", "rb")
train_set_labels = pickle.load(f)
f.close()

# ------------
# loading test set features
f = open("Datasets/test_set_features.pkl", "rb")
test_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=test_set_features2, axis=0)
test_set_features = test_set_features2[:, features_STDs > 48]

# changing the range of data between 0 and 1
test_set_features = np.divide(test_set_features, test_set_features.max())

# loading test set labels
f = open("Datasets/test_set_labels.pkl", "rb")
test_set_labels = pickle.load(f)
f.close()

# ------------
# preparing our training and test sets - joining datasets and lables
train_set = []
test_set = []

for i in range(len(train_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(train_set_labels[i])] = 1
    label = label.reshape(4, 1)
    train_set.append((train_set_features[i].reshape(102, 1), label))

for i in range(len(test_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(test_set_labels[i])] = 1
    label = label.reshape(4, 1)
    test_set.append((test_set_features[i].reshape(102, 1), label))

# shuffle
random.shuffle(train_set)
random.shuffle(test_set)

# print size
# print(len(train_set))  # 1962
# print(len(test_set))  # 662


# second Phase ---> Feed Forward

# randomly select 200 elements of training set
train_random_200_input = random.sample(train_set, 200)
train_random_200_output = []
# initialize weights with normal random values
# TODO normal
w1 = np.random.rand(150, 102)
w2 = np.random.rand(60, 150)
w3 = np.random.rand(4, 60)

# initialize bias vectors zero
b1 = np.zeros((150, 1))
b2 = np.zeros((60, 1))
b3 = np.zeros((4, 1))

true_guess = 0
for (a0, label) in train_random_200_input:

    original_label = np.argmax(label, axis=0)
    temp1 = np.matmul(w1, a0) + b1
    a1 = sigmoid(temp1)
    temp2 = np.matmul(w2, a1) + b2
    a2 = sigmoid(temp2)
    temp3 = np.matmul(w3, a2) + b3
    output = sigmoid(temp3)
    predicted_label = np.argmax(output, axis=0)
    if predicted_label[0] == original_label[0]:
        true_guess += 1
    train_random_200_output.append(output)

Accuracy = (true_guess / 200) * 100
print(Accuracy)
