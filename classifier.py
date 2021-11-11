import math
import numpy as np
import random
import pickle


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def calculate_cost(a3, y):
    return np.sum(np.power(a3 - y, 2))


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
train_random_200_output = np.zeros((4, 200))
original_output = np.zeros((4, 200))
# initialize weights with normal random values

w1 = np.random.normal(0, .5,size=(150, 102))
w2 = np.random.normal(0, .5,size=(60, 150))
w3 = np.random.normal(0, .5,size=(4, 60))

# initialize bias vectors zero
b1 = np.zeros((150, 1))
b2 = np.zeros((60, 1))
b3 = np.zeros((4, 1))

true_guess = 0
count = 0
for (a0, label) in train_random_200_input:

    original_label = np.argmax(label, axis=0)
    original_output[:, count] = label[:, 0]
    temp1 = np.matmul(w1, a0) + b1
    a1 = sigmoid(temp1)
    temp2 = np.matmul(w2, a1) + b2
    a2 = sigmoid(temp2)
    temp3 = np.matmul(w3, a2) + b3
    a3 = sigmoid(temp3)
    # print(a3)
    train_random_200_output[:, count] = a3[:, 0]
    count += 1
    predicted_label = np.argmax(a3, axis=0)
    if predicted_label[0] == original_label[0]:
        true_guess += 1

Accuracy = (true_guess / 200) * 100
print("Accuracy of random init : ")
print(Accuracy)
print("---------------------------------------------")

# third phase

# initialize weights with normal random values

w1 = np.random.standard_normal(size=(150, 102))
print(w1)
w2 = np.random.standard_normal(size=(60, 150))
w3 = np.random.standard_normal(size=(4, 60))

# initialize bias vectors zero
b1 = np.zeros((150, 1))
b2 = np.zeros((60, 1))
b3 = np.zeros((4, 1))

learning_rate = 1
epochs = 5
batch_size = 10
epoch_costs = []
epoch_accuracies = []
for e in range(epochs):
    trueGuess = 0
    accuracy = 0
    current_epoch_cost = 0
    random.shuffle(train_random_200_input)
    batch_num = int(math.ceil(200 / batch_size))
    batches = [train_random_200_input[i:i + batch_size] for i in range(0, 200, batch_size)]
    train_random_200_output = np.zeros((4, 200))
    original_output = np.zeros((4, 200))

    for batch in batches:



        grad_w1 = np.zeros((150, 102))
        grad_w2 = np.zeros((60, 150))
        grad_w3 = np.zeros((4, 60))

        grad_b1 = np.zeros((150, 1))
        grad_b2 = np.zeros((60, 1))
        grad_b3 = np.zeros((4, 1))

        grad_a1 = np.zeros((150, 1))
        grad_a2 = np.zeros((60, 1))

        for image in batch:

            # compute output
            (a0, label) = image

            original_label = np.argmax(label, axis=0)

            z1 = (w1 @ a0) + b1
            a1 = sigmoid(z1)

            z2 = (w2 @ a1) + b2
            a2 = sigmoid(z2)


            z3 = (w3 @ a2) + b3
            a3 = sigmoid(z3)

            cost = calculate_cost(a3, label)
            predicted_label = np.argmax(a3, axis=0)
            if predicted_label[0] == original_label[0]:
                trueGuess += 1
            current_epoch_cost += cost

            # backpropagation phase
            # layer 3 grad_w3 and grad_b3

            for j in range(w3.shape[0]):

                grad_b3[j, 0] += (2 * a3[j, 0] - 2 * label[j, 0]) *  sigmoid_prime(z3[j, 0])

                for k in range(w3.shape[1]):

                    grad_w3[j, k] += (2 * a3[j, 0] - 2 * label[j, 0]) * sigmoid_prime(z3[j, 0]) * a2[k, 0]


            # calculate grad_a2 for the next layer

            for k in range(60):
                for j in range(4):
                    grad_a2[k, 0] += 2 * (2* a3[j, 0] - 2*label[j, 0]) * sigmoid_prime(z3[j, 0]) * w3[j, k]

            # layer 2 grad_w2 and grad_b2

            for j in range(w2.shape[0]):
                grad_b2[j, 0] += grad_a2[j, 0] * sigmoid_prime(z2[j, 0])
                for k in range(w2.shape[1]):
                    grad_w2[j, k] += grad_a2[j, 0] *sigmoid_prime(z2[j, 0])  * a1[k, 0]


            # calculate grad_a1 for the next layer

            for k in range(150):
                for j in range(60):
                    grad_a1[k, 0] = grad_a2[j, 0] * sigmoid_prime(z2[j, 0]) * w2[j, k]

            # layer 1 grad_w1 and grad_b1

            for j in range(w1.shape[0]):
                grad_b1[j, 0] += grad_a1[j, 0] * sigmoid_prime(z1[j, 0]) * 1
                for k in range(w1.shape[1]):
                    grad_w1[j, k] += grad_a1[j, 0] * sigmoid_prime(z1[j, 0]) * a0[k, 0]

        # upgrade w and b

        w1 -= (learning_rate * (grad_w1 / batch_size))


        w2 -= (learning_rate * (grad_w2 / batch_size))
        w3 -= (learning_rate * (grad_w3 / batch_size))


        b1 -= (learning_rate * (grad_b1 / batch_size))
        b2 -= (learning_rate * (grad_b2 / batch_size))
        b3 -=(learning_rate * (grad_b3 / batch_size))
    # add average of epoch cost
    epoch_costs.append(current_epoch_cost / 200)
    accuracy = (trueGuess / 200) * 100
    print(accuracy)
    epoch_accuracies.append(accuracy)

print(epoch_accuracies)
