import math
import numpy as np
import random
import time
import pickle
import statistics
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def calculate_cost(a3, y):
    return np.sum(np.power(a3 - y, 2))


def softmax(z):
    e = np.exp(z)
    return e / np.sum(e, axis=1)


def softmax_prime(s):
    gradian_m = np.diag(s)
    for i in range(len(gradian_m)):
        for j in range(len(gradian_m)):
            if i == j:
                gradian_m[i][j] = s[i] * (1 - s[i])
            else:
                gradian_m[i][j] = -s[i] * s[j]
    return gradian_m


# loading training set features
f = open("D:\\ترم7\\Compuational Inteligence\\project1\\Fruit_classifier\\Emtiyazi\\Datasets\\train_set_features.pkl", "rb")
train_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=train_set_features2, axis=0)
train_set_features = train_set_features2[:, features_STDs > 52.3]

# changing the range of data between 0 and 1
train_set_features = np.divide(train_set_features, train_set_features.max())

# loading training set labels
f = open("D:\\ترم7\\Compuational Inteligence\\project1\\Fruit_classifier\\Emtiyazi\\Datasets\\train_set_labels.pkl", "rb")
train_set_labels = pickle.load(f)
f.close()

# ------------
# loading test set features
f = open("D:\\ترم7\\Compuational Inteligence\\project1\\Fruit_classifier\\Emtiyazi\\Datasets\\test_set_features.pkl", "rb")
test_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=test_set_features2, axis=0)
test_set_features = test_set_features2[:, features_STDs > 48]

# changing the range of data between 0 and 1
test_set_features = np.divide(test_set_features, test_set_features.max())

# loading test set labels
f = open("D:\\ترم7\\Compuational Inteligence\\project1\\Fruit_classifier\\Emtiyazi\\Datasets\\test_set_labels.pkl", "rb")
test_set_labels = pickle.load(f)
f.close()

# ------------
# preparing our training and test sets - joining datasets and lables
train_set = []
test_set = []

for i in range(len(train_set_features)):
    label = np.array([0, 0, 0, 0 ,0])
    label[int(train_set_labels[i])] = 1
    label = label.reshape(5, 1)
    train_set.append((train_set_features[i].reshape(116, 1), label))

for i in range(len(test_set_features)):
    label = np.array([0, 0, 0, 0 ,0])
    label[int(test_set_labels[i])] = 1
    label = label.reshape(5, 1)
    test_set.append((test_set_features[i].reshape(116, 1), label))

# shuffle
random.shuffle(train_set)
random.shuffle(test_set)


# print size
# print(len(train_set))  # 1962
# print(len(test_set))  # 662


def feed_forward(W, b, data_set):
    w1 = W[0]
    w2 = W[1]
    w3 = W[2]
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]
    train_random_200_output = []
    train_random_200_output = np.zeros((5, len(data_set)))
    original_output = np.zeros((5, len(data_set)))

    true_guess = 0
    count = 0
    for (a0, label) in data_set:
        # print(a0.shape)

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

    Accuracy = (true_guess / len(data_set)) * 100
    print("Accuracy of forward feed : ")
    print(Accuracy)
    print("---------------------------------------------")


# 4th phase vectorization
def vectorized_back_propagation(W, b, data_set, epochs, batch_size, learning_rate):
    w1 = W[0]
    w2 = W[1]
    w3 = W[2]
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]

    epoch_costs = []
    epoch_accuracies = []

    for e in range(epochs):
        trueGuess = 0
        accuracy = 0
        current_epoch_cost = 0
        np.random.shuffle(data_set)
        # batch_num = int(math.ceil(200 / batch_size))
        batches = [data_set[i:i + batch_size] for i in range(0, len(data_set), batch_size)]
        train_random_200_output = np.zeros((5, len(data_set)))
        original_output = np.zeros((5, len(data_set)))

        for batch in batches:

            grad_w1 = np.zeros((200, 116))
            grad_w2 = np.zeros((160, 200))
            grad_w3 = np.zeros((5, 160))

            grad_b1 = np.zeros((200, 1))
            grad_b2 = np.zeros((160, 1))
            grad_b3 = np.zeros((5, 1))

            for image in batch:
                # compute output
                (a0, label) = image

                z1 = w1 @ a0 + b1
                a1 = sigmoid(z1)

                z2 = w2 @ a1 + b2
                a2 = sigmoid(z2)

                z3 = w3 @ a2 + b3
                a3 = sigmoid(z3)

                # backpropagation phase
                # layer 3 grad_w3 and grad_b3

                grad_b3 += 2 * (a3 - label) * a3 * (1 - a3)

                grad_w3 += (2 * (a3 - label) * (a3 * (1 - a3))) @ np.transpose(a2)

                # calculate grad_a2 for the next layer

                #
                grad_a2 = np.zeros((160, 1))
                grad_a2 += np.transpose(w3) @ (2 * (a3 - label) * (a3 * (1 - a3)))

                # layer 2 grad_w2 and grad_b2

                grad_b2 += grad_a2 * (a2 * (1 - a2))

                grad_w2 += ((a2 * (1 - a2)) * grad_a2) @ np.transpose(a1)

                # calculate grad_a1 for the next layer
                grad_a1 = np.zeros((200, 1))

                grad_a1 += np.transpose(w2) @ (grad_a2 * (a2 * (1 - a2)))

                # layer 1 grad_w1 and grad_b1

                grad_b1 += grad_a1 * (a1 * (1 - a1))

                grad_w1 += (grad_a1 * (a1 * (1 - a1))) @ np.transpose(a0)

            # upgrade w and b

            w1 -= (learning_rate * (grad_w1 / batch_size))

            w2 -= (learning_rate * (grad_w2 / batch_size))
            w3 -= (learning_rate * (grad_w3 / batch_size))

            b1 -= (learning_rate * (grad_b1 / batch_size))
            b2 -= (learning_rate * (grad_b2 / batch_size))
            b3 -= (learning_rate * (grad_b3 / batch_size))
            # add average of epoch cost
        epoch_cost = 0
        trueGuess = 0
        for data in data_set:
            (a0, label) = data

            z1 = (w1 @ a0) + b1
            a1 = sigmoid(z1)

            z2 = (w2 @ a1) + b2
            a2 = sigmoid(z2)

            z3 = (w3 @ a2) + b3
            a3 = sigmoid(z3)
            c = 0
            for j in range(5):
                c += np.power((a3[j, 0] - label[j, 0]), 2)
            epoch_cost += c

            predicted_label = np.where(a3 == np.amax(a3))
            original_label = np.where(label == np.amax(label))
            if predicted_label == original_label:
                trueGuess += 1

        epoch_costs.append(epoch_cost / len(data_set))
        accuracy = (trueGuess / len(data_set)) * 100
        print(accuracy)
        epoch_accuracies.append(accuracy)

    print(epoch_accuracies)
    print("average accuracy of training")
    print(statistics.mean(epoch_accuracies))
    print("average cost of training")
    print(statistics.mean(epoch_costs))
    plt.plot([x for x in range(epochs)], epoch_costs)
    plt.show()
    return [w1, w2, w3], [b1, b2, b3], statistics.mean(epoch_accuracies)



def main():
    w1 = np.random.standard_normal(size=(200, 116))
    w2 = np.random.standard_normal(size=(160, 200))
    w3 = np.random.standard_normal(size=(5, 160))
    W = [w1, w2, w3]

    # initialize bias vectors zero
    b1 = np.zeros((200, 1))
    b2 = np.zeros((160, 1))
    b3 = np.zeros((5, 1))
    b = [b1, b2, b3]
    epochs = 25
    batch_size = 10
    learning_rate = 0.4
    train_random_200_input = random.sample(train_set, 200)
    print(" Welcome To Fruit Classifier App :)")
    # print("lets feed forward with random weights :")
    # feed_forward(W,b,train_random_200_input)
    # print("now try non-vectorized back prop with 200 train set :")

    W , b , acc = vectorized_back_propagation(W, b, train_set , epochs , batch_size, learning_rate)
    print("Acc Train :")
    print(acc)
    print("===========================================")
    feed_forward(W,b,test_set)
    # print("well now train the model with all train set :")
    # W, b = vectorized_back_propagation(W, b, train_set , epochs , batch_size, learning_rate)
    # print("test set")
    # vectorized_back_propagation(W, b, test_set, epochs, batch_size, learning_rate)


if __name__ == '__main__':
    main()
