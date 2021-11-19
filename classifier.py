import math
import numpy as np
import random
import time
import pickle
import statistics
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def calculate_cost(a3, y):
    c = 0
    for j in range(4):
        c += np.power((a3[j, 0] - y[j, 0]), 2)
    return c


def softmax(z):
    e = np.exp(z)

    return e / np.sum(e)


def softmax_prime(s):
    gradian_m = np.zeros((s.shape[0], s.shape[1]))
    for i in range(gradian_m.shape[0]):
        for j in range(gradian_m.shape[1]):
            if i == j:
                gradian_m[i, j] = s[i] * (1 - s[i])
            else:
                gradian_m[i, j] = -s[i] * s[j]
    return gradian_m


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


def feed_forward(W, b, data_set):
    w1 = W[0]
    w2 = W[1]
    w3 = W[2]
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]
    train_random_200_output = []
    train_random_200_output = np.zeros((4, len(data_set)))
    original_output = np.zeros((4, len(data_set)))

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

        train_random_200_output[:, count] = a3[:, 0]
        count += 1
        predicted_label = np.argmax(a3, axis=0)
        if predicted_label[0] == original_label[0]:
            true_guess += 1

    Accuracy = (true_guess / len(data_set)) * 100
    print("Accuracy of feed forward : ")
    print(Accuracy)
    print("---------------------------------------------")
    return Accuracy


# third phase

def back_propagation(W, b, data_set, epochs, batch_size, learning_rate):
    start_time = time.time()
    w1 = W[0]
    w2 = W[1]
    w3 = W[2]
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]

    epoch_costs = []
    epoch_accuracies = []
    for e in range(epochs):

        np.random.shuffle(data_set)

        batches = [data_set[i:i + batch_size] for i in range(0, len(data_set), batch_size)]

        for batch in batches:

            grad_w1 = np.zeros((150, 102))
            grad_w2 = np.zeros((60, 150))
            grad_w3 = np.zeros((4, 60))

            grad_b1 = np.zeros((150, 1))
            grad_b2 = np.zeros((60, 1))
            grad_b3 = np.zeros((4, 1))

            for image in batch:

                # compute output
                (a0, label) = image

                z1 = (w1 @ a0) + b1
                a1 = sigmoid(z1)

                z2 = (w2 @ a1) + b2
                a2 = sigmoid(z2)

                z3 = (w3 @ a2) + b3
                a3 = sigmoid(z3)

                # backpropagation phase
                # layer 3 grad_w3 and grad_b3

                for j in range(grad_w3.shape[0]):

                    for k in range(grad_w3.shape[1]):
                        grad_w3[j, k] += 2 * (a3[j, 0] - label[j, 0]) * sigmoid_prime(z3[j, 0]) * a2[k, 0]

                for j in range(grad_b3.shape[0]):
                    grad_b3[j, 0] += 2 * (a3[j, 0] - label[j, 0]) * sigmoid_prime(z3[j, 0])

                # calculate grad_a2 for the next layer
                grad_a2 = np.zeros((60, 1))
                for k in range(60):
                    for j in range(4):
                        grad_a2[k, 0] += 2 * (a3[j, 0] - label[j, 0]) * sigmoid_prime(z3[j, 0]) * w3[j, k]

                # layer 2 grad_w2 and grad_b2

                for j in range(grad_w2.shape[0]):

                    for k in range(grad_w2.shape[1]):
                        grad_w2[j, k] += grad_a2[j, 0] * sigmoid_prime(z2[j, 0]) * a1[k, 0]

                for j in range(grad_b2.shape[0]):
                    grad_b2[j, 0] += grad_a2[j, 0] * sigmoid_prime(z2[j, 0])

                # calculate grad_a1 for the next layer
                grad_a1 = np.zeros((150, 1))
                for k in range(150):
                    for j in range(60):
                        grad_a1[k, 0] += grad_a2[j, 0] * sigmoid_prime(z2[j, 0]) * w2[j, k]

                # layer 1 grad_w1 and grad_b1

                for j in range(grad_w1.shape[0]):

                    for k in range(grad_w1.shape[1]):
                        grad_w1[j, k] += grad_a1[j, 0] * sigmoid_prime(z1[j, 0]) * a0[k, 0]

                for j in range(grad_b1.shape[0]):
                    grad_b1[j, 0] += grad_a1[j, 0] * sigmoid_prime(z1[j, 0]) * 1

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

            epoch_cost += calculate_cost(a3, label)
            predicted_label = np.where(a3 == np.amax(a3))
            original_label = np.where(label == np.amax(label))
            if predicted_label == original_label:
                trueGuess += 1

        epoch_costs.append(epoch_cost / len(data_set))
        accuracy = (trueGuess / len(data_set)) * 100
        print(accuracy)
        epoch_accuracies.append(accuracy)

    print(epoch_accuracies)
    # print("average accuracy of training")
    print(statistics.mean(epoch_accuracies))
    # print("average cost of training")
    print(statistics.mean(epoch_costs))

    plt.plot(np.arange(1, 6), epoch_costs)
    plt.show()
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    return [w1, w2, w3], [b1, b2, b3]


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

        np.random.shuffle(data_set)

        batches = [data_set[i:i + batch_size] for i in range(0, len(data_set), batch_size)]

        for batch in batches:

            grad_w1 = np.zeros((150, 102))
            grad_w2 = np.zeros((60, 150))
            grad_w3 = np.zeros((4, 60))

            grad_b1 = np.zeros((150, 1))
            grad_b2 = np.zeros((60, 1))
            grad_b3 = np.zeros((4, 1))

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
                grad_a2 = np.zeros((60, 1))
                grad_a2 += np.transpose(w3) @ (2 * (a3 - label) * (a3 * (1 - a3)))

                # layer 2 grad_w2 and grad_b2

                grad_b2 += grad_a2 * (a2 * (1 - a2))

                grad_w2 += ((a2 * (1 - a2)) * grad_a2) @ np.transpose(a1)

                # calculate grad_a1 for the next layer
                grad_a1 = np.zeros((150, 1))

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
            for j in range(4):
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


def momentum(W, b, data_set, epochs, batch_size, learning_rate):
    momentum_factor = 0.1

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
        train_random_200_output = np.zeros((4, len(data_set)))
        original_output = np.zeros((4, len(data_set)))
        w1_accumulate = 0
        w2_accumulate = 0
        w3_accumulate = 0

        b1_accumulate = 0
        b2_accumulate = 0
        b3_accumulate = 0

        for batch in batches:

            grad_w1 = np.zeros((150, 102))
            grad_w2 = np.zeros((60, 150))
            grad_w3 = np.zeros((4, 60))

            grad_b1 = np.zeros((150, 1))
            grad_b2 = np.zeros((60, 1))
            grad_b3 = np.zeros((4, 1))

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
                grad_a2 = np.zeros((60, 1))
                grad_a2 += np.transpose(w3) @ (2 * (a3 - label) * (a3 * (1 - a3)))

                # layer 2 grad_w2 and grad_b2

                grad_b2 += grad_a2 * (a2 * (1 - a2))

                grad_w2 += ((a2 * (1 - a2)) * grad_a2) @ np.transpose(a1)

                # calculate grad_a1 for the next layer
                grad_a1 = np.zeros((150, 1))

                grad_a1 += np.transpose(w2) @ (grad_a2 * (a2 * (1 - a2)))

                # layer 1 grad_w1 and grad_b1

                grad_b1 += grad_a1 * (a1 * (1 - a1))

                grad_w1 += (grad_a1 * (a1 * (1 - a1))) @ np.transpose(a0)
            # upgrade w and b

            w1 -= ((learning_rate * (grad_w1 / batch_size)) + w1_accumulate)
            w2 -= ((learning_rate * (grad_w2 / batch_size)) + w2_accumulate)
            w3 -= ((learning_rate * (grad_w3 / batch_size)) + w3_accumulate)

            w1_accumulate = ((learning_rate * (grad_w1 / batch_size)) + w1_accumulate) * momentum_factor
            w2_accumulate = ((learning_rate * (grad_w2 / batch_size)) + w2_accumulate) * momentum_factor
            w3_accumulate = ((learning_rate * (grad_w3 / batch_size)) + w3_accumulate) * momentum_factor

            b1 -= ((learning_rate * (grad_b1 / batch_size)) + b1_accumulate)
            b2 -= ((learning_rate * (grad_b2 / batch_size)) + b2_accumulate)
            b3 -= ((learning_rate * (grad_b3 / batch_size)) + b3_accumulate)

            b1_accumulate = ((learning_rate * (grad_b1 / batch_size)) + b1_accumulate) * momentum_factor
            b2_accumulate = ((learning_rate * (grad_b2 / batch_size)) + b2_accumulate) * momentum_factor
            b3_accumulate = ((learning_rate * (grad_b3 / batch_size)) + b3_accumulate) * momentum_factor
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
            for j in range(4):
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


#
#
def back_prop_softmax(W, b, data_set, epochs, batch_size, learning_rate):
    w1 = W[0]
    w2 = W[1]
    w3 = W[2]
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]

    epoch_costs = []
    epoch_accuracies = []
    for time in range(10):
        for e in range(epochs):
            trueGuess = 0
            accuracy = 0
            current_epoch_cost = 0
            np.random.shuffle(data_set)
            # batch_num = int(math.ceil(200 / batch_size))
            batches = [data_set[i:i + batch_size] for i in range(0, len(data_set), batch_size)]
            train_random_200_output = np.zeros((4, len(data_set)))
            original_output = np.zeros((4, len(data_set)))

            for batch in batches:

                grad_w1 = np.zeros((150, 102))
                grad_w2 = np.zeros((60, 150))
                grad_w3 = np.zeros((4, 60))

                grad_b1 = np.zeros((150, 1))
                grad_b2 = np.zeros((60, 1))
                grad_b3 = np.zeros((4, 1))

                for image in batch:
                    # compute output
                    (a0, label) = image

                    z1 = w1 @ a0 + b1

                    a1 = softmax(z1)

                    z2 = w2 @ a1 + b2

                    a2 = softmax(z2)

                    z3 = w3 @ a2 + b3

                    a3 = softmax(z3)

                    # backpropagation phase
                    # layer 3 grad_w3 and grad_b3

                    grad_b3 += 2 * (a3 - label) * (softmax_prime(z3))

                    grad_w3 += (2 * (a3 - label) * (softmax_prime(z3))) @ np.transpose(a2)

                    # calculate grad_a2 for the next layer

                    #
                    grad_a2 = np.zeros((60, 1))
                    grad_a2 += np.transpose(w3) @ (2 * (a3 - label) * (softmax_prime(z3)))

                    # layer 2 grad_w2 and grad_b2

                    grad_b2 += grad_a2 * (softmax_prime(z2))

                    grad_w2 += ((softmax_prime(z2)) * grad_a2) @ np.transpose(a1)

                    # calculate grad_a1 for the next layer
                    grad_a1 = np.zeros((150, 1))

                    grad_a1 += np.transpose(w2) @ (grad_a2 * (softmax_prime(z2)))

                    # layer 1 grad_w1 and grad_b1

                    grad_b1 += grad_a1 * (softmax_prime(z1))

                    grad_w1 += (grad_a1 * (softmax_prime(z1))) @ np.transpose(a0)

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
                a1 = softmax(z1)

                z2 = (w2 @ a1) + b2
                a2 = softmax(z2)

                z3 = (w3 @ a2) + b3
                a3 = softmax(z3)
                e = np.exp(a3)
                cost = (np.log(e/np.sum(e))) * -1
                epoch_cost += cost

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

    # uncomment what ever you want to Run ;)

    w1 = np.random.standard_normal(size=(150, 102))
    w2 = np.random.standard_normal(size=(60, 150))
    w3 = np.random.standard_normal(size=(4, 60))
    W = [w1, w2, w3]

    # initialize bias vectors zero
    b1 = np.zeros((150, 1))
    b2 = np.zeros((60, 1))
    b3 = np.zeros((4, 1))
    b = [b1, b2, b3]
    epochs = 10
    batch_size = 10
    learning_rate = 1
    train_random_200_input = random.sample(train_set, 200)
    print(" Welcome To Fruit Classifier App :)")
    # print("lets feed forward with random weights :")
    # feed_forward(W,b,train_random_200_input)
    # print("now try non-vectorized back prop with 200 train set :")


    # W, b = back_propagation(W, b, train_random_200_input, epochs, batch_size, learning_rate)

    # print("well now train the model with all train set :")
    # avg_acc_train = 0
    # avg_acc_test = 0
    # for i in range(10):
    #     w1 = np.random.standard_normal(size=(150, 102))
    #     w2 = np.random.standard_normal(size=(60, 150))
    #     w3 = np.random.standard_normal(size=(4, 60))
    #     W = [w1, w2, w3]
    #
    #     # initialize bias vectors zero
    #     b1 = np.zeros((150, 1))
    #     b2 = np.zeros((60, 1))
    #     b3 = np.zeros((4, 1))
    #     b = [b1, b2, b3]
    #     W, b, acc = vectorized_back_propagation(W, b, train_set, epochs, batch_size, learning_rate)
    #     avg_acc_train += acc
    #     avg_acc_test += feed_forward(W, b, test_set)
    # print("Avg Train")
    # print(avg_acc_train / 10)
    # print("==============================")
    # print("Avg Test")
    # print(avg_acc_test / 10)

    W, b, acc = back_prop_softmax(W, b, train_set, epochs, batch_size, learning_rate)
    print(acc)
    # acc_test= feed_forward(W, b, test_set)
    # print(acc_test)


if __name__ == '__main__':
    main()
