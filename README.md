# Fruit Classification using Neural Networks

## Project Overview

The goal of this project is to classify different types of fruits using neural networks, specifically Fully Connected Feedforward networks. We will tackle the fruit classification problem using these neural networks, which we have learned about in our course.

In this problem, our input consists of four types of fruits, and the model needs to identify the fruits correctly. The dataset we will use is called "Fruit360." This dataset contains a diverse range of fruit types, with images captured in 360 degrees. However, for simplicity and to improve the algorithm's accuracy, we have selected only four fruit classes for this project.

In the dataset, each class comprises approximately 491 images with dimensions of 100x100 pixels. Considering the direct use of image pixels as the neural network's input would result in a large input layer with 10,000 neurons, making the network overly complex. To address this, we initially employ feature extraction techniques to extract 360 features. We then apply feature vector dimension reduction techniques, reducing the size to 102, which will serve as the input to our neural network.

The structure of the neural network is as follows:

![Image 1](https://github.com/shakibaam/ANN-Project/blob/master/structure.png)

## Formulas

In neural networks, the following formulas are commonly used for calculations:

### Cost Function

The cost function (also known as the loss function) is used to calculate the error between the predicted output and the actual desired output.
 ![Cost-Function Formula](https://github.com/shakibaam/ANN-Project/blob/master/cost%20function.png)


### Gradient Descent

Gradient Descent is an optimization algorithm used to update the weights and biases of the neural network based on the calculated gradients. It is commonly used for minimizing the cost function. The general formula for gradient descent is:

  ![Weight Update Formula](https://github.com/shakibaam/ANN-Project/blob/master/grediant%20descent.png)

  ### Output Calculation

  ![Output Calculation Formula](https://github.com/shakibaam/ANN-Project/blob/master/weight%20update.png)



## Smaple Result:

We trained the neural network model using the following configuration:

- Learning Rate: 0.5
- Number of Samples: 200
- Number of Epochs: 20
- Accuracy: 99%

Here is the cost function output in each epoch of training the ann:

![Image 1](https://github.com/shakibaam/ANN-Project/blob/master/result.png)


