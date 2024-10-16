"""
A script that creates and trains a 1-layer neural network from scratch.

It is a rework of the script developed while following along with
Chapter 5: Neural Network from Scratch, in the course PyTorch Ultimate 2024 -
From Basics to Cutting-Edge by Bert Gollnick. The code for this chapter can be
found at:
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/tree/main/015_NeuralNetworkFromScratch

The goal of this script is to pick apart what was put together very quickly, derive some
basic—if incomplete—understanding, and present it in the hopes that it helps someone else.
"""

from collections import Counter

import numpy as np
import numpy.random
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix

# Constants
RANDOM = numpy.random.default_rng()
LEARNING_RATE = 0.01
ITERATIONS = 10000


class NeuralNetworkFromScratch:
    """
    This class implements a simple, 1-layer neural network from scratch.
    """

    def __init__(self, learning_rate, x_train, y_train, x_test, y_test):
        """
        Creates a new instance
        :param learning_rate: the learning rate of the neural network
        :param x_train: the independent variables of the training set
        :param y_train: the dependent variable of the training set
        :param x_test: the independent variables of the test set
        :param y_test: the dependent variable of the test set
        """

        # Initialize the weights and bias
        self.w = RANDOM.random(x_train.shape[1])
        self.b = RANDOM.random()

        # Set training values from the parameters
        self.learning_rate: float = learning_rate
        self.x_train = x_train
        self.y_train = y_train

        # Set the test values
        self.x_test = x_test
        self.y_test = y_test

        # Create values to hold loss
        self.l_train = []
        self.l_test = []

    @staticmethod
    def activation(x):
        """
        Calculates the activation value for a given x
        :param x: the input value
        :return: the activation value
        """

        # Calculate the activation using a sigmoid function
        return 1 / (1 + np.exp(-x))

    def d_activation(self, x):
        """
        Calculates the derivative of the activation function for a given x
        :param x: the input value
        :return: the derivative of the activation function
        """
        activation = self.activation(x)
        return activation * (1 - activation)

    def hidden(self, x):
        """
        Calculates the hidden value of a given x
        :param x: the input value
        :return: the hidden value of a given x
        """
        return np.dot(x, self.w) + self.b

    def forward(self, x):
        """
        Computes the forward step for a given input x
        :param x: the input value
        :return: the activation of the input x
        """
        hidden = self.hidden(x)
        return self.activation(hidden)

    def backward(self, x, y_true):
        """
        Runs the backward pass of the neural network
        :param x: the data point
        :param y_true: the true y value
        :return: the partial derivatives with respect to the bias and weights
        """

        # Calculate the gradients
        hidden = self.hidden(x)
        y_pred = self.forward(x)

        dl_dpred = 2 * (y_pred - y_true)

        dpred_dhidden = self.d_activation(hidden)

        dhidden_db = 1
        dhidden_dw = x

        dl_db = dl_dpred * dpred_dhidden * dhidden_db
        dl_dw = dl_dpred * dpred_dhidden * dhidden_dw

        return dl_db, dl_dw

    def optimize(self, dl_db, dl_dw):
        """
        Runs the optimization step of backprop, updating the weights and bias

        :param dl_db: the derivative of the loss function with respect to the bias
        :param dl_dw: the derivative of the loss function with respect to the weights
        """

        # Update the bias and weights
        self.b = self.b - (dl_db * self.learning_rate)
        self.w = self.w - (dl_dw * self.learning_rate)

    def train(self, iterations):
        """
        Runs the model training over the given number of iterations
        :param iterations: the number of iterations
        :return:
        """
        for _ in range(iterations):

            # Pick a random position
            random_pos = RANDOM.integers(len(self.x_train))

            # Calculate the forward pass for that value
            x = self.x_train[random_pos]
            y_train_pred = self.forward(x)

            # Calculate the loss
            y_train_true = self.y_train[random_pos]
            loss = np.sum(np.square(y_train_pred - y_train_true))
            self.l_train.append(loss)

            # Calculate the gradients
            dl_db, dl_dw = self.backward(x, y_train_true)

            # Optimize
            self.optimize(dl_db, dl_dw)

            # Calculate error using test data
            l_sum = 0
            for j in range(len(self.x_test)):
                x_test = self.x_test[j]
                y_test_true = self.y_test[j]
                y_test_pred = self.forward(x_test)
                l_sum += np.square(y_test_pred - y_test_true)
            self.l_test.append(l_sum)

        return "Training complete"


def main():
    """
    Creates a simple, 1-layer neural network, trains it, and calculates its accuracy against a test set
    """

    # Load the data
    # Source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
    df = pd.read_csv("../data/heart.csv")
    print(df.head())

    # Pull out the independent and dependent features
    x = np.array(df.loc[:, df.columns != "output"])
    y = np.array(df["output"])
    print(f"\nExtracted features:\nX: {x.shape}\ny: {y.shape}")

    # Split into training and test datasets
    random_state = RANDOM.integers(1000)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    print(
        f"\nSplit datasets:\nX_train: {x_train.shape}\nX_test: {x_test.shape}\ny_train: {y_train.shape}\ny_test: {y_test.shape}"
    )

    # Scale the training and test data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Create an instance of our network
    nn = NeuralNetworkFromScratch(
        learning_rate=LEARNING_RATE,
        x_train=x_train_scaled,
        y_train=y_train,
        x_test=x_test_scaled,
        y_test=y_test,
    )

    # Perform the training
    nn.train(iterations=ITERATIONS)
    print(f"\nFirst loss: {nn.l_test[0]}\nFinal loss: {nn.l_test[-1]}")

    # Plot the losses
    plot = sns.lineplot(x=list(range(len(nn.l_test))), y=nn.l_test)
    figure = plot.get_figure()
    figure.savefig("../figures/nn_from_scratch.png")

    # Iterate over the test data to calculate accuracy
    test_count = x_test_scaled.shape[0]
    correct = 0
    y_preds = []
    for i in range(test_count):

        # Get the true y for this record
        y_true = y_test[i]

        # Create the prediction
        # Because the dependent variable is 0 or 1, we'll round
        y_pred = np.round(nn.forward(x_test_scaled[i]))
        y_preds.append(y_pred)

        # Count correct values
        correct += 1 if y_pred == y_true else 0

    # Calculate accuracy
    accuracy = correct / test_count
    print(f"\nAccuracy: {accuracy}")

    # Sanity check our test data by counting 0/1 values in the test data
    counter = Counter(y_test)
    print(counter)

    # Sanity check our result using a confusion matrix
    matrix = confusion_matrix(y_true=y_test, y_pred=y_preds)
    print(matrix)


if __name__ == "__main__":
    main()
