"""
A script that plays with tensors.

It is a rework of the script demonstrated Chapter 6: Tensors, in the course PyTorch Ultimate 2024 -
From Basics to Cutting-Edge by Bert Gollnick.

The code for this chapter can otherwise be found at:
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/tree/main/020_TensorIntro

The goal of this script is to demonstrate the creation of tensors and the autograd functionality.
"""

from typing import Callable

import torch
import numpy as np
import seaborn as sns


def plot_function(function: Callable, name: str):
    """
    Plots a function

    :param function: the callable function
    :param name: the name under which the plot will be saved
    """

    # Generate the function ranges
    x_range = np.linspace(0, 10, 101)
    y_range = [function(x) for x in x_range]

    # Plot the ranges
    plot = sns.lineplot(x=x_range, y=y_range)

    # Save the plot
    figure = plot.get_figure()
    figure.savefig(f"../figures/testing_tensors_{name}.png")
    figure.clf()


def example_1():
    """
    Runs an example that creates two tensors and shows the gradient of the first
    """

    # Define our y function and plot it
    y_function = lambda x: (x - 3) * (x - 6) * (x - 4)
    plot_function(y_function, "example1")

    # Create our starting tensor
    x = torch.tensor(2.0, requires_grad=True)

    # Create a second tensor based on the first
    y = y_function(x)

    # Run the backward step
    y.backward()

    # Print the x gradient
    print(x.grad)


def example_2():
    """
    Runs an example that creates three tensors and shows the gradient of the first
    """

    # Define our y function and plot it
    y_function = lambda x: x**3
    z_function = lambda y: 5 * y - 4
    plot_function(y_function, "example2a")
    plot_function(z_function, "example2b")

    # Create our starting tensor
    x = torch.tensor(1.0, requires_grad=True)

    # Create our followon tensors
    y = y_function(x)
    z = z_function(y)

    # Run the backward step
    z.backward()

    # Print the x gradient
    print(x.grad)


def example_3():
    """
    Runs an example that creates three layers of tensors and shows the gradients of the first layer
    """

    # Create the starting tensors (the 'input' layer)
    x11 = torch.tensor(2.0, requires_grad=True)
    x21 = torch.tensor(3.0, requires_grad=True)

    # Create the second layer of tensors (the 'hidden' layer)
    x12 = (5 * x11) - (3 * x21)
    x22 = (2 * x11**2) + (2 * x21)

    # Create the third layer of tensors (the 'output' layer)
    y = (4 * x12) + (3 * x22)

    # Run the backward step
    y.backward()

    # Print the gradients
    print(x11.grad)
    print(x21.grad)


if __name__ == "__main__":

    # Run the first example
    example_1()

    # Run the second example
    example_2()

    # Run the third example
    example_3()
