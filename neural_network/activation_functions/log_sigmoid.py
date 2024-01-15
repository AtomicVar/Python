"""
Implements the LogSigmoid function.

The function takes a vector of K real numbers as
input, and then applies the LogSigmoid function to each element of the vector.

For more detailed information, you can refer to the following link:
https://en.wikipedia.org/wiki/Sigmoid_function
https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html
"""

import numpy as np


def log_sigmoid(vector: np.ndarray) -> np.ndarray:
    """
    Implements the LogSigmoid activation function.

    Parameters:
        vector (np.ndarray): The array containing input of LogSigmoid
        activation function.

    Returns:
        log_sigmoid (np.ndarray): The input numpy array after applying
        LogSigmoid activation function.

    The function is defined as:
    f(x) = log(1 / (1 + e^(-x)))

    Examples:
    >>> log_sigmoid(vector=np.array([2.3, 0.6, -2, -3.8]))
    array([-0.09554546, -0.43748795, -2.12692801, -3.82212422])

    >>> log_sigmoid(vector=np.array([-9.2, -0.3, 0.45, -4.56]))
    array([-9.20010103, -0.85435524, -0.49324895, -4.57040771])

    """

    return np.log(1 / (1 + np.exp(-vector)))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
