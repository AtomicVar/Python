"""
Implements the Hardsigmoid function.

The function takes a vector of K real numbers, a real number alpha, and a real number
beta as inputs, and then applies the Hardsigmoid function to each element of the vector.

More details about the activation function can be found on:
https://en.wikipedia.org/wiki/Hard_sigmoid
https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html#torch.nn.Hardsigmoid
"""

import numpy as np


def hardsigmoid(
    vector: np.ndarray, alpha: float = 0.2, beta: float = 0.5
) -> np.ndarray:
    """
    Implements the Hardsigmoid activation function.

    Parameters:
        vector: a vector that consists of numeric values
        alpha: scale factor (default = 0.2)
        beta: offset factor (default = 0.5)

    Returns:
        hardsigmoid (np.ndarray): the input numpy array after applying hardsigmoid.

    The Hardsigmoid activation function is defined as:
        f(x) = max(0, min(1, alpha * x + beta))

    Examples:
    >>> hardsigmoid(vector=np.array([2.3, 0.6, -2, -3.8]))
    array([0.96, 0.62, 0.1 , 0.  ])

    >>> hardsigmoid(vector=np.array([-9.2, -0.3, 0.45, 4.56]))
    array([0.  , 0.44, 0.59, 1.  ])
    """
    return np.clip(alpha * vector + beta, 0, 1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
