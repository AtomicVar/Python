"""
Implements the Hardswish activation function.

The function takes a vector of K real numbers as input and applies the
Hardswish function to each element of the vector.

For more detailed information, you can refer to the following link:
https://serp.ai/hard-swish/
https://arxiv.org/abs/1905.02244
"""

import numpy as np


def hard_swish(vector: np.ndarray) -> np.ndarray:
    """
    Implements the Hardswish activation function.

    Parameters:
        vector: The array containing the input for the Hardswish activation.

    Returns:
        hard_swish (np.array): The input numpy array after applying the
        Hardswish activation function.

    The function is defined as:
    f(x) = x * min(max(x + 3, 0), 6) / 6

    Examples:
    >>> hard_swish(np.array([2.4, -2.4, 0, -3.6]))
    array([ 2.16, -0.24,  0.  , -0.  ])

    >>> hard_swish(np.array([-9.2, -0.3, 0.45, 4.56]))
    array([-0.     , -0.135  ,  0.25875,  4.56   ])
    """

    return vector * np.minimum(np.maximum(vector + 3, 0), 6) / 6


if __name__ == "__main__":
    import doctest

    doctest.testmod()
