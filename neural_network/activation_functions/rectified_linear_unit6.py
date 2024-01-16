"""
Implements the Rectified Linear Unit 6 (ReLU6) activation function. ReLU6 is a variant
of the ReLU function that caps the maximum value at 6.

The function takes a vector of K real numbers as input, and then applies the ReLU6
function to each element of the vector.

For more detailed information, you can refer to the following link:
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
https://serp.ai/relu6/
https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html
"""

import numpy as np


def rectified_linear_unit6(vector: np.ndarray) -> np.ndarray:
    """
        Implements the ReLU6 activation function.

        Parameters:
            vector (np.ndarray): The input array for the ReLU6 activation function.

        Returns:
            np.ndarray: The input array after applying the ReLU6 activation function.

        The function is defined as:
        f(x) = min(max(0, x), 6)

    Examples:
    >>> rectified_linear_unit6(vector=np.array([-2, 0, 2, 4, 8]))
    array([0, 0, 2, 4, 6])

    >>> rectified_linear_unit6(np.array([-3.5, 1.2, 5.6, 7.8]))
    array([0. , 1.2, 5.6, 6. ])

    """
    return np.clip(vector, 0, 6)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
