"""
Implements the Randomized Leaky Rectified Linear Unit (RReLU) activation function.

The function takes a vector of K real numbers as input, and then applies the RReLU
function to each element of the vector.

For more detailed information, you can refer to the following link:
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
https://arxiv.org/abs/1505.00853
https://pytorch.org/docs/stable/generated/torch.nn.RReLU.html
"""

import numpy as np


def randomized_leaky_rectified_linear_unit(
    vector: np.ndarray, lower: float = 0.125, upper: float = 0.333
) -> np.ndarray:
    """
    Implements the Randomized Leaky Rectified Linear Unit (RReLU) activation function.

    Parameters:
        vector (np.ndarray): The input array for the RReLU activation function.
        lower (float): Lower bound of the uniform distribution.
        upper (float): Upper bound of the uniform distribution.

    Returns:
        np.ndarray: The input array after applying the RReLU activation function.

    The function is defined as:
    f(x) = x if x > 0 else alpha * x, where alpha is randomly sampled from the uniform
    distribution.

    Examples:
    >>> np.random.seed(0) # for consistent doctest results
    >>> randomized_leaky_rectified_linear_unit(vector=np.array([-2, 0, 2, 4, -1]))
    array([-0.47830642,  0.        ,  2.        ,  4.        , -0.2131202 ])

    """

    alpha = np.random.uniform(lower, upper, size=vector.shape)
    return np.where(vector > 0, vector, alpha * vector)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
