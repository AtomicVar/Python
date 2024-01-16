"""
Implements the Continuously Differentiable Exponential Linear Unit (CELU)
activation function.

The function takes a vector of K real numbers as input, and then applies the CELU
activation function to each element of the vector.

For more detailed information, you can refer to the following link:
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
https://arxiv.org/abs/1704.07483
https://pytorch.org/docs/stable/generated/torch.nn.CELU.html
"""


import numpy as np


def continuously_differentiable_exponential_linear_unit(
    vector: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """
    Implements the CELU activation function.

    Parameters:
        vector : The input array for the CELU activation function.
        alpha : The alpha value for the CELU activation function. Defaults to 1.0.

    Returns:
        np.ndarray : The input array after applying the CELU activation function.

    The function is defined as:
    f(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))

    Examples:
    >>> continuously_differentiable_exponential_linear_unit(vector=np.array([-2, 0, 2]))
    array([-0.86466472,  0.        ,  2.        ])

    >>> continuously_differentiable_exponential_linear_unit(np.array([-3, 1]), alpha=2)
    array([-1.55373968,  1.        ])

    """
    return np.maximum(0, vector) + np.minimum(0, alpha * (np.exp(vector / alpha) - 1))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
