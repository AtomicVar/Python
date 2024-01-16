"""
Implements the Softsign activation function. The Softsign function is a smoother,
non-linear activation function which is defined as x / (1 + |x|).

The function takes a vector of K real numbers as input, and then applies the Softsign
function to each element of the vector.

For more detailed information, you can refer to the following link:
https://www.gabormelli.com/RKB/Softsign_Activation_Function
https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html
"""

import numpy as np


def softsign(vector: np.ndarray) -> np.ndarray:
    """
    Implements the Softsign activation function.

    Parameters:
        vector (np.ndarray): The input array for the Softsign activation function.

    Returns:
        np.ndarray: The input array after applying the Softsign activation function.

    The function is defined as:
    f(x) = x / (1 + |x|)

    Examples:
    >>> softsign(vector=np.array([-2, 0, 2, 4, 8]))
    array([-0.66666667,  0.        ,  0.66666667,  0.8       ,  0.88888889])

    >>> softsign(np.array([-3.5, 1.2, 5.6, 7.8]))
    array([-0.77777778,  0.54545455,  0.84848485,  0.88636364])

    """
    return vector / (1 + np.abs(vector))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
