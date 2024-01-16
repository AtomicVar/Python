"""
Implements the Softshrink activation function.

The function takes a vector of K real numbers as input, and then applies the Softshrink
function to each element of the vector.

For more detailed information, you can refer to the following link:
https://www.gabormelli.com/RKB/Softshrink_Activation_Function
https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html
"""

import numpy as np


def softshrink(vector: np.ndarray, lambd: float = 0.5) -> np.ndarray:
    """
        Implements the Softshrink activation function.

        Parameters:
            vector (np.ndarray): The input array for the Softshrink activation function.
            lambd (float): The lambda value for the Softshrink function.

        Returns:
            np.ndarray: The input array after applying the Softshrink activation
            function.

        The function is defined as:
            f(x) = x - lambd if x > lambd
                x + lambd if x < -lambd
                0 otherwise

    Examples:
    >>> softshrink(np.array([-2, -0.5, 0, 0.5, 2]))
    array([-1.5,  0. ,  0. ,  0. ,  1.5])

    >>> softshrink(np.array([-3.5, 1.2, 5.6, 7.8]), lambd=1)
    array([-2.5,  0.2,  4.6,  6.8])

    """

    return np.where(np.abs(vector) > lambd, vector - np.sign(vector) * lambd, 0)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
