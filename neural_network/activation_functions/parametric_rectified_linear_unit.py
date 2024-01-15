"""
Implements the Parametric Rectified Linear Unit (PReLU) function.

The function takes a vector of K real numbers as input, and then
applies the PReLU function to each element of the vector.

For more detailed information, you can refer to the following link:
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
https://www.educative.io/answers/what-is-parametric-relu
"""

import numpy as np


def parametric_rectified_linear_unit(vector: np.ndarray, alpha: float) -> np.ndarray:
    """
    Implements the PReLU activation function.

    Parameters:
        vector: the array containing input of PReLU activation function.
        alpha: hyper-parameter.

    return:
    parametric_rectified_linear_unit (np.array): the input numpy array after applying
    PReLU activation function.

    The function is defined as:
    f(x) = x, x>0 else (alpha * x), x<=0

    Examples:
    >>> parametric_rectified_linear_unit(vector=np.array([2.3,0.6,-2,-3.8]), alpha=0.3)
    array([ 2.3 ,  0.6 , -0.6 , -1.14])

    >>> parametric_rectified_linear_unit(vector=np.array([-9.2,-0.3,0.45]), alpha=0.067)
    array([-0.6164, -0.0201,  0.45  ])

    """

    return np.where(vector > 0, vector, alpha * vector)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
