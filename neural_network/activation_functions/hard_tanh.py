"""
Implements the HardTanh (Hard Hyperbolic Tangent) function.

The function takes a vector of K real numbers as input, and applies the
HardTanh function to each element of the vector. The output is clipped
within a specified min_val and max_val range.

For more detailed information, you can refer to the following link:
https://www.gabormelli.com/RKB/HardTanh_Activation_Function
https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
"""

import numpy as np


def hard_tanh(
    vector: np.ndarray, min_val: float = -1.0, max_val: float = 1.0
) -> np.ndarray:
    """
    Implements the HardTanh activation function.

    Parameters:
        vector: the array containing input of HardTanh activation
        min_val: the minimum value of the range, default is -1
        max_val: the maximum value of the range, default is 1

    Returns:
        hard_tanh (np.ndarray): The input numpy array after applying HardTanh.

    Mathematically, f(x) = max(min_val, min(max_val, x))

    Examples:
    >>> hard_tanh(vector=np.array([2.3, 0.6, -2, -3.8]))
    array([ 1. ,  0.6, -1. , -1. ])

    >>> hard_tanh(vector=np.array([-9.2, -0.3, 0.45, 4.56]), min_val=-2, max_val=2)
    array([-2.  , -0.3 ,  0.45,  2.  ])
    """
    return np.clip(vector, min_val, max_val)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
