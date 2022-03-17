import numpy as np


def is_int_or_float(value):
    """
    Check if a value (including a numpy object) has integer or floating data type. Note
    that infinity values have floating data type.
    """
    type_ = type(value)
    return np.issubdtype(type_, np.integer) or np.issubdtype(type_, np.floating)


class OneVec:
    """
    Vector whose elements from position `i` to `j` are set to 1, and 0 otherwise.

    Args:
        length (int): Dimension of the vector.
    """

    def __init__(self, length):
        self.length = length

    def get(self, i, j=None):
        """
        Get the vector.

        Args:
            i (int): Start index of 1 (1<=i<=`length`).
            j (int, optional): End index of 1 (1<=j<=`length`). If None, it returns a
                vector whose `i`th element is set to 1, and 0 otherwise. Defaults to
                None.

        Returns:
            np.ndarray
        """
        vec = np.zeros(self.length)

        if j is None:
            vec[i - 1] = 1
        else:
            vec[i - 1 : j] = 1

        return vec
