import numpy as np
from intervals import poly_lt_zero


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
            vec[i - 1: j] = 1

        return vec


def polytope_to_interval(a_vec, b_vec, A=None, b=None, c=None, tol=1e-10):
    """
    Compute truncation intervals obtained from a polytope `{x'Ax+b'x+c<=0}` as a selection event.

    Args:
        a_vec, b_vec (array-like): Vectors which satisfy `observed_data = a_vec + b_vec * observed_test_static`
        A (array-like, optional): `N`*`N` matrix. Set None if `A` is unused.
            Defaults to None.
        b (array-like, optional): `N` dimensional vector. Set None if `b` is unused.
            Defaults to None.
        c (float, optional): Constant. Set None if `c` is unused. Defaults to None.
        tol (float, optional): Tolerance error parameter. Defaults to 1e-10.

    Returns:
        array-like: truncation intervals [[L1, U1], [L2, U2], ...].
    """
    alp = beta = gam = 0

    if A is not None:
        cA = np.dot(b_vec, A)
        zA = np.dot(a_vec, A)
        alp += np.dot(cA, b_vec)
        beta += np.dot(zA, b_vec) + np.dot(cA, a_vec)
        gam += np.dot(zA, a_vec)

    if b is not None:
        beta += np.dot(b, b_vec)
        gam += np.dot(b, a_vec)

    if c is not None:
        gam += c

    intervals = poly_lt_zero([alp, beta, gam], tol=tol)
    return intervals
