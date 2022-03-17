from collections.abc import Iterable


def _flatten(iterable):
    """
    Flatten array with any shape.

    Args:
        iterable (array): Array with any shape.

    Yields:
        any: Next element of the flattened array.
    """
    for elem in iterable:
        if isinstance(elem, Iterable) and not isinstance(elem, (str, bytes)):
            yield from flatten(elem)
        else:
            yield elem


def flatten(iterable):
    """
    Flatten array with any shape into list.

    Args:
        iterable (array): Array with any shape.
    """
    return list(_flatten(iterable))
