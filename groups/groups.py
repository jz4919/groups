from numbers import Integral
import numpy as np


class Element:
    """An element of the specified group.

    Parameters
    ----------
    group:
        The group of which this is an element.
    value:
        The individual element value.
    """

    def __init__(self, group, value):
        group._validate(value)
        self.group = group
        self.value = value

    def __mul__(self, other):
        """Use * to represent the group operation."""
        return Element(self.group,
                       self.group.operation(self.value,
                                            other.value))

    def __str__(self):
        """Return a string of the form value_group."""
        return f"{self.value}_{self.group}"

    def __repr__(self):
        """Return the canonical string representation of the element."""
        return f"{type(self).__name__}" \
            f"({repr(self.group), repr(self.value)})"


class Group:
    def __init__(self, n):
        self.n = n

    def __call__(self, value):
        """Create an element of this group."""
        return Element(self, value)

    def __repr__(self):
        """Return the canonical string representation of the group."""
        return f"{type(self).__name__}({self.n})"

    def __str__(self):
        """Represent the group as Gd."""
        return f"{self.symbol}{self.n}"


class CyclicGroup(Group):
    """A cyclic group represented by integer addition modulo group n."""

    symbol = "C"

    def _validate(self, value):
        """Ensure that value is a legitimate element value in this group."""
        if not (isinstance(value, Integral) and 0 <= value < self.n):
            raise ValueError("Element value must be an integer"
                             f" in the range [0, {self.n}).")

    def operation(self, a, b):
        """Perform the group operation on two values.

        The group operation is addition modulo n.
        """
        return (a + b) % self.n


class GeneralLinearGroup(Group):
    """The general linear group represented by n x n matrices."""

    symbol = "G"

    def _validate(self, value):
        """Ensure that value is a legitimate element value in this group."""
        if not (isinstance(value, np.ndarray),
                value.shape == (self.n, self.n)):
            raise ValueError("Element value must be an array "
                             f"with shape ({self.n}, {self.n}).")

    def operation(self, a, b):
        """Perform the group operation on two values.

        The group operation is matrix multiplication.
        """
        return a @ b
