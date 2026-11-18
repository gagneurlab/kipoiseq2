import numpy as np

from kipoiseq2.dataclasses import Interval, _Anchor, _check_resize

__all__ = [
    "Compose",
    "DummyAxis",
    "SwapAxes",
    "ResizeInterval",
]


class Compose:
    """Composes several transforms together.

    :param transforms: list of callables; list of transforms to compose.

    Example:

    >>> from kipoiseq2 import Encoder
    >>> from kipoiseq2.transforms import Compose, DummyAxis
    >>> Compose([Encoder().one_hot, DummyAxis(0)])("ACGT").shape
    (1, 4, 4)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# numpy wrappers


class DummyAxis:
    """np.expand_dims wrapper - Insert a dummy axis (calls np.expand_dims)"""

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, x):
        if self.axis is not None:
            return np.expand_dims(x, self.axis)
        else:
            return x


class SwapAxes:
    """np.swapaxes wrapper

    If any of the axes is None, do nothing.
    """

    def __init__(self, axis1=None, axis2=None):
        self.axis1 = axis1
        self.axis2 = axis2

    def __call__(self, x):
        if self.axis1 is None or self.axis2 is None:
            return x
        else:
            return np.swapaxes(x, self.axis1, self.axis2)


# Intervals


class ResizeInterval:
    """Resize intervals with :meth:`Interval.resize <kipoiseq2.dataclasses.Interval.resize>`.

    :param width: width of the resized intervals
    :param anchor: the part of the interval to keep: ``"start"``, ``"center"`` or ``"end"``
    :param use_strand: if False, resize as an unstranded interval. The result keeps the strand.
    :raises TypeError: if ``width`` is not an int
    :raises ValueError: if ``width`` is negative, or if ``anchor`` is not ``"start"``, ``"center"`` or ``"end"``

    Example:

    >>> from kipoiseq2 import Interval
    >>> from kipoiseq2.transforms import ResizeInterval
    >>> ResizeInterval(4, anchor="start")(Interval("chr1", 10, 20, strand="-"))
    Interval('chr1', 16, 20, strand='-')
    """

    def __init__(self, width: int, *, anchor: _Anchor = "center", use_strand: bool = True) -> None:
        _check_resize(width, anchor)
        self.width = width
        self.anchor = anchor
        self.use_strand = use_strand

    def __call__(self, interval: Interval) -> Interval:
        return interval.resize(self.width, anchor=self.anchor, use_strand=self.use_strand)
