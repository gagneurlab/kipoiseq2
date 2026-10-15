import warnings

import numpy as np

from kipoiseq2.transforms import functional as F

__all__ = [
    "Compose",
    "DummyAxis",
    "SwapAxes",
    "ResizeInterval",
    "SplitSplicingSeq",
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
    """Resize the interval with :func:`~kipoiseq2.transforms.functional.resize_interval`.

    :param width: desired width of the output interval
    :param anchor: which part of the interval should be anchored: 'start', 'center' or 'end'.
        See :func:`~kipoiseq2.transforms.functional.resize_interval`.
    """

    def __init__(self, width, anchor="center"):
        self.width = width
        self.anchor = anchor

    def __call__(self, interval):
        return F.resize_interval(interval, self.width, self.anchor)


# Splicing


class SplitSplicingSeq:
    """Split returned splice sequence (exon with flanking intron) to required format.
    It splits into ['intron5prime', 'acceptor', 'exon', 'donor', 'intron3prime'].
    'intron5prime' is the intron 5' of the exon, while 'intron3prime' is from the 3'.

    :param exon_cut_l: when extract exon feature, how many base pair to cut out at the beginning of an exon
    :param exon_cut_r: when extract exon feature, how many base pair to cut out at the end of an exon
        (cut out the part that is considered as acceptor site or donor site)
    :param intron5prime_cut: how many bp to cut out at the end of acceptor intron that consider as acceptor site
    :param intron3prime_cut: how many bp to cut out at the end of donor intron that consider as donor site
    :param acceptor_intron_len: what length in acceptor intron to consider for acceptor site model
    :param acceptor_exon_len: what length in acceptor exon to consider for acceptor site model
    :param donor_intron_len: what length in donor intron to consider for donor site model
    :param donor_exon_len: what length in donor exon to consider for donor site model
    """

    def __init__(
        self,
        exon_cut_l=0,
        exon_cut_r=0,
        intron5prime_cut=6,
        intron3prime_cut=6,
        acceptor_intron_len=50,
        acceptor_exon_len=3,
        donor_exon_len=5,
        donor_intron_len=13,
    ):

        self.exon_cut_l = exon_cut_l
        self.exon_cut_r = exon_cut_r
        self.intron5prime_cut = intron5prime_cut
        self.intron3prime_cut = intron3prime_cut
        self.acceptor_intron_len = acceptor_intron_len
        self.acceptor_exon_len = acceptor_exon_len
        self.donor_exon_len = donor_exon_len
        self.donor_intron_len = donor_intron_len

    def __call__(self, x, intron5prime_len, intron3prime_len):
        """
        :param x: a sequence to split
        :param intron5prime_len: 5' intronic sequence length to take.
        :param intron3prime_len: 3' intronic sequence length to take.
        """
        lackl = self.acceptor_intron_len - intron5prime_len  # need to pad N if left seq not enough long
        if lackl >= 0:
            x = "N" * (lackl + 1) + x
            intron5prime_len += lackl + 1
        lackr = self.donor_intron_len - intron3prime_len
        if lackr >= 0:
            x = x + "N" * (lackr + 1)
            intron3prime_len += lackr + 1

        intron5prime = x[: intron5prime_len - self.intron5prime_cut]
        acceptor = x[(intron5prime_len - self.acceptor_intron_len) : (intron5prime_len + self.acceptor_exon_len)]
        exon = x[(intron5prime_len + self.exon_cut_l) : (-intron3prime_len - self.exon_cut_r)]
        donor = x[(-intron3prime_len - self.donor_exon_len) : (-intron3prime_len + self.donor_intron_len)]
        intron3prime = x[-intron3prime_len + self.intron3prime_cut :]

        if donor[self.donor_exon_len : self.donor_exon_len + 2] != "GT":
            warnings.warn("Non-GT donor", UserWarning)
        if acceptor[self.acceptor_intron_len - 2 : self.acceptor_intron_len] != "AG":
            warnings.warn("Non-AG acceptor", UserWarning)
        if len(exon) == 0:
            exon = "N"

        return {
            "intron5prime": intron5prime,
            "acceptor": acceptor,
            "exon": exon,
            "donor": donor,
            "intron3prime": intron3prime,
        }
