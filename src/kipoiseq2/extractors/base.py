from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from kipoiseq2.dataclasses import Interval

__all__ = ["SequenceExtractor"]


@runtime_checkable
class SequenceExtractor(Protocol):
    """Source of a reference sequence, e.g. a :class:`~kipoiseq2.extractors.FastaStringExtractor`.

    :class:`~kipoiseq2.extractors.VariantSeqExtractor` reads the reference from it.
    Any object with these members fits, without subclassing.
    """

    @property
    def chrom_sizes(self) -> Mapping[str, int]:
        """Length of each chromosome, by name."""
        ...

    def extract(self, interval: Interval) -> str:
        """Return the sequence of ``interval``.

        :param interval: an interval on a chromosome in :attr:`chrom_sizes`, within 0 and the chromosome length.
            For the strand ``"."``, the sequence must be the "+" strand.
        :returns: the sequence, of ``interval.width`` letters
        """
        ...


def _chrom_size(chrom_sizes: Mapping[str, int], chrom: str) -> int:
    """Return the length of ``chrom``.

    :raises ValueError: if ``chrom`` is not in ``chrom_sizes``
    """
    try:
        return chrom_sizes[chrom]
    except KeyError:
        raise ValueError(f"chrom {chrom!r} is not in the reference") from None
