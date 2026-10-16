from typing import Protocol, runtime_checkable

from kipoiseq2.dataclasses import Interval

__all__ = ["SequenceExtractor"]


@runtime_checkable
class SequenceExtractor(Protocol):
    """Source of a reference sequence, e.g. a :class:`~kipoiseq2.extractors.FastaStringExtractor`.

    :class:`~kipoiseq2.extractors.VariantSeqExtractor` reads the reference from it.
    Any object with these members fits, without subclassing.
    """

    def extract(self, interval: Interval) -> str:
        """Return the sequence of ``interval``.

        :param interval: the interval. For the strand ``"."``, the sequence must be the "+" strand.
        :returns: the sequence, of ``interval.width`` letters
        """
        ...
