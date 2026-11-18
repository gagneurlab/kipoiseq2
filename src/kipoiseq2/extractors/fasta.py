import os
from collections.abc import Mapping
from types import MappingProxyType, TracebackType
from typing import Self

from kipoiseq2.dataclasses import Interval, _is_reverse
from kipoiseq2.extractors.base import _chrom_size
from kipoiseq2.transforms.functional import reverse_complement

__all__ = ["FastaStringExtractor"]

_GZIP_MAGIC = b"\x1f\x8b"
"""The first two bytes of a gzip file."""
_BGZF_EXTRA = b"BC\x02\x00"
"""Bytes 12 to 15 of a bgzip file: the ID ``BC`` and the length 2 of the extra subfield, as htslib checks them."""


def _check_compression(fasta_file: str) -> None:
    """Raise an error if pyfaidx cannot read ``fasta_file`` because of its compression.

    pyfaidx reads uncompressed files, and bgzip files whose name ends in ``.gz`` or ``.bgz`` if Biopython is installed.

    :raises ValueError: if the file is compressed with gzip but not with bgzip,
        or if it is compressed with bgzip and its name does not end in ``.gz`` or ``.bgz``
    :raises ImportError: if the file is compressed with bgzip and Biopython is missing
    """
    with open(fasta_file, "rb") as f:
        header = f.read(16)
    if header[:2] != _GZIP_MAGIC:
        return
    if header[12:16] != _BGZF_EXTRA:
        raise ValueError(
            f"{fasta_file} is compressed with gzip. Decompress it, or compress it with bgzip instead, "
            "e.g. 'gunzip genome.fa.gz; bgzip genome.fa'"
        )
    if not fasta_file.lower().endswith((".gz", ".bgz")):
        raise ValueError(f"{fasta_file} is compressed with bgzip, so its name must end in .gz or .bgz")
    try:
        import Bio.bgzf  # noqa: F401
    except ImportError:
        raise ImportError(
            f"{fasta_file} is compressed with bgzip. Reading it requires Biopython, "
            "which kipoiseq2 does not install. Install biopython, or decompress the file."
        ) from None


class FastaStringExtractor:
    """Fasta file extractor

    The file may be uncompressed or compressed with bgzip.
    Reading a bgzip file requires Biopython, which kipoiseq2 does not install.
    pyfaidx writes the index ``.fai`` next to the file if it is missing, and also ``.gzi`` for a bgzip file.

    Close the file with :meth:`close`, or use the extractor as a context manager.

    .. note::
       The extractor is not thread-safe.
       If you wish to use it with multiprocessing,
       create a new extractor object in each process.

    :param fasta_file: path to the fasta_file
    :param use_strand: if True, the extracted sequence
        is reverse complemented in case interval.strand == "-"
    :param uppercase: if True, uppercase the sequences, so that the case-sensitive encoders of
        :mod:`kipoiseq2.encoders` take them as they are.
        If False, keep the case of the file, e.g. the lowercase letters of soft-masked repeats.
    :raises ValueError: if the file is compressed with gzip but not with bgzip,
        or if it is compressed with bgzip and its name does not end in ``.gz`` or ``.bgz``
    :raises ImportError: if the file is compressed with bgzip and Biopython is missing

    Example:

    >>> with FastaStringExtractor("genome.fa", use_strand=True) as fasta:  # doctest: +SKIP
    ...     seq = fasta.extract(Interval("chr1", 10, 20, strand="-"))
    """

    def __init__(self, fasta_file: str | os.PathLike[str], *, use_strand: bool, uppercase: bool = True) -> None:
        from pyfaidx import Fasta

        self.fasta_file = os.fspath(fasta_file)
        _check_compression(self.fasta_file)
        self._use_strand = use_strand
        self._fasta = Fasta(self.fasta_file, sequence_always_upper=uppercase)
        self._chrom_sizes = MappingProxyType({name: record.rlen for name, record in self._fasta.faidx.index.items()})

    @property
    def use_strand(self) -> bool:
        """If True, :meth:`extract` reverse complements the sequence of a "-" interval."""
        return self._use_strand

    @property
    def chrom_sizes(self) -> Mapping[str, int]:
        """Length of each chromosome in the FASTA file, by name, from the index."""
        return self._chrom_sizes

    def extract(self, interval: Interval) -> str:
        """
        Returns the FASTA sequence in some given interval as string

        :param interval: the interval to query
        :returns: sequence of requested interval, reverse complemented for a "-" interval if :attr:`use_strand`
            is True. The letters are uppercase unless the extractor was built with ``uppercase=False``.
        :raises TypeError: if ``interval`` is not an Interval
        :raises ValueError: if ``interval.chrom`` is not in the FASTA file,
            if ``interval`` starts before 0 or ends after the end of the chromosome,
            or if the sequence needs a reverse complement and contains a letter without IUPAC complement, e.g. ``U``
        """
        if not isinstance(interval, Interval):
            raise TypeError(f"interval must be an Interval, not {type(interval).__name__}")
        chrom_size = _chrom_size(self._chrom_sizes, interval.chrom)
        if interval.start < 0 or interval.end > chrom_size:
            raise ValueError(f"interval {interval} must lie within 0 and {chrom_size}, the length of {interval.chrom}")
        # pyfaidx wants a 1-based interval
        seq = str(self._fasta.get_seq(interval.chrom, interval.start + 1, interval.end).seq)
        if _is_reverse(interval, use_strand=self._use_strand):
            return reverse_complement(seq)
        return seq

    def close(self) -> None:
        """Close the FASTA file. Closing twice does not raise."""
        self._fasta.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
