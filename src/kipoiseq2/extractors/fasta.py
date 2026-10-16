import os

from kipoiseq2 import Interval
from kipoiseq2.extractors import BaseExtractor

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


class FastaStringExtractor(BaseExtractor):
    """Fasta file extractor

    The file may be uncompressed or compressed with bgzip.
    Reading a bgzip file requires Biopython, which kipoiseq2 does not install.
    pyfaidx writes the index ``.fai`` next to the file if it is missing, and also ``.gzi`` for a bgzip file.

    .. note::
       The extractor is not thread-safe.
       If you wish to use it with multiprocessing,
       create a new extractor object in each process.

    :param fasta_file: path to the fasta_file
    :param use_strand: if True, the extracted sequence
        is reverse complemented in case interval.strand == "-"
    :param force_upper: Force uppercase output
    :raises ValueError: if the file is compressed with gzip but not with bgzip,
        or if it is compressed with bgzip and its name does not end in ``.gz`` or ``.bgz``
    :raises ImportError: if the file is compressed with bgzip and Biopython is missing
    """

    def __init__(self, fasta_file: str | os.PathLike[str], *, use_strand: bool, force_upper: bool = False):
        from pyfaidx import Fasta

        self.fasta_file = os.fspath(fasta_file)
        _check_compression(self.fasta_file)
        self._use_strand = use_strand
        self.fasta = Fasta(self.fasta_file)
        self.force_upper = force_upper

    def extract(self, interval: Interval, use_strand: bool | None = None) -> str:
        """
        Returns the FASTA sequence in some given interval as string

        :param interval: the interval to query
        :param use_strand: if True, the extracted sequence
            is reverse complemented in case interval.strand == "-".
            Overrides ``self.use_strand``
        :returns: sequence of requested interval
        """
        # reverse-complement seq the negative strand
        if use_strand is None:
            use_strand = self.use_strand
        rc = use_strand and interval.strand == "-"

        # pyfaidx wants a 1-based interval
        seq = str(self.fasta.get_seq(interval.chrom, interval.start + 1, interval.end, rc=rc).seq)

        # optionally, force upper-case letters
        if self.force_upper:
            seq = seq.upper()
        return seq

    def close(self):
        # self.fasta is missing if __init__ failed, e.g. for a missing file
        if hasattr(self, "fasta"):
            return self.fasta.close()


# class BigWigExtractor(BaseExtractor):
#     """Big-wig file extractor

#     NOTE: The extractor is not thread-save.
#     If you with to use it with multiprocessing,
#     create a new extractor object in each process.

#     # Arguments
#       bigwig_file: path to the bigwig file
#     """

#     def __init__(self, bigwig_file):

#         self.bigwig_file = bigwig_file
#         self.batch_extractor = BigwigExtractor(self.bigwig_file)

#     def extract(self, interval):
#         return self.batch_extractor([interval])[0]

#     def close(self):
#         return self.batch_extractor.close()
