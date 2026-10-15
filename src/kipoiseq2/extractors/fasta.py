from kipoiseq2 import Interval
from kipoiseq2.extractors import BaseExtractor

__all__ = ["FastaStringExtractor"]


class FastaStringExtractor(BaseExtractor):
    """Fasta file extractor

    .. note::
       The extractor is not thread-safe.
       If you wish to use it with multiprocessing,
       create a new extractor object in each process.

    :param fasta_file: str; path to the fasta_file
    :param use_strand: bool; if True, the extracted sequence
        is reverse complemented in case interval.strand == "-"
    :param force_upper: bool; Force uppercase output
    """

    def __init__(self, fasta_file, use_strand=False, force_upper=False):
        from pyfaidx import Fasta

        self.fasta_file = fasta_file
        self._use_strand = use_strand
        self.fasta = Fasta(self.fasta_file)
        self.force_upper = force_upper

    def extract(self, interval: Interval, use_strand=None, **kwargs) -> str:
        """
        Returns the FASTA sequence in some given interval as string

        :param interval: the interval to query
        :param use_strand: bool, optional; if True, the extracted sequence
            is reverse complemented in case interval.strand == "-".
            Overrides ``self.use_strand``
        :param ``**kwargs``:
        :returns: sequence of requested interval

        """
        # reverse-complement seq the negative strand
        if use_strand is None:
            use_strand = self.use_strand
        rc = use_strand and interval.strand == "-"

        # pyfaidx wants a 1-based interval
        seq = str(self.fasta.get_seq(interval.chrom, interval.start + 1, interval.stop, rc=rc).seq)

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
