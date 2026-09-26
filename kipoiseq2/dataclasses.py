"""Data classes for major objects:
- Interval
- Variant
"""

# from collections import Mapping, OrderedDict
# from kipoi_utils.data_utils import numpy_collate, numpy_collate_concat
import math
from copy import deepcopy
from typing import Optional

# import numpy as np
# -------------------------------------------
# basepair implementation
# import attr


class Variant:
    def __init__(
        self,
        chrom: str,
        pos: int,  # 1-based
        ref: str,
        alt: str,
        id: str = "",
        qual: float = 0,
        filter: str = "PASS",
        info: Optional[dict] = None,
        source=None,
    ):
        """Variant container.

        See also VCF file definition: http://samtools.github.io/hts-specs/VCFv4.3.pdf
        Note: this class doesn't hold the genotype information.

        Args:
          chrom: CHROM field in the VCF
          pos: POS field in the VCF
          ref: REF field in the VCF
          alt: ALT field in the VCF
          id: ID field in the VCF
          qual: QUAL field in the VCF
          filter: FILTER field in the VCF
          info: INFO field in the VCF
          source: reference to the original source object from which this
            Variant object was created (e.g. `cyvcf2.Variant()` class)
        """
        # main 4 attributes
        # these should be immutable by default to not
        # run into any strange issues downstream.
        self._chrom = str(chrom)
        self._pos = int(pos)
        self._ref = str(ref)
        self._alt = str(alt)

        # other 4 main VCF attributes
        self.id = id
        self.qual = qual
        self.filter = filter
        self.info = info or dict()

        # additional attribute implemented by this class
        self.source = source

    def copy(self):
        return deepcopy(self)

    @property
    def chrom(self):
        return self._chrom

    @property
    def pos(self):
        return self._pos

    @property
    def ref(self):
        return self._ref

    @property
    def alt(self):
        return self._alt

    # convenience properties
    @property
    def start(self):
        """0-based variant start position"""
        return self.pos - 1

    @property
    def end(self):
        """0-based, exclusive variant end position: `start + len(ref)`.

        This equals the 1-based position of the last REF base.
        """
        return self.start + len(self.ref)

    @classmethod
    def from_cyvcf(cls, obj):
        if len(obj.ALT) > 1:
            # TODO - do a proper warning
            print("WARNING: len(obj.ALT) > 1")
        # if there is a deletion
        # empty string
        if len(obj.ALT) == 0:
            obj.ALT = [""]

        return cls(
            chrom=obj.CHROM,
            pos=obj.POS,
            ref=obj.REF,
            alt=obj.ALT[0],  # note. we are using a single one
            id=obj.ID,
            qual=obj.QUAL,
            filter=obj.FILTER,
            info=dict(obj.INFO),
            source=obj,
        )

    @classmethod
    def from_cyvcf_and_given_alt(cls, obj, alt):

        return cls(
            chrom=obj.CHROM,
            pos=obj.POS,
            ref=obj.REF,
            alt=alt,
            id=obj.ID,
            qual=obj.QUAL,
            filter=obj.FILTER,
            info=dict(obj.INFO),
            source=obj,
        )

    def __eq__(self, obj):
        return self.chrom == obj.chrom and self.pos == obj.pos and self.ref == obj.ref and self.alt == obj.alt

    def __hash__(self):
        return hash((self.chrom, self.pos, self.ref, self.alt))

    def __str__(self):
        return "{}:{}:{}>{}".format(self.chrom, self.pos, self.ref, self.alt)

    @classmethod
    def from_str(cls, s: str) -> "Variant":
        chrom, pos, ref_alt = s.split(":")
        ref, alt = ref_alt.split(">")
        return cls(chrom=chrom, pos=int(pos), ref=ref, alt=alt)

    def __repr__(self):
        return "Variant(chrom='{}', pos={}, ref='{}', alt='{}', id='{}',...)".format(
            self.chrom, self.pos, self.ref, self.alt, self.id
        )


class Interval:
    """Container for genomic interval(s)

    Coordinates are 0-based, half-open: the interval covers the bases
    `start` to `end - 1`, and its width is `end - start`. This is the
    BED convention. In 1-based VCF or GTF positions, the interval covers
    `start + 1` to `end`.

    All fields can be either a single values (str or int) or a
    numpy array of values.

    # Arguments
        chrom: Chromosome
        start: 0-based start position (inclusive)
        end: 0-based end position (exclusive)
        name: interval name
        score: interval score
        strand: interval strand ("+", "-" or "." for unknown strand)
        attrs: additional attributes provided as a dictionary
    """

    def __init__(
        self,
        chrom: str,
        start: int,  # 0-based, inclusive
        end: int,  # 0-based, exclusive
        name: str = "",
        score: float = 0,
        strand: str = ".",
        attrs: Optional[dict] = None,
    ):
        self._chrom = chrom
        self._start = start
        self._end = end
        self.name = name
        self.score = score
        self._strand = strand
        self.attrs = attrs or dict()

    # handle chr and stop
    @property
    def chrom(self):
        return self._chrom

    @property
    def chr(self):
        return self.chrom

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def stop(self):
        return self.end

    @property
    def strand(self):
        return self._strand

    @classmethod
    def from_pybedtools(cls, interval):
        """Create the ranges object from `pybedtools.Interval`

        # Arguments
            interval: `pybedtools.Interval` instance
        """
        return cls(
            chrom=interval.chrom,
            start=interval.start,
            end=interval.stop,
            name=interval.name,
            score=interval.score,
            strand=interval.strand,
            attrs=dict(interval.attrs or dict()),
        )

    def to_pybedtools(self):
        import pybedtools

        return pybedtools.create_interval_from_list(
            [self.chrom, self.start, self.end, self.name, self.score, self.strand]
        )

    @property
    def neg_strand(self):
        return self.strand == "-"

    def unstrand(self):
        """Removes strand information."""
        obj = self.copy()
        obj._strand = "."
        return obj

    def center(self, use_strand=True):
        """Compute the center of the interval"""
        if use_strand:
            add_offset = 0 if self.neg_strand else 1
        else:
            add_offset = 0
        delta = (self.end + self.start) % 2
        center = (self.end + self.start) // 2
        return center + add_offset * delta

    def shift(self, x: int, use_strand: bool = True):
        """Shift the interval by x.

        Args:
          x: shift amount
          use_strand (bool)


        This will perform:
        (chrom, start + x, end + x)

        If the strand is negative and use_strand is True, it will return:
        (chrom, start - x, end - x)
        """
        obj = self.copy()
        if use_strand and self.neg_strand:
            x = -x
        obj._start = self.start + x
        obj._end = self.end + x
        return obj

    def swap_strand(self):
        obj = self.copy()
        if obj.strand == "+":
            obj._strand = "-"
        elif obj.strand == "-":
            obj._strand = "+"
        return obj

    def __eq__(self, obj):
        return self.chrom == obj.chrom and self.start == obj.start and self.end == obj.end and self.strand == obj.strand

    def __hash__(self):
        return hash((self.chrom, self.start, self.end, self.strand))

    def __str__(self):
        return "{}:{}-{}:{}".format(self.chrom, self.start, self.end, self.strand)

    def __repr__(self):
        return "Interval(chrom='{}', start={}, end={}, name='{}', strand='{}', ...)".format(
            self.chrom, self.start, self.end, self.name, self.strand
        )

    @classmethod
    def from_str(cls, s):
        chrom, int_range, strand = s.split(":")
        start, end = int_range.split("-")
        return cls(chrom=chrom, start=int(start), end=int(end), strand=strand)

    def copy(self):
        return deepcopy(self)

    def slop(self, upstream=0, downstream=0, use_strand=True):
        """Extend the interval on each strand"""
        obj = self.copy()
        obj._start -= upstream
        obj._end += downstream
        return obj

    def is_valid(self, chrom_len=math.inf):
        """Check that the interval lies within [0, chrom_len).

        Args:
          chrom_len: length of the chromosome. An interval may end at
            `chrom_len`, because the end is exclusive.
        """
        return self.start >= 0 and self.end <= chrom_len

    def truncate(self, chrom_len=math.inf):
        """Truncate the interval to become valid"""
        if self.is_valid(chrom_len):
            return self
        else:
            obj = self.copy()
            obj._start = max(self._start, 0)
            obj._end = min(self.end, chrom_len)
            return obj

    def resize(self, width, use_strand=True):
        obj = deepcopy(self)

        if width is None or self.width() == width:
            # no need to resize
            return obj

        if self.neg_strand and use_strand:
            # negative strand
            obj._start = self.center() - width // 2
            obj._end = self.center() + width // 2 + width % 2
        else:
            # positive strand
            obj._start = self.center() - width // 2 - width % 2
            obj._end = self.center() + width // 2
        return obj

    def width(self):
        return self.end - self.start

    def __len__(self):
        return self.width()

    def trim(self, i, j, use_strand=True):
        if i == 0 and j == self.width():
            return self
        obj = self.copy()
        assert j > i
        if self.strand == "-" and use_strand:
            w = self.width()
            obj._start = self.start + w - j
            obj._end = self.start + w - i
        else:
            obj._start = self.start + i
            obj._end = self.start + j
        return obj
