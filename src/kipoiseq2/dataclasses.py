"""Data classes for the major objects: Interval and Variant."""

import math
from copy import deepcopy
from typing import Optional


class Variant:
    """Variant container.

    See also VCF file definition: http://samtools.github.io/hts-specs/VCFv4.3.pdf

    .. note::
       this class doesn't hold the genotype information.

    :param chrom: CHROM field in the VCF
    :param pos: POS field in the VCF
    :param ref: REF field in the VCF
    :param alt: ALT field in the VCF
    :param id: ID field in the VCF
    :param qual: QUAL field in the VCF
    :param filter: FILTER field in the VCF
    :param info: INFO field in the VCF
    :param source: reference to the original source object from which this
        Variant object was created
    """

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
        """0-based, exclusive variant end position: ``start + len(ref)``.

        This equals the 1-based position of the last REF base.
        """
        return self.start + len(self.ref)

    def __eq__(self, obj):
        try:
            return self.chrom == obj.chrom and self.pos == obj.pos and self.ref == obj.ref and self.alt == obj.alt
        except AttributeError:
            return NotImplemented

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
    ``start`` to ``end - 1``, and its width is ``end - start``. This is the
    BED convention. In 1-based VCF or GTF positions, the interval covers
    ``start + 1`` to ``end``.

    :param chrom: Chromosome
    :param start: 0-based start position (inclusive)
    :param end: 0-based end position (exclusive)
    :param name: interval name
    :param score: interval score
    :param strand: interval strand ("+", "-" or "." for unknown strand)
    :param attrs: additional attributes provided as a dictionary
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

    @property
    def neg_strand(self):
        return self.strand == "-"

    def unstrand(self):
        """Removes strand information."""
        obj = self.copy()
        obj._strand = "."
        return obj

    def center(self, use_strand=True):
        """Compute the center of the interval.

        If ``start + end`` is odd, the center lies in the middle of a base.
        Then the center is rounded towards the 3' end: up on "+" and "." intervals, and down on "-" intervals.

        :param use_strand: if False, compute the center of the unstranded interval, which is rounded up on all strands
        :returns: the 0-based center position
        """
        delta = (self.end + self.start) % 2
        center = (self.end + self.start) // 2
        if use_strand and self.neg_strand:
            return center
        return center + delta

    def shift(self, x: int, use_strand: bool = True):
        """Shift the interval by x.

        This will perform:
        (chrom, start + x, end + x)

        If the strand is negative and use_strand is True, it will return:
        (chrom, start - x, end - x)

        :param x: shift amount
        :param use_strand:
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
        try:
            return (
                self.chrom == obj.chrom
                and self.start == obj.start
                and self.end == obj.end
                and self.strand == obj.strand
            )
        except AttributeError:
            return NotImplemented

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
        """Extend the interval upstream and downstream.

        :param upstream: number of bases to add upstream. A negative value removes bases.
        :param downstream: number of bases to add downstream. A negative value removes bases.
        :param use_strand: if True, upstream and downstream follow the strand:
            on a "-" interval, ``upstream`` moves ``end`` and ``downstream`` moves ``start``.
            On "+" and "." intervals, and if False, ``upstream`` moves ``start`` and ``downstream`` moves ``end``.
        :returns: a new interval
        """
        if use_strand and self.neg_strand:
            upstream, downstream = downstream, upstream
        obj = self.copy()
        obj._start -= upstream
        obj._end += downstream
        return obj

    def is_valid(self, chrom_len=math.inf):
        """Check that the interval lies within [0, chrom_len).

        :param chrom_len: length of the chromosome. An interval may end at
            ``chrom_len``, because the end is exclusive.
        """
        return self.start >= 0 and self.end <= chrom_len

    def truncate(self, chrom_len=math.inf):
        """Truncate the interval to become valid.

        :param chrom_len: length of the chromosome
        :returns: a new interval with ``start`` at least 0 and ``end`` at most ``chrom_len``
        """
        obj = self.copy()
        obj._start = max(self.start, 0)
        obj._end = min(self.end, chrom_len)
        return obj

    def resize(self, width, use_strand=True):
        """Resize the interval to ``width`` around its :meth:`center`.

        If ``width`` is odd, the result has one more base upstream of the center than downstream.
        Upstream is towards ``start`` on "+" and "." intervals, and towards ``end`` on "-" intervals.

        :param width: width of the resized interval. If None, return a copy.
        :param use_strand: if False, resize as an unstranded interval. The result keeps the strand.
        :returns: a new interval
        """
        obj = deepcopy(self)

        if width is None or self.width() == width:
            # no need to resize
            return obj

        center = self.center(use_strand=use_strand)
        if self.neg_strand and use_strand:
            # negative strand
            obj._start = center - width // 2
            obj._end = center + width // 2 + width % 2
        else:
            # positive strand
            obj._start = center - width // 2 - width % 2
            obj._end = center + width // 2
        return obj

    def width(self):
        return self.end - self.start

    def __len__(self):
        return self.width()

    def trim(self, i, j, use_strand=True):
        """Keep the bases ``i`` to ``j - 1`` of the interval.

        :param i: 0-based position of the first base to keep
        :param j: 0-based position after the last base to keep
        :param use_strand: if True, count the positions from ``end`` on "-" intervals.
            On "+" and "." intervals, and if False, count them from ``start``.
        :returns: a new interval
        """
        obj = self.copy()
        if i == 0 and j == self.width():
            return obj
        assert j > i
        if self.strand == "-" and use_strand:
            w = self.width()
            obj._start = self.start + w - j
            obj._end = self.start + w - i
        else:
            obj._start = self.start + i
            obj._end = self.start + j
        return obj
