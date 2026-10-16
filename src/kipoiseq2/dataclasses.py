"""Data classes for the major objects: Interval and Variant."""

import math
import numbers
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional, cast, get_args


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


class _ReadOnlyDict(Mapping[str, Any]):
    """Read-only copy of a mapping, for :attr:`Interval.attrs`.

    Unlike a ``types.MappingProxyType``, it can be pickled and deep-copied, e.g. by ``dataclasses.asdict``.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __reduce__(self) -> tuple[type["_ReadOnlyDict"], tuple[dict[str, Any]]]:
        return _ReadOnlyDict, (self._data,)


_Strand = Literal["+", "-", "."]
"""Type of the strand of an :class:`Interval`, with ``"."`` for an unknown strand."""
_STRANDS = get_args(_Strand)
"""The values of ``_Strand``, for the checks at runtime."""

_Anchor = Literal["start", "center", "end"]
"""Type of the part that :meth:`Interval.resize` keeps of an interval."""
_ANCHORS = get_args(_Anchor)
"""The values of ``_Anchor``, for the checks at runtime."""


def _check_int(value: object, name: str) -> None:
    """Raise a TypeError if ``value`` is not an int. A bool does not count as an int."""
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, not {type(value).__name__}")


def _check_resize(width: int, anchor: str, *, width_name: str = "width") -> None:
    """Raise a TypeError or ValueError for invalid arguments of :meth:`Interval.resize`.

    :param width_name: the name of ``width`` in the error messages
    """
    _check_int(width, width_name)
    if width < 0:
        raise ValueError(f"{width_name} must be at least 0, got {width}")
    if anchor not in _ANCHORS:
        raise ValueError(f'anchor must be "start", "center" or "end", got {anchor!r}')


def _center(start: int, end: int, *, reverse: bool) -> int:
    """Return the center of the range from ``start`` to ``end``, rounded down if ``reverse``, and up otherwise."""
    return (start + end) // 2 if reverse else (start + end + 1) // 2


def _resize_bounds(start: int, end: int, width: int, anchor: str, *, reverse: bool) -> tuple[int, int]:
    """Return the start and end of the range from ``start`` to ``end``, resized like :meth:`Interval.resize`.

    :param reverse: if True, resize like a "-" interval, and if False, like a "+" interval
    """
    if anchor == "center":
        center = _center(start, end, reverse=reverse)
        if reverse:
            return center - width // 2, center + width // 2 + width % 2
        return center - width // 2 - width % 2, center + width // 2
    # on a "-" interval, the start is at the genomic end
    if (anchor == "start") != reverse:
        return start, start + width
    return end - width, end


@dataclass(frozen=True, init=False, repr=False)
class Interval:
    """Genomic interval.

    Coordinates are 0-based, half-open: the interval covers the bases
    ``start`` to ``end - 1``, and its width is ``end - start``. This is the
    BED convention. In 1-based VCF or GTF positions, the interval covers
    ``start + 1`` to ``end``.

    An Interval is immutable. The constructor checks the fields once,
    and methods such as :meth:`resize` return a new Interval.
    Two intervals are equal if all their fields are equal.
    The hash leaves out ``attrs``, so an Interval with any ``attrs`` can be a dict key.

    :param chrom: chromosome
    :param start: 0-based start position (inclusive). It may be negative, e.g. after :meth:`shift`.
    :param end: 0-based end position (exclusive), at least ``start``
    :param strand: ``"+"``, ``"-"`` or ``"."`` for an unknown strand
    :param name: interval name
    :param score: interval score
    :param attrs: additional attributes. The interval keeps a read-only copy.
    :raises TypeError: if ``chrom`` or ``name`` is not a str, if ``start`` or ``end`` is not an int,
        if ``strand`` is not a str, if ``score`` is not a number, or if ``attrs`` is not a mapping
    :raises ValueError: if ``start`` is greater than ``end``, if ``strand`` is not ``"+"``, ``"-"`` or ``"."``,
        or if ``score`` is NaN

    Example:

    >>> interval = Interval("chr1", 10, 20, strand="-", attrs={"gene_id": "g1"})
    >>> interval.width
    10
    >>> interval.shift(5)
    Interval('chr1', 5, 15, strand='-', name='', score=0.0, attrs={'gene_id': 'g1'})
    """

    chrom: str
    start: int
    end: int
    strand: _Strand
    name: str
    score: float
    attrs: Mapping[str, Any] = field(hash=False)

    def __init__(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        strand: _Strand = ".",
        name: str = "",
        score: float = 0,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        for value, field_name in ((chrom, "chrom"), (strand, "strand"), (name, "name")):
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a str, not {type(value).__name__}")
        _check_int(start, "start")
        _check_int(end, "end")
        if not isinstance(score, numbers.Real) or isinstance(score, bool):
            raise TypeError(f"score must be a number, not {type(score).__name__}")
        if attrs is None:
            attrs = {}
        elif not isinstance(attrs, Mapping):
            raise TypeError(f"attrs must be a mapping, not {type(attrs).__name__}")
        if start > end:
            raise ValueError(f"start must not be greater than end, got start={start}, end={end}")
        if strand not in _STRANDS:
            raise ValueError(f'strand must be "+", "-" or ".", got {strand!r}')
        # a NaN score would make the interval unequal to itself
        if math.isnan(score):
            raise ValueError("score must not be NaN")
        # the dataclass is frozen, so the fields are set with object.__setattr__
        object.__setattr__(self, "chrom", chrom)
        object.__setattr__(self, "start", int(start))
        object.__setattr__(self, "end", int(end))
        object.__setattr__(self, "strand", strand)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "score", float(score))
        object.__setattr__(self, "attrs", _ReadOnlyDict(attrs))

    @property
    def width(self) -> int:
        """Number of bases of the interval, ``end - start``."""
        return self.end - self.start

    @property
    def neg_strand(self) -> bool:
        """True if the strand is ``"-"``."""
        return self.strand == "-"

    def unstrand(self) -> "Interval":
        """Return the interval with the strand ``"."``."""
        return replace(self, strand=".")

    def swap_strand(self) -> "Interval":
        """Return the interval with the strand ``"-"`` for ``"+"``, ``"+"`` for ``"-"``, and ``"."`` for ``"."``."""
        if self.strand == ".":
            return replace(self)
        return replace(self, strand="+" if self.strand == "-" else "-")

    def center(self, *, use_strand: bool = True) -> int:
        """Compute the center of the interval.

        If ``start + end`` is odd, the center lies in the middle of a base.
        Then the center is rounded towards the 3' end: up on "+" and "." intervals, and down on "-" intervals.

        :param use_strand: if False, compute the center of the unstranded interval, which is rounded up on all strands
        :returns: the 0-based center position
        """
        return _center(self.start, self.end, reverse=use_strand and self.neg_strand)

    def shift(self, offset: int, *, use_strand: bool = True) -> "Interval":
        """Shift the interval downstream by ``offset`` bases.

        :param offset: number of bases to shift. A negative value shifts upstream.
        :param use_strand: if True, a positive ``offset`` moves "-" intervals towards ``start``.
            Otherwise it moves intervals towards ``end``.
        :returns: a new interval
        :raises TypeError: if ``offset`` is not an int

        Example:

        >>> Interval("chr1", 10, 20, strand="-").shift(5)
        Interval('chr1', 5, 15, strand='-', name='', score=0.0, attrs={})
        """
        _check_int(offset, "offset")
        if use_strand and self.neg_strand:
            offset = -offset
        return replace(self, start=self.start + offset, end=self.end + offset)

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}:{self.strand}"

    def __repr__(self) -> str:
        return (
            f"Interval({self.chrom!r}, {self.start}, {self.end}, strand={self.strand!r}, "
            f"name={self.name!r}, score={self.score!r}, attrs={dict(self.attrs)!r})"
        )

    @classmethod
    def from_str(cls, s: str) -> "Interval":
        """Parse an interval from its str, e.g. ``"chr1:10-20:-"``.

        :param s: the string ``"chrom:start-end:strand"``
        :returns: the interval
        :raises ValueError: if ``s`` has another format, or for the reasons of the constructor
        """
        # rsplit keeps the colons of chromosome names such as HLA-A*01:01:01:01
        parts = s.rsplit(":", 2)
        match = re.fullmatch(r"(-?\d+)-(-?\d+)", parts[1]) if len(parts) == 3 else None
        if match is None:
            raise ValueError(f's must have the format "chrom:start-end:strand", got {s!r}')
        # the constructor checks the strand
        return cls(parts[0], int(match[1]), int(match[2]), strand=cast(_Strand, parts[2]))

    def slop(self, *, upstream: int = 0, downstream: int = 0, use_strand: bool = True) -> "Interval":
        """Extend the interval upstream and downstream.

        :param upstream: number of bases to add upstream. A negative value removes bases.
        :param downstream: number of bases to add downstream. A negative value removes bases.
        :param use_strand: if True, upstream and downstream follow the strand:
            on a "-" interval, ``upstream`` moves ``end`` and ``downstream`` moves ``start``.
            Otherwise ``upstream`` moves ``start`` and ``downstream`` moves ``end``.
        :returns: a new interval
        :raises TypeError: if ``upstream`` or ``downstream`` is not an int
        :raises ValueError: if the new ``start`` is greater than the new ``end``
        """
        _check_int(upstream, "upstream")
        _check_int(downstream, "downstream")
        if use_strand and self.neg_strand:
            upstream, downstream = downstream, upstream
        return replace(self, start=self.start - upstream, end=self.end + downstream)

    def is_valid(self, chrom_len: float = math.inf) -> bool:
        """Check that the interval lies within [0, chrom_len).

        :param chrom_len: length of the chromosome. An interval may end at
            ``chrom_len``, because the end is exclusive.
        """
        return self.start >= 0 and self.end <= chrom_len

    def truncate(self, chrom_len: float = math.inf) -> "Interval":
        """Truncate the interval to become valid.

        :param chrom_len: length of the chromosome
        :returns: a new interval with ``start`` at least 0 and ``end`` at most ``chrom_len``
        :raises ValueError: if ``end`` is negative, or if ``start`` is greater than ``chrom_len``
        """
        # int turns a float chrom_len into an int, and min never returns the default math.inf
        return replace(self, start=max(self.start, 0), end=int(min(self.end, chrom_len)))

    def resize(self, width: int, *, anchor: _Anchor = "center", use_strand: bool = True) -> "Interval":
        """Resize the interval to ``width``, keeping its start, center or end.

        ``"start"`` keeps the 5' end and ``"end"`` keeps the 3' end of the interval.
        On a "-" interval, the 5' end is ``end``, unless ``use_strand`` is False.
        ``"center"`` keeps the :meth:`center`.
        If ``width`` is odd, the result has one more base upstream of the center than downstream.
        Upstream is towards ``start``, or towards ``end`` on "-" intervals if ``use_strand`` is True.

        :param width: width of the resized interval
        :param anchor: the part of the interval to keep: ``"start"``, ``"center"`` or ``"end"``
        :param use_strand: if False, resize as an unstranded interval. The result keeps the strand.
        :returns: a new interval
        :raises TypeError: if ``width`` is not an int
        :raises ValueError: if ``width`` is negative, or if ``anchor`` is not ``"start"``, ``"center"`` or ``"end"``

        Example:

        >>> interval = Interval("chr1", 10, 20, strand="-")
        >>> interval.resize(4, anchor="start")
        Interval('chr1', 16, 20, strand='-', name='', score=0.0, attrs={})
        >>> interval.resize(4, anchor="start", use_strand=False)
        Interval('chr1', 10, 14, strand='-', name='', score=0.0, attrs={})
        """
        _check_resize(width, anchor)
        start, end = _resize_bounds(self.start, self.end, width, anchor, reverse=use_strand and self.neg_strand)
        return replace(self, start=start, end=end)

    def slice(self, i: int, j: int, *, use_strand: bool = True) -> "Interval":
        """Keep the bases ``i`` to ``j - 1`` of the interval, like ``seq[i:j]`` keeps them of its sequence.

        :param i: 0-based position of the first base to keep
        :param j: 0-based position after the last base to keep
        :param use_strand: if True, count the positions from ``end`` on "-" intervals.
            Otherwise count them from ``start``.
        :returns: a new interval of width ``j - i``
        :raises TypeError: if ``i`` or ``j`` is not an int
        :raises ValueError: unless ``0 <= i < j <= width``

        Example:

        >>> Interval("chr1", 10, 20, strand="-").slice(0, 3)
        Interval('chr1', 17, 20, strand='-', name='', score=0.0, attrs={})
        """
        _check_int(i, "i")
        _check_int(j, "j")
        if not 0 <= i < j <= self.width:
            raise ValueError(f"i and j must fulfill 0 <= i < j <= width, got i={i}, j={j} and width={self.width}")
        if use_strand and self.neg_strand:
            return replace(self, start=self.end - j, end=self.end - i)
        return replace(self, start=self.start + i, end=self.start + j)
