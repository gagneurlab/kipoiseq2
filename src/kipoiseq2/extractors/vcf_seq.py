from collections.abc import Iterable
from typing import Optional

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors.base import SequenceExtractor, _chrom_size

__all__ = ["VariantSeqExtractor"]

# IUPAC complement, the same table as pyfaidx.complement. Other characters
# are deleted, so a length change reveals them.
_COMPLEMENT_FROM = "ACTGNactgnYRWSKMDVHBXyrwskmdvhbx"
_COMPLEMENT_TO = "TGACNtgacnRYWSMKHBDVXrywsmkhbdvx"
_COMPLEMENT = str.maketrans(
    _COMPLEMENT_FROM, _COMPLEMENT_TO, "".join(chr(c) for c in range(256) if chr(c) not in _COMPLEMENT_FROM)
)


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence with IUPAC codes.

    :param seq: DNA sequence. Upper and lower case are kept.
    :returns: The reverse complement of ``seq``.
    :raises ValueError: if ``seq`` contains a character without IUPAC complement.
    """
    comp = seq.translate(_COMPLEMENT)
    if len(comp) != len(seq):
        invalid = sorted(set(seq) - set(_COMPLEMENT_FROM))
        raise ValueError("Sequence contains non-DNA characters: {}".format(invalid))
    return comp[::-1]


_OUTSIDE = "\0"
"""Placeholder for a base outside the chromosome. ``extract`` turns it into N or raises a ValueError."""

_Edit = tuple[int, int, str]
"""A variant as ``(start, end, alt)``: the ALT sequence replaces the reference from ``start`` to ``end``, 0-based, half-open."""

_Piece = tuple[int, int] | str
"""A part of the result: a reference range ``(start, end)``, 0-based and half-open, or an ALT sequence.

Overlapping variants can give a range whose ``start`` is greater than its ``end``. It stands for no bases.
"""


class VariantSeqExtractor:
    """Extractor of sequences with variants applied.

    :param reference: the extractor of the reference sequence, e.g. a
        :class:`~kipoiseq2.extractors.FastaStringExtractor`.
        :meth:`extract` passes it intervals with the strand ``"."``, so it returns the "+" strand
        also with ``use_strand=True``. VariantSeqExtractor does not close it.
    :param use_strand: if True, :meth:`extract` reverse complements the sequence of a "-" interval
    :raises TypeError: if ``reference`` is not a :class:`~kipoiseq2.extractors.base.SequenceExtractor`

    Example:

    >>> with FastaStringExtractor("genome.fa", use_strand=False) as fasta:  # doctest: +SKIP
    ...     extractor = VariantSeqExtractor(fasta, use_strand=True)
    ...     seq = extractor.extract(interval, variants, anchor=interval.start)
    """

    def __init__(self, reference: SequenceExtractor, *, use_strand: bool) -> None:
        if not isinstance(reference, SequenceExtractor):
            raise TypeError(f"reference must be a SequenceExtractor, not {type(reference).__name__}")
        self._reference = reference
        self._use_strand = use_strand

    @property
    def use_strand(self) -> bool:
        """If True, :meth:`extract` reverse complements the sequence of a "-" interval."""
        return self._use_strand

    def extract(
        self,
        interval: Interval,
        variants: Iterable[Variant],
        anchor: int,
        fixed_len: bool = True,
        use_strand: Optional[bool] = None,
        is_padding: bool = False,
    ) -> str:
        """
        :param interval: the region of interest from
            which to query the sequence. 0-based
        :param variants: variants overlapping the ``interval``.
            can also be indels. 1-based
        :param anchor: absolution position w.r.t. the interval start. (0-based).
            E.g. for an interval of ``chr1:10-20`` the anchor of 10 denotes
            the point chr1:10 in the 0-based coordinate system.
        :param fixed_len: if True, the return sequence will have the same length
            as the ``interval`` (e.g. ``interval.end - interval.start``)
        :param use_strand: if True, the extracted sequence
            is reverse complemented in case interval.strand == "-".
            Overrides ``self.use_strand``
        :param is_padding: if True, use N for the bases before 0 or after the end of the chromosome.
            Otherwise raise a ValueError if the sequence needs such bases, e.g. because the interval
            ends after the end of the chromosome, or because a deletion pulls in bases from beyond it.
            The length of the chromosome comes from ``reference.chrom_sizes``.
        :returns: A single sequence (``str``) with all the variants applied.
        :raises TypeError: if ``interval`` is not an Interval, or if a variant is not a Variant
        :raises ValueError: if ``interval.chrom`` is not in ``reference.chrom_sizes``,
            if a variant lies on another chromosome than ``interval`` or outside the chromosome,
            if the REF of a variant differs from the reference, ignoring case,
            or if the sequence needs bases outside the chromosome and ``is_padding`` is False
        """
        if not isinstance(interval, Interval):
            raise TypeError(f"interval must be an Interval, not {type(interval).__name__}")
        chrom_size = _chrom_size(self._reference.chrom_sizes, interval.chrom)
        variants = list(variants)
        for v in variants:
            if not isinstance(v, Variant):
                raise TypeError(f"variants must be Variants, not {type(v).__name__}")
            if v.chrom != interval.chrom:
                raise ValueError(f"variant {v} lies on another chromosome than the interval {interval}")
            if v.start < 0 or v.end > chrom_size:
                raise ValueError(f"variant {v} lies outside {v.chrom}, which has {chrom_size} bases")

        anchor = max(min(anchor, interval.end), interval.start)
        edits = self._split([(v.start, v.end, v.alt) for v in variants], anchor)
        if not fixed_len:
            # keep only the parts of the variants within the interval
            edits = self._split(edits, interval.start, keep="right")
            edits = self._split(edits, interval.end, keep="left")

        # the variants right of the anchor in genomic order, and those left of it in reverse order
        right = sorted((e for e in edits if e[0] >= anchor), key=lambda e: e[0])
        left = sorted((e for e in edits if e[0] < anchor), key=lambda e: e[0], reverse=True)

        if fixed_len:
            # deletions pull in reference bases from beyond the interval
            istart = interval.start + sum(min(0, len(alt) - (end - start)) for start, end, alt in left)
            iend = interval.end - sum(min(0, len(alt) - (end - start)) for start, end, alt in right)
        else:
            istart, iend = interval.start, interval.end

        # from the anchor outwards, the reference ranges between the variants and their ALT sequences
        left_pieces: list[_Piece] = []
        prev = anchor
        for start, end, alt in left:
            if end <= istart:
                break
            left_pieces += [(end, prev), alt]
            prev = start
        left_pieces.append((istart, prev))
        left_pieces.reverse()

        right_pieces: list[_Piece] = []
        prev = anchor
        for start, end, alt in right:
            if start >= iend:
                break
            right_pieces += [(prev, start), alt]
            prev = end
        right_pieces.append((prev, iend))

        # one fetch covers the pieces and the REF of each variant
        fetch_start = min([istart, *(v.start for v in variants)])
        fetch_end = max([iend, *(v.end for v in variants)])
        ref_seq = self._fetch(interval.chrom, fetch_start, fetch_end, chrom_size)
        for v in variants:
            ref = ref_seq[v.start - fetch_start : v.end - fetch_start]
            # a reference read with uppercase=False keeps soft-masked bases in lowercase
            if ref.upper() != v.ref.upper():
                raise ValueError(f"the REF of variant {v} differs from the reference {ref!r}")
        down_str = self._join(left_pieces, ref_seq, fetch_start)
        up_str = self._join(right_pieces, ref_seq, fetch_start)

        if fixed_len:
            # the deletions pulled in enough bases, so both parts are long enough.
            # down_str[-n:] would keep the whole string for n == 0
            down_str = down_str[len(down_str) - (anchor - interval.start) :]
            up_str = up_str[: interval.end - anchor]

        seq = down_str + up_str
        if _OUTSIDE in seq:
            if not is_padding:
                raise ValueError(
                    f"the sequence of {interval} needs bases outside {interval.chrom}, which has {chrom_size} bases. "
                    "Pass is_padding=True to fill them with N."
                )
            seq = seq.replace(_OUTSIDE, "N")

        if use_strand is None:
            use_strand = self.use_strand
        if use_strand and interval.strand == "-":
            seq = reverse_complement(seq)

        return seq

    @staticmethod
    def _split(edits: list[_Edit], pos: int, *, keep: str = "both") -> list[_Edit]:
        """Split the variants whose reference range contains ``pos`` at ``pos``.

        :param keep: the parts of a split variant to keep: ``"left"``, ``"right"`` or ``"both"``
        """
        result = []
        for start, end, alt in edits:
            if start < pos < end:
                mid = pos - start
                if keep != "right":
                    result.append((start, pos, alt[:mid]))
                if keep != "left":
                    result.append((pos, end, alt[mid:]))
            else:
                result.append((start, end, alt))
        return result

    def _fetch(self, chrom: str, start: int, end: int, chrom_size: int) -> str:
        """Return the "+" strand of the reference from ``start`` to ``end``, with ``_OUTSIDE`` outside the chromosome."""
        n_before = max(0, min(end, 0) - start)
        n_after = max(0, end - max(start, chrom_size))
        # the strand "." keeps a reference with use_strand=True from reverse complementing
        fetched = Interval(chrom, min(max(start, 0), chrom_size), min(max(end, 0), chrom_size))
        return _OUTSIDE * n_before + self._reference.extract(fetched) + _OUTSIDE * n_after

    @staticmethod
    def _join(pieces: list[_Piece], ref_seq: str, ref_start: int) -> str:
        """Join the pieces, with the reference ranges cut from ``ref_seq``, the reference from ``ref_start`` on."""
        parts = []
        for piece in pieces:
            if isinstance(piece, str):
                parts.append(piece)
            elif piece[0] < piece[1]:
                parts.append(ref_seq[piece[0] - ref_start : piece[1] - ref_start])
        return "".join(parts)
