import math
from collections.abc import Iterable
from typing import Optional

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors.base import SequenceExtractor

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
        chrom_len: float = math.inf,
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
        :param chrom_len: length of the chromosome. If chrom_len == math.inf, the length of the chromosome is not
            checked.
        :param is_padding: if True, the sequence is padded with 'N's if sequence can't extend to the fixed length,
        :returns: A single sequence (``str``) with all the variants applied.
        """
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

        # the fetch interval is half-open, so it may end at chrom_len
        istart = max(istart, 0)
        # int keeps the end an int for a float chrom_len
        iend = int(min(iend, chrom_len))

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

        # fetch the reference, ignoring the strand
        ref_seq = self._reference.extract(Interval(interval.chrom, istart, iend))
        down_str = self._join(left_pieces, ref_seq, istart, iend)
        up_str = self._join(right_pieces, ref_seq, istart, iend)

        if fixed_len:
            down_str, up_str = self._cut_to_fix_len(down_str, up_str, interval, anchor, is_padding=is_padding)

        seq = down_str + up_str

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

    @staticmethod
    def _join(pieces: list[_Piece], ref_seq: str, istart: int, iend: int) -> str:
        """Join the pieces, with the reference ranges cut from ``ref_seq``, the reference from ``istart`` to ``iend``."""
        parts = []
        for piece in pieces:
            if isinstance(piece, str):
                parts.append(piece)
            else:
                start, end = max(piece[0], istart), min(piece[1], iend)
                if start < end:
                    parts.append(ref_seq[start - istart : end - istart])
        return "".join(parts)

    @staticmethod
    def _cut_to_fix_len(down_str, up_str, interval, anchor, is_padding=False):
        down_len = anchor - interval.start
        down_diff = len(down_str) - down_len
        if down_diff > 0:
            # down_str[-down_len:] would keep the whole string for down_len == 0
            down_str = down_str[len(down_str) - down_len :]
        elif down_diff < 0:
            if is_padding:
                down_str = "N" * abs(down_diff) + down_str
            else:
                raise ValueError("is_padding should be set to True, if the sequence can't extend to the fixed length")

        up_len = interval.end - anchor
        up_diff = len(up_str) - up_len
        if up_diff > 0:
            up_str = up_str[:up_len]
        elif up_diff < 0:
            if is_padding:
                up_str = up_str + "N" * abs(up_diff)
            else:
                raise ValueError("is_padding should be set to True, if the sequence can't extend to the fixed length")

        return down_str, up_str
