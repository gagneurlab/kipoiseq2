import itertools
from collections.abc import Iterable
from typing import Literal

from kipoiseq2.dataclasses import Interval, Variant, _check_int, _is_reverse
from kipoiseq2.extractors.base import SequenceExtractor, _chrom_size
from kipoiseq2.transforms import reverse_complement

__all__ = ["VariantSeqExtractor"]

_OUTSIDE = "\0"
"""Placeholder for a base outside the chromosome. ``extract`` turns it into N or raises a ValueError."""

_Edit = tuple[int, int, str]
"""A variant as ``(start, end, alt)``: the ALT sequence replaces the reference from ``start`` to ``end``, 0-based, half-open."""

_Piece = tuple[int, int] | str
"""A part of the result: a reference range ``(start, end)``, 0-based and half-open, or an ALT sequence.

A variant that spans an end of the sequence can give a range whose ``start`` is greater than its ``end``.
It stands for no bases.
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
    ...     seq = extractor.extract(interval, variants, anchor_pos=interval.start)
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
        *,
        anchor_pos: int | None = None,
        pad_with_n: bool = False,
    ) -> str:
        """Return the sequence of ``interval`` with the variants applied.

        If ``anchor_pos`` is None, the sequence covers the reference from ``interval.start`` to ``interval.end``
        with the parts of the variants within the interval, so indels change its length.

        If ``anchor_pos`` is an int, the sequence keeps the width of the interval.
        The variants left of ``anchor_pos`` change the part from ``interval.start`` to ``anchor_pos``.
        This part then keeps its length: it gets cut at ``interval.start``, or extended there with reference bases.
        The variants right of ``anchor_pos`` change the part from ``anchor_pos`` to ``interval.end``,
        which gets cut or extended at ``interval.end``.
        So a deletion pulls in reference bases from beyond the interval, and an insertion pushes bases out of it.
        A variant lies left of ``anchor_pos`` if its REF does.
        A variant whose REF spans ``anchor_pos`` is split there: the first ``anchor_pos - variant.start`` bases
        of its REF and ALT lie left, and the rest right.
        For example, the insertion A>AGGG with its REF at ``anchor_pos - 1`` lies left entirely.
        GenomeKit normalizes it to an insertion at ``anchor_pos`` and puts it right.

        :param interval: the region of the sequence
        :param variants: the variants to apply, on the chromosome of ``interval``.
            Variants whose REF does not overlap ``interval`` have no effect.
            Two variants must not overlap: their REF must not share a base.
        :param anchor_pos: None, or a 0-based position from ``interval.start`` to ``interval.end``.
            If None, the length of the sequence follows the variants.
            Otherwise the sequence has the width of ``interval``, anchored at ``anchor_pos`` as described above.
        :param pad_with_n: if True, use N for the bases before 0 or after the end of the chromosome.
            Otherwise raise a ValueError if the sequence needs such bases.
            It needs them if ``interval`` runs past an end of the chromosome.
            With ``anchor_pos``, it also needs them if a deletion pulls in bases from beyond an end of the chromosome.
            The length of the chromosome comes from ``reference.chrom_sizes``.
        :returns: the sequence, reverse complemented for a "-" interval if :attr:`use_strand` is True
        :raises TypeError: if ``interval`` is not an Interval, if a variant is not a Variant,
            or if ``anchor_pos`` is neither None nor an int
        :raises ValueError: if ``anchor_pos`` lies outside ``interval``,
            if ``interval.chrom`` is not in ``reference.chrom_sizes``,
            if a variant lies on another chromosome than ``interval`` or outside the chromosome,
            if two variants overlap, if the REF of a variant differs from the reference, ignoring case,
            or if the sequence needs bases outside the chromosome and ``pad_with_n`` is False
        """
        if not isinstance(interval, Interval):
            raise TypeError(f"interval must be an Interval, not {type(interval).__name__}")
        if anchor_pos is not None:
            _check_int(anchor_pos, "anchor_pos")
            if not interval.start <= anchor_pos <= interval.end:
                raise ValueError(
                    f"anchor_pos must lie within {interval.start} and {interval.end}, the bounds of {interval}, "
                    f"got {anchor_pos}"
                )
        chrom_size = _chrom_size(self._reference.chrom_sizes, interval.chrom)
        variants = list(variants)
        for v in variants:
            if not isinstance(v, Variant):
                raise TypeError(f"variants must be Variants, not {type(v).__name__}")
            if v.chrom != interval.chrom:
                raise ValueError(f"variant {v} lies on another chromosome than the interval {interval}")
            if v.start < 0 or v.end > chrom_size:
                raise ValueError(f"variant {v} lies outside {v.chrom}, which has {chrom_size} bases")
        # sorted by start, two variants overlap only if two neighbors overlap
        for v1, v2 in itertools.pairwise(sorted(variants, key=lambda v: v.start)):
            if v2.start < v1.end:
                raise ValueError(f"variants {v1} and {v2} overlap, but the REF of two variants must not share a base")

        edits = [(v.start, v.end, v.alt) for v in variants]
        if anchor_pos is None:
            # keep only the parts of the variants within the interval, which all lie right of its start
            anchor = interval.start
            edits = self._split(self._split(edits, interval.start, keep="right"), interval.end, keep="left")
        else:
            anchor = anchor_pos
            edits = self._split(edits, anchor)

        # the variants right of the anchor in genomic order, and those left of it in reverse order
        right = sorted((e for e in edits if e[0] >= anchor), key=lambda e: e[0])
        left = sorted((e for e in edits if e[0] < anchor), key=lambda e: e[0], reverse=True)

        if anchor_pos is not None:
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

        if anchor_pos is not None:
            # the deletions pulled in enough bases, so both parts are long enough.
            # down_str[-n:] would keep the whole string for n == 0
            down_str = down_str[len(down_str) - (anchor - interval.start) :]
            up_str = up_str[: interval.end - anchor]

        seq = down_str + up_str
        if _OUTSIDE in seq:
            if not pad_with_n:
                raise ValueError(
                    f"the sequence of {interval} needs bases outside {interval.chrom}, which has {chrom_size} bases. "
                    "Pass pad_with_n=True to fill them with N."
                )
            seq = seq.replace(_OUTSIDE, "N")

        if _is_reverse(interval, use_strand=self._use_strand):
            seq = reverse_complement(seq)

        return seq

    @staticmethod
    def _split(edits: list[_Edit], pos: int, *, keep: Literal["left", "right", "both"] = "both") -> list[_Edit]:
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
