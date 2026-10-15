import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors import BaseExtractor, FastaStringExtractor

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


@dataclass(frozen=True)
class Subsequence:
    """A sequence together with its 0-based, half-open coordinates [start, end).

    Slicing keeps the coordinates in sync with the sequence, so a slice of a
    REF allele still knows where it starts and ends on the chromosome.
    """

    seq: str
    start: int
    end: int

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, key: slice) -> "Subsequence":
        i, j, step = key.indices(len(self.seq))
        if step != 1:
            raise ValueError("Subsequence supports only slices with step 1")
        j = max(i, j)
        return Subsequence(self.seq[i:j], self.start + i, self.start + j)


class IntervalSeqBuilder(list):
    """
    String builder for ``Subsequence`` objects and reference ranges.

    A reference range is a tuple ``(start, end)`` of 0-based, half-open coordinates on the reference.
    Overlapping variants can give a range whose ``start`` is greater than its ``end``; it stands for no bases.
    """

    def restore(self, sequence: Subsequence):
        """Replace every reference range in the builder by its part of ``sequence``.

        :param sequence: the reference sequence that covers all ranges
            in the builder.
        """
        for i, item in enumerate(self):
            if isinstance(item, tuple):
                range_start, range_end = item
                start = range_start - sequence.start
                self[i] = sequence[start : start + max(0, range_end - range_start)]

    def _concat(self):
        for sequence in self:
            if type(sequence) is not Subsequence:
                raise TypeError(
                    "Reference ranges should be restored with `restore` method before calling concat method!"
                )
            yield sequence.seq

    def concat(self):
        """
        Build the string from sequence objects.

        :returns: str; the final sequence.
        """
        return "".join(self._concat())


class VariantSeqExtractor(BaseExtractor):
    """
    Sequence extractor which allows to obtain the alternative sequence,
    given some interval and variants inside this interval.

    :param fasta_file: path to the fasta file (can be gzipped)
    :param reference_sequence: extractor returning the reference sequence given some interval
    :param use_strand: if True, the extracted sequence
        is reverse complemented in case interval.strand == "-"
    :raises ValueError: if both or none of ``fasta_file`` and ``reference_sequence`` are given
    """

    _ref_seq_extractor: BaseExtractor

    def __init__(
        self,
        fasta_file: Optional[str] = None,
        reference_sequence: Optional[BaseExtractor] = None,
        *,
        use_strand: bool,
    ):
        self._use_strand = use_strand

        if fasta_file is not None:
            if reference_sequence is not None:
                raise ValueError("Give exactly one of `fasta_file` and `reference_sequence`")
            self._ref_seq_extractor = FastaStringExtractor(fasta_file, use_strand=False)
        else:
            if reference_sequence is None:
                raise ValueError("Give exactly one of `fasta_file` and `reference_sequence`")
            self._ref_seq_extractor = reference_sequence

    @property
    def fasta(self):
        """Deprecated alias of :attr:`ref_seq_extractor`."""
        warnings.warn("`fasta` is deprecated, use `ref_seq_extractor` instead", DeprecationWarning, stacklevel=2)
        return self._ref_seq_extractor

    @property
    def ref_seq_extractor(self) -> BaseExtractor:
        """
        :returns: The reference sequence extractor of this object
        """
        return self._ref_seq_extractor

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
        # Preprocessing
        anchor = max(min(anchor, interval.end), interval.start)
        variant_pairs = self._variant_to_sequence(variants)

        # 1. Split variants overlapping with anchor
        # and interval start end if not fixed_len
        variant_pairs = self._split_overlapping(variant_pairs, anchor)

        if not fixed_len:
            variant_pairs = self._split_overlapping(variant_pairs, interval.start, which="right")
            variant_pairs = self._split_overlapping(variant_pairs, interval.end, which="left")

        variant_pairs = list(variant_pairs)

        # 2. split the variants into upstream and downstream
        # and sort the variants in each interval
        upstream_variants = sorted(filter(lambda x: x[0].start >= anchor, variant_pairs), key=lambda x: x[0].start)

        downstream_variants = sorted(
            filter(lambda x: x[0].start < anchor, variant_pairs), key=lambda x: x[0].start, reverse=True
        )

        # 3. Extend start and end position for deletions
        if fixed_len:
            istart, iend = self._updated_interval(interval, upstream_variants, downstream_variants)
        else:
            istart, iend = interval.start, interval.end

        # the fetch interval is half-open, so it may end at chrom_len
        istart = max(istart, 0)
        # int keeps the end an int for a float chrom_len
        iend = int(min(iend, chrom_len))

        # 4. Iterate from the anchor point outwards. At each
        # register the interval from which to take the reference sequence
        # as well as the interval for the variant
        down_sb = self._downstream_builder(downstream_variants, interval, anchor, istart)

        up_sb = self._upstream_builder(upstream_variants, interval, anchor, iend)

        # 5. fetch the sequence and restore intervals in builder
        seq = self._fetch(interval, istart, iend)
        up_sb.restore(seq)
        down_sb.restore(seq)

        # 6. Concate sequences from the upstream and downstream splits. Concat
        # upstream and downstream sequence. Cut to fix the length.
        down_str = down_sb.concat()
        up_str = up_sb.concat()

        if fixed_len:
            down_str, up_str = self._cut_to_fix_len(down_str, up_str, interval, anchor, is_padding=is_padding)

        seq = down_str + up_str

        if use_strand is None:
            use_strand = self.use_strand
        if use_strand and interval.strand == "-":
            seq = reverse_complement(seq)

        return seq

    @staticmethod
    def _variant_to_sequence(variants):
        """
        Convert :class:`~kipoiseq2.dataclasses.Variant` objects to (REF, ALT) pairs of ``Subsequence`` objects.
        """
        for v in variants:
            ref = Subsequence(seq=v.ref, start=v.start, end=v.start + len(v.ref))
            alt = Subsequence(seq=v.alt, start=v.start, end=v.start + len(v.alt))
            yield ref, alt

    @staticmethod
    def _split_overlapping(variant_pairs, anchor, which="both"):
        """
        Split the variants hitting the anchor into two
        """
        for ref, alt in variant_pairs:
            if ref.start < anchor < ref.end:
                mid = anchor - ref.start
                if which == "left" or which == "both":
                    yield ref[:mid], alt[:mid]
                if which == "right" or which == "both":
                    yield ref[mid:], alt[mid:]
            else:
                yield ref, alt

    @staticmethod
    def _updated_interval(interval, up_variants, down_variants):
        istart = interval.start
        iend = interval.end

        for ref, alt in up_variants:
            diff_len = len(alt) - len(ref)
            if diff_len < 0:
                iend -= diff_len

        for ref, alt in down_variants:
            diff_len = len(alt) - len(ref)
            if diff_len < 0:
                istart += diff_len

        return istart, iend

    @staticmethod
    def _downstream_builder(down_variants, interval, anchor, istart):
        down_sb = IntervalSeqBuilder()

        prev = anchor
        for ref, alt in down_variants:
            if ref.end <= istart:
                break
            down_sb.append((ref.end, prev))
            down_sb.append(alt)
            prev = ref.start
        down_sb.append((istart, prev))
        down_sb.reverse()

        return down_sb

    @staticmethod
    def _upstream_builder(up_variants, interval, anchor, iend):
        up_sb = IntervalSeqBuilder()

        prev = anchor
        for ref, alt in up_variants:
            if ref.start >= iend:
                break
            up_sb.append((prev, ref.start))
            up_sb.append(alt)
            prev = ref.end
        up_sb.append((prev, iend))

        return up_sb

    def _fetch(self, interval, istart, iend):
        # fetch interval, ignore strand
        seq = self.ref_seq_extractor.extract(Interval(interval.chrom, istart, iend))
        seq = Subsequence(seq=seq, start=istart, end=iend)
        return seq

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
