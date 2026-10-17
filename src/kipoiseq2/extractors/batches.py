"""Build batches of reference and alternative sequences as polars DataFrames.

:func:`iter_variant_seqs` applies the variants of interval-variant pairs, e.g.
of :func:`~kipoiseq2.extractors.vcf_matching.scan_variant_pairs`, to the
reference, and encodes the sequences for model input. It needs polars, which
the ``ranges`` extra installs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from kipoiseq2.dataclasses import _ANCHORS, Interval, Variant, _Anchor, _check_int, _is_reverse
from kipoiseq2.encoders import Encoder, Tokenizer
from kipoiseq2.extractors.base import SequenceExtractor
from kipoiseq2.extractors.vcf import _check_names
from kipoiseq2.extractors.vcf_seq import VariantSeqExtractor
from kipoiseq2.transforms import reverse_complement

if TYPE_CHECKING:
    import polars as pl
    from numpy.typing import NDArray

__all__ = ["iter_variant_seqs"]

_INTERVAL_COLUMNS = ("chrom", "start", "end", "strand")
"""The columns of the pairs that hold the interval."""

_VARIANT_COLUMNS = ("variant_pos", "variant_ref", "variant_alt")
"""The columns of the pairs that hold the variant."""

_SEQ_COLUMNS = ("ref_seq", "alt_seq", "ref", "alt")
"""The columns that :func:`iter_variant_seqs` adds to the group keys and the interval."""

_CARRIER_COLUMN = "variant_carrier"
"""The column of the pairs that tells whether the sample carries the ALT allele,
from the carrier column of :func:`~kipoiseq2.extractors.vcf.scan_vcf_genotypes`."""


def iter_variant_seqs(
    pairs: pl.DataFrame | pl.LazyFrame,
    reference: SequenceExtractor,
    *,
    use_strand: bool,
    anchor: _Anchor | None = "center",
    group_by: Sequence[str] = ("interval_idx", "variant_idx"),
    encoding: Tokenizer | Encoder | None = Tokenizer(),
    pad_with_n: bool = False,
    batch_size: int = 64,
) -> Iterator[pl.DataFrame]:
    """Yield the reference and alternative sequences of interval-variant pairs as DataFrame batches.

    The rows of ``pairs`` with equal values in the columns ``group_by`` form a group.
    Each group gives one output row: its variants are applied to its interval with
    :meth:`VariantSeqExtractor.extract <kipoiseq2.extractors.vcf_seq.VariantSeqExtractor.extract>`.
    All rows of a group must have the same interval, and the variants of a group must not overlap.
    The default ``group_by`` gives one row per pair, e.g. for variant effect predictions.
    ``group_by=["interval_idx"]`` applies all variants of an interval together.
    For the haplotypes of the samples, pass ``scan_vcf_genotypes(path, haplotypes=True)`` as the variants of
    :func:`~kipoiseq2.extractors.vcf_matching.scan_variant_pairs`,
    and group by ``["interval_idx", "variant_sample", "variant_haplotype"]``.
    An interval without variants has no pairs, so it gives no row.
    Each pair applies its ALT allele, so ``pairs`` must not contain samples that do not carry it.
    For the genotypes of :func:`~kipoiseq2.extractors.vcf.scan_vcf_genotypes`, keep ``carriers_only=True``:
    a False or a null in the column variant_carrier raises a ValueError.

    ``pairs`` is collected into memory when ``iter_variant_seqs`` is called, and the groups are checked then.
    The sequences of a batch are built when the iterator reaches it.
    Within a batch, the reference of each interval is fetched once.

    The encoded columns are polars Array columns if all sequences have the same length,
    i.e. if ``anchor`` is not None and all intervals of ``pairs`` have the same width.
    Then ``batch["alt"].to_numpy()`` returns a read-only array of shape ``(rows, length)`` for a Tokenizer,
    or ``(rows, length, n_columns)`` for an Encoder, without a copy unless the dtype is bool.
    Otherwise the encoded columns of every batch are polars List columns, with one item per letter.

    :param pairs: DataFrame or LazyFrame with the columns chrom, start, end, strand,
        variant_pos, variant_ref, variant_alt and the columns of ``group_by``,
        e.g. of :func:`~kipoiseq2.extractors.vcf_matching.scan_variant_pairs`
    :param reference: the reference sequence, e.g. a :class:`~kipoiseq2.extractors.fasta.FastaStringExtractor`.
        It gets intervals with the strand ``"."``, as from :class:`~kipoiseq2.extractors.vcf_seq.VariantSeqExtractor`.
    :param use_strand: if True, reverse complement the sequences of "-" intervals, and let ``anchor`` follow the strand
    :param anchor: the part of the interval that keeps its position: ``"start"``, ``"center"`` or ``"end"``.
        Then alt_seq has the width of the interval. An indel pulls in or pushes out bases at the end of the interval
        on its side of the anchor.
        ``anchor`` picks the ``anchor_pos`` of ``VariantSeqExtractor.extract`` by the rule of
        :meth:`Interval.resize <kipoiseq2.dataclasses.Interval.resize>`:
        ``"start"`` is the 5' end, which is ``end`` on "-" intervals if ``use_strand`` is True, and ``start`` otherwise.
        ``"end"`` is the other end. ``"center"`` is ``interval.center(use_strand=use_strand)``.
        None gives alt_seq the length that the variants give, as ``anchor_pos=None``.
    :param group_by: the columns of ``pairs`` that form the groups
    :param encoding: a :class:`~kipoiseq2.encoders.Tokenizer` for the tokens of
        :meth:`~kipoiseq2.encoders.Tokenizer.tokenize_series`, an :class:`~kipoiseq2.encoders.Encoder` for the one-hot
        arrays of :meth:`~kipoiseq2.encoders.Encoder.one_hot_series`, or None for no encoded columns.
        The dtype of ``encoding`` sets the dtype of the encoded columns, e.g. ``Encoder(dtype=np.float16)``.
    :param pad_with_n: if True, use N for the bases outside the chromosome. Otherwise raise a ValueError for them.
    :param batch_size: number of rows per batch. The last batch can have fewer rows.
    :returns: iterator of DataFrames, sorted by the columns of ``group_by``, with the columns:

        - the columns of ``group_by``
        - chrom, start, end and strand of the interval, unless ``group_by`` contains them
        - ref_seq: the reference sequence of the interval
        - alt_seq: the sequence of the interval with the variants of the group applied
        - ref and alt: ref_seq and alt_seq encoded, unless ``encoding`` is None.
          For a Tokenizer, they are ``pl.Array(dtype, length)`` or ``pl.List(dtype)``, with the polars type of its
          dtype, e.g. ``pl.UInt8`` for the default ``Tokenizer()``.
          For an Encoder, they are ``pl.Array(dtype, (length, n_columns))`` or ``pl.List(pl.Array(dtype, n_columns))``.
          length is the number of letters.
    :raises TypeError: if ``pairs`` is not a polars DataFrame or LazyFrame,
        if ``reference`` is not a :class:`~kipoiseq2.extractors.base.SequenceExtractor`,
        if ``group_by`` is not a sequence of strings, if ``encoding`` is not a Tokenizer, an Encoder or None,
        or if ``batch_size`` is not an int
    :raises ValueError: if ``anchor`` is not ``"start"``, ``"center"``, ``"end"`` or None,
        if ``group_by`` is empty, repeats a column or names ref_seq, alt_seq, ref or alt,
        if ``pairs`` lacks a column, if a required column of ``pairs`` contains nulls,
        if the column variant_carrier of ``pairs`` contains False or null,
        if a group has several intervals, or if ``batch_size`` is less than 1.
        While iterating, it raises the errors of ``VariantSeqExtractor.extract`` and of the encoding,
        with a note that names the group, e.g. a ValueError for overlapping variants,
        for a REF that differs from the reference, or for a letter that ``encoding`` does not contain.

    Example:

    >>> pairs = scan_variant_pairs(windows, scan_vcf_variants("variants.vcf.gz"))  # doctest: +SKIP
    >>> with FastaStringExtractor("genome.fa", use_strand=False) as fasta:  # doctest: +SKIP
    ...     for batch in iter_variant_seqs(pairs, fasta, use_strand=True, batch_size=32):
    ...         alt = batch["alt"].to_numpy()  # uint8 tokens of shape (32, window width)
    """
    import polars as pl

    if not isinstance(pairs, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(f"pairs must be a polars DataFrame or LazyFrame, not {type(pairs).__name__}")
    if not isinstance(reference, SequenceExtractor):
        raise TypeError(f"reference must be a SequenceExtractor, not {type(reference).__name__}")
    if anchor is not None and anchor not in _ANCHORS:
        raise ValueError(f'anchor must be "start", "center", "end" or None, got {anchor!r}')
    _check_names(group_by, "group_by")
    keys = list(group_by)
    if not keys:
        raise ValueError("group_by must name at least one column")
    if len(set(keys)) < len(keys):
        raise ValueError(f"group_by must not repeat a column, got {keys}")
    clash = [c for c in keys if c in _SEQ_COLUMNS]
    if clash:
        raise ValueError(f"group_by must not contain {clash}, because the batches use these names for the sequences")
    # the method for a Series encodes the batches, and the method for one sequence finds the group of an error
    encode_series: Callable[[pl.Series], pl.Series] | None = None
    encode: Callable[[str], NDArray[Any]] | None = None
    if isinstance(encoding, Tokenizer):
        encode_series, encode = encoding.tokenize_series, encoding.tokenize
    elif isinstance(encoding, Encoder):
        encode_series, encode = encoding.one_hot_series, encoding.one_hot
    elif encoding is not None:
        raise TypeError(f"encoding must be a Tokenizer, an Encoder or None, not {type(encoding).__name__}")
    _check_int(batch_size, "batch_size")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    columns = pairs.collect_schema().names()
    missing = [c for c in (*_INTERVAL_COLUMNS, *_VARIANT_COLUMNS, *keys) if c not in columns]
    if missing:
        raise ValueError(f"pairs lacks the columns {missing}")

    # The keys get private names, so they cannot clash with the other columns.
    key_names = [f"key_{i}" for i in range(len(keys))]
    carrier = [_CARRIER_COLUMN] if _CARRIER_COLUMN in columns else []
    frame = (
        pairs.lazy()
        .select(*(pl.col(c).alias(k) for c, k in zip(keys, key_names)), *_INTERVAL_COLUMNS, *_VARIANT_COLUMNS, *carrier)
        .collect()
    )
    if carrier and not frame.get_column(_CARRIER_COLUMN).fill_null(False).all():
        raise ValueError(
            f"pairs has rows whose {_CARRIER_COLUMN} is False or null, "
            "but iter_variant_seqs applies the ALT allele of every row. "
            "Scan the genotypes with scan_vcf_genotypes(..., carriers_only=True)."
        )
    null_counts = frame.select(pl.col(*_INTERVAL_COLUMNS, *_VARIANT_COLUMNS).null_count()).row(0, named=True)
    nulls = [c for c, n in null_counts.items() if n]
    if nulls:
        raise ValueError(f"pairs has nulls in the columns {nulls}")
    groups = (
        frame.group_by(key_names)
        .agg(
            pl.struct(_INTERVAL_COLUMNS).n_unique().alias("n_intervals"),
            pl.col(_INTERVAL_COLUMNS).first(),
            pl.col(_VARIANT_COLUMNS),
            # extract fetches at most the sum of the REF lengths beyond each end of the interval
            pl.col("variant_ref").str.len_chars().sum().alias("margin"),
        )
        .sort(key_names)
    )
    several = groups.filter(pl.col("n_intervals") > 1)
    if len(several):
        group = dict(zip(keys, several.select(key_names).row(0)))
        raise ValueError(
            f"the group {group} of pairs has {several['n_intervals'][0]} intervals, but a group needs one interval. "
            "Add interval_idx to group_by."
        )
    same_length = anchor is not None and groups.select((pl.col("end") - pl.col("start")).n_unique()).item() <= 1
    return _iter_batches(
        groups,
        reference,
        keys=keys,
        key_names=key_names,
        use_strand=use_strand,
        anchor=anchor,
        encode_series=encode_series,
        encode=encode,
        same_length=same_length,
        pad_with_n=pad_with_n,
        batch_size=batch_size,
    )


def _iter_batches(
    groups: pl.DataFrame,
    reference: SequenceExtractor,
    *,
    keys: list[str],
    key_names: list[str],
    use_strand: bool,
    anchor: _Anchor | None,
    encode_series: Callable[[pl.Series], pl.Series] | None,
    encode: Callable[[str], NDArray[Any]] | None,
    same_length: bool,
    pad_with_n: bool,
    batch_size: int,
) -> Iterator[pl.DataFrame]:
    """Yield the batches of :func:`iter_variant_seqs`.

    :param groups: one row per group, sorted, with the keys, n_intervals, the interval columns,
        the variant columns as lists, and margin
    :param keys: the columns of ``group_by``
    :param key_names: the names of the keys in ``groups``
    :param encode_series: the method that encodes a Series of sequences, or None for no encoded columns
    :param encode: the method that encodes one sequence, or None for no encoded columns
    :param same_length: if True, all sequences have the same length, so the encoded columns are Array columns.
        Otherwise they are List columns.
    """
    import polars as pl

    window_reference = _WindowReference(reference)
    extractor = VariantSeqExtractor(window_reference, use_strand=use_strand)
    for batch in groups.iter_slices(batch_size):
        # Series.to_list and the DataFrame constructor run no polars query, unlike select
        chroms, starts, ends, strands, positions, ref_alleles, alt_alleles, margins = (
            batch.get_column(c).to_list() for c in (*_INTERVAL_COLUMNS, *_VARIANT_COLUMNS, "margin")
        )
        window_margins: dict[tuple[str, int, int], int] = {}
        for coords, margin in zip(zip(chroms, starts, ends), margins):
            window_margins[coords] = max(margin, window_margins.get(coords, 0))
        windows = {coords: window_reference.fetch(*coords, margin) for coords, margin in window_margins.items()}

        ref_seqs: list[str] = []
        alt_seqs: list[str] = []
        rows = zip(chroms, starts, ends, strands, positions, ref_alleles, alt_alleles)
        for i, (chrom, start, end, strand, row_positions, row_refs, row_alts) in enumerate(rows):
            try:
                window_reference.window = windows[chrom, start, end]
                interval = Interval(chrom, start, end, strand=strand)
                ref_seq = _ref_seq(window_reference, extractor, interval, pad_with_n=pad_with_n)
                variants = [Variant(chrom, *v) for v in zip(row_positions, row_refs, row_alts)]
                # a resize to width 0 keeps only the position of the anchor
                anchor_pos = None if anchor is None else interval.resize(0, anchor=anchor, use_strand=use_strand).start
                alt_seq = extractor.extract(interval, variants, anchor_pos=anchor_pos, pad_with_n=pad_with_n)
            except (TypeError, ValueError) as e:
                e.add_note(_group_note(batch, i, keys, key_names))
                raise
            ref_seqs.append(ref_seq)
            alt_seqs.append(alt_seq)

        ref_series = pl.Series("ref_seq", ref_seqs, dtype=pl.String)
        alt_series = pl.Series("alt_seq", alt_seqs, dtype=pl.String)
        columns = [
            *(batch.get_column(name).alias(k) for k, name in zip(keys, key_names)),
            *(batch.get_column(c) for c in _INTERVAL_COLUMNS if c not in keys),
            ref_series,
            alt_series,
        ]
        if encode_series is not None and encode is not None:
            try:
                columns += [
                    _encoded(encode_series, ref_series, same_length=same_length).alias("ref"),
                    _encoded(encode_series, alt_series, same_length=same_length).alias("alt"),
                ]
            except ValueError:
                _raise_with_group(encode, ref_seqs, alt_seqs, batch, keys, key_names)
                raise
        yield pl.DataFrame(columns)


def _encoded(encode_series: Callable[[pl.Series], pl.Series], seqs: pl.Series, *, same_length: bool) -> pl.Series:
    """Return the encoded ``seqs``, as an Array column if ``same_length`` is True, and as a List column otherwise.

    :raises ValueError: if ``encode_series`` cannot encode a sequence
    """
    import polars as pl

    series = encode_series(seqs)
    if not same_length and isinstance(series.dtype, pl.Array):
        # the sequences of this batch have the same length by chance, but all batches share one schema
        series = series.cast(pl.List(series.dtype.inner))
    return series


def _raise_with_group(
    encode: Callable[[str], NDArray[Any]],
    ref_seqs: list[str],
    alt_seqs: list[str],
    batch: pl.DataFrame,
    keys: list[str],
    key_names: list[str],
) -> None:
    """Encode the sequences row by row, and raise the first ValueError with a note that names the group of its row.

    If no sequence raises a ValueError, it returns, and the caller raises its own error.
    """
    for i, (ref_seq, alt_seq) in enumerate(zip(ref_seqs, alt_seqs)):
        try:
            encode(ref_seq)
            encode(alt_seq)
        except ValueError as e:
            e.add_note(_group_note(batch, i, keys, key_names))
            raise


def _group_note(batch: pl.DataFrame, i: int, keys: list[str], key_names: list[str]) -> str:
    """Return the note that names the group of row ``i`` of ``batch`` for an error."""
    group = {k: batch.get_column(name)[i] for k, name in zip(keys, key_names)}
    return f"in the group {group} of pairs"


def _ref_seq(
    window_reference: _WindowReference, extractor: VariantSeqExtractor, interval: Interval, *, pad_with_n: bool
) -> str:
    """Return the reference sequence of ``interval``, as ``extractor.extract(interval, [])``.

    It cuts the sequence from the window if the window covers ``interval``.
    The window covers every interval within the chromosome, so only an interval past an end of the chromosome
    needs ``extract``, which pads it or raises the error.
    """
    seq = window_reference.cut(interval)
    if seq is None:
        return extractor.extract(interval, [], pad_with_n=pad_with_n)
    if _is_reverse(interval, use_strand=extractor.use_strand):
        return reverse_complement(seq)
    return seq


class _WindowReference:
    """The reference of :func:`iter_variant_seqs`, which serves the sequences of an interval from one fetch.

    :meth:`extract` cuts the sequence from :attr:`window` if the window covers it,
    and fetches it from the reference otherwise.

    :param reference: the reference sequence
    """

    def __init__(self, reference: SequenceExtractor) -> None:
        self._reference = reference
        self.window: tuple[str, int, str] = ("", 0, "")
        """The current window: its chromosome, its start and its sequence on the "+" strand."""

    @property
    def chrom_sizes(self) -> Mapping[str, int]:
        """Length of each chromosome, by name."""
        return self._reference.chrom_sizes

    def fetch(self, chrom: str, start: int, end: int, margin: int) -> tuple[str, int, str]:
        """Fetch the window from ``start - margin`` to ``end + margin``, within the chromosome.

        :returns: the window for :attr:`window`. For a chromosome that is not in the reference, an empty window.
        """
        chrom_size = self.chrom_sizes.get(chrom)
        if chrom_size is None:
            # extract raises the error for the chromosome
            return chrom, 0, ""
        window_start = min(max(start - margin, 0), chrom_size)
        window_end = max(min(end + margin, chrom_size), window_start)
        return chrom, window_start, self._reference.extract(Interval(chrom, window_start, window_end))

    def cut(self, interval: Interval) -> str | None:
        """Return the "+" strand of ``interval`` if :attr:`window` covers it, and None otherwise."""
        chrom, window_start, seq = self.window
        start = interval.start - window_start
        end = interval.end - window_start
        if interval.chrom == chrom and start >= 0 and end <= len(seq):
            return seq[start:end]
        return None

    def extract(self, interval: Interval) -> str:
        """Return the "+" strand of ``interval``.

        :param interval: an interval with the strand ``"."``, from VariantSeqExtractor
        """
        seq = self.cut(interval)
        return self._reference.extract(interval) if seq is None else seq
