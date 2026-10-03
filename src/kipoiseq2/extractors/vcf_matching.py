"""Match variants with the intervals they overlap.

`SingleVariantMatcher` and `MultiVariantsMatcher` find the interval-variant
pairs with one polars-bio overlap join. The variants come from a VCF file,
a polars DataFrame or LazyFrame, or Variant objects. Intervals and variants
use 0-based, half-open coordinates: a variant overlaps an interval if
`variant.start < interval.end` and `interval.start < variant.end`, with
`variant.start = pos - 1` and `variant.end = variant.start + len(ref)`.

The matchers need the `ranges` extra (polars and polars-bio).
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, cast

from kipoiseq2.dataclasses import Interval, Variant

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "intervals_to_polars",
    "variants_to_polars",
    "overlap_variants",
    "BaseVariantMatcher",
    "SingleVariantMatcher",
    "MultiVariantsMatcher",
]


def _import_polars():
    try:
        import polars as pl
        import polars_bio as pb
    except ImportError as e:
        raise ImportError(
            "VCF reading and variant matching need the `ranges` extra: pip install 'kipoiseq2[ranges]'"
        ) from e
    return pl, pb


def intervals_to_polars(intervals: Iterable[Interval]) -> pl.DataFrame:
    """Convert intervals to a polars DataFrame.

    Args:
      intervals: Interval objects.

    Returns:
      DataFrame with the columns chrom, start, end (0-based, half-open)
      and strand, one row per interval.
    """
    pl, _ = _import_polars()
    intervals = list(intervals)
    return pl.DataFrame(
        {
            "chrom": [i.chrom for i in intervals],
            "start": [i.start for i in intervals],
            "end": [i.end for i in intervals],
            "strand": [i.strand for i in intervals],
        },
        schema={"chrom": pl.String, "start": pl.Int64, "end": pl.Int64, "strand": pl.String},
    )


def variants_to_polars(variants: Iterable[Variant]) -> pl.DataFrame:
    """Convert variants to a polars DataFrame.

    Args:
      variants: Variant objects.

    Returns:
      DataFrame with the columns chrom, start (`pos - 1`), end
      (`start + len(ref)`), pos (the 1-based VCF POS), ref and alt, one row
      per variant.
    """
    pl, _ = _import_polars()
    variants = list(variants)
    return pl.DataFrame(
        {
            "chrom": [v.chrom for v in variants],
            "start": [v.start for v in variants],
            "end": [v.end for v in variants],
            "pos": [v.pos for v in variants],
            "ref": [v.ref for v in variants],
            "alt": [v.alt for v in variants],
        },
        schema={
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "pos": pl.Int64,
            "ref": pl.String,
            "alt": pl.String,
        },
    )


def overlap_variants(intervals: pl.DataFrame, variants: pl.DataFrame) -> pl.DataFrame:
    """Find all interval-variant pairs that overlap, with one polars-bio overlap join.

    Args:
      intervals: DataFrame with the columns chrom, start and end in 0-based,
        half-open coordinates, e.g. from `intervals_to_polars`.
      variants: DataFrame with the columns chrom, start and end in 0-based,
        half-open coordinates, e.g. from `variants_to_polars`.

    Returns:
      One row per overlapping pair, sorted by variant_idx and then by
      interval_idx. The columns are:

      - all columns of `intervals` under their own names
      - interval_idx: row number of the interval in `intervals`
      - all columns of `variants` except chrom, with the prefix `variant_`
        (e.g. variant_start, variant_end, variant_ref)
      - variant_idx: row number of the variant in `variants`
    """
    # the order of the polars-bio output is not deterministic
    return _overlap_lazy(intervals, variants).collect().sort(["variant_idx", "interval_idx"])


def _overlap_lazy(intervals: pl.DataFrame, variants: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
    """Return the pairs of `overlap_variants` as a LazyFrame, without sorting.

    polars-bio indexes the second input (df2) in memory and streams the first
    input (df1). So the variants go into df1, and a large VCF does not need an
    index in memory.
    """
    pl, pb = _import_polars()
    df1 = variants.with_row_index("variant_idx")
    df2 = intervals.with_row_index("interval_idx")
    # polars-bio reads the coordinate system from per-frame metadata.
    # pb.set_option would change the default of the whole process instead.
    # polars-bio registers the config_meta namespace at import, so mypy does not know it.
    df1.config_meta.set(coordinate_system_zero_based=True)  # type: ignore[union-attr]
    df2.config_meta.set(coordinate_system_zero_based=True)  # type: ignore[attr-defined]
    joined = pb.overlap(
        df1,
        df2,
        cols1=["chrom", "start", "end"],
        cols2=["chrom", "start", "end"],
        suffixes=("_1", "_2"),
        overlap_output="join",
        output_type="polars.LazyFrame",
    )
    columns = [pl.col(f"{c}_2").alias(c) for c in (*intervals.columns, "interval_idx")]
    columns += [pl.col(f"{c}_1").alias(f"variant_{c}") for c in variants.collect_schema().names() if c != "chrom"]
    columns.append(pl.col("variant_idx_1").alias("variant_idx"))
    return joined.select(columns)


class BaseVariantMatcher:
    """
    Base variant intervals matcher
    """

    def __init__(
        self,
        vcf_file: Optional[str] = None,
        variants: Union[pl.DataFrame, pl.LazyFrame, Iterable[Variant], None] = None,
        *,
        intervals: Union[pl.DataFrame, Iterable[Interval]],
        interval_attrs: Optional[Sequence[str]] = None,
        variant_batch_size: int = 10000,
    ):
        """
        Give one source of variants (`vcf_file` or `variants`) and the intervals.

        Args:
          vcf_file: path of a VCF file, read with `scan_vcf_variants`.
          variants: either a polars DataFrame or LazyFrame, or Variant
            objects. A frame needs the columns chrom, pos (1-based), ref and
            alt, with one ALT allele per row. Its other columns pass through
            to the pairs.
          intervals: either a polars DataFrame with the columns chrom, start
            and end (0-based, half-open), an optional strand column and the
            columns in `interval_attrs`, or Interval objects (any iterable,
            e.g. a list or a generator). The matchers
            yield Interval objects as given, so they keep their name and attrs.
          interval_attrs: columns of the `intervals` DataFrame to copy into
            `Interval.attrs`. Not valid with Interval objects.
          variant_batch_size: number of variants per overlap join when
            `SingleVariantMatcher` yields the pairs in batches or one by one.
        """
        self._variants: Optional[List[Variant]] = None
        self._variant_frame = self._read_variants(vcf_file, variants)
        self.interval_attrs = list(interval_attrs or [])
        pl, _ = _import_polars()
        self._intervals: Optional[List[Interval]] = None
        if not isinstance(intervals, pl.DataFrame):
            # a generator can be consumed only once
            self._intervals = intervals = list(cast(Iterable[Interval], intervals))
        self._interval_frame = self._read_intervals(intervals, self.interval_attrs)
        self.variant_batch_size = variant_batch_size

    def _read_variants(self, vcf_file=None, variants=None) -> pl.LazyFrame:
        """Return the variants as a LazyFrame with chrom, start, end, pos, ref, alt and the other columns."""
        pl, _ = _import_polars()
        if (vcf_file is None) == (variants is None):
            raise ValueError("Give exactly one of `vcf_file` and `variants`")
        if vcf_file is not None:
            from kipoiseq2.extractors.vcf import scan_vcf_variants

            return scan_vcf_variants(vcf_file)
        if not isinstance(variants, (pl.DataFrame, pl.LazyFrame)):
            # __iter__ yields these objects
            self._variants = list(variants)
            return variants_to_polars(self._variants).lazy()

        columns = variants.collect_schema().names()
        missing = [c for c in ("chrom", "pos", "ref", "alt") if c not in columns]
        if missing:
            raise ValueError("`variants` lacks the columns {}".format(missing))
        start = pl.col("pos").cast(pl.Int64) - 1
        return variants.lazy().select(
            pl.col("chrom").cast(pl.String),
            start.alias("start"),
            (start + pl.col("ref").cast(pl.String).str.len_chars()).alias("end"),
            pl.col("pos").cast(pl.Int64),
            pl.col("ref").cast(pl.String),
            pl.col("alt").cast(pl.String),
            *[c for c in columns if c not in ("chrom", "start", "end", "pos", "ref", "alt")],
        )

    @staticmethod
    def _read_intervals(intervals, interval_attrs=()) -> pl.DataFrame:
        """Return the intervals as a DataFrame with chrom, start, end, strand and the interval_attrs columns."""
        pl, _ = _import_polars()
        if not isinstance(intervals, pl.DataFrame):
            if interval_attrs:
                raise ValueError("`interval_attrs` is not valid with Interval objects")
            return intervals_to_polars(intervals)

        missing = [c for c in ("chrom", "start", "end", *interval_attrs) if c not in intervals.columns]
        if missing:
            raise ValueError("`intervals` lacks the columns {}".format(missing))
        strand = pl.col("strand") if "strand" in intervals.columns else pl.lit(".")
        return intervals.select(
            pl.col("chrom").cast(pl.String),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            strand.cast(pl.String).alias("strand"),
            *[pl.col(a) for a in interval_attrs],
        )

    def _intervals_of(self, rows: pl.DataFrame) -> List[Interval]:
        """Interval objects of the interval columns in `rows` (the interval frame or overlap_variants output)."""
        if self._intervals is not None:
            return [self._intervals[i] for i in rows.get_column("interval_idx")]
        attrs = self.interval_attrs
        return [
            Interval(chrom, start, end, strand=strand, attrs=dict(zip(attrs, values)))
            for chrom, start, end, strand, *values in rows.select("chrom", "start", "end", "strand", *attrs).iter_rows()
        ]

    def _variants_of(self, pairs: pl.DataFrame) -> List[Variant]:
        """Variant objects of the variant columns in `pairs` (the output of `pairs` or `iter_batches`)."""
        if self._variants is not None:
            return [self._variants[i] for i in pairs.get_column("variant_idx")]
        # an overlapping variant has the chrom of its interval
        columns = pairs.select("chrom", "variant_pos", "variant_ref", "variant_alt")
        return [Variant(chrom, pos, ref, alt) for chrom, pos, ref, alt in columns.iter_rows()]

    def scan_pairs(self) -> pl.LazyFrame:
        """Return all interval-variant pairs as a LazyFrame, without sorting.

        With a VCF file or a LazyFrame as the variant source, the variants
        stream through the overlap join. So for a large VCF, write the pairs
        with `sink_parquet`, or process them with `collect_batches`.

        Returns:
          The pairs as described in `overlap_variants`, in no fixed order.
          variant_idx counts the variants in the order of the variant source.
        """
        return _overlap_lazy(self._interval_frame, self._variant_frame)

    def pairs(self) -> pl.DataFrame:
        """Return all interval-variant pairs from one overlap join.

        Returns:
          The pairs as described in `overlap_variants`. variant_idx counts
          the variants in the order of the variant source.
        """
        return self.scan_pairs().collect().sort(["variant_idx", "interval_idx"])

    def __iter__(self):
        raise NotImplementedError()


class SingleVariantMatcher(BaseVariantMatcher):
    """
    Match each variant with each interval it overlaps.

    Iterating yields (Interval, Variant) pairs. `pairs()` returns all pairs
    as one polars DataFrame, and `scan_pairs()` as a LazyFrame.
    """

    def iter_batches(self) -> Iterator[pl.DataFrame]:
        """Yield the pairs of `variant_batch_size` variants at a time.

        Each batch costs one overlap join. variant_idx counts the variants
        over all batches, and each batch is sorted by variant_idx and then
        by interval_idx.
        """
        pl, _ = _import_polars()
        offset = 0
        for batch in self._variant_frame.collect_batches(chunk_size=self.variant_batch_size):
            pairs = _overlap_lazy(self._interval_frame, batch).collect()
            yield pairs.with_columns(pl.col("variant_idx") + offset).sort(["variant_idx", "interval_idx"])
            offset += batch.height

    def __iter__(self) -> Iterator[Tuple[Interval, Variant]]:
        """
        Yield (Interval, Variant) for each overlapping pair.

        The pairs come in the order of the variant source, and per variant
        in the order of the intervals. With Variant objects as `variants`,
        it yields these objects. Otherwise it builds Variant objects from
        chrom, pos, ref and alt.
        """
        for pairs in self.iter_batches():
            yield from zip(self._intervals_of(pairs), self._variants_of(pairs))


class MultiVariantsMatcher(BaseVariantMatcher):
    """
    Match each interval with all variants that overlap it.

    Iterating yields (Interval, variants) pairs in the order of the intervals,
    also for intervals without variants. The variants of an interval come in
    the order of the variant source.
    """

    def __iter__(self) -> Iterator[Tuple[Interval, Iterator[Variant]]]:
        pairs = self.pairs()
        variants = defaultdict(list)
        for interval_idx, variant in zip(pairs.get_column("interval_idx"), self._variants_of(pairs)):
            variants[interval_idx].append(variant)
        intervals = self._intervals if self._intervals is not None else self._intervals_of(self._interval_frame)
        for interval_idx, interval in enumerate(intervals):
            yield interval, iter(variants[interval_idx])
