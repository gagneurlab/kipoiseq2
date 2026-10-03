"""Match variants with the intervals they overlap.

`SingleVariantMatcher` finds the interval-variant pairs with a polars-bio
overlap join. Intervals and variants use 0-based, half-open coordinates:
a variant overlaps an interval if `variant.start < interval.end` and
`interval.start < variant.end`, with `variant.start = pos - 1` and
`variant.end = variant.start + len(ref)`.

The matchers need the `ranges` extra (polars and polars-bio).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, cast

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.variant_source import VariantFetcher

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "intervals_to_polars",
    "variants_to_polars",
    "overlap_variants",
    "VariantListFetcher",
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


class VariantListFetcher(VariantFetcher):
    """Variant fetcher over Variant objects held in memory."""

    def __init__(self, variants: Iterable[Variant]):
        self.variants = list(variants)

    def fetch_variants(self, interval: Union[Interval, Iterable[Interval]]) -> Iterator[Variant]:
        """Yield the variants that overlap the interval(s), per interval in the given order."""
        intervals = [interval] if isinstance(interval, Interval) else interval
        for i in intervals:
            for v in self.variants:
                if v.chrom == i.chrom and v.start < i.end and i.start < v.end:
                    yield v

    def __iter__(self) -> Iterator[Variant]:
        yield from self.variants


class BaseVariantMatcher:
    """
    Base variant intervals matcher
    """

    def __init__(
        self,
        vcf_file: Optional[str] = None,
        variants: Optional[Sequence[Variant]] = None,
        variant_fetcher: Optional[VariantFetcher] = None,
        *,
        intervals: Union[pl.DataFrame, Iterable[Interval]],
        interval_attrs: Optional[Sequence[str]] = None,
        vcf_lazy: bool = True,
        variant_batch_size: int = 10000,
    ):
        """
        Give one source of variants (`vcf_file`, `variants` or
        `variant_fetcher`) and the intervals.

        Args:
          vcf_file: path of a VCF file, read with `MultiSampleVCF`
            (needs the `vcf` extra).
          variants: Variant objects.
          variant_fetcher: a VariantFetcher, e.g. a `MultiSampleVCF`.
          intervals: either a polars DataFrame with the columns chrom, start
            and end (0-based, half-open), an optional strand column and the
            columns in `interval_attrs`, or Interval objects (any iterable,
            e.g. a list or a generator). The matchers
            yield Interval objects as given, so they keep their name and attrs.
          interval_attrs: columns of the `intervals` DataFrame to copy into
            `Interval.attrs`. Not valid with Interval objects.
          vcf_lazy: passed to `MultiSampleVCF` as `lazy`.
          variant_batch_size: number of variants per overlap join when
            iterating over the pairs.
        """
        self.variant_fetcher = self._read_variants(vcf_file, variants, variant_fetcher, vcf_lazy)
        self.interval_attrs = list(interval_attrs or [])
        pl, _ = _import_polars()
        self._intervals: Optional[List[Interval]] = None
        if not isinstance(intervals, pl.DataFrame):
            # a generator can be consumed only once
            self._intervals = intervals = list(cast(Iterable[Interval], intervals))
        self._interval_frame = self._read_intervals(intervals, self.interval_attrs)
        self.variant_batch_size = variant_batch_size

    @staticmethod
    def _read_variants(
        vcf_file=None,
        variants=None,
        variant_fetcher=None,
        vcf_lazy: bool = True,
    ) -> VariantFetcher:
        if vcf_file is not None:
            from kipoiseq2.extractors import MultiSampleVCF

            return MultiSampleVCF(vcf_file, lazy=vcf_lazy)
        elif variant_fetcher is not None:
            assert isinstance(variant_fetcher, VariantFetcher), "Wrong type of variant fetcher: %s" % type(
                variant_fetcher
            )
            return variant_fetcher
        elif variants is not None:
            return VariantListFetcher(variants)
        else:
            raise ValueError("No source of variants was specified!")

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

    def __iter__(self):
        raise NotImplementedError()


class SingleVariantMatcher(BaseVariantMatcher):
    """
    Match each variant with each interval it overlaps.

    Iterating yields (Interval, Variant) pairs. `pairs()` returns all pairs
    as one polars DataFrame.
    """

    def pairs(self) -> pl.DataFrame:
        """Return all interval-variant pairs from one overlap join.

        This reads all variants of the source into one polars DataFrame,
        but keeps no Variant objects.

        Returns:
          The pairs as described in `overlap_variants`. variant_idx counts
          the variants in the order of the variant source.
        """
        pl, _ = _import_polars()
        batches = [variants_to_polars(b) for b in self.variant_fetcher.batch_iter(self.variant_batch_size)]
        variants = pl.concat(batches) if batches else variants_to_polars([])
        return overlap_variants(self._interval_frame, variants)

    def iter_batches(self) -> Iterator[Tuple[List[Variant], pl.DataFrame]]:
        """Yield each batch of variants together with its pairs.

        Each batch holds up to `variant_batch_size` variants and costs one
        overlap join. In the pairs, variant_idx indexes into the batch.
        """
        for batch in self.variant_fetcher.batch_iter(self.variant_batch_size):
            batch = list(batch)
            yield batch, overlap_variants(self._interval_frame, variants_to_polars(batch))

    def __iter__(self) -> Iterator[Tuple[Interval, Variant]]:
        """
        Yield (Interval, Variant) for each overlapping pair.

        The pairs come in the order of the variant source, and per variant
        in the order of the intervals.
        """
        for batch, pairs in self.iter_batches():
            intervals = self._intervals_of(pairs)
            for interval, variant_idx in zip(intervals, pairs.get_column("variant_idx")):
                yield interval, batch[variant_idx]


class MultiVariantsMatcher(BaseVariantMatcher):
    """
    Match each interval with all variants that overlap it.

    Iterating yields (Interval, variants) pairs in the order of the intervals.
    The variants come from `VariantFetcher.fetch_variants`.
    """

    def __iter__(self) -> Iterator[Tuple[Interval, Iterator[Variant]]]:
        intervals = self._intervals if self._intervals is not None else self._intervals_of(self._interval_frame)
        for i in intervals:
            yield i, self.variant_fetcher.fetch_variants(i)
