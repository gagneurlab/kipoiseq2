"""Match variants with the intervals they overlap.

:func:`scan_variant_pairs` finds the interval-variant pairs with one polars-bio
overlap join. The variants come from a polars DataFrame or LazyFrame, e.g.
from :func:`~kipoiseq2.extractors.vcf.scan_vcf_variants`. Intervals and
variants use 0-based, half-open coordinates: a variant overlaps an interval if
``variant.start < interval.end`` and ``interval.start < variant.end``, with
``variant.start = pos - 1`` and ``variant.end = variant.start + len(ref)``.

It needs the ``ranges`` extra (polars and polars-bio).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

from kipoiseq2.extractors._polars import _import_polars

if TYPE_CHECKING:
    import polars as pl

__all__ = ["scan_variant_pairs"]


def _read_intervals(intervals: pl.DataFrame) -> pl.DataFrame:
    """Return the intervals with the columns chrom, start, end, strand and the other columns.

    :raises ValueError: if ``intervals`` lacks chrom, start or end, or has a column
        named interval_idx or starting with ``variant_``
    """
    import polars as pl

    missing = [c for c in ("chrom", "start", "end") if c not in intervals.columns]
    if missing:
        raise ValueError("`intervals` lacks the columns {}".format(missing))
    reserved = [c for c in intervals.columns if c == "interval_idx" or c.startswith("variant_")]
    if reserved:
        raise ValueError(
            f"`intervals` has the columns {reserved}, "
            "but the pairs use the name interval_idx and the prefix variant_: rename these columns"
        )
    # a null strand is unknown, as a missing strand column
    strand = pl.col("strand").cast(pl.String).fill_null(".") if "strand" in intervals.columns else pl.lit(".")
    return intervals.select(
        pl.col("chrom").cast(pl.String),
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
        strand.alias("strand"),
        pl.exclude("chrom", "start", "end", "strand"),
    )


def _read_variants(variants: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
    """Return the variants as a LazyFrame with chrom, start, end, pos, ref, alt and the other columns.

    :raises ValueError: if ``variants`` lacks chrom, pos, ref or alt, or has a column named idx or variant_idx
    """
    import polars as pl

    columns = variants.collect_schema().names()
    missing = [c for c in ("chrom", "pos", "ref", "alt") if c not in columns]
    if missing:
        raise ValueError("`variants` lacks the columns {}".format(missing))
    reserved = [c for c in columns if c in ("idx", "variant_idx")]
    if reserved:
        raise ValueError(
            f"`variants` has the columns {reserved}, but the pairs use the prefix variant_ for the variant columns "
            "and the name variant_idx for the row number of the variant: rename these columns"
        )
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


def scan_variant_pairs(intervals: pl.DataFrame, variants: Union[pl.DataFrame, pl.LazyFrame]) -> pl.LazyFrame:
    """Find all interval-variant pairs that overlap, with one polars-bio overlap join.

    polars-bio holds ``intervals`` in memory and streams ``variants`` through
    the join. So for a large VCF, write the pairs with ``sink_parquet``, or
    process them with ``collect_batches``. The pairs come in no fixed order:
    ``.collect().sort("variant_idx", "interval_idx")`` sorts them by variant,
    and per variant by interval.

    :param intervals: DataFrame with the columns chrom, start and end (0-based,
        half-open), an optional strand column and any other columns. A null
        strand gives the strand ``"."``, the same as a missing strand column.
    :param variants: DataFrame or LazyFrame with the columns chrom, pos
        (1-based), ref and alt, with one ALT allele per row, e.g.
        ``scan_vcf_variants(path)``. Its other columns pass through to the pairs.
    :returns: LazyFrame with one row per overlapping pair. The columns are:

        - chrom, start, end, strand and the other columns of ``intervals``, under their own names
        - interval_idx: row number of the interval in ``intervals``
        - variant_start (``pos - 1``), variant_end (``start + len(ref)``), variant_pos,
          variant_ref, variant_alt and the other columns of ``variants`` except chrom,
          with the prefix ``variant_``, e.g. variant_id for ``scan_vcf_variants``
        - variant_idx: row number of the variant in ``variants``
    :raises TypeError: if ``intervals`` is not a polars DataFrame, or if
        ``variants`` is not a polars DataFrame or LazyFrame
    :raises ValueError: if a frame lacks a required column, if ``intervals``
        has a column named interval_idx or starting with ``variant_``, or if
        ``variants`` has a column named idx or variant_idx. The pairs use
        these names for the row numbers and for the variant columns.

    Example:

    >>> import polars as pl
    >>> intervals = pl.DataFrame({"chrom": ["chr1"], "start": [10], "end": [20], "gene_id": ["g1"]})
    >>> variants = pl.DataFrame({"chrom": ["chr1", "chr1"], "pos": [15, 30], "ref": ["A", "C"], "alt": ["T", "G"]})
    >>> pairs = scan_variant_pairs(intervals, variants).collect()
    >>> pairs.select("gene_id", "interval_idx", "variant_pos", "variant_alt", "variant_idx").rows()
    [('g1', 0, 15, 'T', 0)]
    """
    pl, pb = _import_polars()
    if not isinstance(intervals, pl.DataFrame):
        raise TypeError(f"intervals must be a polars DataFrame, not {type(intervals).__name__}")
    if isinstance(variants, (str, os.PathLike)):
        raise TypeError("variants must be a polars DataFrame or LazyFrame, not a path: pass scan_vcf_variants(path)")
    if not isinstance(variants, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(f"variants must be a polars DataFrame or LazyFrame, not {type(variants).__name__}")
    intervals = _read_intervals(intervals)
    variants = _read_variants(variants)

    # polars-bio indexes the second input (df2) in memory and streams the first input (df1).
    # So the variants go into df1, and a large VCF does not need an index in memory.
    df1 = variants.with_row_index("variant_idx")
    df2 = intervals.with_row_index("interval_idx")
    # polars-bio reads the coordinate system from per-frame metadata.
    # pb.set_option would change the default of the whole process instead.
    # polars-bio registers the config_meta namespace at import, so mypy does not know it.
    df1.config_meta.set(coordinate_system_zero_based=True)  # type: ignore[attr-defined]
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
