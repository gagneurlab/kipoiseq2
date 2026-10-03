"""Read VCF files as polars tables.

`scan_vcf_variants` reads a VCF file with polars-bio as a polars LazyFrame,
with one row per ALT allele. `scan_vcf_genotypes` adds one row per sample,
with the FORMAT fields of the sample and whether it carries the ALT allele.
Both need the `ranges` extra (polars and polars-bio).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional, Sequence

from kipoiseq2.extractors.vcf_matching import _import_polars

if TYPE_CHECKING:
    import polars as pl

__all__ = ["scan_vcf_variants", "scan_vcf_genotypes"]

# the columns of polars_bio.scan_vcf without INFO and FORMAT fields
_SCAN_VCF_COLUMNS = ("chrom", "start", "end", "id", "ref", "alt", "qual", "filter")


def scan_vcf_variants(path: str, **scan_vcf_kwargs) -> pl.LazyFrame:
    """Read the variants of a VCF file as a polars LazyFrame, with one row per ALT allele.

    The file is read lazily with `polars_bio.scan_vcf`. A record with
    several ALT alleles gives one row per ALT allele. ALT alleles that
    contain `N` or `*` are dropped, as in kipoiseq's `MultiSampleVCF`. An
    ALT of `.` gives an empty alt.

    Args:
      path: path of the VCF file.
      **scan_vcf_kwargs: keyword arguments of `polars_bio.scan_vcf`, e.g.
        `info_fields=["AF"]`. They override the defaults
        `use_zero_based=True, info_fields=[], format_fields=[]`.

    Returns:
      LazyFrame with the columns chrom, start (`pos - 1`), end
      (`start + len(ref)`), pos (the 1-based VCF POS), id, ref, alt,
      allele_idx (the 1-based index of the ALT allele in the record), qual,
      filter and the requested INFO and FORMAT fields. start and end ignore
      INFO/END. polars-bio gives a missing ID or FILTER (`.`) as an empty
      string and a missing QUAL as null. Several FILTER values stay joined
      with `;`, e.g. "q10;s50". A field with one value per ALT allele
      (Number=A) keeps all values of the record, so select the value of the
      row with `pl.col("AF").list.get(pl.col("allele_idx") - 1)`.
    """
    pl, pb = _import_polars()
    scan = pb.scan_vcf(str(path), **{"use_zero_based": True, "info_fields": [], "format_fields": [], **scan_vcf_kwargs})
    fields = [c for c in scan.collect_schema().names() if c not in _SCAN_VCF_COLUMNS]
    # with use_zero_based=False, start is the 1-based POS
    zero_based = scan.config_meta.get_metadata()["coordinate_system_zero_based"]  # type: ignore[attr-defined]
    # polars-bio joins the ALT alleles of a record with "|"
    alts = pl.col("alt").str.split("|")
    start = pl.col("pos") - 1
    return (
        scan.with_columns(
            pos=pl.col("start").cast(pl.Int64) + int(zero_based),
            alt=alts,
            allele_idx=pl.int_ranges(1, alts.list.len() + 1),
        )
        .explode("alt", "allele_idx", empty_as_null=False)
        .filter(~pl.col("alt").str.contains("[N*]"))
        .with_columns(start=start, end=start + pl.col("ref").str.len_chars())
        .select("chrom", "start", "end", "pos", "id", "ref", "alt", "allele_idx", "qual", "filter", *fields)
    )


def scan_vcf_genotypes(
    path: str,
    samples: Optional[Sequence[str]] = None,
    format_fields: Sequence[str] = ("GT",),
    carriers_only: bool = True,
    **scan_vcf_kwargs,
) -> pl.LazyFrame:
    """Read the genotypes of a VCF file as a polars LazyFrame, with one row per ALT allele and sample.

    A sample carries an ALT allele if its GT contains the allele_idx of the
    row. GT may be haploid or phased, and missing alleles (`.`) are ignored.
    So a sample with GT 0/2 carries the second ALT allele but not the first
    one, and a sample with GT ./1 carries the first ALT allele.

    Args:
      path: path of the VCF file.
      samples: names of the samples to read. The default reads all samples.
      format_fields: FORMAT fields to read. GT is always read.
      carriers_only: keep only the rows where the sample carries the ALT allele.
      **scan_vcf_kwargs: more keyword arguments of `polars_bio.scan_vcf`,
        as in `scan_vcf_variants`.

    Returns:
      LazyFrame with the columns of `scan_vcf_variants`, then sample, GT,
      the other FORMAT fields with the value of the sample, and carrier.
      FORMAT fields with several values per sample, such as AD, are lists.
    """
    pl, _ = _import_polars()
    fields = ["GT", *(f for f in format_fields if f != "GT")]
    variants = scan_vcf_variants(
        path, samples=None if samples is None else list(samples), format_fields=fields, **scan_vcf_kwargs
    )
    sample_names = json.loads(variants.config_meta.get_metadata()["source_header"])["sample_names"]  # type: ignore[attr-defined]
    if not sample_names:
        raise ValueError("{} has no samples, or none of the requested ones".format(path))
    columns = variants.collect_schema().names()
    if "genotypes" in columns:
        # for a multi-sample VCF, also with samples=[one], polars-bio returns the FORMAT fields
        # as a struct of lists, with one list item per sample
        columns.remove("genotypes")
        genotypes = (
            variants.with_columns(sample=pl.lit(sample_names, dtype=pl.List(pl.String)))
            .unnest("genotypes")
            .explode("sample", *fields, empty_as_null=False)
        )
    else:
        # for a single-sample VCF, polars-bio returns the FORMAT fields as columns
        columns = [c for c in columns if c not in fields]
        genotypes = variants.with_columns(sample=pl.lit(sample_names[0]))
    alleles = pl.col("GT").str.extract_all(r"\d+")
    carrier = alleles.list.contains(pl.col("allele_idx").cast(pl.String)).fill_null(False)
    genotypes = genotypes.select(*columns, "sample", *fields, carrier.alias("carrier"))
    return genotypes.filter("carrier") if carriers_only else genotypes
