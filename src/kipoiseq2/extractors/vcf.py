"""Read VCF files as polars tables.

:func:`scan_vcf_variants` reads a VCF file with polars-bio as a polars LazyFrame,
with one row per ALT allele. :func:`scan_vcf_genotypes` adds one row per sample,
or per haplotype of a sample, with the FORMAT fields of the sample and whether
it carries the ALT allele.
Both need the ``ranges`` extra (polars and polars-bio).
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional, Sequence

from kipoiseq2.extractors._polars import _import_polars

if TYPE_CHECKING:
    import polars as pl
    from polars import LazyFrame

__all__ = ["scan_vcf_variants", "scan_vcf_genotypes"]

# the columns of polars_bio.scan_vcf without INFO and FORMAT fields
_SCAN_VCF_COLUMNS = ("chrom", "start", "end", "id", "ref", "alt", "qual", "filter")

_SEQUENCE_ALT = "(?i)^[ACGTN]+$"
"""Regex of the ALT alleles that are sequences of bases: A, C, G, T and N, in any case."""


def _check_names(names: object, name: str) -> None:
    """Check that ``names`` is a sequence of strings, but not a str.

    :param names: the value of the argument ``name``
    :param name: the name of the argument in the error message, e.g. ``"info_fields"``
    :raises TypeError: if ``names`` is a str or not a sequence, or if it contains an element that is not a str
    """
    if isinstance(names, str):
        raise TypeError(f"{name} must be a sequence of strings, not a str: pass [{names!r}]")
    if not isinstance(names, Sequence):
        raise TypeError(f"{name} must be a sequence of strings, not {type(names).__name__}")
    for n in names:
        if not isinstance(n, str):
            raise TypeError(f"{name} must be a sequence of strings, but contains {n!r}")


def _scan_vcf(
    path: str | os.PathLike[str],
    *,
    info_fields: Sequence[str],
    format_fields: Sequence[str],
    samples: Optional[Sequence[str]],
    sequence_alts_only: bool,
) -> pl.LazyFrame:
    """Return the rows of :func:`scan_vcf_variants`, with the FORMAT fields of ``polars_bio.scan_vcf``."""
    _check_names(info_fields, "info_fields")
    pl, pb = _import_polars()
    # polars-bio has no type annotations. The local pl is a variable, so the annotation uses LazyFrame, not pl.LazyFrame.
    scan: LazyFrame = pb.scan_vcf(
        os.fspath(path),
        use_zero_based=True,
        info_fields=list(info_fields),
        format_fields=list(format_fields),
        samples=None if samples is None else list(samples),
    )
    fields = [c for c in scan.collect_schema().names() if c not in _SCAN_VCF_COLUMNS]
    # polars-bio joins the ALT alleles of a record with "|"
    alts = pl.col("alt").str.split("|")
    start = pl.col("start").cast(pl.Int64)
    variants = scan.with_columns(
        start=start,
        end=start + pl.col("ref").str.len_chars(),
        pos=start + 1,
        ref=pl.col("ref").str.to_uppercase(),
        alt=alts,
        allele_idx=pl.int_ranges(1, alts.list.len() + 1),
    ).explode("alt", "allele_idx", empty_as_null=False)
    # VCF bases are case-insensitive, but the encoders of kipoiseq2.encoders are case-sensitive, so ref and the
    # sequence ALT alleles become uppercase. The other ALT alleles keep their text, e.g. the chromosome of a breakend.
    alt = pl.col("alt")
    is_sequence = alt.str.contains(_SEQUENCE_ALT)
    if sequence_alts_only:
        variants = variants.filter(is_sequence).with_columns(alt=alt.str.to_uppercase())
    else:
        # polars-bio gives ALT "." as an empty string
        variants = variants.with_columns(
            alt=pl.when(is_sequence).then(alt.str.to_uppercase()).when(alt != "").then(alt)
        )
    variants = variants.select(
        "chrom", "start", "end", "pos", "id", "ref", "alt", "allele_idx", "qual", "filter", *fields
    )
    # polars-bio range operations read this metadata
    variants.config_meta.set(coordinate_system_zero_based=True)  # type: ignore[attr-defined]
    return variants


def scan_vcf_variants(
    path: str | os.PathLike[str],
    *,
    info_fields: Sequence[str] = (),
    sequence_alts_only: bool = True,
) -> pl.LazyFrame:
    """Read the variants of a VCF file as a polars LazyFrame, with one row per ALT allele.

    The file is read lazily with ``polars_bio.scan_vcf``. A record with
    several ALT alleles gives one row per ALT allele. By default, only the
    ALT alleles that are sequences of bases give a row. The VCF
    specification defines the bases as A, C, G, T and N, in any case. So
    these ALT alleles give no row: a missing ALT (``.``), the ``*`` allele,
    symbolic ALT alleles such as ``<DEL>``, and breakends. ref and alt are
    uppercase, so the case-sensitive encoders of :mod:`kipoiseq2.encoders`
    take them as they are. allele_idx counts the dropped ALT alleles too.

    :param path: path of the VCF file.
    :param info_fields: INFO fields to read, e.g. ``["AF"]``.
    :param sequence_alts_only: keep only the ALT alleles that are sequences of
        bases. With False, every ALT allele gives a row, and a missing ALT
        (``.``) gives a null alt. The ALT alleles that are not sequences of
        bases keep their text, e.g. ``<DEL>``. Use False only for tables: a
        :class:`~kipoiseq2.dataclasses.Variant` needs a str alt, and
        :class:`~kipoiseq2.extractors.vcf_seq.VariantSeqExtractor` inserts
        symbolic ALT alleles such as ``<DEL>`` as text.
    :returns: LazyFrame with the columns chrom, start (``pos - 1``), end
        (``start + len(ref)``), pos (the 1-based VCF POS), id, ref, alt,
        allele_idx (the 1-based index of the ALT allele in the record), qual,
        filter and the ``info_fields``. start and end ignore INFO/END.
        polars-bio gives a missing ID or FILTER (``.``) as an empty string and a
        missing QUAL as null. Several FILTER values stay joined with ``;``, e.g.
        "q10;s50". A field with one value per ALT allele (Number=A) keeps all
        values of the record, so select the value of the row with
        ``pl.col("AF").list.get(pl.col("allele_idx") - 1)``.
    :raises TypeError: if ``info_fields`` is not a sequence of strings, e.g. a list, or if it is a str.
    """
    return _scan_vcf(
        path, info_fields=info_fields, format_fields=(), samples=None, sequence_alts_only=sequence_alts_only
    )


def scan_vcf_genotypes(
    path: str | os.PathLike[str],
    *,
    samples: Optional[Sequence[str]] = None,
    format_fields: Sequence[str] = ("GT",),
    info_fields: Sequence[str] = (),
    sequence_alts_only: bool = True,
    carriers_only: bool = True,
    haplotypes: bool = False,
) -> pl.LazyFrame:
    """Read the genotypes of a VCF file as a polars LazyFrame, with one row per ALT allele and sample.

    A sample carries an ALT allele if its GT contains the allele_idx of the
    row. GT may be haploid or phased, and missing alleles (``.``) are ignored.
    So a sample with GT 0/2 carries the second ALT allele but not the first
    one, and a sample with GT ./1 carries the first ALT allele.
    The rows cover the same ALT alleles as :func:`scan_vcf_variants`. For
    the ALT alleles ``A,<DEL>,C``, a sample with GT 0/3 carries C, and a
    sample with GT 0/2 carries none of the kept ALT alleles.

    With ``haplotypes=True``, each allele of GT gives a row, with its
    position in GT as haplotype: 0 or 1 for a diploid GT. carrier then
    tells whether this haplotype carries the ALT allele. So GT 0|1 gives a
    row for haplotype 0 that does not carry the ALT allele, and a row for
    haplotype 1 that carries it. haplotype ignores the phasing, as
    ``bcftools consensus -H`` does: for an unphased GT such as 0/1, the
    order of the alleles in GT sets the haplotype.
    ``~pl.col("GT").str.contains("/")`` keeps the phased and haploid GTs.
    A missing GT gives one row with a null haplotype.

    :param path: path of the VCF file.
    :param samples: names of the samples to read. The default reads all samples.
    :param format_fields: FORMAT fields to read. GT is always read.
    :param info_fields: INFO fields to read, as in :func:`scan_vcf_variants`.
    :param sequence_alts_only: keep only the ALT alleles that are sequences of
        bases, as in :func:`scan_vcf_variants`. With False, a sample with GT
        0/2 for the ALT alleles ``A,<DEL>,C`` carries ``<DEL>``.
    :param carriers_only: keep only the rows where the sample carries the ALT allele.
        With ``haplotypes=True``, keep only the haplotypes that carry it.
    :param haplotypes: give one row per ALT allele, sample and haplotype, as described above.
    :returns: LazyFrame with the columns of :func:`scan_vcf_variants`, then sample,
        haplotype (only with ``haplotypes=True``), GT,
        the other FORMAT fields with the value of the sample, and carrier.
        FORMAT fields with several values per sample, such as AD, are lists.
    :raises TypeError: if ``samples``, ``format_fields`` or ``info_fields`` is not
        a sequence of strings, e.g. a list, or if it is a str.
    :raises ValueError: if the VCF file has no samples, or none of ``samples``.
    """
    if samples is not None:
        _check_names(samples, "samples")
    _check_names(format_fields, "format_fields")
    pl, _ = _import_polars()
    fields = ["GT", *(f for f in format_fields if f != "GT")]
    variants = _scan_vcf(
        path, info_fields=info_fields, format_fields=fields, samples=samples, sequence_alts_only=sequence_alts_only
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
    allele_idx = pl.col("allele_idx").cast(pl.String)
    if haplotypes:
        # the missing alleles (.) give a row too, so that haplotype is the position of the allele in GT
        alleles = pl.col("GT").str.extract_all(r"\d+|\.")
        # carrier holds the allele of the haplotype until the select compares it with allele_idx
        genotypes = genotypes.with_columns(carrier=alleles, haplotype=pl.int_ranges(alleles.list.len())).explode(
            "carrier", "haplotype", empty_as_null=False
        )
        carrier = pl.col("carrier") == allele_idx
        haplotype_columns = ["haplotype"]
    else:
        carrier = pl.col("GT").str.extract_all(r"\d+").list.contains(allele_idx)
        haplotype_columns = []
    genotypes = genotypes.select(
        *columns, "sample", *haplotype_columns, *fields, carrier.fill_null(False).alias("carrier")
    )
    return genotypes.filter("carrier") if carriers_only else genotypes
