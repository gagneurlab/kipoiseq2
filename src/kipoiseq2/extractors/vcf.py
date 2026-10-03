"""Read VCF files.

`scan_vcf_variants` reads a VCF file with polars-bio as a polars LazyFrame,
with one row per ALT allele. It needs the `ranges` extra (polars and
polars-bio).

`MultiSampleVCF` reads a VCF file with cyvcf2 and yields Variant objects.
It needs the `vcf` extra.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Union

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors.vcf_matching import _import_polars
from kipoiseq2.extractors.vcf_query import VariantIntervalQueryable
from kipoiseq2.utils import batch_iter
from kipoiseq2.variant_source import VariantFetcher

if TYPE_CHECKING:
    import polars as pl

try:
    from cyvcf2 import VCF
except ImportError:
    VCF = object

__all__ = ["scan_vcf_variants", "MultiSampleVCF"]

# the columns of polars_bio.scan_vcf without INFO and FORMAT fields
_SCAN_VCF_COLUMNS = ("chrom", "start", "end", "id", "ref", "alt", "qual", "filter")


def scan_vcf_variants(path: str, **scan_vcf_kwargs) -> pl.LazyFrame:
    """Read the variants of a VCF file as a polars LazyFrame, with one row per ALT allele.

    The file is read lazily with `polars_bio.scan_vcf`. A record with
    several ALT alleles gives one row per ALT allele. ALT alleles that
    contain `N` or `*` are dropped, as in `MultiSampleVCF`. An ALT of `.`
    gives an empty alt.

    Args:
      path: path of the VCF file.
      **scan_vcf_kwargs: keyword arguments of `polars_bio.scan_vcf`, e.g.
        `info_fields=["AF"]`. They override the defaults
        `use_zero_based=True, info_fields=[], format_fields=[]`.

    Returns:
      LazyFrame with the columns chrom, start (`pos - 1`), end
      (`start + len(ref)`), pos (the 1-based VCF POS), ref, alt, allele_idx
      (the 1-based index of the ALT allele in the record) and the requested
      INFO and FORMAT fields. start and end ignore INFO/END. A field with
      one value per ALT allele (Number=A) keeps all values of the record,
      so select the value of the row with
      `pl.col("AF").list.get(pl.col("allele_idx") - 1)`.
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
        .select("chrom", "start", "end", "pos", "ref", "alt", "allele_idx", *fields)
    )


class MultiSampleVCF(VariantFetcher, VCF):
    def __init__(self, *args, **kwargs):
        try:
            from cyvcf2 import VCF
        except ImportError as e:
            raise ImportError("MultiSampleVCF needs the `vcf` extra: pip install 'kipoiseq2[vcf]'") from e

        VCF.__init__(self, *args, **kwargs, strict_gt=True)
        self.sample_mapping = dict(zip(self.samples, range(len(self.samples))))

    def fetch_variants(self, interval, sample_id=None):
        for cy_variant in self(self._region(interval)):
            for variant in self._variants_from_cyvcf2(cy_variant):
                if sample_id is None or self.has_variant(variant, sample_id):
                    yield variant

    def _variants_from_cyvcf2(self, cy_variant):
        # in case deletion is present
        ALTs = cy_variant.ALT or [""]
        # single REF can have multiple ALT
        for alt in ALTs:
            v = Variant.from_cyvcf_and_given_alt(cy_variant, alt)
            if "N" in alt or "*" in alt:
                logging.warning("Undefined variant %s are not supported: Skip" % str(v))
                continue
            yield v

    @staticmethod
    def _region(interval: Interval):
        return "%s:%d-%d" % (interval.chrom, interval.start + 1, interval.end)

    def has_variant(self, variant: Variant, sample_id: str) -> bool:
        gt_type = variant.source.gt_types[self.sample_mapping[sample_id]]
        return self._has_variant_gt(gt_type)

    @staticmethod
    def _has_variant_gt(gt_type: int) -> bool:
        return gt_type != 0 and gt_type != 2

    def __next__(self):
        return Variant.from_cyvcf(super().__next__())

    def __iter__(self) -> Iterator[Variant]:
        while True:
            try:
                cy_variant = super().__next__()
            except StopIteration:
                break
            yield from self._variants_from_cyvcf2(cy_variant)

    def batch_iter(self, batch_size=10000) -> Iterator[Iterable[Variant]]:
        """Iterates variants in vcf file.

        # Arguments
            vcf_file: path of vcf file.
            batch_size: size of each batch.
        """
        variants = iter(self)
        yield from batch_iter(variants, batch_size=batch_size)

    def query_variants(self, intervals: List[Interval], sample_id=None) -> VariantIntervalQueryable:
        """Fetch variants for given multi-intervals from vcf file
        for sample if sample id is given.

        # Arguments
            intervals (List[Interval]): list of Interval objects
            sample_id (str, optional): sample id in vcf file.

        # Returns
          VCFQueryable: queryable object whihc allow you to query the
            fetched variatns.

        # Example
            To fetch variants if quality more than 10 and there is
            a variant in interval
            ```
              >>> MultiSampleVCF(vcf_path) \
                    .query_variants(intervals) \
                    .filter(lambda variant: variant.qual > 10) \
                    .filter_range(NumberVariantQuery(max_num=1)) \
                    .to_vcf(output_path)
            ```
        """
        pairs = [(self.fetch_variants(i, sample_id=sample_id), i) for i in intervals]
        return VariantIntervalQueryable(self, pairs)

    def query_all(self) -> VariantIntervalQueryable:
        """Convert all variants into queryable object without interval so
        interval filters will not work.

        # Example
            To fetch variants if quality more than 10
            ```
              >>> MultiSampleVCF(vcf_path) \
                    .query_all() \
                    .filter(lambda variant: variant.qual > 10) \
                    .to_vcf(output_path)
            ```
        """
        pairs = [(iter(self), None)]
        return VariantIntervalQueryable(self, pairs)

    def get_variant(self, variant: Union[Variant, str]) -> Variant:
        """Returns variant from vcf file. Lets you use vcf file as dict.

        # Arguments:
            variant: variant object or variant id as string.

        # Returns
            Variant object.

        # Example
            ```python
              >>> MultiSampleVCF(vcf_path).get_variant("chr1:4:T:['C']")
            ```
        """
        if isinstance(variant, str):
            variant = Variant.from_str(variant)

        variants = self.fetch_variants(Interval(variant.chrom, variant.pos - 1, variant.pos))
        for v in variants:
            if v.ref == variant.ref and v.alt == variant.alt:
                return v
        raise KeyError("Variant %s not found in vcf file." % str(variant))

    def get_variants(
        self, variants: Iterable[Union[str, Variant]], regions=None, variant_gap=150
    ) -> List[Optional[Variant]]:
        """Returns list of variants from vcf file. Lets you use vcf file as dict.

        # Arguments:
            variants: list of variants
            regions: list of regions to seek for variants.
              Automatically generated from variants if not given.
            variant_gap: only used if `regions` is not given. Variants on
              the same chromosome closer than this many bases share one
              generated region.

        # Returns
           List of variants
        """
        parsed_variants = [Variant.from_str(v) if isinstance(v, str) else v for v in variants]
        regions = regions or self._regions_from_variants(parsed_variants, variant_gap=variant_gap)
        variant_map = dict()

        for r in regions:
            r_variants = self.fetch_variants(r)
            for v in r_variants:
                variant_map[v] = v

        return [variant_map.get(v) for v in parsed_variants]

    def _regions_from_variants(self, variants: List[Variant], variant_gap=150):
        regions = list()

        for chrom, vs in self._group_variants_by_chrom(variants).items():
            starts = sorted(v.pos for v in vs)

            start_i = starts[0]
            prev_i = starts[0]
            for i in starts[1:]:
                if prev_i + variant_gap < i:
                    regions.append(Interval(chrom, start_i - 1, prev_i))
                    start_i = i
                prev_i = i

            regions.append(Interval(chrom, start_i - 1, prev_i))

        return regions

    def _group_variants_by_chrom(self, variants: List[Variant]):
        chroms = defaultdict(set)

        for v in variants:
            chroms[v.chrom].add(v)

        return dict(chroms)

    def get_samples(self, variant: Variant) -> Dict[str, int]:
        """Fetchs sample names which have given variants

        # Arguments
            variant: variant object.

        # Returns
            Dict[str, int]: Dict of sample which have variant and gt as value.
        """
        return dict(filter(lambda x: self._has_variant_gt(x[1]), zip(self.samples, variant.source.gt_types)))
