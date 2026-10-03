from typing import Iterable, Iterator, List, Union

import polars as pl
import pytest
from conftest import vcf_file

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors.vcf import MultiSampleVCF
from kipoiseq2.extractors.vcf_matching import (
    BaseVariantMatcher,
    MultiVariantsMatcher,
    SingleVariantMatcher,
    VariantFetcher,
    VariantListFetcher,
    intervals_to_polars,
    overlap_variants,
    variants_to_polars,
)

intervals = [
    Interval("chr1", 1, 10, strand="+"),
    Interval("chr1", 23, 30, strand="-"),
    Interval("chr10", 1, 30, strand="+"),
]

variants = [Variant("chr1", 4, "T", "C"), Variant("chr1", 5, "A", "GA"), Variant("chr1", 25, "AACG", "GA")]

interval_frame = pl.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr1", "chr10"],
        "start": [1, 23, 5, 1],
        "end": [10, 30, 50, 30],
        "strand": ["+", "-", ".", "+"],
        "gene_id": ["g1", "g2", "g3", "g4"],
    }
)


class VariantFetcherProxy(VariantFetcher):
    def __init__(self, variant_fetcher: VariantFetcher):
        self.variant_fetcher = variant_fetcher

    def fetch_variants(self, interval: Union[Interval, Iterable[Interval]]) -> Iterator[Variant]:
        yield from self.variant_fetcher.fetch_variants(interval)

    def batch_iter(self, batch_size=10000) -> Iterator[List[Variant]]:
        yield from self.variant_fetcher.batch_iter(batch_size)

    def __iter__(self) -> Iterator[Variant]:
        yield from self.variant_fetcher


# make sure that kipoiseq2 only uses the VariantFetcher API
read_variants_fn = BaseVariantMatcher._read_variants


@staticmethod
def proxy_fn(*args, **kwargs):
    vf = VariantFetcherProxy(read_variants_fn(*args, **kwargs))
    return vf


BaseVariantMatcher._read_variants = proxy_fn


def test_variants_to_polars():
    vcf = MultiSampleVCF(vcf_file)
    vcf_variants = list(vcf)
    df = variants_to_polars(vcf_variants)
    assert df.height == len(vcf_variants)
    assert df.row(0, named=True) == {"chrom": "chr1", "start": 3, "end": 4, "pos": 4, "ref": "T", "alt": "C"}
    # the deletion AACG>GA at POS 25 covers the 0-based bases 24 to 27
    assert df.row(2, named=True)["start"] == 24
    assert df.row(2, named=True)["end"] == 28


def test_intervals_to_polars():
    df = intervals_to_polars(intervals)
    assert df.columns == ["chrom", "start", "end", "strand"]
    assert df["chrom"].to_list() == ["chr1", "chr1", "chr10"]
    assert df["start"].to_list() == [1, 23, 1]
    assert df["end"].to_list() == [10, 30, 30]
    assert df["strand"].to_list() == ["+", "-", "+"]


def test_overlap_variants():
    pairs = overlap_variants(interval_frame, variants_to_polars(variants))
    assert pairs.columns == [
        "chrom",
        "start",
        "end",
        "strand",
        "gene_id",
        "interval_idx",
        "variant_start",
        "variant_end",
        "variant_pos",
        "variant_ref",
        "variant_alt",
        "variant_idx",
    ]
    # sorted by variant, then by interval
    assert pairs.select("variant_idx", "interval_idx").rows() == [(0, 0), (1, 0), (2, 1), (2, 2)]
    assert pairs.row(3, named=True) == {
        "chrom": "chr1",
        "start": 5,
        "end": 50,
        "strand": ".",
        "gene_id": "g3",
        "interval_idx": 2,
        "variant_start": 24,
        "variant_end": 28,
        "variant_pos": 25,
        "variant_ref": "AACG",
        "variant_alt": "GA",
        "variant_idx": 2,
    }


def test_overlap_variants_empty():
    pairs = overlap_variants(interval_frame, variants_to_polars([]))
    assert pairs.height == 0
    assert "variant_idx" in pairs.columns
    assert overlap_variants(interval_frame.clear(), variants_to_polars(variants)).height == 0


def test_BaseVariantMatcher__read_intervals():
    with pytest.raises(ValueError):
        BaseVariantMatcher._read_intervals(intervals=intervals, interval_attrs=["gene_id"])

    with pytest.raises(ValueError, match="gene_name"):
        BaseVariantMatcher._read_intervals(intervals=interval_frame, interval_attrs=["gene_name"])

    df = BaseVariantMatcher._read_intervals(intervals=interval_frame)
    assert df.columns == ["chrom", "start", "end", "strand"]
    assert df["start"].to_list() == [1, 23, 5, 1]

    df = BaseVariantMatcher._read_intervals(intervals=interval_frame.drop("strand"), interval_attrs=["gene_id"])
    assert df.columns == ["chrom", "start", "end", "strand", "gene_id"]
    assert df["strand"].to_list() == ["."] * 4

    df = BaseVariantMatcher._read_intervals(intervals=intervals)
    assert df["chrom"].to_list() == ["chr1", "chr1", "chr10"]
    assert df["strand"].to_list() == ["+", "-", "+"]


def test_BaseVariantMatcher_intervals_input():
    from_frame = SingleVariantMatcher(variants=variants, intervals=interval_frame)
    assert from_frame._intervals is None
    assert from_frame._interval_frame.height == interval_frame.height

    from_objects = SingleVariantMatcher(variants=variants, intervals=intervals)
    assert from_objects._intervals == intervals
    assert from_objects._interval_frame["chrom"].to_list() == ["chr1", "chr1", "chr10"]

    with pytest.raises(TypeError):
        SingleVariantMatcher(variants=variants)


def test_SingleVariantMatcher__iter__():
    inters = intervals + [Interval("chr1", 5, 50)]
    expected = [
        (inters[0], variants[0]),
        (inters[0], variants[1]),
        (inters[1], variants[2]),
        (inters[3], variants[2]),
    ]

    assert list(SingleVariantMatcher(vcf_file, intervals=interval_frame)) == expected
    assert list(SingleVariantMatcher(variants=variants, intervals=interval_frame)) == expected
    assert (
        list(SingleVariantMatcher(variant_fetcher=VariantListFetcher(variants), intervals=interval_frame)) == expected
    )
    assert list(SingleVariantMatcher(vcf_file, intervals=inters)) == expected
    # one variant per overlap join gives the same pairs
    assert list(SingleVariantMatcher(vcf_file, intervals=interval_frame, variant_batch_size=1)) == expected


def test_SingleVariantMatcher_generator_of_intervals():
    inters = intervals + [Interval("chr1", 5, 50)]
    expected = [
        (inters[0], variants[0]),
        (inters[0], variants[1]),
        (inters[1], variants[2]),
        (inters[3], variants[2]),
    ]
    assert list(SingleVariantMatcher(variants=variants, intervals=(i for i in inters))) == expected


def test_SingleVariantMatcher_interval_attrs():
    pairs = list(SingleVariantMatcher(vcf_file, intervals=interval_frame, interval_attrs=["gene_id"]))
    assert [i.attrs for i, _ in pairs] == [{"gene_id": "g1"}, {"gene_id": "g1"}, {"gene_id": "g2"}, {"gene_id": "g3"}]


def test_SingleVariantMatcher_yields_given_intervals():
    named = [Interval("chr1", 1, 10, name="a", attrs={"x": 1})]
    ((interval, variant), *_) = list(SingleVariantMatcher(vcf_file, intervals=named))
    assert interval is named[0]
    assert variant.source is not None  # the cyvcf2 record, e.g. for genotypes


def test_SingleVariantMatcher_pairs():
    matcher = SingleVariantMatcher(vcf_file, intervals=interval_frame, interval_attrs=["gene_id"], variant_batch_size=2)
    pairs = matcher.pairs()
    assert pairs.select("gene_id", "variant_pos", "variant_ref", "variant_alt").rows() == [
        ("g1", 4, "T", "C"),
        ("g1", 5, "A", "GA"),
        ("g2", 25, "AACG", "GA"),
        ("g3", 25, "AACG", "GA"),
    ]
    # variant_idx counts over all batches
    assert pairs["variant_idx"].to_list() == [0, 1, 2, 2]


def test_SingleVariantMatcher_iter_batches():
    matcher = SingleVariantMatcher(variants=variants, intervals=interval_frame, variant_batch_size=2)
    batches = list(matcher.iter_batches())
    assert [len(b) for b, _ in batches] == [2, 1]
    assert [p["variant_idx"].to_list() for _, p in batches] == [[0, 1], [0, 0]]


def test_MultiVariantMatcher__iter__():
    matcher = MultiVariantsMatcher(vcf_file, intervals=intervals)
    pairs = list(matcher)

    assert pairs[0][0] == intervals[0]
    assert list(pairs[0][1]) == [variants[0], variants[1]]
    assert pairs[1][0] == intervals[1]
    assert list(pairs[1][1]) == [variants[2]]

    matcher = MultiVariantsMatcher(vcf_file, intervals=interval_frame)
    pairs = list(matcher)

    assert pairs[0][0] == intervals[0]
    assert list(pairs[0][1]) == [variants[0], variants[1]]
    assert pairs[1][0] == intervals[1]
    assert list(pairs[1][1]) == [variants[2]]

    matcher = MultiVariantsMatcher(variants=variants, intervals=interval_frame)
    pairs = list(matcher)

    assert pairs[0][0] == intervals[0]
    assert list(pairs[0][1]) == [variants[0], variants[1]]
    assert pairs[1][0] == intervals[1]
    assert list(pairs[1][1]) == [variants[2]]
    assert list(pairs[2][1]) == [variants[2]]


# Interval chr1:[10, 20) is 0-based, half-open: it covers the 1-based positions 11 to 20.
# The expected values equal the output of the pyranges-based matcher of kipoiseq 0.7.1.
BOUNDARY_INTERVAL = Interval("chr1", 10, 20, strand="+")
BOUNDARY_CASES = [
    ("snv_before_first_base", Variant("chr1", 10, "A", "C"), False),
    ("snv_first_base", Variant("chr1", 11, "A", "C"), True),
    ("snv_last_base", Variant("chr1", 20, "A", "C"), True),
    ("snv_after_last_base", Variant("chr1", 21, "A", "C"), False),
    ("deletion_ending_before_start", Variant("chr1", 8, "AAA", "A"), False),
    ("deletion_overlapping_start", Variant("chr1", 8, "AAAA", "A"), True),
    ("deletion_overlapping_end", Variant("chr1", 19, "AAAA", "A"), True),
    ("deletion_starting_after_end", Variant("chr1", 21, "AA", "A"), False),
    ("deletion_spanning_interval", Variant("chr1", 5, "A" * 20, "A"), True),
    ("insertion_anchor_before_start", Variant("chr1", 10, "A", "AT"), False),
    ("insertion_anchor_first_base", Variant("chr1", 11, "A", "AT"), True),
    ("insertion_anchor_last_base", Variant("chr1", 20, "A", "AT"), True),
    ("insertion_anchor_after_end", Variant("chr1", 21, "A", "AT"), False),
    # an empty REF gives a zero-length variant, which overlaps only strictly inside
    ("empty_ref_at_start", Variant("chr1", 11, "", "T"), False),
    ("empty_ref_inside", Variant("chr1", 15, "", "T"), True),
    ("empty_ref_at_end", Variant("chr1", 21, "", "T"), False),
    ("other_chrom", Variant("chr2", 15, "A", "C"), False),
]


@pytest.mark.parametrize("variant, overlaps", [c[1:] for c in BOUNDARY_CASES], ids=[c[0] for c in BOUNDARY_CASES])
def test_SingleVariantMatcher_boundaries(variant, overlaps):
    pairs = list(SingleVariantMatcher(variants=[variant], intervals=[BOUNDARY_INTERVAL]))
    assert pairs == ([(BOUNDARY_INTERVAL, variant)] if overlaps else [])
    # the list fetcher of MultiVariantsMatcher uses the same rule
    ((_, fetched),) = list(MultiVariantsMatcher(variants=[variant], intervals=[BOUNDARY_INTERVAL]))
    assert list(fetched) == ([variant] if overlaps else [])


def test_SingleVariantMatcher_boundaries_in_one_batch():
    all_variants = [v for _, v, _ in BOUNDARY_CASES]
    pairs = list(SingleVariantMatcher(variants=all_variants, intervals=[BOUNDARY_INTERVAL]))
    assert [v for _, v in pairs] == [v for _, v, overlaps in BOUNDARY_CASES if overlaps]
