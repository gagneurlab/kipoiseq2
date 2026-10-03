import polars as pl
import pytest
from conftest import test_with_multiple_variants, vcf_file

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors.vcf_matching import (
    BaseVariantMatcher,
    MultiVariantsMatcher,
    SingleVariantMatcher,
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


def test_variants_to_polars():
    df = variants_to_polars(variants)
    assert df.height == len(variants)
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
    assert from_frame._interval_objects is None
    assert from_frame.intervals.height == interval_frame.height

    from_objects = SingleVariantMatcher(variants=variants, intervals=intervals)
    assert from_objects._interval_objects == intervals
    assert from_objects.intervals["chrom"].to_list() == ["chr1", "chr1", "chr10"]

    with pytest.raises(TypeError):
        SingleVariantMatcher(variants=variants)


def test_BaseVariantMatcher_intervals_attribute():
    matcher = MultiVariantsMatcher(
        variants=variants, intervals=interval_frame.drop("strand"), interval_attrs=["gene_id"]
    )
    assert isinstance(matcher.intervals, pl.DataFrame)
    assert matcher.intervals.columns == ["chrom", "start", "end", "strand", "gene_id"]
    assert matcher.intervals.rows() == [
        ("chr1", 1, 10, ".", "g1"),
        ("chr1", 23, 30, ".", "g2"),
        ("chr1", 5, 50, ".", "g3"),
        ("chr10", 1, 30, ".", "g4"),
    ]

    matcher = SingleVariantMatcher(variants=variants, intervals=intervals)
    assert isinstance(matcher.intervals, pl.DataFrame)
    assert matcher.intervals.columns == ["chrom", "start", "end", "strand"]
    assert matcher.intervals.rows() == [("chr1", 1, 10, "+"), ("chr1", 23, 30, "-"), ("chr10", 1, 30, "+")]


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
    assert list(SingleVariantMatcher(variants=variants_to_polars(variants), intervals=interval_frame)) == expected
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


def test_SingleVariantMatcher_yields_given_objects():
    named = [Interval("chr1", 1, 10, name="a", attrs={"x": 1})]
    ((interval, variant), *_) = list(SingleVariantMatcher(variants=variants, intervals=named))
    assert interval is named[0]
    assert variant is variants[0]

    # a table gives new Variant objects
    ((_, variant), *_) = list(SingleVariantMatcher(vcf_file, intervals=named))
    assert variant == variants[0]
    assert variant.source is None


def test_SingleVariantMatcher_pairs():
    matcher = SingleVariantMatcher(vcf_file, intervals=interval_frame, interval_attrs=["gene_id"])
    pairs = matcher.pairs()
    assert pairs.select("gene_id", "variant_pos", "variant_ref", "variant_alt").rows() == [
        ("g1", 4, "T", "C"),
        ("g1", 5, "A", "GA"),
        ("g2", 25, "AACG", "GA"),
        ("g3", 25, "AACG", "GA"),
    ]
    assert pairs["variant_idx"].to_list() == [0, 1, 2, 2]
    # of the columns of scan_vcf_variants, the matcher keeps the coordinates, ref, alt and allele_idx
    assert [c for c in pairs.columns if c.startswith("variant_")] == [
        "variant_start",
        "variant_end",
        "variant_pos",
        "variant_ref",
        "variant_alt",
        "variant_allele_idx",
        "variant_idx",
    ]
    assert pairs["variant_allele_idx"].to_list() == [1, 1, 1, 1]


@pytest.mark.parametrize(
    "source",
    [{"vcf_file": vcf_file}, {"variants": variants}, {"variants": variants_to_polars(variants).lazy()}],
    ids=["vcf_file", "objects", "frame"],
)
def test_SingleVariantMatcher_iter_batches(source):
    matcher = SingleVariantMatcher(**source, intervals=interval_frame, variant_batch_size=2)
    batches = list(matcher.iter_batches())
    # variant_idx counts over all batches
    assert [b["variant_idx"].to_list() for b in batches] == [[0, 1], [2, 2]]
    assert pl.concat(batches).equals(matcher.pairs())


def test_SingleVariantMatcher_scan_pairs():
    matcher = SingleVariantMatcher(vcf_file, intervals=interval_frame)
    lazy_pairs = matcher.scan_pairs()
    assert isinstance(lazy_pairs, pl.LazyFrame)
    assert lazy_pairs.collect().sort(["variant_idx", "interval_idx"]).equals(matcher.pairs())


def test_BaseVariantMatcher_variant_sources():
    with pytest.raises(ValueError, match="exactly one"):
        SingleVariantMatcher(intervals=intervals)
    with pytest.raises(ValueError, match="exactly one"):
        SingleVariantMatcher(vcf_file, variants=variants, intervals=intervals)


def test_SingleVariantMatcher_variant_frame():
    frame = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "pos": [4, 5, 25],
            "ref": ["T", "A", "AACG"],
            "alt": ["C", "GA", "GA"],
            "variant_id": ["v1", "v2", "v3"],
            # start and end are recomputed from pos and ref
            "start": [0, 0, 0],
        }
    )
    pairs = SingleVariantMatcher(variants=frame, intervals=interval_frame).pairs()
    assert pairs.select("variant_start", "variant_end", "variant_variant_id").rows() == [
        (3, 4, "v1"),
        (4, 5, "v2"),
        (24, 28, "v3"),
        (24, 28, "v3"),
    ]
    assert pairs.equals(SingleVariantMatcher(variants=frame.lazy(), intervals=interval_frame).pairs())

    with pytest.raises(ValueError, match="alt"):
        SingleVariantMatcher(variants=frame.drop("alt"), intervals=interval_frame)


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
    # the interval on chr10 has no variants
    assert pairs[3][0] == intervals[2]
    assert list(pairs[3][1]) == []


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
    expected = [(BOUNDARY_INTERVAL, variant)] if overlaps else []
    assert list(SingleVariantMatcher(variants=[variant], intervals=[BOUNDARY_INTERVAL])) == expected
    frame = pl.DataFrame({"chrom": [variant.chrom], "pos": [variant.pos], "ref": [variant.ref], "alt": [variant.alt]})
    assert list(SingleVariantMatcher(variants=frame, intervals=[BOUNDARY_INTERVAL])) == expected
    ((_, matched),) = list(MultiVariantsMatcher(variants=[variant], intervals=[BOUNDARY_INTERVAL]))
    assert list(matched) == ([variant] if overlaps else [])


def test_SingleVariantMatcher_boundaries_in_one_batch():
    all_variants = [v for _, v, _ in BOUNDARY_CASES]
    pairs = list(SingleVariantMatcher(variants=all_variants, intervals=[BOUNDARY_INTERVAL]))
    assert [v for _, v in pairs] == [v for _, v, overlaps in BOUNDARY_CASES if overlaps]


def test_SingleVariantMatcher_boundaries_vcf_file(tmp_path):
    # a VCF record cannot have an empty REF
    cases = [(v, overlaps) for _, v, overlaps in BOUNDARY_CASES if v.ref]
    path = tmp_path / "boundaries.vcf"
    records = ["\t".join([v.chrom, str(v.pos), ".", v.ref, v.alt, ".", ".", "."]) for v, _ in cases]
    path.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + "\n".join(records) + "\n")
    pairs = list(SingleVariantMatcher(str(path), intervals=[BOUNDARY_INTERVAL]))
    assert [v for _, v in pairs] == [v for v, overlaps in cases if overlaps]


# Intervals around the variants of the test VCFs, including one without variants
CYVCF2_INTERVALS = [
    Interval("chr1", 0, 15, strand="+"),
    Interval("chr1", 9, 30, strand="-"),
    Interval("chr1", 25, 50),
    Interval("chr2", 0, 10),
    Interval("chr3", 0, 10),
]


# The overlapping pairs of CYVCF2_INTERVALS and the Variants of the cyvcf2-based MultiSampleVCF,
# in VCF order and per variant in the order of the intervals: the index into CYVCF2_INTERVALS and the variant.
CYVCF2_PAIRS = {
    vcf_file: [
        (0, "chr1:4:T>C"),
        (0, "chr1:5:A>GA"),
        (1, "chr1:25:AACG>GA"),
        (2, "chr1:25:AACG>GA"),
    ],
    test_with_multiple_variants: [
        (0, "chr1:4:T>C"),
        (0, "chr1:4:T>A"),
        (0, "chr1:4:T>G"),
        (0, "chr1:5:A>GA"),
        (0, "chr1:5:A>"),
        (0, "chr1:12:T>"),
        (1, "chr1:12:T>"),
        (1, "chr1:25:AACG>GA"),
        (2, "chr1:25:AACG>GA"),
    ],
    "edge_case_vcf": [
        (0, "chr1:10:A>C"),
        (1, "chr1:10:A>C"),
        (0, "chr1:10:A>G"),
        (1, "chr1:10:A>G"),
        (1, "chr1:20:ACGT><DEL>"),
        (1, "chr1:30:T>"),
        (2, "chr1:30:T>"),
        (3, "chr2:5:TA>T"),
        (3, "chr2:5:TA>TAA"),
    ],
}


@pytest.mark.parametrize("path", [vcf_file, test_with_multiple_variants, "edge_case_vcf"])
def test_SingleVariantMatcher_equals_cyvcf2(path, request):
    expected = [(CYVCF2_INTERVALS[i], variant) for i, variant in CYVCF2_PAIRS[path]]
    if path == "edge_case_vcf":
        path = request.getfixturevalue(path)
    matcher = SingleVariantMatcher(path, intervals=CYVCF2_INTERVALS, variant_batch_size=2)
    assert [(interval, str(variant)) for interval, variant in matcher] == expected
    pairs = matcher.pairs()
    actual = [
        (CYVCF2_INTERVALS[i], "{}:{}:{}>{}".format(chrom, pos, ref, alt))
        for i, chrom, pos, ref, alt in pairs.select(
            "interval_idx", "chrom", "variant_pos", "variant_ref", "variant_alt"
        ).iter_rows()
    ]
    assert actual == expected
