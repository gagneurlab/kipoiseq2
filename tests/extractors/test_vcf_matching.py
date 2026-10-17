import polars as pl
import pytest
from conftest import test_with_multiple_variants, vcf_file

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors.vcf import scan_vcf_variants
from kipoiseq2.extractors.vcf_matching import scan_variant_pairs

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

variant_frame = pl.DataFrame(variants)


def sorted_pairs(intervals, variants):
    """Return the pairs of scan_variant_pairs as a DataFrame, sorted by variant and then by interval."""
    return scan_variant_pairs(intervals, variants).collect().sort("variant_idx", "interval_idx")


def test_scan_variant_pairs():
    lazy_pairs = scan_variant_pairs(interval_frame, variant_frame)
    assert isinstance(lazy_pairs, pl.LazyFrame)
    pairs = lazy_pairs.collect().sort("variant_idx", "interval_idx")
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
    assert pairs.select("variant_idx", "interval_idx").rows() == [(0, 0), (1, 0), (2, 1), (2, 2)]
    # the deletion AACG>GA at POS 25 covers the 0-based bases 24 to 27
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


def test_scan_variant_pairs_empty():
    no_variants = variant_frame.clear()
    pairs = scan_variant_pairs(interval_frame, no_variants).collect()
    assert pairs.height == 0
    assert "variant_idx" in pairs.columns
    assert scan_variant_pairs(interval_frame.clear(), variant_frame).collect().height == 0


def test_scan_variant_pairs_lazy_variants():
    # a DataFrame and a LazyFrame of variants give the same pairs
    assert sorted_pairs(interval_frame, variant_frame.lazy()).equals(sorted_pairs(interval_frame, variant_frame))


def test_scan_variant_pairs_collects_again():
    # each collect starts a new pass over the variants
    lazy_pairs = scan_variant_pairs(interval_frame, scan_vcf_variants(vcf_file))
    pairs = lazy_pairs.collect().sort("variant_idx", "interval_idx")
    assert lazy_pairs.collect().sort("variant_idx", "interval_idx").equals(pairs)


def test_scan_variant_pairs_collect_batches():
    lazy_pairs = scan_variant_pairs(interval_frame, scan_vcf_variants(vcf_file))
    batches = list(lazy_pairs.collect_batches(chunk_size=2))
    assert all(batch.height <= 2 for batch in batches)
    pairs = pl.concat(batches).sort("variant_idx", "interval_idx")
    assert pairs.equals(lazy_pairs.collect().sort("variant_idx", "interval_idx"))


def test_scan_variant_pairs_interval_columns():
    with pytest.raises(ValueError, match="end"):
        scan_variant_pairs(interval_frame.drop("end"), variant_frame)

    # chrom, start, end and strand come first, the other columns follow in their order
    pairs = sorted_pairs(interval_frame.select("gene_id", "end", "start", "chrom"), variant_frame)
    assert pairs.columns[:6] == ["chrom", "start", "end", "strand", "gene_id", "interval_idx"]
    # without a strand column, the strand is unknown
    assert pairs["strand"].to_list() == ["."] * 4


def test_scan_variant_pairs_null_strand():
    # a null strand is unknown, as a missing strand column
    frame = interval_frame.with_columns(strand=pl.Series(["+", None, None, "-"]))
    pairs = sorted_pairs(frame, variant_frame)
    assert pairs.select("gene_id", "strand").rows() == [("g1", "+"), ("g1", "+"), ("g2", "."), ("g3", ".")]
    frame = interval_frame.with_columns(strand=pl.lit(None))
    assert sorted_pairs(frame, variant_frame)["strand"].to_list() == ["."] * 4


def test_scan_variant_pairs_arguments():
    with pytest.raises(TypeError, match="scan_vcf_variants"):
        scan_variant_pairs(interval_frame, vcf_file)
    with pytest.raises(TypeError, match="^variants must be a polars DataFrame or LazyFrame, not list$"):
        scan_variant_pairs(interval_frame, variants)
    with pytest.raises(TypeError, match="^intervals must be a polars DataFrame, not LazyFrame$"):
        scan_variant_pairs(interval_frame.lazy(), variant_frame)
    with pytest.raises(TypeError, match="^intervals must be a polars DataFrame, not list$"):
        scan_variant_pairs(intervals, variant_frame)
    with pytest.raises(ValueError, match="alt"):
        scan_variant_pairs(interval_frame, variant_frame.drop("alt"))


def test_scan_variant_pairs_interval_column_names():
    # interval columns keep their names in the pairs, and the variant columns have the prefix variant_
    frame = interval_frame.with_columns(pos=pl.lit(1), ref=pl.lit("r"), id=pl.lit("i"))
    pairs = sorted_pairs(frame, scan_vcf_variants(vcf_file))
    columns = ("pos", "ref", "id", "variant_pos", "variant_ref", "variant_id")
    assert pairs.select(columns).row(0) == (1, "r", "i", 4, "T", "")

    # the pairs use the name interval_idx and the prefix variant_, so these interval columns raise a ValueError
    for column in ("interval_idx", "variant_idx", "variant_ref", "variant_id"):
        with pytest.raises(ValueError, match=column):
            scan_variant_pairs(interval_frame.with_columns(pl.lit("x").alias(column)), variant_frame)


def test_scan_variant_pairs_variant_column_names():
    frame = variant_frame.with_columns(
        variant_id=pl.Series(["v1", "v2", "v3"]),
        # start and end are recomputed from pos and ref
        start=pl.lit(0),
    )
    pairs = sorted_pairs(interval_frame, frame)
    assert pairs.select("variant_start", "variant_end", "variant_variant_id").rows() == [
        (3, 4, "v1"),
        (4, 5, "v2"),
        (24, 28, "v3"),
        (24, 28, "v3"),
    ]

    # with the prefix variant_, idx would become variant_idx, the name of the row number of the variant
    for column in ("idx", "variant_idx"):
        with pytest.raises(ValueError, match=f"columns \\['{column}'\\]"):
            scan_variant_pairs(interval_frame, variant_frame.with_columns(pl.lit(1).alias(column)))


def test_scan_variant_pairs_vcf():
    pairs = sorted_pairs(interval_frame, scan_vcf_variants(vcf_file))
    assert pairs.select("gene_id", "variant_pos", "variant_ref", "variant_alt").rows() == [
        ("g1", 4, "T", "C"),
        ("g1", 5, "A", "GA"),
        ("g2", 25, "AACG", "GA"),
        ("g3", 25, "AACG", "GA"),
    ]
    assert pairs["variant_idx"].to_list() == [0, 1, 2, 2]
    # the pairs keep all columns of scan_vcf_variants
    assert [c for c in pairs.columns if c.startswith("variant_")] == [
        "variant_start",
        "variant_end",
        "variant_pos",
        "variant_ref",
        "variant_alt",
        "variant_id",
        "variant_allele_idx",
        "variant_qual",
        "variant_filter",
        "variant_idx",
    ]
    assert pairs["variant_allele_idx"].to_list() == [1, 1, 1, 1]


def test_scan_variant_pairs_vcf_columns(edge_case_vcf):
    # id, qual and filter of scan_vcf_variants pass through to the pairs
    pairs = sorted_pairs(pl.DataFrame([Interval("chr1", 0, 50)]), scan_vcf_variants(edge_case_vcf))
    assert pairs.select("variant_pos", "variant_alt", "variant_id", "variant_qual", "variant_filter").rows() == [
        (10, "C", "rs1", 50.0, "PASS"),
        (10, "G", "rs1", 50.0, "PASS"),
        (40, "N", "", None, ""),
    ]


def test_scan_variant_pairs_null_alt(alt_kinds_vcf):
    # the ALT "." gives a null alt, which passes through to the pairs
    variants = scan_vcf_variants(alt_kinds_vcf, sequence_alts_only=False)
    pairs = sorted_pairs(pl.DataFrame([Interval("chr1", 6, 9)]), variants)
    assert pairs.select("variant_pos", "variant_alt").rows() == [(8, None)]


def test_scan_variant_pairs_to_objects():
    # a loop over the rows of the pairs builds Interval and Variant objects
    pairs = sorted_pairs(pl.DataFrame(intervals), scan_vcf_variants(vcf_file))
    objects = [
        (
            Interval(r["chrom"], r["start"], r["end"], strand=r["strand"]),
            Variant(r["chrom"], r["variant_pos"], r["variant_ref"], r["variant_alt"]),
        )
        for r in pairs.iter_rows(named=True)
    ]
    assert objects == [(intervals[0], variants[0]), (intervals[0], variants[1]), (intervals[1], variants[2])]


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
    ("other_chrom", Variant("chr2", 15, "A", "C"), False),
]


def matched_variants(pairs):
    """Return the variants of the pairs as Variant objects."""
    return [
        Variant(r["chrom"], r["variant_pos"], r["variant_ref"], r["variant_alt"]) for r in pairs.iter_rows(named=True)
    ]


@pytest.mark.parametrize("variant, overlaps", [c[1:] for c in BOUNDARY_CASES], ids=[c[0] for c in BOUNDARY_CASES])
def test_scan_variant_pairs_boundaries(variant, overlaps):
    pairs = sorted_pairs(pl.DataFrame([BOUNDARY_INTERVAL]), pl.DataFrame([variant]))
    assert matched_variants(pairs) == ([variant] if overlaps else [])


def test_scan_variant_pairs_boundaries_in_one_frame():
    all_variants = pl.DataFrame([v for _, v, _ in BOUNDARY_CASES])
    pairs = sorted_pairs(pl.DataFrame([BOUNDARY_INTERVAL]), all_variants)
    assert matched_variants(pairs) == [v for _, v, overlaps in BOUNDARY_CASES if overlaps]


def test_scan_variant_pairs_boundaries_vcf_file(tmp_path):
    cases = [(v, overlaps) for _, v, overlaps in BOUNDARY_CASES]
    path = tmp_path / "boundaries.vcf"
    records = ["\t".join([v.chrom, str(v.pos), ".", v.ref, v.alt, ".", ".", "."]) for v, _ in cases]
    path.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + "\n".join(records) + "\n")
    pairs = sorted_pairs(pl.DataFrame([BOUNDARY_INTERVAL]), scan_vcf_variants(path))
    assert matched_variants(pairs) == [v for v, overlaps in cases if overlaps]


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
# As in CYVCF2_VARIANTS of test_vcf.py, the ALT alleles follow scan_vcf_variants instead of MultiSampleVCF.
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
        (0, "chr1:5:A>N"),
        (1, "chr1:25:AACG>GA"),
        (2, "chr1:25:AACG>GA"),
    ],
    "edge_case_vcf": [
        (0, "chr1:10:A>C"),
        (1, "chr1:10:A>C"),
        (0, "chr1:10:A>G"),
        (1, "chr1:10:A>G"),
        (2, "chr1:40:G>N"),
        (3, "chr2:5:TA>T"),
        (3, "chr2:5:TA>TAA"),
        (3, "chr2:5:TA>N"),
    ],
}


@pytest.mark.parametrize("path", [vcf_file, test_with_multiple_variants, "edge_case_vcf"])
def test_scan_variant_pairs_equals_cyvcf2(path, request):
    expected = CYVCF2_PAIRS[path]
    if path == "edge_case_vcf":
        path = request.getfixturevalue(path)
    pairs = sorted_pairs(pl.DataFrame(CYVCF2_INTERVALS), scan_vcf_variants(path))
    actual = [
        (i, "{}:{}:{}>{}".format(chrom, pos, ref, alt))
        for i, chrom, pos, ref, alt in pairs.select(
            "interval_idx", "chrom", "variant_pos", "variant_ref", "variant_alt"
        ).iter_rows()
    ]
    assert actual == expected
