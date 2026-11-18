from pathlib import Path

import polars as pl
import pytest
from conftest import EDGE_CASE_VCF, test_with_multiple_variants, vcf_file

from kipoiseq2.extractors.vcf import scan_vcf_genotypes, scan_vcf_variants

# The output of kipoiseq's MultiSampleVCF, which read VCF files with cyvcf2:
# chrom, start, end, pos, ref and alt of the Variants it yielded.
# scan_vcf_variants keeps other ALT alleles than MultiSampleVCF, and these lists follow it:
# they add the ALT alleles that contain N, and they drop symbolic ALT alleles and ALT `.`.
CYVCF2_VARIANTS = {
    vcf_file: [
        ("chr1", 3, 4, 4, "T", "C"),
        ("chr1", 4, 5, 5, "A", "GA"),
        ("chr1", 24, 28, 25, "AACG", "GA"),
    ],
    test_with_multiple_variants: [
        ("chr1", 3, 4, 4, "T", "C"),
        ("chr1", 3, 4, 4, "T", "A"),
        ("chr1", 3, 4, 4, "T", "G"),
        ("chr1", 4, 5, 5, "A", "GA"),
        ("chr1", 4, 5, 5, "A", "N"),
        ("chr1", 24, 28, 25, "AACG", "GA"),
    ],
    "edge_case_vcf": [
        ("chr1", 9, 10, 10, "A", "C"),
        ("chr1", 9, 10, 10, "A", "G"),
        ("chr1", 39, 40, 40, "G", "N"),
        ("chr2", 4, 6, 5, "TA", "T"),
        ("chr2", 4, 6, 5, "TA", "TAA"),
        ("chr2", 4, 6, 5, "TA", "N"),
    ],
}

# chrom, pos, ref, alt and sample of the samples that MultiSampleVCF.get_samples returned
CYVCF2_CARRIERS = {
    vcf_file: [
        ("chr1", 4, "T", "C", "NA00003"),
        ("chr1", 25, "AACG", "GA", "NA00002"),
    ],
    test_with_multiple_variants: [
        ("chr1", 4, "T", "C", "NA00003"),
        ("chr1", 4, "T", "A", "NA00003"),
        ("chr1", 4, "T", "G", "NA00003"),
        ("chr1", 25, "AACG", "GA", "NA00002"),
    ],
}


@pytest.mark.parametrize("path", [vcf_file, test_with_multiple_variants])
def test_scan_vcf_variants_equals_cyvcf2(path):
    df = scan_vcf_variants(path).collect()
    assert df.select("chrom", "start", "end", "pos", "ref", "alt").rows() == CYVCF2_VARIANTS[path]


def test_scan_vcf_variants_edge_cases(edge_case_vcf):
    df = scan_vcf_variants(edge_case_vcf).collect()
    assert df.select("chrom", "start", "end", "pos", "ref", "alt").rows() == CYVCF2_VARIANTS["edge_case_vcf"]
    # one row per ALT allele that is a sequence of bases
    assert df.select("pos", "alt", "allele_idx").rows() == [
        (10, "C", 1),
        (10, "G", 2),
        (40, "N", 1),
        (5, "T", 1),
        (5, "TAA", 2),
        (5, "N", 3),
    ]
    # a missing ID or FILTER is an empty string, and a missing QUAL is null
    assert df.filter(pl.col("alt").is_in(["G", "TAA"])).select("id", "qual", "filter").rows() == [
        ("rs1", 50.0, "PASS"),
        ("", None, ""),
    ]


def test_scan_vcf_variants_alt_kinds(alt_kinds_vcf):
    df = scan_vcf_variants(alt_kinds_vcf).collect()
    # ALT ".", "*", symbolic ALT alleles and the breakend are dropped, and ref and alt are uppercase.
    # allele_idx counts the dropped <DEL>.
    assert df.select("pos", "ref", "alt", "allele_idx").rows() == [
        (2, "C", "A", 1),
        (4, "T", "ACG", 1),
        (6, "A", "AN", 1),
        (23, "G", "A", 1),
        (23, "G", "C", 3),
    ]
    # end follows REF, not INFO/END=25
    assert df.filter(pl.col("pos") == 23).select("start", "end").rows() == [(22, 23), (22, 23)]


def test_scan_vcf_variants_columns():
    lf = scan_vcf_variants(vcf_file, info_fields=["DP"])
    assert lf.collect_schema() == pl.Schema(
        {
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "pos": pl.Int64,
            "id": pl.String,
            "ref": pl.String,
            "alt": pl.String,
            "allele_idx": pl.Int64,
            "qual": pl.Float64,
            "filter": pl.String,
            "DP": pl.Int32,
        }
    )


def test_scan_vcf_variants_zero_based(edge_case_vcf):
    variants = scan_vcf_variants(vcf_file)
    assert variants.collect().select("start", "end", "pos").row(0) == (3, 4, 4)
    # polars-bio range operations read the coordinate system from the metadata
    assert variants.config_meta.get_metadata()["coordinate_system_zero_based"] is True
    for carriers_only in (True, False):
        genotypes = scan_vcf_genotypes(edge_case_vcf, carriers_only=carriers_only)
        assert genotypes.config_meta.get_metadata()["coordinate_system_zero_based"] is True
        assert genotypes.collect().select("start", "end", "pos").row(0) == (9, 10, 10)


def test_scan_vcf_variants_all_alts(alt_kinds_vcf):
    df = scan_vcf_variants(alt_kinds_vcf, sequence_alts_only=False).collect()
    # every ALT allele gives a row, and ALT "." gives a null alt.
    # The sequences are uppercase, but the breakend keeps its text, so chr1 stays lowercase.
    assert df.select("pos", "alt", "allele_idx").rows() == [
        (2, "A", 1),
        (4, "ACG", 1),
        (6, "AN", 1),
        (8, None, 1),
        (10, "*", 1),
        (12, "<DEL>", 1),
        (14, "<DUP>", 1),
        (16, "<INS>", 1),
        (18, "<NON_REF>", 1),
        (20, "A]chr1:30]", 1),
        (23, "A", 1),
        (23, "<DEL>", 2),
        (23, "C", 3),
    ]
    sequence_alts = scan_vcf_variants(alt_kinds_vcf).collect()
    assert df.filter(pl.col("alt").str.contains("(?i)^[ACGTN]+$")).equals(sequence_alts)


def test_scan_vcf_variants_path(edge_case_vcf):
    assert scan_vcf_variants(Path(edge_case_vcf)).collect().equals(scan_vcf_variants(edge_case_vcf).collect())


def test_scan_vcf_variants_arguments():
    with pytest.raises(TypeError, match="info_fields"):
        scan_vcf_variants(vcf_file, info_fields="DP")
    with pytest.raises(TypeError, match="info_fields"):
        scan_vcf_variants(vcf_file, info_fields=[1])
    with pytest.raises(TypeError, match="info_fields"):
        scan_vcf_variants(vcf_file, info_fields={"DP"})
    # the options are keyword-only, and there are no options of polars_bio.scan_vcf
    with pytest.raises(TypeError):
        scan_vcf_variants(vcf_file, ["DP"])
    with pytest.raises(TypeError):
        scan_vcf_variants(vcf_file, use_zero_based=False)
    with pytest.raises(TypeError):
        scan_vcf_variants(vcf_file, format_fields=["GT"])


@pytest.mark.parametrize("path", [vcf_file, test_with_multiple_variants])
def test_scan_vcf_genotypes_equals_cyvcf2(path):
    # these VCFs have one ALT allele per record, so carriers per record and per ALT allele are the same
    df = scan_vcf_genotypes(path).collect()
    assert df.select("chrom", "pos", "ref", "alt", "sample").rows() == CYVCF2_CARRIERS[path]


def test_scan_vcf_genotypes_per_allele(edge_case_vcf):
    df = scan_vcf_genotypes(edge_case_vcf).collect()
    assert df.select("pos", "alt", "sample", "GT").rows() == [
        (10, "C", "S2", "1|1"),
        # 0/2 carries the second ALT allele only
        (10, "G", "S1", "0/2"),
        (40, "N", "S1", "0/1"),
        # haploid
        (5, "T", "S1", "1"),
        # the missing allele is ignored
        (5, "T", "S3", ".|1"),
        (5, "TAA", "S2", "3/2"),
        (5, "N", "S2", "3/2"),
    ]
    assert df["carrier"].all()


def test_scan_vcf_genotypes_all_rows(edge_case_vcf):
    df = scan_vcf_genotypes(edge_case_vcf, format_fields=["GQ", "AD"], carriers_only=False).collect()
    assert df.columns == [
        "chrom",
        "start",
        "end",
        "pos",
        "id",
        "ref",
        "alt",
        "allele_idx",
        "qual",
        "filter",
        "sample",
        "GT",
        "GQ",
        "AD",
        "carrier",
    ]
    # 6 ALT alleles times 3 samples
    assert df.height == 18
    assert df.filter(pl.col("pos") == 10).select("alt", "sample", "GT", "GQ", "AD", "carrier").rows() == [
        ("C", "S1", "0/2", 30, [5, 0, 5], False),
        ("C", "S2", "1|1", 20, [0, 8, 0], True),
        ("C", "S3", "./.", None, None, False),
        ("G", "S1", "0/2", 30, [5, 0, 5], True),
        ("G", "S2", "1|1", 20, [0, 8, 0], False),
        ("G", "S3", "./.", None, None, False),
    ]
    # a missing GT is not a carrier
    assert df.filter((pl.col("pos") == 40) & (pl.col("sample") == "S3")).select("GT", "carrier").row(0) == (None, False)


def test_scan_vcf_genotypes_haplotypes(edge_case_vcf):
    df = scan_vcf_genotypes(edge_case_vcf, haplotypes=True).collect()
    assert df.select("pos", "alt", "sample", "haplotype", "GT").rows() == [
        (10, "C", "S2", 0, "1|1"),
        (10, "C", "S2", 1, "1|1"),
        # haplotype 1 of 0/2 carries the second ALT allele
        (10, "G", "S1", 1, "0/2"),
        (40, "N", "S1", 1, "0/1"),
        # haploid
        (5, "T", "S1", 0, "1"),
        # the missing allele keeps its position in GT
        (5, "T", "S3", 1, ".|1"),
        # unphased: the order in GT sets the haplotype
        (5, "TAA", "S2", 1, "3/2"),
        (5, "N", "S2", 0, "3/2"),
    ]
    assert df["carrier"].all()
    # the carriers per sample are the carriers per haplotype without haplotype
    per_sample = scan_vcf_genotypes(edge_case_vcf).collect()
    assert df.drop("haplotype").unique(maintain_order=True).equals(per_sample)


def test_scan_vcf_genotypes_haplotypes_all_rows(edge_case_vcf):
    df = scan_vcf_genotypes(edge_case_vcf, format_fields=["GQ", "AD"], carriers_only=False, haplotypes=True).collect()
    assert df.columns[10:] == ["sample", "haplotype", "GT", "GQ", "AD", "carrier"]
    assert df.schema["haplotype"] == pl.Int64
    # the ALT alleles C and G of 3 diploid samples, N of 2 diploid samples and 1 missing GT,
    # and T, TAA and N of 1 haploid and 2 diploid samples
    assert df.height == 2 * 3 * 2 + (2 * 2 + 1) + 3 * (1 + 2 * 2)
    assert df.filter(pl.col("alt") == "C").select("sample", "haplotype", "GT", "GQ", "carrier").rows() == [
        ("S1", 0, "0/2", 30, False),
        ("S1", 1, "0/2", 30, False),
        ("S2", 0, "1|1", 20, True),
        ("S2", 1, "1|1", 20, True),
        ("S3", 0, "./.", None, False),
        ("S3", 1, "./.", None, False),
    ]
    # a missing GT gives one row with a null haplotype
    missing = df.filter((pl.col("pos") == 40) & (pl.col("sample") == "S3"))
    assert missing.select("haplotype", "GT", "carrier").rows() == [(None, None, False)]


@pytest.mark.parametrize("haplotypes", [False, True])
def test_scan_vcf_genotypes_alt_kinds(alt_kinds_vcf, haplotypes):
    df = scan_vcf_genotypes(alt_kinds_vcf, haplotypes=haplotypes).collect()
    # for the ALT alleles A,<DEL>,C, S1 with GT 2/3 carries C, and S2 with GT 0/1 carries A.
    # Each carrier has one haplotype with the ALT allele, so haplotypes=True gives the same rows.
    assert df.select("pos", "alt", "allele_idx", "sample", "GT").rows() == [
        (2, "A", 1, "S1", "0/1"),
        (4, "ACG", 1, "S1", "0/1"),
        (6, "AN", 1, "S1", "0/1"),
        (23, "A", 1, "S2", "0/1"),
        (23, "C", 3, "S1", "2/3"),
    ]


def test_scan_vcf_genotypes_all_alts(alt_kinds_vcf):
    df = scan_vcf_genotypes(alt_kinds_vcf, sequence_alts_only=False).collect()
    # for the ALT alleles A,<DEL>,C, S1 with GT 2/3 carries <DEL> and C
    assert df.filter(pl.col("pos") == 23).select("alt", "allele_idx", "sample", "GT").rows() == [
        ("A", 1, "S2", "0/1"),
        ("<DEL>", 2, "S1", "2/3"),
        ("C", 3, "S1", "2/3"),
    ]


def test_scan_vcf_genotypes_info_fields(edge_case_vcf):
    df = scan_vcf_genotypes(edge_case_vcf, info_fields=["DP"]).collect()
    assert df.columns[:11] == scan_vcf_variants(edge_case_vcf, info_fields=["DP"]).collect_schema().names()
    assert df.filter(pl.col("pos") == 10)["DP"].to_list() == [10, 10]


@pytest.mark.parametrize("argument", ["samples", "format_fields", "info_fields"])
def test_scan_vcf_genotypes_arguments(edge_case_vcf, argument):
    with pytest.raises(TypeError, match=argument):
        scan_vcf_genotypes(edge_case_vcf, **{argument: "GQ"})
    with pytest.raises(TypeError, match=argument):
        scan_vcf_genotypes(edge_case_vcf, **{argument: [None]})


def test_scan_vcf_genotypes_keyword_only(edge_case_vcf):
    with pytest.raises(TypeError):
        scan_vcf_genotypes(edge_case_vcf, ["S1"])
    with pytest.raises(TypeError):
        scan_vcf_genotypes(edge_case_vcf, use_zero_based=False)


@pytest.mark.parametrize("haplotypes", [False, True])
def test_scan_vcf_genotypes_samples(edge_case_vcf, tmp_path, haplotypes):
    options = {"format_fields": ["GQ", "AD"], "carriers_only": False, "haplotypes": haplotypes}
    s2 = scan_vcf_genotypes(edge_case_vcf, samples=["S2"], **options).collect()
    all_samples = scan_vcf_genotypes(edge_case_vcf, **options).collect()
    assert s2.equals(all_samples.filter(pl.col("sample") == "S2"))

    # polars-bio returns the FORMAT fields of a single-sample VCF as columns instead of a struct
    lines = [line.split("\t") for line in EDGE_CASE_VCF.splitlines()]
    single_sample_vcf = tmp_path / "single_sample.vcf"
    single_sample_vcf.write_text("".join("\t".join(line[:9] + line[10:11]) + "\n" for line in lines))
    single = scan_vcf_genotypes(str(single_sample_vcf), **options).collect()
    assert single.equals(s2)


def test_scan_vcf_genotypes_without_samples(tmp_path):
    lines = [line.split("\t") for line in EDGE_CASE_VCF.splitlines() if not line.startswith("##FORMAT")]
    sites_only_vcf = tmp_path / "sites_only.vcf"
    sites_only_vcf.write_text("".join("\t".join(line[:8]) + "\n" for line in lines))
    with pytest.raises(ValueError, match="no samples"):
        scan_vcf_genotypes(str(sites_only_vcf))
