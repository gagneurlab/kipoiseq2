import polars as pl
import pytest
from conftest import EDGE_CASE_VCF, test_with_multiple_variants, vcf_file

from kipoiseq2.extractors.vcf import scan_vcf_genotypes, scan_vcf_variants

# The output of kipoiseq's MultiSampleVCF, which read VCF files with cyvcf2:
# chrom, start, end, pos, ref and alt of the Variants it yielded.
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
        ("chr1", 4, 5, 5, "A", ""),
        ("chr1", 11, 12, 12, "T", ""),
        ("chr1", 24, 28, 25, "AACG", "GA"),
    ],
    "edge_case_vcf": [
        ("chr1", 9, 10, 10, "A", "C"),
        ("chr1", 9, 10, 10, "A", "G"),
        ("chr1", 19, 23, 20, "ACGT", "<DEL>"),
        ("chr1", 29, 30, 30, "T", ""),
        ("chr2", 4, 6, 5, "TA", "T"),
        ("chr2", 4, 6, 5, "TA", "TAA"),
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
    # one row per ALT allele, without the ALT alleles that contain N or *
    assert df.select("pos", "alt", "allele_idx").rows() == [
        (10, "C", 1),
        (10, "G", 2),
        (20, "<DEL>", 1),
        (30, "", 1),
        (5, "T", 1),
        (5, "TAA", 2),
    ]
    # end follows REF, not INFO/END=40
    assert df.filter(pl.col("alt") == "<DEL>").select("start", "end").row(0) == (19, 23)


def test_scan_vcf_variants_columns():
    lf = scan_vcf_variants(vcf_file, info_fields=["DP"])
    assert lf.collect_schema() == pl.Schema(
        {
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "pos": pl.Int64,
            "ref": pl.String,
            "alt": pl.String,
            "allele_idx": pl.Int64,
            "DP": pl.Int32,
        }
    )
    # start and end do not depend on the coordinate system of polars-bio
    assert scan_vcf_variants(vcf_file, use_zero_based=False).collect().equals(scan_vcf_variants(vcf_file).collect())


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
        (20, "<DEL>", "S1", "0/1"),
        (20, "<DEL>", "S2", "1/2"),
        # haploid
        (5, "T", "S1", "1"),
        # the missing allele is ignored
        (5, "T", "S3", ".|1"),
        (5, "TAA", "S2", "3/2"),
    ]
    assert df["carrier"].all()


def test_scan_vcf_genotypes_all_rows(edge_case_vcf):
    df = scan_vcf_genotypes(edge_case_vcf, format_fields=["GQ", "AD"], carriers_only=False).collect()
    assert df.columns == [
        "chrom",
        "start",
        "end",
        "pos",
        "ref",
        "alt",
        "allele_idx",
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
    assert df.filter((pl.col("pos") == 30) & (pl.col("sample") == "S3")).select("GT", "carrier").row(0) == (None, False)


def test_scan_vcf_genotypes_samples(edge_case_vcf, tmp_path):
    s2 = scan_vcf_genotypes(edge_case_vcf, samples=["S2"], format_fields=["GQ", "AD"], carriers_only=False).collect()
    all_samples = scan_vcf_genotypes(edge_case_vcf, format_fields=["GQ", "AD"], carriers_only=False).collect()
    assert s2.equals(all_samples.filter(pl.col("sample") == "S2"))

    # polars-bio returns the FORMAT fields of a single-sample VCF as columns instead of a struct
    lines = [line.split("\t") for line in EDGE_CASE_VCF.splitlines()]
    single_sample_vcf = tmp_path / "single_sample.vcf"
    single_sample_vcf.write_text("".join("\t".join(line[:9] + line[10:11]) + "\n" for line in lines))
    single = scan_vcf_genotypes(str(single_sample_vcf), format_fields=["GQ", "AD"], carriers_only=False).collect()
    assert single.equals(s2)


def test_scan_vcf_genotypes_without_samples(tmp_path):
    lines = [line.split("\t") for line in EDGE_CASE_VCF.splitlines() if not line.startswith("##FORMAT")]
    sites_only_vcf = tmp_path / "sites_only.vcf"
    sites_only_vcf.write_text("".join("\t".join(line[:8]) + "\n" for line in lines))
    with pytest.raises(ValueError, match="no samples"):
        scan_vcf_genotypes(str(sites_only_vcf))
