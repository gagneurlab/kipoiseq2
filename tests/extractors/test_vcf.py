import sys

import polars as pl
import pytest
from conftest import EDGE_CASE_VCF, sample_5kb_fasta_file, test_with_multiple_variants, vcf_file

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors.vcf import MultiSampleVCF, scan_vcf_genotypes, scan_vcf_variants
from kipoiseq2.extractors.vcf_query import NumberVariantQuery

fasta_file = sample_5kb_fasta_file

intervals = [Interval("chr1", 3, 10), Interval("chr1", 4, 30), Interval("chr1", 19, 30)]


@pytest.fixture
def multi_sample_vcf():
    return MultiSampleVCF(vcf_file)


def test_MultiSampleVCF_without_vcf_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "cyvcf2", None)
    with pytest.raises(ImportError, match=r"kipoiseq2\[vcf\]"):
        MultiSampleVCF(vcf_file)


def test_MultiSampleVCF__next__(multi_sample_vcf):
    variant = next(multi_sample_vcf)
    assert variant.chrom == "chr1"
    assert variant.pos == 4
    assert variant.ref == "T"
    assert variant.alt == "C"


def test_MultiSampleVCF__iter__(multi_sample_vcf):
    variant = list(multi_sample_vcf)[0]
    assert variant.chrom == "chr1"
    assert variant.pos == 4
    assert variant.ref == "T"
    assert variant.alt == "C"


def test_MultiSampleVCF_fetch_variant(multi_sample_vcf):
    interval = Interval("chr1", 3, 5)
    assert len(list(multi_sample_vcf.fetch_variants(interval))) == 2
    assert len(list(multi_sample_vcf.fetch_variants(interval, "NA00003"))) == 1
    assert len(list(multi_sample_vcf.fetch_variants(interval, "NA00001"))) == 0

    interval = Interval("chr1", 7, 12)
    assert len(list(multi_sample_vcf.fetch_variants(interval))) == 0
    assert len(list(multi_sample_vcf.fetch_variants(interval, "NA00003"))) == 0


def test_MultiSampleVCF_query_variants(multi_sample_vcf):
    vq = multi_sample_vcf.query_variants(intervals)
    variants = list(vq)

    assert len(variants) == 5
    assert variants[0].pos == 4
    assert variants[1].pos == 5

    msvcf = MultiSampleVCF(test_with_multiple_variants)
    vq = msvcf.query_variants([Interval("chr1", 3, 10)])
    variants = list(vq)

    assert len(variants) == 5
    assert variants[0].ref == "T"
    assert variants[1].ref == "T"
    assert variants[2].ref == "T"
    assert variants[0].alt == "C"
    assert variants[1].alt == "A"
    assert variants[2].alt == "G"
    assert variants[4].alt == ""

    vq = msvcf.query_variants([Interval("chr1", 11, 14)])
    variants = list(vq)

    assert len(variants) == 1
    assert variants[0].ref == "T"
    assert variants[0].alt == ""


def test_MultiSampleVCF_get_samples(multi_sample_vcf):
    variants = list(multi_sample_vcf)
    samples = multi_sample_vcf.get_samples(variants[0])
    assert samples == {"NA00003": 3}


def test_MultiSampleVCF_get_variant(multi_sample_vcf):

    variant = multi_sample_vcf.get_variant("chr1:4:T>C")
    assert variant.chrom == "chr1"
    assert variant.pos == 4
    assert variant.ref == "T"
    assert variant.alt == "C"

    variant = multi_sample_vcf.get_variant(Variant("chr1", 4, "T", "C"))
    assert variant.chrom == "chr1"
    assert variant.pos == 4
    assert variant.ref == "T"
    assert variant.alt == "C"

    with pytest.raises(KeyError):
        multi_sample_vcf.get_variant("chr1:4:A>C")


def test_MultiSampleVCF_get_variants(multi_sample_vcf):
    variants = multi_sample_vcf.get_variants(["chr1:4:T>C"], intervals)
    assert len(variants) == 1

    variant = variants[0]
    assert variant.chrom == "chr1"
    assert variant.pos == 4
    assert variant.ref == "T"
    assert variant.alt == "C"

    variants = multi_sample_vcf.get_variants(["chr1:4:T>C", "chr1:25:AACG>GA"])
    assert len(variants) == 2

    variant = variants[0]
    assert variant.chrom == "chr1"
    assert variant.pos == 4
    assert variant.ref == "T"
    assert variant.alt == "C"


def test_MultiSampleVCF__regions_from_variants(multi_sample_vcf):
    variants = [
        Variant("chr1", 4, "T", "C"),
        Variant("chr1", 25, "AACG", "GA"),
        Variant("chr1", 55525, "AACG", "GA"),
        Variant("chr10", 55525, "AACG", "GA"),
    ]
    regions = multi_sample_vcf._regions_from_variants(variants)

    assert set(regions) == set(
        [Interval("chr1", 3, 25), Interval("chr1", 55524, 55525), Interval("chr10", 55524, 55525)]
    )


def test_MultiSampleVCF__regions_from_variants_variant_gap(multi_sample_vcf):
    variants = [Variant("chr1", 4, "T", "C"), Variant("chr1", 25, "AACG", "GA")]

    assert multi_sample_vcf._regions_from_variants(variants, variant_gap=150) == [Interval("chr1", 3, 25)]
    assert set(multi_sample_vcf._regions_from_variants(variants, variant_gap=10)) == {
        Interval("chr1", 3, 4),
        Interval("chr1", 24, 25),
    }


def test_MultiSampleVCF_get_variants_variant_gap(multi_sample_vcf, monkeypatch):
    seen = []
    monkeypatch.setattr(
        multi_sample_vcf, "_regions_from_variants", lambda v, variant_gap: seen.append(variant_gap) or []
    )

    multi_sample_vcf.get_variants(["chr1:4:T>C"], variant_gap=10)
    assert seen == [10]


def test_MultiSampleVCF_VariantQueryable_to_vcf(tmpdir, multi_sample_vcf):
    output_vcf_file = str(tmpdir / "output.vcf")

    multi_sample_vcf.query_variants(intervals).filter_range(NumberVariantQuery(max_num=1)).to_vcf(output_vcf_file)

    vcf = MultiSampleVCF(output_vcf_file)
    variants = list(vcf)
    assert len(variants) == 1
    assert variants[0].ref == "AACG"
    assert variants[0].alt == "GA"


def test_to_vcf_without_vcf_extra(monkeypatch, tmpdir, multi_sample_vcf):
    queryable = multi_sample_vcf.query_all()
    monkeypatch.setitem(sys.modules, "cyvcf2", None)
    with pytest.raises(ImportError, match=r"kipoiseq2\[vcf\]"):
        queryable.to_vcf(str(tmpdir / "output.vcf"))


def test_batch_iter_vcf(multi_sample_vcf):
    batchs = list(multi_sample_vcf.batch_iter(10))
    assert sum(len(i) for i in batchs) == 3


def test_MultiSampleVCF_query_all(multi_sample_vcf):
    variants = list(multi_sample_vcf.query_all())
    assert len(variants) == 3


def _cyvcf2_variants(path):
    """chrom, start, end, pos, ref and alt of the Variants of MultiSampleVCF."""
    return [(v.chrom, v.start, v.end, v.pos, v.ref, v.alt) for v in MultiSampleVCF(path)]


@pytest.mark.parametrize("path", [vcf_file, test_with_multiple_variants])
def test_scan_vcf_variants_equals_MultiSampleVCF(path):
    df = scan_vcf_variants(path).collect()
    assert df.select("chrom", "start", "end", "pos", "ref", "alt").rows() == _cyvcf2_variants(path)


def test_scan_vcf_variants_edge_cases(edge_case_vcf):
    df = scan_vcf_variants(edge_case_vcf).collect()
    assert df.select("chrom", "start", "end", "pos", "ref", "alt").rows() == _cyvcf2_variants(edge_case_vcf)
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


def _cyvcf2_carriers(path):
    """chrom, pos, ref, alt and sample of each sample that MultiSampleVCF.get_samples returns."""
    vcf = MultiSampleVCF(path)
    return [(v.chrom, v.pos, v.ref, v.alt, sample) for v in vcf for sample in vcf.get_samples(v)]


@pytest.mark.parametrize("path", [vcf_file, test_with_multiple_variants])
def test_scan_vcf_genotypes_equals_MultiSampleVCF_get_samples(path):
    # these VCFs have one ALT allele per record, so carriers per record and per ALT allele are the same
    df = scan_vcf_genotypes(path).collect()
    assert df.select("chrom", "pos", "ref", "alt", "sample").rows() == _cyvcf2_carriers(path)


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
