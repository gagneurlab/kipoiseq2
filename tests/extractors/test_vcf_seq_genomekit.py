"""Compare VariantSeqExtractor with GenomeKit 7.7.0.

The expected sequences are the outputs of GenomeKit for the same reference and variants.
"""

import pytest

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors import FastaStringExtractor, VariantSeqExtractor

FASTA = """\
>chr1
GCTAAAGACAATTACATAACATACACGTCAGCACGAAACTTGTTGGCCCAGTGTGAATCG
CTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGGAC
"""
"""A random 120 bp chromosome."""

SNV = Variant("chr1", 30, "A", "C")
INSERTION = Variant("chr1", 50, "A", "AGGG")
DELETION = Variant("chr1", 70, "TAAG", "T")

REGIONS = {"all": (20, 80, [SNV, INSERTION, DELETION]), "del_only": (55, 115, [DELETION])}
"""The regions of the haplotypes by name, as start, end and the variants within the region."""

HAPLOTYPES = {
    # GenomeKit gives ATACACGTCCGCACGAAACTTGTTGGCCCAGGGGTGTGAATCGCTTAAGGGTTTAAGTGT
    ("all", "+", "center"): "CACGTCCGCACGAAACTTGTTGGCCCAGGGGTGTGAATCGCTTAAGGGTTTAAGTGTGAT",
    ("all", "+", "5p"): "ATACACGTCCGCACGAAACTTGTTGGCCCAGGGGTGTGAATCGCTTAAGGGTTTAAGTGT",
    ("all", "+", "start"): "ATACACGTCCGCACGAAACTTGTTGGCCCAGGGGTGTGAATCGCTTAAGGGTTTAAGTGT",
    # GenomeKit gives ACACTTAAACCCTTAAGCGATTCACACCCCTGGGCCAACAAGTTTCGTGCGGACGTGTAT
    ("all", "-", "center"): "ATCACACTTAAACCCTTAAGCGATTCACACCCCTGGGCCAACAAGTTTCGTGCGGACGTG",
    ("all", "-", "5p"): "ACACTTAAACCCTTAAGCGATTCACACCCCTGGGCCAACAAGTTTCGTGCGGACGTGTAT",
    ("all", "-", "start"): "ACACTTAAACCCTTAAGCGATTCACACCCCTGGGCCAACAAGTTTCGTGCGGACGTGTAT",
    ("del_only", "+", "center"): "GTGAATCGCTTAAGGGTTTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCAT",
    ("del_only", "+", "5p"): "AATCGCTTAAGGGTTTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGG",
    ("del_only", "+", "start"): "AATCGCTTAAGGGTTTAAGTGTGATGCATACGCCTTTACTTGCTGTGTCCACCCCATCGG",
    ("del_only", "-", "center"): "ATGGGGTGGACACAGCAAGTAAAGGCGTATGCATCACACTTAAACCCTTAAGCGATTCAC",
    ("del_only", "-", "5p"): "ATGGGGTGGACACAGCAAGTAAAGGCGTATGCATCACACTTAAACCCTTAAGCGATTCAC",
    ("del_only", "-", "start"): "CCGATGGGGTGGACACAGCAAGTAAAGGCGTATGCATCACACTTAAACCCTTAAGCGATT",
}
"""The haplotypes from kipoiseq2 by region, strand and anchor.

They equal those from GenomeKit, except for the region "all" with the anchor "center".
There the REF of the insertion A>AGGG ends at the anchor 50, so kipoiseq2 puts the insertion left of the anchor.
GenomeKit normalizes it to an insertion of GGG at 50 and puts it right, see VariantSeqExtractor.extract.
"""


@pytest.fixture
def fasta_file(tmp_path):
    path = tmp_path / "ref.fa"
    path.write_text(FASTA)
    return path


@pytest.fixture
def reference(fasta_file):
    with FastaStringExtractor(fasta_file, use_strand=True) as extractor:
        yield extractor


@pytest.fixture
def variant_seq_extractor(fasta_file):
    with FastaStringExtractor(fasta_file, use_strand=False) as fasta:
        yield VariantSeqExtractor(fasta, use_strand=True)


@pytest.mark.parametrize(
    ("variant", "ref", "alt"),
    [
        pytest.param(SNV, "ACGTCAGCACG", "ACGTCCGCACG", id="snv"),
        pytest.param(INSERTION, "GGCCCAGTGTG", "GGCCCAGGGGT", id="insertion"),
        pytest.param(DELETION, "AGGGTTAAGTA", "AGGGTTTAAGT", id="deletion"),
    ],
)
def test_window_anchored_at_variant_start(reference, variant_seq_extractor, variant, ref, alt):
    # 5 bp left of the variant start and 6 bp from it on, so the alt sequence keeps 11 bp
    interval = Interval("chr1", variant.start - 5, variant.start + 6)
    assert reference.extract(interval) == ref
    assert variant_seq_extractor.extract(interval, [variant], anchor_pos=variant.start) == alt


@pytest.mark.parametrize(
    ("variant", "ref", "alt"),
    [
        pytest.param(SNV, "ACGTCAGCACG", "ACGTCCGCACG", id="snv"),
        pytest.param(INSERTION, "GGCCCAGTGTG", "GGCCCAGGGGTGTG", id="insertion"),
        pytest.param(DELETION, "AGGGTTAAGTAAGT", "AGGGTTTAAGT", id="deletion"),
    ],
)
def test_window_around_variant(reference, variant_seq_extractor, variant, ref, alt):
    # 5 bp on each side of the REF. Without an anchor, the length of the alt sequence follows the variant.
    interval = Interval("chr1", variant.start - 5, variant.end + 5)
    assert reference.extract(interval) == ref
    assert variant_seq_extractor.extract(interval, [variant], anchor_pos=None) == alt


@pytest.mark.parametrize("anchor", ["center", "5p", "start"])
@pytest.mark.parametrize("strand", ["+", "-"], ids=["plus", "minus"])
@pytest.mark.parametrize("region", ["all", "del_only"])
def test_haplotype(variant_seq_extractor, region, strand, anchor):
    start, end, variants = REGIONS[region]
    interval = Interval("chr1", start, end, strand=strand)
    # the anchors of GenomeKit: "5p" is the 5' end, and "start" is the genomic start on both strands
    anchor_pos = {"center": interval.center(), "5p": start if strand == "+" else end, "start": start}[anchor]
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=anchor_pos)
    assert seq == HAPLOTYPES[region, strand, anchor]
