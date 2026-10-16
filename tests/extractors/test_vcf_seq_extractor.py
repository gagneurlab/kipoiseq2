import pytest
from conftest import sample_5kb_fasta_file

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors import FastaStringExtractor
from kipoiseq2.extractors.vcf_seq import VariantSeqExtractor, reverse_complement

fasta_file = sample_5kb_fasta_file

intervals = [Interval("chr1", 4, 10), Interval("chr1", 5, 30), Interval("chr1", 20, 30)]


def test_reverse_complement():
    assert reverse_complement("ACGTNacgtn") == "nacgtNACGT"
    assert reverse_complement("RYKM") == "KMRY"
    with pytest.raises(ValueError):
        reverse_complement("AC-GT")


@pytest.fixture
def variant_seq_extractor():
    return VariantSeqExtractor(fasta_file, use_strand=True)


def test_variant_seq_extractor_reference_arguments():
    with pytest.raises(ValueError, match="exactly one of `fasta_file` and `reference_sequence`"):
        VariantSeqExtractor(use_strand=True)
    with pytest.raises(ValueError, match="exactly one of `fasta_file` and `reference_sequence`"):
        VariantSeqExtractor(
            fasta_file, reference_sequence=FastaStringExtractor(fasta_file, use_strand=False), use_strand=True
        )


def test_variant_seq_extractor_requires_use_strand_keyword():
    with pytest.raises(TypeError, match="use_strand"):
        VariantSeqExtractor(fasta_file)
    with pytest.raises(TypeError, match="positional"):
        VariantSeqExtractor(fasta_file, None, True)


def test_variant_seq_extractor_extract_rejects_unknown_arguments(variant_seq_extractor):
    with pytest.raises(TypeError, match="fixed_length"):
        variant_seq_extractor.extract(Interval("chr1", 0, 3), [], anchor=0, fixed_length=False)


def test_split():
    # a deletion at 3 to 6 split at 5
    assert VariantSeqExtractor._split([(3, 6, "T")], 5) == [(3, 5, "T"), (5, 6, "")]
    # an insertion at 3 to 5 split at 4
    assert VariantSeqExtractor._split([(3, 5, "AAA")], 4) == [(3, 4, "A"), (4, 5, "AA")]
    assert VariantSeqExtractor._split([(3, 5, "AAA")], 4, keep="left") == [(3, 4, "A")]
    assert VariantSeqExtractor._split([(3, 5, "AAA")], 4, keep="right") == [(4, 5, "AA")]
    # a variant that ends or starts at the position stays whole
    assert VariantSeqExtractor._split([(3, 5, "A"), (5, 6, "C")], 5) == [(3, 5, "A"), (5, 6, "C")]


def test_extract(variant_seq_extractor):
    # the variants of tests/data/test.vcf.gz
    variants = [Variant("chr1", 4, "T", "C"), Variant("chr1", 5, "A", "GA"), Variant("chr1", 25, "AACG", "GA")]

    interval = Interval("chr1", 0, 30)

    seq = variant_seq_extractor.extract(interval, variants, anchor=28, fixed_len=True, is_padding=True)
    assert len(seq) == interval.end - interval.start
    assert seq == "NACGCGAACGTAACGTAACGTAACGTGATA"

    interval = Interval("chr1", 2, 9)

    seq = variant_seq_extractor.extract(interval, variants, anchor=5)
    assert len(seq) == interval.end - interval.start
    assert seq == "CGAACGT"

    interval = Interval("chr1", 2, 9, strand="-")
    seq = variant_seq_extractor.extract(interval, variants, anchor=5)
    assert len(seq) == interval.end - interval.start
    assert seq == "ACGTTCG"

    interval = Interval("chr1", 4, 14)
    seq = variant_seq_extractor.extract(interval, variants, anchor=7)
    assert len(seq) == interval.end - interval.start
    assert seq == "AACGTAACGT"

    interval = Interval("chr1", 4, 14)
    seq = variant_seq_extractor.extract(interval, variants, anchor=4)
    assert len(seq) == interval.end - interval.start
    assert seq == "GAACGTAACG"

    interval = Interval("chr1", 2, 5)
    seq = variant_seq_extractor.extract(interval, variants, anchor=3)
    assert len(seq) == interval.end - interval.start
    assert seq == "GCG"

    interval = Interval("chr1", 24, 34)
    seq = variant_seq_extractor.extract(interval, variants, anchor=27)
    assert len(seq) == interval.end - interval.start
    assert seq == "TGATAACGTA"

    interval = Interval("chr1", 25, 35)
    seq = variant_seq_extractor.extract(interval, variants, anchor=34)
    assert len(seq) == interval.end - interval.start
    assert seq == "TGATAACGTA"

    interval = Interval("chr1", 34, 44)
    seq = variant_seq_extractor.extract(interval, variants, anchor=37)
    assert len(seq) == interval.end - interval.start
    assert seq == "AACGTAACGT"

    interval = Interval("chr1", 34, 44)
    seq = variant_seq_extractor.extract(interval, variants, anchor=100)
    assert len(seq) == interval.end - interval.start
    assert seq == "AACGTAACGT"

    interval = Interval("chr1", 5, 11, strand="+")
    seq = variant_seq_extractor.extract(interval, variants, anchor=10, fixed_len=False)
    assert seq == "ACGTAA"

    interval = Interval("chr1", 0, 3, strand="+")
    seq = variant_seq_extractor.extract(interval, variants, anchor=10, fixed_len=False)
    assert seq == "ACG"

    interval = Interval("chr1", 0, 3, strand="+")
    ref_seq_extractor = FastaStringExtractor(fasta_file, use_strand=True)
    seq = VariantSeqExtractor(reference_sequence=ref_seq_extractor, use_strand=True).extract(
        interval, variants, anchor=10, fixed_len=False
    )
    assert seq == "ACG"


@pytest.mark.parametrize("anchor", [15, 20, 25])
def test_extract_deletion_before_interval(variant_seq_extractor, anchor):
    # the deletion lies before the interval, so the sequence is the reference of the interval.
    # The anchor 15 is clamped to the interval start.
    interval = Interval("chr1", 20, 30)
    deletion = Variant("chr1", 12, "CG", "C")
    seq = variant_seq_extractor.extract(interval, [deletion], anchor=anchor, fixed_len=True)
    assert len(seq) == interval.end - interval.start
    assert seq == "ACGTAACGTA"


# sample.5kb.fa has one 5000 bp chromosome that repeats ACGTA
CHROM_LEN = 5000


@pytest.mark.parametrize("is_padding", [False, True])
def test_extract_interval_at_chromosome_end(variant_seq_extractor, is_padding):
    # the interval covers the last 10 bases, so no padding is needed
    interval = Interval("chr1", CHROM_LEN - 10, CHROM_LEN)
    seq = variant_seq_extractor.extract(interval, [], anchor=CHROM_LEN - 5, chrom_len=CHROM_LEN, is_padding=is_padding)
    assert seq == "ACGTAACGTA"

    # a deletion upstream of the anchor pulls in 2 bases after the chromosome end
    deletion = Variant("chr1", CHROM_LEN - 7, "GTA", "G")
    if is_padding:
        seq = variant_seq_extractor.extract(
            interval, [deletion], anchor=CHROM_LEN - 10, chrom_len=CHROM_LEN, is_padding=True
        )
        assert seq == "ACGACGTANN"
    else:
        with pytest.raises(ValueError, match="is_padding"):
            variant_seq_extractor.extract(interval, [deletion], anchor=CHROM_LEN - 10, chrom_len=CHROM_LEN)


def test_extract_interval_beyond_chromosome_end(variant_seq_extractor):
    interval = Interval("chr1", CHROM_LEN - 5, CHROM_LEN + 5)
    seq = variant_seq_extractor.extract(interval, [], anchor=CHROM_LEN, chrom_len=CHROM_LEN, is_padding=True)
    assert seq == "ACGTANNNNN"
    # chrom_len may be a float
    float_len = float(CHROM_LEN)
    assert variant_seq_extractor.extract(interval, [], anchor=CHROM_LEN, chrom_len=float_len, is_padding=True) == seq
