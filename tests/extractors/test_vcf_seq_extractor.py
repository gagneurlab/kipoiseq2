import pytest
from conftest import sample_5kb_fasta_file

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors import FastaStringExtractor
from kipoiseq2.extractors.vcf_seq import (
    IntervalSeqBuilder,
    Subsequence,
    VariantSeqExtractor,
    reverse_complement,
)

fasta_file = sample_5kb_fasta_file

intervals = [Interval("chr1", 4, 10), Interval("chr1", 5, 30), Interval("chr1", 20, 30)]


@pytest.fixture
def interval_seq_builder():
    return IntervalSeqBuilder(
        [
            Interval("chr1", 10, 13),
            Interval("chr1", 13, 14),
            Subsequence(seq="TAGC", start=14, end=18),
            Interval("chr1", 18, 20),
        ]
    )


def test_interval_seq_builder_restore(interval_seq_builder):
    sequence = Subsequence(seq="CCCCATCGTT", start=10, end=20)
    interval_seq_builder.restore(sequence)
    assert interval_seq_builder[0].seq == "CCC"
    assert interval_seq_builder[1].seq == "C"
    assert interval_seq_builder[2].seq == "TAGC"
    assert interval_seq_builder[3].seq == "TT"

    interval_seq_builder.append(Interval("chr1", 5, 10))
    interval_seq_builder.restore(sequence)
    assert interval_seq_builder[4].seq == ""

    interval_seq_builder.append(Interval("chr1", 20, 25))
    interval_seq_builder.restore(sequence)
    assert interval_seq_builder[5].seq == ""

    interval_seq_builder.append(Interval("chr1", 10, 5))
    interval_seq_builder.restore(sequence)
    assert interval_seq_builder[6].seq == ""

    interval_seq_builder.append(Interval("chr1", 25, 20))
    interval_seq_builder.restore(sequence)
    assert interval_seq_builder[7].seq == ""


def test_interval_seq_builder_concat(interval_seq_builder):
    with pytest.raises(TypeError):
        interval_seq_builder.concat()

    sequence = Subsequence(seq="CCCCATCGNN", start=10, end=20)
    interval_seq_builder.restore(sequence)
    assert interval_seq_builder.concat() == "CCCCTAGCNN"


def test_subsequence_slice_keeps_coordinates():
    s = Subsequence(seq="ACGTA", start=10, end=15)
    assert s[1:3] == Subsequence("CG", 11, 13)
    assert s[:2] == Subsequence("AC", 10, 12)
    assert s[3:] == Subsequence("TA", 13, 15)
    # slices beyond the sequence are clamped, as for str
    assert s[4:9] == Subsequence("A", 14, 15)
    assert s[7:9] == Subsequence("", 15, 15)


def test_reverse_complement():
    assert reverse_complement("ACGTNacgtn") == "nacgtNACGT"
    assert reverse_complement("RYKM") == "KMRY"
    with pytest.raises(ValueError):
        reverse_complement("AC-GT")


@pytest.fixture
def variant_seq_extractor():
    return VariantSeqExtractor(fasta_file)


def test__split_overlapping(variant_seq_extractor):
    pair = (Subsequence(seq="AAA", start=3, end=6), Subsequence(seq="T", start=3, end=4))
    splited_pairs = list(variant_seq_extractor._split_overlapping([pair], 5))

    assert splited_pairs[0][0].seq == "AA"
    assert splited_pairs[0][1].seq == "T"
    assert splited_pairs[1][0].seq == "A"
    assert splited_pairs[1][1].seq == ""

    pair = (Subsequence(seq="TT", start=3, end=5), Subsequence(seq="AAA", start=3, end=6))
    splited_pairs = list(variant_seq_extractor._split_overlapping([pair], 4))

    assert splited_pairs[0][0].seq == "T"
    assert splited_pairs[0][1].seq == "A"
    assert splited_pairs[1][0].seq == "T"
    assert splited_pairs[1][1].seq == "AA"


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
    seq = VariantSeqExtractor(reference_sequence=ref_seq_extractor).extract(
        interval, variants, anchor=10, fixed_len=False
    )
    assert seq == "ACG"


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
        with pytest.raises(ValueError):
            variant_seq_extractor.extract(interval, [deletion], anchor=CHROM_LEN - 10, chrom_len=CHROM_LEN)


def test_extract_interval_beyond_chromosome_end(variant_seq_extractor):
    interval = Interval("chr1", CHROM_LEN - 5, CHROM_LEN + 5)
    seq = variant_seq_extractor.extract(interval, [], anchor=CHROM_LEN, chrom_len=CHROM_LEN, is_padding=True)
    assert seq == "ACGTANNNNN"
