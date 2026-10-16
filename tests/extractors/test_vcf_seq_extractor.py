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
    with FastaStringExtractor(fasta_file, use_strand=False) as fasta:
        yield VariantSeqExtractor(fasta, use_strand=True)


def test_variant_seq_extractor_rejects_other_references():
    # a path is no SequenceExtractor: pass a FastaStringExtractor
    with pytest.raises(TypeError, match="^reference must be a SequenceExtractor, not str$"):
        VariantSeqExtractor(fasta_file, use_strand=True)


def test_variant_seq_extractor_takes_any_sequence_extractor():
    class Reference:
        chrom_sizes = {"chr1": 5}

        def extract(self, interval):
            return "ACGTA"[interval.start : interval.end]

    extractor = VariantSeqExtractor(Reference(), use_strand=True)
    assert extractor.extract(Interval("chr1", 0, 5), [Variant("chr1", 2, "C", "T")], anchor=0) == "ATGTA"


def test_variant_seq_extractor_requires_use_strand_keyword():
    reference = FastaStringExtractor(fasta_file, use_strand=False)
    with pytest.raises(TypeError, match="use_strand"):
        VariantSeqExtractor(reference)
    with pytest.raises(TypeError, match="positional"):
        VariantSeqExtractor(reference, True)
    reference.close()


def test_variant_seq_extractor_extract_rejects_unknown_arguments(variant_seq_extractor):
    with pytest.raises(TypeError, match="fixed_length"):
        variant_seq_extractor.extract(Interval("chr1", 0, 3), [], anchor=0, fixed_length=False)


@pytest.mark.parametrize("strand", ["+", "-", "."])
def test_variant_seq_extractor_reference_with_use_strand(variant_seq_extractor, strand):
    # the reference gets intervals with the strand ".", so a reference with use_strand=True
    # does not reverse complement the sequence a second time
    variants = [Variant("chr1", 4, "T", "C"), Variant("chr1", 5, "A", "GA")]
    interval = Interval("chr1", 2, 9, strand=strand)
    expected = variant_seq_extractor.extract(interval, variants, anchor=5)
    assert expected == ("ACGTTCG" if strand == "-" else "CGAACGT")
    with FastaStringExtractor(fasta_file, use_strand=True) as reference:
        extractor = VariantSeqExtractor(reference, use_strand=True)
        assert extractor.extract(interval, variants, anchor=5) == expected
        assert extractor.extract(interval, variants, anchor=5, fixed_len=False) == variant_seq_extractor.extract(
            interval, variants, anchor=5, fixed_len=False
        )


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
    seq = variant_seq_extractor.extract(interval, [], anchor=CHROM_LEN - 5, is_padding=is_padding)
    assert seq == "ACGTAACGTA"

    # a deletion upstream of the anchor pulls in 2 bases after the chromosome end
    deletion = Variant("chr1", CHROM_LEN - 7, "GTA", "G")
    if is_padding:
        seq = variant_seq_extractor.extract(interval, [deletion], anchor=CHROM_LEN - 10, is_padding=True)
        assert seq == "ACGACGTANN"
    else:
        with pytest.raises(ValueError, match="needs bases outside chr1, which has 5000 bases. Pass is_padding=True"):
            variant_seq_extractor.extract(interval, [deletion], anchor=CHROM_LEN - 10)


def test_extract_interval_at_chromosome_start(variant_seq_extractor):
    # a deletion downstream of the anchor pulls in 2 bases before the chromosome start
    interval = Interval("chr1", 0, 10)
    deletion = Variant("chr1", 3, "GTA", "G")
    assert variant_seq_extractor.extract(interval, [deletion], anchor=10, is_padding=True) == "NNACGACGTA"
    with pytest.raises(ValueError, match="needs bases outside chr1"):
        variant_seq_extractor.extract(interval, [deletion], anchor=10)


@pytest.mark.parametrize("anchor", [CHROM_LEN - 5, CHROM_LEN, CHROM_LEN + 3])
def test_extract_interval_beyond_chromosome_end(variant_seq_extractor, anchor):
    # the N stand at the positions after the chromosome end, wherever the anchor is
    interval = Interval("chr1", CHROM_LEN - 5, CHROM_LEN + 5)
    assert variant_seq_extractor.extract(interval, [], anchor=anchor, is_padding=True) == "ACGTANNNNN"
    assert variant_seq_extractor.extract(interval, [], anchor=anchor, fixed_len=False, is_padding=True) == "ACGTANNNNN"
    with pytest.raises(ValueError, match="needs bases outside chr1"):
        variant_seq_extractor.extract(interval, [], anchor=anchor)


def test_extract_interval_before_chromosome_start(variant_seq_extractor):
    interval = Interval("chr1", -5, 5)
    assert variant_seq_extractor.extract(interval, [], anchor=-5, is_padding=True) == "NNNNNACGTA"
    assert variant_seq_extractor.extract(interval, [], anchor=0, fixed_len=False, is_padding=True) == "NNNNNACGTA"


def test_extract_unknown_chrom(variant_seq_extractor):
    with pytest.raises(ValueError, match="^chrom 'chr2' is not in the reference$"):
        variant_seq_extractor.extract(Interval("chr2", 0, 10), [], anchor=0)
