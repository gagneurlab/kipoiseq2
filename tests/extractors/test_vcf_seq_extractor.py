import polars as pl
import pytest
from conftest import sample_5kb_fasta_file

from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.extractors import FastaStringExtractor, scan_variant_pairs, scan_vcf_variants
from kipoiseq2.extractors.vcf_seq import VariantSeqExtractor

fasta_file = sample_5kb_fasta_file

intervals = [Interval("chr1", 4, 10), Interval("chr1", 5, 30), Interval("chr1", 20, 30)]


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
    assert extractor.extract(Interval("chr1", 0, 5), [Variant("chr1", 2, "C", "T")], anchor_pos=0) == "ATGTA"


def test_variant_seq_extractor_requires_use_strand_keyword():
    reference = FastaStringExtractor(fasta_file, use_strand=False)
    with pytest.raises(TypeError, match="use_strand"):
        VariantSeqExtractor(reference)
    with pytest.raises(TypeError, match="positional"):
        VariantSeqExtractor(reference, True)
    reference.close()


@pytest.mark.parametrize("name", ["fixed_length", "anchor", "fixed_len", "is_padding", "use_strand"])
def test_variant_seq_extractor_extract_rejects_unknown_arguments(variant_seq_extractor, name):
    # the options of 0.1 are gone too
    with pytest.raises(TypeError, match=name):
        variant_seq_extractor.extract(Interval("chr1", 0, 3), [], **{name: False})


def test_variant_seq_extractor_extract_options_are_keyword_only(variant_seq_extractor):
    with pytest.raises(TypeError, match="positional"):
        variant_seq_extractor.extract(Interval("chr1", 0, 3), [], 0)


@pytest.mark.parametrize("strand", ["+", "-", "."])
def test_variant_seq_extractor_reference_with_use_strand(variant_seq_extractor, strand):
    # the reference gets intervals with the strand ".", so a reference with use_strand=True
    # does not reverse complement the sequence a second time
    variants = [Variant("chr1", 4, "T", "C"), Variant("chr1", 5, "A", "GA")]
    interval = Interval("chr1", 2, 9, strand=strand)
    expected = variant_seq_extractor.extract(interval, variants, anchor_pos=5)
    assert expected == ("ACGTTCG" if strand == "-" else "CGAACGT")
    with FastaStringExtractor(fasta_file, use_strand=True) as reference:
        extractor = VariantSeqExtractor(reference, use_strand=True)
        assert extractor.extract(interval, variants, anchor_pos=5) == expected
        assert extractor.extract(interval, variants) == variant_seq_extractor.extract(interval, variants)


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

    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=28, pad_with_n=True)
    assert len(seq) == interval.end - interval.start
    assert seq == "NACGCGAACGTAACGTAACGTAACGTGATA"

    interval = Interval("chr1", 2, 9)

    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=5)
    assert len(seq) == interval.end - interval.start
    assert seq == "CGAACGT"

    interval = Interval("chr1", 2, 9, strand="-")
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=5)
    assert len(seq) == interval.end - interval.start
    assert seq == "ACGTTCG"

    interval = Interval("chr1", 4, 14)
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=7)
    assert len(seq) == interval.end - interval.start
    assert seq == "AACGTAACGT"

    interval = Interval("chr1", 4, 14)
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=4)
    assert len(seq) == interval.end - interval.start
    assert seq == "GAACGTAACG"

    interval = Interval("chr1", 2, 5)
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=3)
    assert len(seq) == interval.end - interval.start
    assert seq == "GCG"

    interval = Interval("chr1", 24, 34)
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=27)
    assert len(seq) == interval.end - interval.start
    assert seq == "TGATAACGTA"

    interval = Interval("chr1", 25, 35)
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=34)
    assert len(seq) == interval.end - interval.start
    assert seq == "TGATAACGTA"

    interval = Interval("chr1", 34, 44)
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=37)
    assert len(seq) == interval.end - interval.start
    assert seq == "AACGTAACGT"

    interval = Interval("chr1", 34, 44)
    seq = variant_seq_extractor.extract(interval, variants, anchor_pos=44)
    assert len(seq) == interval.end - interval.start
    assert seq == "AACGTAACGT"

    # without anchor_pos, the insertion with its REF before the interval has no effect
    interval = Interval("chr1", 5, 11, strand="+")
    seq = variant_seq_extractor.extract(interval, variants)
    assert seq == "ACGTAA"

    # the insertion with its REF in the interval makes the sequence longer
    interval = Interval("chr1", 4, 11, strand="+")
    seq = variant_seq_extractor.extract(interval, variants)
    assert seq == "GAACGTAA"

    # only the part of the deletion within the interval applies
    interval = Interval("chr1", 26, 30, strand="+")
    seq = variant_seq_extractor.extract(interval, variants)
    assert seq == "TA"

    interval = Interval("chr1", 0, 3, strand="+")
    seq = variant_seq_extractor.extract(interval, variants)
    assert seq == "ACG"


def test_extract_vcf_variants(variant_seq_extractor, alt_kinds_vcf):
    # the interval covers the records of ALT_KINDS_VCF up to the breakend
    interval = Interval("chr1", 0, 21)
    pairs = scan_variant_pairs(pl.DataFrame([interval]), scan_vcf_variants(alt_kinds_vcf)).collect()
    variants = [
        Variant(r["chrom"], r["variant_pos"], r["variant_ref"], r["variant_alt"])
        for r in pairs.sort("variant_idx").iter_rows(named=True)
    ]
    seq = variant_seq_extractor.extract(interval, variants)
    # the reference ACGTAACGTAACGTAACGTAA with C>A at 2, T>ACG at 4 and A>AN at 6.
    # The ALT ".", "*", symbolic ALT alleles and the breakend change nothing.
    assert seq == "AAGACGAANCGTAACGTAACGTAA"


def test_extract_anchor_rule(variant_seq_extractor):
    # the reference of the interval is ACGTAACGTA
    interval = Interval("chr1", 0, 10)
    # the REF of the insertion lies at anchor_pos - 1, so the insertion lies left and pushes out bases at the start.
    # GenomeKit would put it right of anchor_pos.
    insertion = Variant("chr1", 5, "A", "AGGG")
    assert variant_seq_extractor.extract(interval, [insertion], anchor_pos=5) == "TAGGGACGTA"
    # the REF of the insertion lies at anchor_pos, so the insertion lies right and pushes out bases at the end
    assert variant_seq_extractor.extract(interval, [insertion], anchor_pos=4) == "ACGTAGGGAC"
    # the REF TAA spans anchor_pos: TA becomes GC left of anchor_pos, and the last A is deleted right of it
    mnv = Variant("chr1", 4, "TAA", "GC")
    assert variant_seq_extractor.extract(interval, [mnv], anchor_pos=5) == "ACGGCCGTAA"


@pytest.mark.parametrize("anchor_pos", [20, 25, 30])
def test_extract_deletion_before_interval(variant_seq_extractor, anchor_pos):
    # the deletion lies before the interval, so the sequence is the reference of the interval
    interval = Interval("chr1", 20, 30)
    deletion = Variant("chr1", 12, "CG", "C")
    seq = variant_seq_extractor.extract(interval, [deletion], anchor_pos=anchor_pos)
    assert len(seq) == interval.end - interval.start
    assert seq == "ACGTAACGTA"


# sample.5kb.fa has one 5000 bp chromosome that repeats ACGTA
CHROM_LEN = 5000


@pytest.mark.parametrize("pad_with_n", [False, True])
def test_extract_interval_at_chromosome_end(variant_seq_extractor, pad_with_n):
    # the interval covers the last 10 bases, so no padding is needed
    interval = Interval("chr1", CHROM_LEN - 10, CHROM_LEN)
    seq = variant_seq_extractor.extract(interval, [], anchor_pos=CHROM_LEN - 5, pad_with_n=pad_with_n)
    assert seq == "ACGTAACGTA"

    # a deletion upstream of the anchor pulls in 2 bases after the chromosome end
    deletion = Variant("chr1", CHROM_LEN - 7, "GTA", "G")
    if pad_with_n:
        seq = variant_seq_extractor.extract(interval, [deletion], anchor_pos=CHROM_LEN - 10, pad_with_n=True)
        assert seq == "ACGACGTANN"
    else:
        with pytest.raises(ValueError, match="needs bases outside chr1, which has 5000 bases. Pass pad_with_n=True"):
            variant_seq_extractor.extract(interval, [deletion], anchor_pos=CHROM_LEN - 10)


def test_extract_interval_at_chromosome_start(variant_seq_extractor):
    # a deletion downstream of the anchor pulls in 2 bases before the chromosome start
    interval = Interval("chr1", 0, 10)
    deletion = Variant("chr1", 3, "GTA", "G")
    assert variant_seq_extractor.extract(interval, [deletion], anchor_pos=10, pad_with_n=True) == "NNACGACGTA"
    with pytest.raises(ValueError, match="needs bases outside chr1"):
        variant_seq_extractor.extract(interval, [deletion], anchor_pos=10)


@pytest.mark.parametrize("anchor_pos", [CHROM_LEN - 5, CHROM_LEN, CHROM_LEN + 3])
def test_extract_interval_beyond_chromosome_end(variant_seq_extractor, anchor_pos):
    # the N stand at the positions after the chromosome end, wherever the anchor is
    interval = Interval("chr1", CHROM_LEN - 5, CHROM_LEN + 5)
    assert variant_seq_extractor.extract(interval, [], anchor_pos=anchor_pos, pad_with_n=True) == "ACGTANNNNN"
    # also without anchor_pos, the interval needs bases after the chromosome end
    assert variant_seq_extractor.extract(interval, [], pad_with_n=True) == "ACGTANNNNN"
    with pytest.raises(ValueError, match="needs bases outside chr1"):
        variant_seq_extractor.extract(interval, [], anchor_pos=anchor_pos)


def test_extract_interval_before_chromosome_start(variant_seq_extractor):
    interval = Interval("chr1", -5, 5)
    assert variant_seq_extractor.extract(interval, [], anchor_pos=-5, pad_with_n=True) == "NNNNNACGTA"
    assert variant_seq_extractor.extract(interval, [], pad_with_n=True) == "NNNNNACGTA"


def test_extract_unknown_chrom(variant_seq_extractor):
    with pytest.raises(ValueError, match="^chrom 'chr2' is not in the reference$"):
        variant_seq_extractor.extract(Interval("chr2", 0, 10), [], anchor_pos=0)


def test_extract_rejects_wrong_types(variant_seq_extractor):
    with pytest.raises(TypeError, match="^interval must be an Interval, not str$"):
        variant_seq_extractor.extract("chr1:0-10:+", [], anchor_pos=0)
    with pytest.raises(TypeError, match="^variants must be Variants, not str$"):
        variant_seq_extractor.extract(Interval("chr1", 0, 10), ["chr1:2:C>T"], anchor_pos=0)


def test_extract_variant_on_other_chrom(variant_seq_extractor):
    with pytest.raises(ValueError, match="^variant chr2:2:C>T lies on another chromosome than the interval chr1:0-10"):
        variant_seq_extractor.extract(Interval("chr1", 0, 10), [Variant("chr2", 2, "C", "T")], anchor_pos=0)


def test_extract_variant_outside_chromosome(variant_seq_extractor):
    variant = Variant("chr1", CHROM_LEN, "AC", "A")
    with pytest.raises(ValueError, match="^variant chr1:5000:AC>A lies outside chr1, which has 5000 bases$"):
        variant_seq_extractor.extract(Interval("chr1", CHROM_LEN - 10, CHROM_LEN), [variant], anchor_pos=CHROM_LEN)


def test_extract_ref_differs(variant_seq_extractor):
    # the reference has a G at position 3
    with pytest.raises(ValueError, match="^the REF of variant chr1:3:A>T differs from the reference 'G'$"):
        variant_seq_extractor.extract(Interval("chr1", 0, 10), [Variant("chr1", 3, "A", "T")], anchor_pos=0)
    # the REF of a variant outside the interval is checked too
    with pytest.raises(ValueError, match="differs from the reference 'GT'"):
        variant_seq_extractor.extract(Interval("chr1", 20, 30), [Variant("chr1", 13, "TA", "T")], anchor_pos=20)


# the reference from 0 to 10 is ACGTAACGTA
OVERLAPPING_VARIANTS = {
    "same_base": [Variant("chr1", 5, "A", "C"), Variant("chr1", 5, "A", "G")],
    "snv_in_deletion": [Variant("chr1", 5, "AAC", "A"), Variant("chr1", 6, "A", "T")],
    "snv_at_deletion_end": [Variant("chr1", 2, "CGTAAC", "C"), Variant("chr1", 7, "C", "T")],
    "insertions_at_one_base": [Variant("chr1", 5, "A", "AT"), Variant("chr1", 5, "A", "AGG")],
    "duplicate": [Variant("chr1", 5, "A", "C"), Variant("chr1", 5, "A", "C")],
}


@pytest.mark.parametrize("variants", OVERLAPPING_VARIANTS.values(), ids=OVERLAPPING_VARIANTS.keys())
@pytest.mark.parametrize("anchor_pos", [None, 0, 5, 10])
def test_extract_overlapping_variants(variant_seq_extractor, variants, anchor_pos):
    # in 0.1, both ALTs were inserted, so the sequence could be too long
    for order in (variants, variants[::-1]):
        with pytest.raises(ValueError, match=r"^variants chr1:\S+ and chr1:\S+ overlap, but the REF of two variants"):
            variant_seq_extractor.extract(Interval("chr1", 0, 10), order, anchor_pos=anchor_pos)


def test_extract_adjacent_variants(variant_seq_extractor):
    # variants that touch but do not share a base do not overlap
    variants = [Variant("chr1", 5, "A", "T"), Variant("chr1", 6, "AC", "A"), Variant("chr1", 4, "T", "TT")]
    assert variant_seq_extractor.extract(Interval("chr1", 0, 10), variants) == "ACGTTTAGTA"


@pytest.mark.parametrize("ref", ["C", "c"])
def test_extract_ref_ignores_case(ref):
    # sample.fasta has the sequence ACGTacgt, so with uppercase=False the reference has a lowercase c at position 6
    with FastaStringExtractor("tests/data/sample.fasta", use_strand=False, uppercase=False) as fasta:
        extractor = VariantSeqExtractor(fasta, use_strand=True)
        assert extractor.extract(Interval("chr1", 0, 8), [Variant("chr1", 6, ref, "T")], anchor_pos=0) == "ACGTaTgt"


@pytest.mark.parametrize("anchor_pos", [19, 31, -1])
def test_extract_anchor_pos_outside_interval(variant_seq_extractor, anchor_pos):
    with pytest.raises(
        ValueError, match=f"^anchor_pos must lie within 20 and 30, the bounds of chr1:20-30:., got {anchor_pos}$"
    ):
        variant_seq_extractor.extract(Interval("chr1", 20, 30), [], anchor_pos=anchor_pos)


@pytest.mark.parametrize("anchor_pos", [20.0, "20", True])
def test_extract_anchor_pos_not_int(variant_seq_extractor, anchor_pos):
    with pytest.raises(TypeError, match="^anchor_pos must be an int, not"):
        variant_seq_extractor.extract(Interval("chr1", 20, 30), [], anchor_pos=anchor_pos)
