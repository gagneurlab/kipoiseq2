import random

import numpy as np
import polars as pl
import pytest
from conftest import sample_5kb_fasta_file

from kipoiseq2.alphabets import Alphabet
from kipoiseq2.dataclasses import Interval, Variant
from kipoiseq2.encoders import Encoder, Tokenizer
from kipoiseq2.extractors import (
    FastaStringExtractor,
    VariantSeqExtractor,
    iter_variant_seqs,
    scan_variant_pairs,
    scan_vcf_genotypes,
    scan_vcf_variants,
)
from kipoiseq2.transforms import reverse_complement

FLOAT32_ONE_HOT = Encoder(ambiguous="fractional", dtype=np.float32)
"""float32 one-hot arrays, with N as 0.25 in every column."""

# random sequences, so that a shifted sequence differs from the original
_rng = random.Random(0)
CHROMS = {"chr1": "".join(_rng.choices("ACGT", k=200)), "chr2": "".join(_rng.choices("ACGT", k=60))}


class SeqReference:
    """SequenceExtractor of the sequences in CHROMS, which records the intervals that it extracts."""

    def __init__(self, seqs=CHROMS):
        self.seqs = seqs
        self.calls = []

    @property
    def chrom_sizes(self):
        return {chrom: len(seq) for chrom, seq in self.seqs.items()}

    def extract(self, interval):
        self.calls.append(interval)
        return self.seqs[interval.chrom][interval.start : interval.end]


def variant(chrom, pos, ref_len, alt):
    """Return the variant at ``pos`` with the REF of CHROMS. ``alt`` may use ``{ref}`` and ``{first}``."""
    ref = CHROMS[chrom][pos - 1 : pos - 1 + ref_len]
    return Variant(chrom, pos, ref, alt.format(ref=ref, first=ref[0]))


# The variants do not overlap each other, so all variants of an interval can apply together.
VARIANTS = [
    variant("chr1", 9, 3, "{first}"),  # deletion across the start of the first interval
    variant("chr1", 12, 1, "N"),
    variant("chr1", 20, 1, "{first}GAT"),
    variant("chr1", 25, 3, "{first}"),
    variant("chr1", 31, 2, "NN"),
    variant("chr1", 38, 5, "{first}"),  # deletion across the end of the first interval
    variant("chr1", 55, 1, "{first}CC"),
    variant("chr1", 101, 1, "N"),
    variant("chr1", 190, 1, "N"),  # in no interval
    variant("chr2", 8, 1, "N"),
    variant("chr2", 15, 6, "{first}"),
    variant("chr2", 29, 1, "{first}A"),
]

# widths 30, 21, 30, 30, 1 and 10, the last interval without variants
INTERVALS = pl.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr1", "chr2", "chr1", "chr1"],
        "start": [10, 15, 40, 6, 100, 150],
        "end": [40, 36, 70, 36, 101, 160],
        "strand": ["+", "-", ".", "-", "+", "+"],
        "name": ["a", "b", "c", "d", "e", "f"],
    }
)

# the intervals of the same width 30
SAME_WIDTH = INTERVALS.filter(pl.col("end") - pl.col("start") == 30)


def pairs_of(intervals, variants=VARIANTS):
    return scan_variant_pairs(intervals, pl.DataFrame(variants))


def collect(batches):
    return pl.concat(list(batches))


def expected_anchor_pos(interval, anchor, use_strand):
    """Return the position that Interval.resize keeps for ``anchor``."""
    return interval.resize(0, anchor=anchor, use_strand=use_strand).start


@pytest.mark.parametrize("use_strand", [True, False])
@pytest.mark.parametrize("anchor", ["start", "center", "end", None])
@pytest.mark.parametrize("group_by", [("interval_idx", "variant_idx"), ("interval_idx",)])
def test_iter_variant_seqs_equals_extract(use_strand, anchor, group_by):
    pairs = pairs_of(INTERVALS).collect()
    reference = SeqReference()
    out = collect(iter_variant_seqs(pairs, reference, use_strand=use_strand, anchor=anchor, group_by=group_by))
    # one batch, so one fetch per interval
    assert len(reference.calls) == out.select("chrom", "start", "end").n_unique()

    assert len(out) == pairs.select(group_by).n_unique()
    assert out.select(group_by).rows() == sorted(out.select(group_by).rows())
    extractor = VariantSeqExtractor(reference, use_strand=use_strand)
    for row in out.iter_rows(named=True):
        group = pairs.filter(*(pl.col(k) == row[k] for k in group_by))
        variants = [
            Variant(r["chrom"], r["variant_pos"], r["variant_ref"], r["variant_alt"])
            for r in group.iter_rows(named=True)
        ]
        interval = Interval(row["chrom"], row["start"], row["end"], strand=row["strand"])
        anchor_pos = None if anchor is None else expected_anchor_pos(interval, anchor, use_strand)
        assert row["alt_seq"] == extractor.extract(interval, variants, anchor_pos=anchor_pos)
        ref_seq = reference.extract(interval)
        assert row["ref_seq"] == (reverse_complement(ref_seq) if use_strand and interval.strand == "-" else ref_seq)
        assert row["ref"] == Tokenizer().tokenize(row["ref_seq"]).tolist()
        assert row["alt"] == Tokenizer().tokenize(row["alt_seq"]).tolist()


@pytest.mark.parametrize("use_strand", [True, False])
@pytest.mark.parametrize("anchor", ["start", "center", "end", None])
def test_iter_variant_seqs_equals_fasta_extract(alt_kinds_vcf, use_strand, anchor):
    # the reference is a FastaStringExtractor with use_strand=True, which gets intervals with the strand "."
    intervals = pl.DataFrame({"chrom": ["chr1"] * 3, "start": [0, 3, 10], "end": [12, 13, 22], "strand": list("+-.")})
    pairs = scan_variant_pairs(intervals, scan_vcf_variants(alt_kinds_vcf))
    with FastaStringExtractor(sample_5kb_fasta_file, use_strand=True) as reference:
        out = collect(iter_variant_seqs(pairs, reference, use_strand=use_strand, anchor=anchor))
    pairs = pairs.collect()
    assert len(out) == len(pairs)
    with FastaStringExtractor(sample_5kb_fasta_file, use_strand=use_strand) as fasta:
        extractor = VariantSeqExtractor(fasta, use_strand=use_strand)
        for row in out.iter_rows(named=True):
            pair = pairs.row(
                by_predicate=(pl.col("interval_idx") == row["interval_idx"])
                & (pl.col("variant_idx") == row["variant_idx"]),
                named=True,
            )
            interval = Interval(row["chrom"], row["start"], row["end"], strand=row["strand"])
            variants = [Variant(row["chrom"], pair["variant_pos"], pair["variant_ref"], pair["variant_alt"])]
            anchor_pos = None if anchor is None else expected_anchor_pos(interval, anchor, use_strand)
            assert row["alt_seq"] == extractor.extract(interval, variants, anchor_pos=anchor_pos)
            assert row["ref_seq"] == fasta.extract(interval)


def test_iter_variant_seqs_haplotypes(tmp_path):
    # S1 carries the first variant on haplotype 1 and the other two on haplotype 0. S3 carries none.
    records = [
        (variant("chr1", 12, 1, "N"), "0|1", "1|1", "0|0"),
        (variant("chr1", 20, 1, "{first}GAT"), "1|0", "0|0", "0|0"),
        (variant("chr1", 25, 3, "{first}"), "1|0", "0|0", "0|0"),
    ]
    header = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1,length=200>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3",
    ]
    lines = ["\t".join(["chr1", str(v.pos), ".", v.ref, v.alt, ".", ".", ".", "GT", *gts]) for v, *gts in records]
    vcf = tmp_path / "haplotypes.vcf"
    vcf.write_text("\n".join(header + lines) + "\n")
    intervals = pl.DataFrame({"chrom": ["chr1"], "start": [10], "end": [40], "strand": ["+"]})
    pairs = scan_variant_pairs(intervals, scan_vcf_genotypes(str(vcf), haplotypes=True))
    reference = SeqReference()
    group_by = ["interval_idx", "variant_sample", "variant_haplotype"]
    out = collect(iter_variant_seqs(pairs, reference, use_strand=True, group_by=group_by))
    # the haplotypes without variants in the interval, e.g. those of S3, give no row
    assert out.select("variant_sample", "variant_haplotype").rows() == [("S1", 0), ("S1", 1), ("S2", 0), ("S2", 1)]
    v1, v2, v3 = (v for v, *_ in records)
    interval = Interval("chr1", 10, 40, strand="+")
    extractor = VariantSeqExtractor(reference, use_strand=True)
    expected = [extractor.extract(interval, vs, anchor_pos=interval.center()) for vs in [[v2, v3], [v1], [v1], [v1]]]
    assert out["alt_seq"].to_list() == expected


@pytest.mark.parametrize(
    "anchor, use_strand, alt_seq",
    [
        # anchor_pos 12, the 5' end of the "-" interval: the deletion pulls in the base at 3
        ("start", True, reverse_complement("ACCCGGGG")),
        # anchor_pos 4: the deletion pulls in the base at 12
        ("end", True, reverse_complement("CCCGGGGT")),
        ("start", False, "CCCGGGGT"),
        ("end", False, "ACCCGGGG"),
        # the center 8 lies right of the deletion
        ("center", True, reverse_complement("ACCCGGGG")),
        (None, True, reverse_complement("CCCGGGG")),
    ],
)
def test_iter_variant_seqs_anchor_follows_strand(anchor, use_strand, alt_seq):
    reference = SeqReference({"t": "AAAACCCCGGGGTTTT"})
    pairs = pl.DataFrame(
        {
            "chrom": ["t"],
            "start": [4],
            "end": [12],
            "strand": ["-"],
            "interval_idx": [0],
            "variant_pos": [7],
            "variant_ref": ["CC"],
            "variant_alt": ["C"],
            "variant_idx": [0],
        }
    )
    (batch,) = iter_variant_seqs(pairs, reference, use_strand=use_strand, anchor=anchor)
    assert batch["ref_seq"].to_list() == [reverse_complement("CCCCGGGG") if use_strand else "CCCCGGGG"]
    assert batch["alt_seq"].to_list() == [alt_seq]


def test_iter_variant_seqs_columns():
    pairs = pairs_of(SAME_WIDTH)
    (batch,) = iter_variant_seqs(pairs, SeqReference(), use_strand=True)
    assert batch.schema == pl.Schema(
        {
            "interval_idx": pl.UInt32,
            "variant_idx": pl.UInt32,
            "chrom": pl.String,
            "start": pl.Int64,
            "end": pl.Int64,
            "strand": pl.String,
            "ref_seq": pl.String,
            "alt_seq": pl.String,
            "ref": pl.Array(pl.UInt8, 30),
            "alt": pl.Array(pl.UInt8, 30),
        }
    )


def test_iter_variant_seqs_group_by_interval_columns():
    # the keys come first, and the interval columns that are keys appear once
    pairs = pairs_of(SAME_WIDTH)
    (batch,) = iter_variant_seqs(pairs, SeqReference(), use_strand=True, group_by=["name", "chrom"], encoding=None)
    assert batch.columns == ["name", "chrom", "start", "end", "strand", "ref_seq", "alt_seq"]
    assert batch["name"].to_list() == ["a", "c", "d"]


def test_iter_variant_seqs_several_variants_per_interval():
    pairs = pairs_of(INTERVALS.filter(pl.col("name") == "a"))
    (batch,) = iter_variant_seqs(pairs, SeqReference(), use_strand=True, anchor=None, group_by=["interval_idx"])
    # the deletions across the ends of the interval keep only their bases within the interval
    seq = CHROMS["chr1"]
    parts = ["N", seq[12:19], seq[19], "GAT", seq[20:24], seq[24], seq[27:30], "NN", seq[32:37], seq[37]]
    assert batch["alt_seq"].to_list() == ["".join(parts)]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 7, 8, 100])
def test_iter_variant_seqs_batch_size(batch_size):
    pairs = pairs_of(INTERVALS)
    n_rows = len(pairs.collect())
    batches = list(iter_variant_seqs(pairs, SeqReference(), use_strand=True, batch_size=batch_size))
    assert [len(b) for b in batches] == [min(batch_size, n_rows - i) for i in range(0, n_rows, batch_size)]
    assert collect(batches).equals(
        collect(iter_variant_seqs(pairs, SeqReference(), use_strand=True, batch_size=n_rows))
    )


def test_iter_variant_seqs_fetches_each_interval_once_per_batch():
    # the first interval has 6 pairs
    pairs = pairs_of(INTERVALS.filter(pl.col("name") == "a"))
    for batch_size, n_batches in [(6, 1), (4, 2), (1, 6)]:
        reference = SeqReference()
        batches = list(iter_variant_seqs(pairs, reference, use_strand=True, batch_size=batch_size))
        assert len(batches) == n_batches
        assert len(reference.calls) == n_batches


def test_iter_variant_seqs_empty_pairs():
    pairs = pairs_of(INTERVALS.filter(pl.col("name") == "f"))
    assert pairs.collect().is_empty()
    assert list(iter_variant_seqs(pairs, SeqReference(), use_strand=True)) == []


def test_iter_variant_seqs_tokens_round_trip():
    (batch,) = iter_variant_seqs(pairs_of(SAME_WIDTH), SeqReference(), use_strand=True, batch_size=100)
    for column, seq_column in [("ref", "ref_seq"), ("alt", "alt_seq")]:
        tokens = batch[column].to_numpy()
        assert tokens.shape == (len(batch), 30)
        assert tokens.dtype == np.uint8
        # np.eye(15, 4) has the row of Encoder() for each token of DNA, and decode gives N for a row of zeros
        assert [Encoder().decode(np.eye(15, 4, dtype=np.uint8)[t]) for t in tokens] == batch[seq_column].to_list()


def test_iter_variant_seqs_one_hot_round_trip():
    (batch,) = iter_variant_seqs(
        pairs_of(SAME_WIDTH), SeqReference(), use_strand=True, encoding=FLOAT32_ONE_HOT, batch_size=100
    )
    assert batch.schema["alt"] == pl.Array(pl.Float32, (30, 4))
    alt = batch["alt"].to_numpy()
    assert alt.shape == (len(batch), 30, 4)
    assert alt.dtype == np.float32
    assert [FLOAT32_ONE_HOT.decode(a) for a in alt] == batch["alt_seq"].to_list()
    np.testing.assert_array_equal(alt, np.stack([FLOAT32_ONE_HOT.one_hot(s) for s in batch["alt_seq"]]))


@pytest.mark.parametrize("encoding", [Tokenizer(), FLOAT32_ONE_HOT])
def test_iter_variant_seqs_to_numpy_without_copy(encoding):
    (batch,) = iter_variant_seqs(pairs_of(SAME_WIDTH), SeqReference(), use_strand=True, encoding=encoding)
    # each call returns a view of the same polars buffer
    assert np.shares_memory(batch["alt"].to_numpy(), batch["alt"].to_numpy())
    assert not batch["alt"].to_numpy().flags.writeable


# a Tokenizer gives tokens and an Encoder gives one-hot arrays, both in their dtype
@pytest.mark.parametrize(
    "encoding, polars_dtype",
    [
        (Tokenizer(), pl.Array(pl.UInt8, 30)),
        (Tokenizer(dtype=np.int64), pl.Array(pl.Int64, 30)),
        (Encoder(), pl.Array(pl.UInt8, (30, 4))),
        (FLOAT32_ONE_HOT, pl.Array(pl.Float32, (30, 4))),
        (Encoder(ambiguous="fractional", dtype="float64"), pl.Array(pl.Float64, (30, 4))),
        (Encoder(dtype=np.float16), pl.Array(pl.Float16, (30, 4))),
    ],
)
def test_iter_variant_seqs_dtype(encoding, polars_dtype):
    batches = iter_variant_seqs(pairs_of(SAME_WIDTH), SeqReference(), use_strand=True, encoding=encoding)
    assert collect(batches).schema["alt"] == polars_dtype


@pytest.mark.parametrize("anchor", ["center", None])
def test_iter_variant_seqs_letters_of_two_bases(anchor):
    base_pairs = Tokenizer(Alphabet([a + b for a in "ACGTN" for b in "ACGTN"]))
    batches = iter_variant_seqs(
        pairs_of(SAME_WIDTH), SeqReference(), use_strand=True, anchor=anchor, encoding=base_pairs
    )
    if anchor is None:
        # the first pair has the deletion of 2 bases across the start of the interval, which leaves 29 bases
        with pytest.raises(ValueError, match="multiple of the letter length 2") as excinfo:
            list(batches)
        assert excinfo.value.__notes__ == ["in the group {'interval_idx': 0, 'variant_idx': 0} of pairs"]
    else:
        (batch,) = batches
        assert batch.schema["alt"] == pl.Array(pl.UInt8, 15)
        assert batch["alt"].to_list() == [base_pairs.tokenize(s).tolist() for s in batch["alt_seq"]]


@pytest.mark.parametrize("encoding", [Tokenizer(), FLOAT32_ONE_HOT])
def test_iter_variant_seqs_unknown_letter(encoding):
    # X at position 50 lies in the second interval of SAME_WIDTH, but in no REF
    seq = CHROMS["chr1"]
    reference = SeqReference({**CHROMS, "chr1": seq[:50] + "X" + seq[51:]})
    with pytest.raises(ValueError, match=r"not in the alphabet: \['X'\]") as excinfo:
        list(iter_variant_seqs(pairs_of(SAME_WIDTH), reference, use_strand=True, encoding=encoding))
    assert excinfo.value.__notes__ == ["in the group {'interval_idx': 1, 'variant_idx': 5} of pairs"]


@pytest.mark.parametrize("encoding", [Tokenizer(), FLOAT32_ONE_HOT])
def test_iter_variant_seqs_variable_length(encoding):
    # with anchor=None, the indels change the length
    out = collect(
        iter_variant_seqs(pairs_of(SAME_WIDTH), SeqReference(), use_strand=True, anchor=None, encoding=encoding)
    )
    inner = pl.UInt8 if isinstance(encoding, Tokenizer) else pl.Array(pl.Float32, 4)
    assert out.schema["alt"] == pl.List(inner)
    assert out["alt"].list.len().to_list() == out["alt_seq"].str.len_chars().to_list()
    assert len(set(out["alt_seq"].str.len_chars())) > 1
    for row in out.iter_rows(named=True):
        expected = (
            encoding.tokenize(row["alt_seq"]) if isinstance(encoding, Tokenizer) else encoding.one_hot(row["alt_seq"])
        )
        np.testing.assert_array_equal(np.array(row["alt"]), expected)


def test_iter_variant_seqs_list_for_different_widths():
    # with an anchor, alt_seq has the width of the interval, but the widths differ
    out = collect(iter_variant_seqs(pairs_of(INTERVALS), SeqReference(), use_strand=True))
    assert out.schema["alt"] == pl.List(pl.UInt8)
    assert out["alt"].list.len().to_list() == (out["end"] - out["start"]).to_list()


@pytest.mark.parametrize(
    "encoding, polars_dtype", [(Tokenizer(), pl.List(pl.UInt8)), (FLOAT32_ONE_HOT, pl.List(pl.Array(pl.Float32, 4)))]
)
def test_iter_variant_seqs_list_with_equal_lengths_in_batch(encoding, polars_dtype):
    # a batch whose sequences have equal lengths still gives a List column
    batches = list(
        iter_variant_seqs(pairs_of(INTERVALS), SeqReference(), use_strand=True, encoding=encoding, batch_size=1)
    )
    assert {b.schema[c] for b in batches for c in ["ref", "alt"]} == {polars_dtype}
    out = pl.concat(batches)
    encode = encoding.tokenize if isinstance(encoding, Tokenizer) else encoding.one_hot
    for column, seq_column in [("ref", "ref_seq"), ("alt", "alt_seq")]:
        assert out[column].to_list() == [encode(seq).tolist() for seq in out[seq_column]]


def test_iter_variant_seqs_no_encoding():
    (batch,) = iter_variant_seqs(pairs_of(SAME_WIDTH), SeqReference(), use_strand=True, encoding=None)
    assert batch.columns[-2:] == ["ref_seq", "alt_seq"]


@pytest.mark.parametrize("pad_with_n", [False, True])
def test_iter_variant_seqs_pad_with_n(pad_with_n):
    # the deletion pulls in bases from beyond the end of chr2
    intervals = pl.DataFrame({"chrom": ["chr2"], "start": [50], "end": [60], "strand": ["+"]})
    pairs = pairs_of(intervals, [variant("chr2", 55, 3, "{first}")])
    batches = iter_variant_seqs(pairs, SeqReference(), use_strand=True, anchor="start", pad_with_n=pad_with_n)
    if pad_with_n:
        assert collect(batches)["alt_seq"].to_list() == [CHROMS["chr2"][50:55] + CHROMS["chr2"][57:] + "NN"]
    else:
        with pytest.raises(ValueError, match="needs bases outside chr2") as excinfo:
            list(batches)
        assert excinfo.value.__notes__ == ["in the group {'interval_idx': 0, 'variant_idx': 0} of pairs"]


def test_iter_variant_seqs_ref_seq_past_chromosome_end():
    # the window misses an interval past the end of chr2, so each pair gets its ref_seq from extract
    intervals = pl.DataFrame({"chrom": ["chr2"], "start": [50], "end": [70], "strand": ["-"]})
    pairs = pairs_of(intervals, [variant("chr2", 52, 1, "N"), variant("chr2", 56, 1, "N")])
    reference = SeqReference()
    out = collect(iter_variant_seqs(pairs, reference, use_strand=True, pad_with_n=True))
    interval = Interval("chr2", 50, 70, strand="-")
    expected = VariantSeqExtractor(reference, use_strand=True).extract(interval, [], pad_with_n=True)
    assert expected == reverse_complement(CHROMS["chr2"][50:] + "N" * 10)
    assert out["ref_seq"].to_list() == [expected, expected]


def test_iter_variant_seqs_rejects_non_carriers(alt_kinds_vcf):
    # with carriers_only=False, the genotypes contain samples that do not carry the ALT allele
    intervals = pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [30]})
    genotypes = scan_vcf_genotypes(alt_kinds_vcf, carriers_only=False)
    pairs = scan_variant_pairs(intervals, genotypes)
    message = "^pairs has rows whose variant_carrier is False or null, .* carriers_only=True"
    with pytest.raises(ValueError, match=message):
        iter_variant_seqs(pairs, SeqReference(), use_strand=True)
    carriers = pairs.filter("variant_carrier").collect()
    assert len(carriers) == len(scan_variant_pairs(intervals, scan_vcf_genotypes(alt_kinds_vcf)).collect())
    # a null does not tell that the sample carries the ALT allele
    unknown = carriers.with_columns(pl.lit(None, dtype=pl.Boolean).alias("variant_carrier"))
    with pytest.raises(ValueError, match=message):
        iter_variant_seqs(unknown, SeqReference(), use_strand=True)
    with FastaStringExtractor(sample_5kb_fasta_file, use_strand=False) as fasta:
        out = collect(iter_variant_seqs(carriers, fasta, use_strand=True, encoding=None))
    assert len(out) == len(carriers)


def test_iter_variant_seqs_overlapping_variants_in_group():
    # the variants of interval_idx 0 do not overlap, but these two do
    variants = [variant("chr1", 12, 1, "A"), variant("chr1", 12, 1, "C"), variant("chr1", 30, 1, "N")]
    pairs = pairs_of(INTERVALS.filter(pl.col("name") == "a"), variants)
    # each pair alone is fine
    assert len(collect(iter_variant_seqs(pairs, SeqReference(), use_strand=True))) == 3
    with pytest.raises(ValueError, match="overlap") as excinfo:
        list(iter_variant_seqs(pairs, SeqReference(), use_strand=True, group_by=["interval_idx"]))
    assert excinfo.value.__notes__ == ["in the group {'interval_idx': 0} of pairs"]


def test_iter_variant_seqs_ref_differs():
    pairs = pairs_of(SAME_WIDTH).with_columns(variant_ref=pl.lit("NNN"))
    with pytest.raises(ValueError, match="differs from the reference"):
        list(iter_variant_seqs(pairs, SeqReference(), use_strand=True))


def test_iter_variant_seqs_unknown_chrom():
    pairs = pairs_of(SAME_WIDTH)
    reference = SeqReference({"chr1": CHROMS["chr1"]})
    with pytest.raises(ValueError, match="^chrom 'chr2' is not in the reference") as excinfo:
        list(iter_variant_seqs(pairs, reference, use_strand=True))
    assert excinfo.value.__notes__ == ["in the group {'interval_idx': 2, 'variant_idx': 9} of pairs"]


def test_iter_variant_seqs_group_with_several_intervals():
    pairs = pairs_of(INTERVALS)
    with pytest.raises(
        ValueError, match=r"^the group \{'chrom': 'chr1'\} of pairs has 4 intervals, but a group needs one interval"
    ):
        iter_variant_seqs(pairs, SeqReference(), use_strand=True, group_by=["chrom"])


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"pairs": [1]}, TypeError, "^pairs must be a polars DataFrame or LazyFrame, not list$"),
        ({"reference": sample_5kb_fasta_file}, TypeError, "^reference must be a SequenceExtractor, not str$"),
        ({"group_by": "interval_idx"}, TypeError, "^group_by must be a sequence of strings, not a str"),
        ({"group_by": [1]}, TypeError, "^group_by must be a sequence of strings, but contains 1$"),
        ({"encoding": {"A": [1]}}, TypeError, "^encoding must be a Tokenizer, an Encoder or None, not dict$"),
        ({"encoding": Alphabet(["A"])}, TypeError, "^encoding must be a Tokenizer, an Encoder or None, not Alphabet$"),
        ({"dtype": np.uint8}, TypeError, "unexpected keyword argument 'dtype'"),
        ({"batch_size": 1.0}, TypeError, "^batch_size must be an int, not float$"),
        ({"anchor": "middle"}, ValueError, '^anchor must be "start", "center", "end" or None, got \'middle\'$'),
        ({"group_by": []}, ValueError, "^group_by must name at least one column$"),
        ({"group_by": ["interval_idx"] * 2}, ValueError, "^group_by must not repeat a column"),
        ({"group_by": ["interval_idx", "ref"]}, ValueError, r"^group_by must not contain \['ref'\]"),
        ({"group_by": ["gene_id"]}, ValueError, r"^pairs lacks the columns \['gene_id'\]$"),
        ({"batch_size": 0}, ValueError, "^batch_size must be at least 1, got 0$"),
    ],
)
def test_iter_variant_seqs_invalid_arguments(kwargs, error, match):
    arguments = {"pairs": pairs_of(SAME_WIDTH), "reference": SeqReference(), "use_strand": True, **kwargs}
    pairs = arguments.pop("pairs")
    reference = arguments.pop("reference")
    with pytest.raises(error, match=match):
        iter_variant_seqs(pairs, reference, **arguments)


@pytest.mark.parametrize("column", ["strand", "variant_pos", "variant_alt"])
def test_iter_variant_seqs_missing_column(column):
    pairs = pairs_of(SAME_WIDTH).drop(column)
    with pytest.raises(ValueError, match=rf"^pairs lacks the columns \['{column}'\]$"):
        iter_variant_seqs(pairs, SeqReference(), use_strand=True)


def test_iter_variant_seqs_nulls():
    pairs = pairs_of(SAME_WIDTH).with_columns(variant_alt=pl.lit(None, dtype=pl.String))
    with pytest.raises(ValueError, match=r"^pairs has nulls in the columns \['variant_alt'\]$"):
        iter_variant_seqs(pairs, SeqReference(), use_strand=True)


def test_iter_variant_seqs_options_are_keyword_only():
    with pytest.raises(TypeError, match="positional"):
        iter_variant_seqs(pairs_of(SAME_WIDTH), SeqReference(), True)
    with pytest.raises(TypeError, match="use_strand"):
        iter_variant_seqs(pairs_of(SAME_WIDTH), SeqReference())
