import gzip
import struct
import sys
import zlib

import pytest

from kipoiseq2.dataclasses import Interval
from kipoiseq2.extractors import FastaStringExtractor, SequenceExtractor

comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
comp.update({k.lower(): v.lower() for k, v in comp.items()})


@pytest.mark.parametrize("use_strand", [True, False])
@pytest.mark.parametrize("uppercase", [True, False])
def test_fastareader(use_strand, uppercase):
    fp = "tests/data/sample.fasta"
    with open(fp, "r") as ifh:
        for i, s in enumerate(ifh):
            if i == 1:
                fasta_str = s.lstrip()
    fr = FastaStringExtractor(fp, use_strand=use_strand, uppercase=uppercase)
    intervals = Interval("chr1", 0, 2, strand="-"), Interval("chr1", 3, 4)

    for interval in intervals:
        seq = fr.extract(interval)
        ref_seq = fasta_str[interval.start : interval.end]
        if use_strand and interval.strand == "-":
            ref_seq = list(ref_seq)[::-1]
            ref_seq = "".join([comp[el] for el in ref_seq])
        assert seq == (ref_seq.upper() if uppercase else ref_seq)


def test_fasta_extractor_uppercases_by_default(tmp_path):
    fasta_file = tmp_path / "soft.fa"
    fasta_file.write_text(">chr1\nACgtnN\n")
    interval = Interval("chr1", 0, 6, strand="-")
    assert FastaStringExtractor(fasta_file, use_strand=True).extract(interval) == "NNACGT"
    soft = FastaStringExtractor(fasta_file, use_strand=True, uppercase=False)
    assert soft.extract(interval) == "NnacGT"
    # uppercase is keyword-only
    with pytest.raises(TypeError):
        FastaStringExtractor(fasta_file, True, False)


def test_fasta_extractor_is_a_sequence_extractor():
    with FastaStringExtractor("tests/data/sample.fasta", use_strand=False) as fasta:
        assert isinstance(fasta, SequenceExtractor)


def test_fasta_extractor_chrom_sizes():
    with FastaStringExtractor("tests/data/sample.5kb.fa", use_strand=False) as fasta:
        assert fasta.chrom_sizes == {"chr1": 5000}
        with pytest.raises(TypeError):
            fasta.chrom_sizes["chr2"] = 10


@pytest.mark.parametrize(
    "interval", [Interval("chr1", 0, 0), Interval("chr1", 4990, 5000), Interval("chr1", 5000, 5000)]
)
def test_fasta_extractor_interval_at_chromosome_ends(interval):
    with FastaStringExtractor("tests/data/sample.5kb.fa", use_strand=False) as fasta:
        assert len(fasta.extract(interval)) == interval.width


@pytest.mark.parametrize(
    "interval", [Interval("chr1", -1, 5), Interval("chr1", 4990, 5001), Interval("chr1", 5001, 5010)]
)
def test_fasta_extractor_rejects_intervals_outside_the_chromosome(interval):
    # in 0.1, an interval past the chromosome end gave a shorter sequence
    with FastaStringExtractor("tests/data/sample.5kb.fa", use_strand=False) as fasta:
        with pytest.raises(ValueError, match="must lie within 0 and 5000, the length of chr1$"):
            fasta.extract(interval)


def test_fasta_extractor_rejects_unknown_chrom_and_other_types():
    with FastaStringExtractor("tests/data/sample.5kb.fa", use_strand=False) as fasta:
        with pytest.raises(ValueError, match="^chrom 'chr2' is not in the reference$"):
            fasta.extract(Interval("chr2", 0, 5))
        with pytest.raises(TypeError, match="^interval must be an Interval, not str$"):
            fasta.extract("chr1:0-5:+")


def test_fasta_extractor_missing_file():
    with pytest.raises(FileNotFoundError):
        FastaStringExtractor("tests/data/missing.fa", use_strand=False)


def test_fasta_extractor_context_manager():
    with FastaStringExtractor("tests/data/sample.fasta", use_strand=False) as fasta:
        assert fasta.use_strand is False
    with pytest.raises(ValueError, match="closed file"):
        fasta.extract(Interval("chr1", 0, 2))
    # closing twice does not raise
    fasta.close()


def test_fasta_extractor_requires_use_strand_keyword():
    fp = "tests/data/sample.fasta"
    with pytest.raises(TypeError, match="use_strand"):
        FastaStringExtractor(fp)
    with pytest.raises(TypeError, match="positional"):
        FastaStringExtractor(fp, True)


def test_fasta_extractor_extract_rejects_unknown_arguments():
    fr = FastaStringExtractor("tests/data/sample.fasta", use_strand=False)
    with pytest.raises(TypeError, match="fixed_length"):
        fr.extract(Interval("chr1", 0, 2), fixed_length=False)


FASTA = ">chr1\nACGTACGTAC\nGTACG\n"


def _bgzip(data: bytes) -> bytes:
    """Compress ``data`` into one BGZF block, followed by the BGZF end-of-file block."""
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(data) + compressor.flush()
    # the header holds the block size minus 1 in the extra subfield "BC"
    header = b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00"
    block_size = len(header) + 2 + len(compressed) + 8
    block = header + struct.pack("<H", block_size - 1) + compressed + struct.pack("<II", zlib.crc32(data), len(data))
    return block + bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


@pytest.mark.parametrize("name", ["genome.fa.gz", "genome.fa"])
def test_fasta_extractor_rejects_gzip(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(gzip.compress(FASTA.encode()))
    with pytest.raises(ValueError, match="is compressed with gzip. Decompress it, or compress it with bgzip"):
        FastaStringExtractor(path, use_strand=False)
    # pyfaidx did not write an index
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]


def test_fasta_extractor_bgzip_requires_suffix(tmp_path):
    path = tmp_path / "genome.fa"
    path.write_bytes(_bgzip(FASTA.encode()))
    with pytest.raises(ValueError, match=r"is compressed with bgzip, so its name must end in \.gz or \.bgz$"):
        FastaStringExtractor(path, use_strand=False)


def test_fasta_extractor_bgzip_requires_biopython(tmp_path, monkeypatch):
    path = tmp_path / "genome.fa.gz"
    path.write_bytes(_bgzip(FASTA.encode()))
    # None in sys.modules makes the import fail
    monkeypatch.setitem(sys.modules, "Bio", None)
    monkeypatch.setitem(sys.modules, "Bio.bgzf", None)
    with pytest.raises(ImportError, match="is compressed with bgzip. Reading it requires Biopython"):
        FastaStringExtractor(path, use_strand=False)


@pytest.mark.parametrize("name", ["genome.fa.gz", "genome.fa.bgz"])
def test_fasta_extractor_reads_bgzip(tmp_path, name):
    pytest.importorskip("Bio.bgzf")
    path = tmp_path / name
    path.write_bytes(_bgzip(FASTA.encode()))
    fasta = FastaStringExtractor(path, use_strand=False)
    assert fasta.extract(Interval("chr1", 8, 13)) == "ACGTA"
    fasta.close()
