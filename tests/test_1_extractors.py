import gc
import gzip
import struct
import sys
import zlib

import pytest

from kipoiseq2.dataclasses import Interval
from kipoiseq2.extractors import BaseExtractor, FastaStringExtractor

comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
comp.update({k.lower(): v.lower() for k, v in comp.items()})


@pytest.mark.parametrize("use_strand", [True, False])
@pytest.mark.parametrize("force_upper", [True, False])
def test_fastareader(use_strand, force_upper):
    fp = "tests/data/sample.fasta"
    with open(fp, "r") as ifh:
        for i, s in enumerate(ifh):
            if i == 1:
                fasta_str = s.lstrip()
    fr = FastaStringExtractor(fp, use_strand=use_strand, force_upper=force_upper)
    intervals = Interval("chr1", 0, 2, strand="-"), Interval("chr1", 3, 4)

    for interval in intervals:
        seq = fr.extract(interval)
        ref_seq = fasta_str[interval.start : interval.end]
        if use_strand and interval.strand == "-":
            ref_seq = list(ref_seq)[::-1]
            ref_seq = "".join([comp[el] for el in ref_seq])
        if force_upper:
            assert seq == ref_seq.upper()
        else:
            assert seq == ref_seq


def test_base_extractor_extract_is_abstract():
    class NoExtract(BaseExtractor):
        pass

    with pytest.raises(TypeError, match="extract"):
        BaseExtractor()
    with pytest.raises(TypeError, match="extract"):
        NoExtract()


def test_fasta_extractor_missing_file(monkeypatch):
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    with pytest.raises(OSError):
        FastaStringExtractor("tests/data/missing.fa", use_strand=False)
    gc.collect()
    # __del__ calls close() on the half-initialized extractor
    assert unraisable == []


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
