import gc
import sys

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
