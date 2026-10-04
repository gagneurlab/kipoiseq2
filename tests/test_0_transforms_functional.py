import numpy as np
import pytest

from kipoiseq2.dataclasses import Interval
from kipoiseq2.transforms.functional import (
    fixed_len,
    one_hot,
    one_hot_dna,
    pad,
    resize_interval,
    token2one_hot,
    tokenize,
    translate,
    trim,
)
from kipoiseq2.transforms.transforms import ResizeInterval
from kipoiseq2.utils import DNA


def test_tokenize():
    assert np.all(tokenize("ACGTTA", DNA, neutral_alphabet="N") == [0, 1, 2, 3, 3, 0])
    assert np.all(tokenize("ACGTGATGA", ["ACG", "TGA"], neutral_alphabet="NNN") == [0, 1, 1])
    assert np.all(tokenize("ACGTGATGA", ["ACG"], neutral_alphabet="TGA") == [0, -1, -1])
    with pytest.raises(Exception):
        tokenize("ACGTGATGA", ["ACG"], neutral_alphabet="NNN")


def test_token2one_hot():
    assert np.array_equal(token2one_hot(np.array([0, 1, -1]), 2), np.array([[1, 0], [0, 1], [0.25, 0.25]]))


def test_tokenize_one_hot():
    assert one_hot("ACG", DNA, "N").shape == (3, 4)

    et = tokenize("ACG", DNA, "N")
    assert et.shape == (3,)
    assert np.array_equal(et, np.array([0, 1, 2]))

    et = tokenize("TGTN", DNA, "N")
    assert np.array_equal(et, np.array([3, 2, 3, -1]))  # N mapped to -1


def test_one_hot():

    seq = "ACGTTTATNT"
    assert len(seq) == 10

    assert one_hot_dna(seq).shape == (10, 4)
    assert one_hot(seq).shape == (10, 4)

    assert np.all(one_hot_dna(seq) == one_hot(seq))

    assert one_hot(pad(seq, 20)).shape == (20, 4)

    assert one_hot(fixed_len(seq, 20)).shape == (20, 4)
    assert one_hot(fixed_len(seq, 5)).shape == (5, 4)
    assert trim(seq, 5) == "TTTAT"
    assert trim(seq, 5, "start") == "ACGTT"
    assert trim(seq, 5, "end") == "TATNT"
    with pytest.raises(Exception):
        assert pad(seq, 5, "end") == "TATNT"

    assert np.all(one_hot(seq)[0] == np.array([1, 0, 0, 0]))
    assert np.all(one_hot(seq)[1] == np.array([0, 1, 0, 0]))
    assert np.all(one_hot(seq)[2] == np.array([0, 0, 1, 0]))
    assert np.all(one_hot(seq)[3] == np.array([0, 0, 0, 1]))
    assert np.all(one_hot(seq)[4] == np.array([0, 0, 0, 1]))
    assert np.all(one_hot(seq)[-1] == np.array([0, 0, 0, 1]))
    assert np.all(one_hot(seq)[-2] == np.array([0.25, 0.25, 0.25, 0.25]))

    with pytest.raises(ValueError):
        one_hot(["A", "C"])

    with pytest.raises(ValueError):
        one_hot_dna(["A", "C"])


def test_fixed_len():
    seq = "ACGTTTATNT"
    assert len(fixed_len(seq, 20, value="N", anchor="end")) == 20
    assert fixed_len([1, 2, 3, 4], 6, value=[0], anchor="end") == [0, 0, 1, 2, 3, 4]
    assert fixed_len([1, 2, 3, 4], 2, value=[0], anchor="end") == [3, 4]

    # expect error
    assert fixed_len(seq, length=3, value="NNN", anchor="end") == "TNT"


def test_pad_sequences():
    seq = "CTTACTCAGA"
    assert fixed_len(seq, 4, anchor="center", value="N") == "ACTC"
    assert fixed_len(seq, 9, anchor="start", value="N") == "CTTACTCAG"

    assert fixed_len(seq, 10, anchor="start", value="N") == seq
    assert fixed_len(seq, 10, anchor="end", value="N") == "CTTACTCAGA"


def test_translate():
    assert translate("ATGTGGTAA") == "MW_"
    # TGA is a stop codon, and translation continues after it
    assert translate("ATGTGACCC") == "M_P"
    with pytest.raises(ValueError):
        translate("ATGT")
    with pytest.raises(KeyError):
        translate("ATGNNN")


def test_translate_transl_table():
    seq = "AGAAGGATATGA"
    assert translate(seq) == "RRI_"
    assert translate(seq, transl_table=11) == translate(seq, transl_table=1)
    assert translate(seq, transl_table=2) == "__MW"
    with pytest.raises(ValueError, match=r"\[1, 2, 11\]"):
        translate(seq, transl_table=3)


def test_translate_transl_except():
    # selenocysteine at codon 2, and the TGA at codon 3 stays a stop
    assert translate("ATGTGATGATAA", transl_except={2: "U"}) == "MU__"
    # non-AUG start
    assert translate("ACGTGG", transl_except={1: "M"}) == "MW"
    # an exception codon is not looked up in the table
    assert translate("ATGNNN", transl_except={2: "X"}) == "MX"


def test_translate_transl_except_partial_codon():
    # mitochondrial stop that poly(A) completes
    assert translate("ATGT", transl_table=2, transl_except={2: "_"}) == "M_"
    with pytest.raises(ValueError):
        translate("ATGT", transl_table=2)


@pytest.mark.parametrize("transl_except", [{0: "U"}, {3: "U"}, {1: "Sec"}, {1: ""}, {1: "u"}])
def test_translate_invalid_transl_except(transl_except):
    with pytest.raises(ValueError):
        translate("ATGTGA", transl_except=transl_except)


@pytest.mark.parametrize("anchor", ["start", "end", "center"])
@pytest.mark.parametrize("ilen", [3, 4])
def test_resize_interval(anchor, ilen):
    dummy_start, dummy_end = 10, 20
    dummy_center = int((dummy_start + dummy_end) / 2)

    dummy_inter = Interval("chr2", dummy_start, dummy_end, name="intname", strand="-")
    ret_inter = resize_interval(dummy_inter, ilen, anchor)

    # the original interval was left intact
    assert dummy_inter.chrom == "chr2"
    assert dummy_inter.start == dummy_start
    assert dummy_inter.end == dummy_end
    assert dummy_inter.name == "intname"

    # metadata kept
    assert ret_inter.chrom == dummy_inter.chrom
    assert ret_inter.name == "intname"
    assert ret_inter.strand == "-"

    # desired output width
    assert ret_inter.end - ret_inter.start == ilen

    # correct anchor point
    if anchor == "start":
        assert ret_inter.start == dummy_start
    elif anchor == "end":
        assert ret_inter.end == dummy_end
    elif anchor == "center":
        assert int((ret_inter.start + ret_inter.end) / 2) == dummy_center


def test_ResizeInterval():
    """Same test as before"""
    dummy_start, dummy_end = 10, 20
    dummy_center = int((dummy_start + dummy_end) / 2)
    ilen = 4
    dummy_inter = Interval("chr2", dummy_start, dummy_end, name="intname", strand="-")
    ri = ResizeInterval(ilen, "center")
    ret_inter = ri(dummy_inter)
    assert int((ret_inter.start + ret_inter.end) / 2) == dummy_center

    # the original interval was left intact
    assert dummy_inter.chrom == "chr2"
    assert dummy_inter.start == dummy_start
    assert dummy_inter.end == dummy_end
    assert dummy_inter.name == "intname"

    # metadata kept
    assert ret_inter.chrom == dummy_inter.chrom
    assert ret_inter.name == "intname"
    assert ret_inter.strand == "-"

    # desired output width
    assert ret_inter.end - ret_inter.start == ilen
