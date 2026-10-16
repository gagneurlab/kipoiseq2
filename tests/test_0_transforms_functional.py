import pytest

from kipoiseq2.transforms.functional import fixed_len, pad, translate, trim


def test_trim():
    seq = "ACGTTTATNT"
    assert len(pad(seq, 20)) == 20
    assert len(fixed_len(seq, 20)) == 20
    assert len(fixed_len(seq, 5)) == 5
    assert trim(seq, 5) == "TTTAT"
    assert trim(seq, 5, "start") == "ACGTT"
    assert trim(seq, 5, "end") == "TATNT"
    with pytest.raises(Exception):
        assert pad(seq, 5, "end") == "TATNT"


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


def test_pad_value_with_several_elements():
    assert pad("A", 4, value="NM", anchor="start") == "ANMN"
    assert pad("A", 4, value="NM", anchor="end") == "NMNA"
    assert pad("A", 4, value="NM", anchor="center") == "NMAN"
    assert pad([1], 4, value=[0, 9], anchor="end") == [0, 9, 0, 1]
    assert pad([[1, 0]], 4, value=[[0, 0], [9, 9]], anchor="start") == [[1, 0], [0, 0], [9, 9], [0, 0]]
    assert fixed_len("A", 4, value="NN") == "NNAN"


@pytest.mark.parametrize("anchor", ["start", "end", "center"])
def test_pad_value_with_several_elements_length(anchor):
    for length in range(1, 12):
        for value in ["NM", "NMK"]:
            assert len(pad("A", length, value=value, anchor=anchor)) == length
            assert len(pad(["A"], length, value=list(value), anchor=anchor)) == length
            assert len(fixed_len("A", length, value=value, anchor=anchor)) == length


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
