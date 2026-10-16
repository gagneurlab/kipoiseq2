import itertools
import re
import string

import numpy as np
import pytest

from kipoiseq2.dataclasses import Interval
from kipoiseq2.transforms import TRANSLATION_TABLE, fixed_len, pad, reverse_complement, translate, trim


def test_trim():
    seq = "ACGTTTATNT"
    assert len(pad(seq, 20)) == 20
    assert len(fixed_len(seq, 20)) == 20
    assert len(fixed_len(seq, 5)) == 5
    assert trim(seq, 5) == "GTTTA"
    assert trim(seq, 5, anchor="start") == "ACGTT"
    assert trim(seq, 5, anchor="end") == "TATNT"


def test_reverse_complement():
    assert reverse_complement("ACGTNacgtn") == "nacgtNACGT"
    assert reverse_complement("RYKM") == "KMRY"
    with pytest.raises(ValueError):
        reverse_complement("AC-GT")


@pytest.mark.parametrize("seq", ["ACU", "AC-", "ACé", "ACĀ"])
def test_reverse_complement_rejects_characters_without_complement(seq):
    # the translation table keeps the characters above 255, such as Ā
    with pytest.raises(ValueError, match=f"non-DNA characters: \\['{seq[-1]}'\\]"):
        reverse_complement(seq)


def test_fixed_len():
    seq = "ACGTTTATNT"
    assert len(fixed_len(seq, 20, value="N", anchor="end")) == 20
    assert fixed_len(seq, length=3, value="NNN", anchor="end") == "TNT"
    assert fixed_len(seq, 10) == seq


def test_trim_to_zero():
    # seq[-0:] returned the whole sequence
    assert trim("ACGT", 0, anchor="end") == ""
    assert trim("ACGT", 0, anchor="start") == ""
    assert trim("ACGT", 0) == ""
    assert fixed_len("ACGT", 0, anchor="end") == ""


@pytest.mark.parametrize("anchor", ["start", "center", "end"])
def test_fixed_len_keeps_the_letters_of_interval_resize(anchor):
    for seq_len in range(16):
        seq = string.ascii_lowercase[:seq_len]
        for length in range(16):
            interval = Interval("chr1", 0, seq_len).resize(length, anchor=anchor)
            # the positions outside 0 to seq_len are padded
            expected = "".join(seq[i] if 0 <= i < seq_len else "N" for i in range(interval.start, interval.end))
            assert fixed_len(seq, length, anchor=anchor) == expected
            if length >= seq_len:
                assert pad(seq, length, anchor=anchor) == expected
            if length <= seq_len:
                assert trim(seq, length, anchor=anchor) == expected


def test_trim_center_rounds_like_interval_resize():
    # Interval("chr1", 0, 10).resize(7) is 1-8, and trim kept 2-9
    assert trim("0123456789", 7) == "1234567"
    assert pad("A", 4) == "NANN"


@pytest.mark.parametrize(
    "func, args, kwargs, error, match",
    [
        (pad, ("ACGT", 2), {}, ValueError, re.escape("length must be at least len(seq)=4, got 2")),
        (trim, ("ACGT", 5), {}, ValueError, re.escape("length must be at most len(seq)=4, got 5")),
        (fixed_len, ("ACGT", -1), {}, ValueError, "^length must be at least 0, got -1$"),
        (pad, ("ACGT", 5.0), {}, TypeError, "^length must be an int, not float$"),
        (trim, (["A", "C"], 1), {}, TypeError, "^seq must be a str, not list$"),
        (fixed_len, (["A", "C"], 4), {"value": ["N"]}, TypeError, "^seq must be a str, not list$"),
        (pad, ("ACGT", 6), {"value": ""}, ValueError, "^value must not be empty$"),
        # fixed_len checks value also if it trims
        (fixed_len, ("ACGT", 2), {"value": ""}, ValueError, "^value must not be empty$"),
        (pad, ("ACGT", 6), {"value": ["N"]}, TypeError, "^value must be a str, not list$"),
        (trim, ("ACGT", 2), {"anchor": "middle"}, ValueError, "^anchor must be "),
        # anchor and value are keyword-only
        (pad, ("ACGT", 6, "N"), {}, TypeError, "positional argument"),
        (trim, ("ACGT", 2, "end"), {}, TypeError, "positional argument"),
        (fixed_len, ("ACGT", 6, "end"), {}, TypeError, "positional argument"),
    ],
)
def test_pad_trim_fixed_len_invalid(func, args, kwargs, error, match):
    with pytest.raises(error, match=match):
        func(*args, **kwargs)


def test_pad_sequences():
    seq = "CTTACTCAGA"
    assert fixed_len(seq, 4, anchor="center", value="N") == "ACTC"
    assert fixed_len(seq, 9, anchor="start", value="N") == "CTTACTCAG"

    assert fixed_len(seq, 10, anchor="start", value="N") == seq
    assert fixed_len(seq, 10, anchor="end", value="N") == "CTTACTCAGA"


def test_pad_value_with_several_elements():
    assert pad("A", 4, value="NM", anchor="start") == "ANMN"
    assert pad("A", 4, value="NM", anchor="end") == "NMNA"
    assert pad("A", 4, value="NM", anchor="center") == "NANM"
    assert fixed_len("A", 4, value="NN") == "NANN"


@pytest.mark.parametrize("anchor", ["start", "end", "center"])
def test_pad_value_with_several_elements_length(anchor):
    for length in range(1, 12):
        for value in ["NM", "NMK"]:
            assert len(pad("A", length, value=value, anchor=anchor)) == length
            assert len(fixed_len("A", length, value=value, anchor=anchor)) == length


def test_translate():
    assert translate("ATGTGGTAA") == "MW_"
    # TGA is a stop codon, and translation continues after it
    assert translate("ATGTGACCC") == "M_P"
    assert translate("") == ""
    with pytest.raises(ValueError):
        translate("ATGT")


def test_translate_is_case_sensitive():
    # the message names the lowercase codons and hints at their uppercase forms
    expected = (
        "seq contains codons that are not in genetic code 1: ['AtG', 'atg']. "
        "The lookup is case-sensitive, but genetic code 1 has the uppercase forms of ['AtG', 'atg']: "
        "uppercase the sequence with seq.upper(), or read it with FastaStringExtractor, which uppercases by default"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        translate("atgAtGTAA")
    with pytest.raises(ValueError, match=re.escape("not in genetic code 2: ['aga']")):
        translate("AGAaga", transl_table=2)
    # a translation exception covers a lowercase codon
    assert translate("atgTGGTAA", transl_except={1: "M"}) == "MW_"


def test_translate_unknown_codons():
    # the message names the codons as they are in seq
    with pytest.raises(ValueError, match=re.escape("not in genetic code 1: ['ATN', 'atn']")):
        translate("ATGATNatnATN")
    with pytest.raises(ValueError, match=re.escape("['NNN']")):
        translate("ATGNNN")


def test_translate_keyword_only():
    with pytest.raises(TypeError):
        translate("ATG", 1)
    with pytest.raises(TypeError):
        translate("ATG", 1, {1: "M"})


@pytest.mark.parametrize(
    "seq, kwargs, match",
    [
        (b"ATG", {}, "^seq must be a str, not bytes$"),
        ("ATG", {"transl_table": "1"}, "^transl_table must be an int, not str$"),
        ("ATG", {"transl_table": True}, "^transl_table must be an int, not bool$"),
        ("ATG", {"transl_except": [(1, "M")]}, "^transl_except must be a mapping or None, not list$"),
        ("ATG", {"transl_except": {1.0: "M"}}, "^transl_except position must be an int, not float$"),
        ("ATG", {"transl_except": {True: "M"}}, "^transl_except position must be an int, not bool$"),
        ("ATG", {"transl_except": {1: 77}}, "^transl_except amino acid must be a str, not int$"),
    ],
)
def test_translate_invalid_types(seq, kwargs, match):
    with pytest.raises(TypeError, match=match):
        translate(seq, **kwargs)


def test_translate_transl_table():
    seq = "AGAAGGATATGA"
    assert translate(seq) == "RRI_"
    assert translate(seq, transl_table=11) == translate(seq, transl_table=1)
    assert translate(seq, transl_table=2) == "__MW"
    assert translate(seq, transl_table=np.int64(2)) == "__MW"
    with pytest.raises(ValueError, match=r"\[1, 2, 11\]"):
        translate(seq, transl_table=3)


def test_translate_transl_except():
    # selenocysteine at codon 2, and the TGA at codon 3 stays a stop
    assert translate("ATGTGATGATAA", transl_except={2: "U"}) == "MU__"
    # non-AUG start
    assert translate("ACGTGG", transl_except={1: "M"}) == "MW"
    # an exception codon is not looked up in the genetic code
    assert translate("ATGNNN", transl_except={2: "X"}) == "MX"
    assert translate("ATGTGA", transl_except={np.int64(2): "U"}) == "MU"


def test_translate_transl_except_partial_codon():
    # mitochondrial stop that poly(A) completes
    assert translate("ATGT", transl_table=2, transl_except={2: "_"}) == "M_"
    with pytest.raises(ValueError):
        translate("ATGT", transl_table=2)


@pytest.mark.parametrize("transl_except", [{0: "U"}, {3: "U"}, {1: "Sec"}, {1: ""}, {1: "u"}])
def test_translate_invalid_transl_except(transl_except):
    with pytest.raises(ValueError):
        translate("ATGTGA", transl_except=transl_except)


def test_translation_table():
    codons = ["".join(codon) for codon in itertools.product("ACGT", repeat=3)]
    assert sorted(TRANSLATION_TABLE) == codons
    assert translate("".join(codons)) == "".join(TRANSLATION_TABLE[codon] for codon in codons)
    with pytest.raises(TypeError):
        TRANSLATION_TABLE["TGA"] = "U"
