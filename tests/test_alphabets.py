import copy
import dataclasses
import pickle
import re
from collections import OrderedDict
from types import MappingProxyType

import pytest

import kipoiseq2
from kipoiseq2.alphabets import AMINO_ACIDS, DNA, RNA, Alphabet


def test_alphabet():
    alphabet = Alphabet(["A", "C", "G"], ambiguous={"N": ["A", "C", "G"], "R": ["A", "G"]})
    assert alphabet.letters == ("A", "C", "G")
    assert alphabet.ambiguous == {"N": ("A", "C", "G"), "R": ("A", "G")}
    assert alphabet == Alphabet(("A", "C", "G"), {"N": ("A", "C", "G"), "R": ("A", "G")})
    assert hash(alphabet) == hash(Alphabet(("A", "C", "G"), {"N": ("A", "C", "G"), "R": ("A", "G")}))
    assert repr(alphabet) == "Alphabet(letters=('A', 'C', 'G'), ambiguous={'N': ('A', 'C', 'G'), 'R': ('A', 'G')})"


def test_alphabet_without_codes():
    alphabet = Alphabet(["A", "C"])
    assert alphabet.ambiguous == {}
    assert alphabet == Alphabet(["A", "C"], ambiguous={})
    assert hash(alphabet) == hash(Alphabet(["A", "C"], ambiguous={}))
    assert repr(alphabet) == "Alphabet(letters=('A', 'C'), ambiguous={})"


def test_alphabet_code_letters_in_the_order_of_the_letters():
    # equal sets of letters compare equal
    alphabet = Alphabet(["A", "C", "G", "T"], ambiguous={"R": ["G", "A"], "N": ("T", "G", "C", "A")})
    assert alphabet.ambiguous == {"R": ("A", "G"), "N": ("A", "C", "G", "T")}
    assert alphabet == Alphabet(["A", "C", "G", "T"], ambiguous={"R": ["A", "G"], "N": ["A", "C", "G", "T"]})


def test_alphabet_keeps_the_order_of_the_codes():
    # the order of the codes sets their tokens, so alphabets with another order differ
    alphabet = Alphabet(["A", "C", "G"], ambiguous={"R": ["A", "G"], "N": ["A", "C", "G"]})
    assert list(alphabet.ambiguous) == ["R", "N"]
    assert alphabet != Alphabet(["A", "C", "G"], ambiguous={"N": ["A", "C", "G"], "R": ["A", "G"]})
    assert alphabet == Alphabet(["A", "C", "G"], ambiguous=OrderedDict([("R", ["A", "G"]), ("N", ["A", "C", "G"])]))


def test_alphabet_ambiguous_any_mapping():
    alphabet = Alphabet(["A", "C"], ambiguous=MappingProxyType({"N": ["A", "C"]}))
    assert alphabet == Alphabet(["A", "C"], ambiguous={"N": ["A", "C"]})


def test_alphabet_letters_with_several_characters():
    alphabet = Alphabet(["ACG", "TGA"], ambiguous={"NNN": ["ACG", "TGA"]})
    assert alphabet.letters == ("ACG", "TGA")
    assert alphabet.ambiguous == {"NNN": ("ACG", "TGA")}


def test_alphabet_frozen():
    letters = ["A", "C"]
    ambiguous = {"N": ["A", "C"]}
    alphabet = Alphabet(letters, ambiguous=ambiguous)
    letters.append("G")
    ambiguous["N"].append("G")
    ambiguous["M"] = ["A", "C"]
    assert alphabet.letters == ("A", "C")
    assert alphabet.ambiguous == {"N": ("A", "C")}
    with pytest.raises(dataclasses.FrozenInstanceError):
        alphabet.letters = ("A",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        alphabet.ambiguous = {}
    with pytest.raises(TypeError):
        alphabet.ambiguous["N"] = ("A", "C")


def test_alphabet_default_ambiguous_is_read_only():
    with pytest.raises(TypeError):
        Alphabet(["A"]).ambiguous["N"] = ("A",)
    assert Alphabet(["A"]).ambiguous == {}


@pytest.mark.parametrize("alphabet", [DNA, RNA, AMINO_ACIDS, Alphabet(["A"]), Alphabet(["ACG", "TGA"])])
def test_alphabet_pickle(alphabet):
    for copied in [pickle.loads(pickle.dumps(alphabet)), copy.copy(alphabet), copy.deepcopy(alphabet)]:
        assert copied == alphabet
        assert hash(copied) == hash(alphabet)
        assert list(copied.ambiguous) == list(alphabet.ambiguous)


def test_alphabet_replace_validates_again():
    alphabet = Alphabet(["A", "C", "G"], ambiguous={"R": ["A", "G"]})
    assert dataclasses.replace(alphabet, letters=["A", "G"]) == Alphabet(["A", "G"], ambiguous={"R": ["A", "G"]})
    with pytest.raises(ValueError, match=re.escape("the letters of ambiguous['R'] must be in letters, but ['G'] are")):
        dataclasses.replace(alphabet, letters=["A", "C"])


def test_dna_and_rna():
    # N and the other IUPAC ambiguity codes, with N first
    codes = {
        "N": "ACGT",
        "R": "AG",
        "Y": "CT",
        "S": "CG",
        "W": "AT",
        "K": "GT",
        "M": "AC",
        "B": "CGT",
        "D": "AGT",
        "H": "ACT",
        "V": "ACG",
    }
    assert DNA == Alphabet(list("ACGT"), ambiguous={code: list(letters) for code, letters in codes.items()})
    rna_codes = {code: list(letters.replace("T", "U")) for code, letters in codes.items()}
    assert RNA == Alphabet(list("ACGU"), ambiguous=rna_codes)


def test_amino_acids():
    # the 20 standard amino acids. N is asparagine, and X is any amino acid.
    assert AMINO_ACIDS.letters == tuple("ARNDCEQGHILKMFPSTWYV")
    assert AMINO_ACIDS.ambiguous == {"X": AMINO_ACIDS.letters, "B": ("N", "D"), "Z": ("E", "Q")}
    assert list(AMINO_ACIDS.ambiguous) == ["X", "B", "Z"]


@pytest.mark.parametrize("alphabet", [DNA, RNA, AMINO_ACIDS])
def test_presets_list_the_code_for_all_letters_first(alphabet):
    # the first code stands for all letters, so Encoder.decode gives it for a row of zeros
    assert next(iter(alphabet.ambiguous.values())) == alphabet.letters


def test_package_exports_the_alphabets():
    assert kipoiseq2.Alphabet is Alphabet
    assert kipoiseq2.DNA is DNA
    assert kipoiseq2.RNA is RNA
    assert kipoiseq2.AMINO_ACIDS is AMINO_ACIDS


@pytest.mark.parametrize(
    "letters, ambiguous, name, value",
    [
        ("ACGT", {}, "letters", "ACGT"),
        (["A", "C"], {"N": "AC"}, "ambiguous['N']", "AC"),
        (["A", "C"], {"M": "N"}, "ambiguous['M']", "N"),
    ],
)
def test_alphabet_str(letters, ambiguous, name, value):
    # a str is ambiguous: "NRY" can mean three letters or one letter of three characters
    expected = f"{name} must be a list or tuple of strings, not a str: "
    expected += f"pass a list of letters, e.g. list({value!r}) or [{value!r}]"
    with pytest.raises(TypeError, match=f"^{re.escape(expected)}$"):
        Alphabet(letters, ambiguous=ambiguous)


@pytest.mark.parametrize(
    "letters, ambiguous, match",
    [
        (["A", "C", "G", 1], {}, "^letters must be a list or tuple of strings, but contains 1$"),
        ([b"A", b"C"], {}, "^letters must be a list or tuple of strings, but contains b'A'$"),
        # a set has no order
        ({"A", "C", "G", "T"}, {}, "^letters must be a list or tuple of strings, not set$"),
        (None, {}, "^letters must be a list or tuple of strings, not NoneType$"),
        (["A", "C"], [("N", ["A", "C"])], "^ambiguous must be a mapping from codes to letters, not list$"),
        (["A", "C"], None, "^ambiguous must be a mapping from codes to letters, not NoneType$"),
        (["A", "C"], {1: ["A", "C"]}, "^the codes of ambiguous must be strings, but ambiguous has the key 1$"),
        (["A", "C"], {b"N": ["A", "C"]}, "^the codes of ambiguous must be strings, but ambiguous has the key b'N'$"),
        (["A", "C"], {"N": {"A", "C"}}, re.escape("ambiguous['N'] must be a list or tuple of strings, not set")),
        (["A", "C"], {"N": None}, re.escape("ambiguous['N'] must be a list or tuple of strings, not NoneType")),
        (["A", "C"], {"N": ["A", 1]}, re.escape("ambiguous['N'] must be a list or tuple of strings, but contains 1")),
    ],
)
def test_alphabet_wrong_types(letters, ambiguous, match):
    with pytest.raises(TypeError, match=match):
        Alphabet(letters, ambiguous=ambiguous)


@pytest.mark.parametrize(
    "letters, ambiguous, message",
    [
        ([], {}, "letters must not be empty"),
        ([], {"N": ["A", "C"]}, "letters must not be empty"),
        ([""], {}, "letters must contain non-empty strings of the same length, got ('',)"),
        (["ACG", "TG"], {}, "letters must contain non-empty strings of the same length, got ('ACG', 'TG')"),
        (list("ACGA"), {}, "letters must not repeat a letter, but repeat ['A']"),
        (list("ACGTCT"), {}, "letters must not repeat a letter, but repeat ['C', 'T']"),
        (list("ACGT"), {"T": ["A", "C"]}, "the codes of ambiguous must not be letters, but 'T' is a letter"),
        (["ACG", "TGA"], {"ACG": ["ACG", "TGA"]}, "the codes of ambiguous must not be letters, but 'ACG' is a letter"),
        # one code is one letter, so "NRY" is not three codes
        (list("ACGT"), {"NRY": ["A", "C"]}, "the codes of ambiguous must have the length 1 of the letters, got 'NRY'"),
        (list("ACGT"), {"": ["A", "C"]}, "the codes of ambiguous must have the length 1 of the letters, got ''"),
        (
            ["ACG", "TGA"],
            {"N": ["ACG", "TGA"]},
            "the codes of ambiguous must have the length 3 of the letters, got 'N'",
        ),
        (list("ACGT"), {"R": ["A", "U"]}, "the letters of ambiguous['R'] must be in letters, but ['U'] are not"),
        # letters are case-sensitive
        (list("ACGT"), {"R": ["a", "g"]}, "the letters of ambiguous['R'] must be in letters, but ['a', 'g'] are not"),
        (list("ACGT"), {"R": ["A", "G", "A"]}, "ambiguous['R'] must not repeat a letter, but repeats ['A']"),
        (list("ACGT"), {"R": ["A", "A"]}, "ambiguous['R'] must not repeat a letter, but repeats ['A']"),
        # a code of one letter is that letter
        (list("ACGT"), {"N": ["A"]}, "ambiguous['N'] must have at least 2 letters, got ('A',)"),
        (list("ACGT"), {"N": []}, "ambiguous['N'] must have at least 2 letters, got ()"),
    ],
)
def test_alphabet_invalid(letters, ambiguous, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        Alphabet(letters, ambiguous=ambiguous)


def test_alphabet_codes_with_the_same_letters():
    # e.g. N and * for any base
    alphabet = Alphabet(DNA.letters, ambiguous={"N": DNA.letters, "*": DNA.letters})
    assert alphabet.ambiguous == {"N": DNA.letters, "*": DNA.letters}


def test_alphabet_is_case_sensitive():
    # "A" and "a" are different letters, e.g. for separate columns of soft-masked bases
    alphabet = Alphabet(list("ACGTacgt"), ambiguous={"N": list("ACGT"), "n": list("acgt")})
    assert alphabet.letters == tuple("ACGTacgt")
    assert alphabet != Alphabet(list("ACGTacgt"), ambiguous={"N": list("ACGT"), "n": list("ACGT")})
