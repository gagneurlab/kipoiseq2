import re

import numpy as np
import pytest

import kipoiseq2.transforms
from kipoiseq2.transforms.transforms import Compose, OneHot, ReorderedOneHot, SplitSplicingSeq


# --------------------------------------------
def test_compose():
    c = Compose([OneHot()])
    print(str(c))
    assert c("ACGT").shape == (4, 4)


def test_ReorderedOneHot():
    seqlen = 10
    seq = "A" * seqlen

    test_pairs = [
        ((list("ACGT"), 1, None), (10, 4)),
        ((list("ACGT"), 2, 1), (10, 1, 4)),
        ((list("ACGTD"), 2, 1), (10, 1, 5)),
        ((list("ACGT"), 0, None), (4, 10)),
        ((list("ACGT"), 0, 1), (4, 1, 10)),
        ((list("ACGT"), 0, 2), (4, 10, 1)),
    ]

    for args, result in test_pairs:
        tr = ReorderedOneHot(alphabet=args[0], alphabet_axis=args[1], dummy_axis=args[2])
        out = tr(seq)
        assert out.shape == tr.get_output_shape(seqlen)
        assert out.shape == result

    with pytest.raises(ValueError, match="dummy_axis can't be the same as alphabet_axis"):
        ReorderedOneHot(alphabet_axis=1, dummy_axis=1)

    with pytest.raises(ValueError, match="dummy_axis can't be the same as alphabet_axis"):
        ReorderedOneHot(dummy_axis=1)


@pytest.mark.parametrize("alphabet_axis, dummy_axis", [(-1, None), (2, None), (3, 1), (-1, 2)])
def test_ReorderedOneHot_invalid_alphabet_axis(alphabet_axis, dummy_axis):
    expected = f"alphabet_axis can be 0 or 1, or 2 if dummy_axis is given, got {alphabet_axis}"
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        ReorderedOneHot(alphabet_axis=alphabet_axis, dummy_axis=dummy_axis)


# OneHot() encodes with one_hot_dna. OneHot(alphabet=list("ACGT")) encodes with one_hot,
# because a list never equals the tuple DNA.
@pytest.mark.parametrize("seq", ["", "ACGTN", "acgtn", "AcGtN"])
@pytest.mark.parametrize("dtype", [None, np.float32, np.int8])
def test_OneHot_paths_agree(seq, dtype):
    expected = OneHot(alphabet=list("ACGT"), dtype=dtype)(seq)
    result = OneHot(dtype=dtype)(seq)
    assert result.dtype == expected.dtype
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("seq", ["ACGTX", "acgtx", "ACGT\xff", "ACGTé"])
def test_OneHot_paths_agree_unknown_letters(seq):
    with pytest.raises(ValueError) as excinfo:
        OneHot(alphabet=list("ACGT"))(seq)
    with pytest.raises(ValueError, match=re.escape(str(excinfo.value))):
        OneHot()(seq)


@pytest.mark.parametrize("alphabet", [list("ACGT"), ("A", "C", "G", "T")])
def test_ReorderedOneHot_ignores_case(alphabet):
    tr = ReorderedOneHot(alphabet=alphabet, alphabet_axis=0)
    assert np.array_equal(tr("acgtn"), tr("ACGTN"))
    with pytest.raises(ValueError, match="X"):
        tr("ACGTX")


@pytest.mark.parametrize("transform", [OneHot, ReorderedOneHot])
@pytest.mark.parametrize("name, value", [("alphabet", "ACGT"), ("neutral_alphabet", "N"), ("neutral_alphabet", "NRY")])
def test_OneHot_alphabet_str(transform, name, value):
    expected = f"{name} must be a list or tuple of strings, not a str: "
    expected += f"pass a list of letters, e.g. list({value!r}) or [{value!r}]"
    with pytest.raises(TypeError, match=f"^{re.escape(expected)}$"):
        transform(**{name: value})


@pytest.mark.parametrize("transform", [OneHot, ReorderedOneHot])
@pytest.mark.parametrize("name, value", [("alphabet", ["A", "C", "G", 1]), ("neutral_alphabet", [None])])
def test_OneHot_alphabet_not_strings(transform, name, value):
    with pytest.raises(TypeError, match=f"^{name} must be a list or tuple of strings"):
        transform(**{name: value})


@pytest.mark.parametrize("transform", [OneHot, ReorderedOneHot])
def test_OneHot_letters_of_different_lengths(transform):
    with pytest.raises(ValueError, match="^all letters of alphabet and neutral_alphabet must have the same length"):
        transform(neutral_alphabet=["NRY"])("ACGT")


def test_OneHot_letters_with_several_characters():
    # each element of the list is one letter
    tr = OneHot(alphabet=["ACG", "TGA"], neutral_alphabet=["NNN"])
    assert np.array_equal(tr("ACGTGANNN"), [[1, 0], [0, 1], [0.25, 0.25]])


def test_OneHot_defaults_encode_with_one_hot_dna(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("OneHot() encoded with one_hot")

    monkeypatch.setattr(kipoiseq2.transforms.functional, "one_hot", fail)
    assert OneHot()("ACGTN").shape == (5, 4)


def test_SplitSplicingSeq():
    split = SplitSplicingSeq(
        exon_cut_l=0,
        exon_cut_r=0,
        intron5prime_cut=3,
        intron3prime_cut=3,
        acceptor_intron_len=2,
        acceptor_exon_len=3,
        donor_exon_len=3,
        donor_intron_len=2,
    )

    # seq = 'TAAAG GTAGTAGA GTCCC'
    seq = "TAAAGGTAGTAGAGTCCC"
    splited = split(seq, 5, 5)
    assert splited["intron5prime"] == "TA"
    assert splited["acceptor"] == "AGGTA"
    assert splited["exon"] == "GTAGTAGA"
    assert splited["donor"] == "AGAGT"
    assert splited["intron3prime"] == "CC"

    # no AG before the exon and no GT after it
    with pytest.warns(UserWarning) as record:
        split("TAACCGTAGTAGACCCCC", 5, 5)
    assert [str(w.message) for w in record] == ["Non-GT donor", "Non-AG acceptor"]


def test_ResizeInterval():
    pass


def test_transforms_namespace():
    public = {name for name in vars(kipoiseq2.transforms) if not name.startswith("_")}
    # functional and transforms are the submodules
    assert public == {
        "functional",
        "transforms",
        "Compose",
        "DummyAxis",
        "SwapAxes",
        "ResizeInterval",
        "OneHot",
        "ReorderedOneHot",
        "SplitSplicingSeq",
    }
