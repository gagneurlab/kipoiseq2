import re

import numpy as np
import pytest

import kipoiseq2.transforms
from kipoiseq2.transforms.transforms import Compose, OneHot, SplitSplicingSeq


# --------------------------------------------
def test_compose():
    c = Compose([OneHot()])
    print(str(c))
    assert c("ACGT").shape == (4, 4)


# the layouts of the former ReorderedOneHot(alphabet_axis=..., dummy_axis=...) for a sequence of length 10
@pytest.mark.parametrize(
    "dummy_axis, swap, shape",
    [
        (None, None, (10, 4)),  # alphabet_axis=1
        (None, (0, 1), (4, 10)),  # alphabet_axis=0
        (0, None, (1, 10, 4)),  # alphabet_axis=2, dummy_axis=0
        (1, None, (10, 1, 4)),  # alphabet_axis=2, dummy_axis=1
        (2, None, (10, 4, 1)),  # alphabet_axis=1, dummy_axis=2
        (0, (2, 1), (1, 4, 10)),  # alphabet_axis=1, dummy_axis=0
        (1, (0, 2), (4, 1, 10)),  # alphabet_axis=0, dummy_axis=1
        (2, (1, 0), (4, 10, 1)),  # alphabet_axis=0, dummy_axis=2
    ],
)
def test_OneHot_axes(dummy_axis, swap, shape):
    x = OneHot()("ACGTNACGTA")
    out = x if dummy_axis is None else np.expand_dims(x, dummy_axis)
    out = out if swap is None else np.swapaxes(out, *swap)
    assert out.shape == shape
    # the alphabet axis holds the one-hot vectors
    alphabet_axis = shape.index(4)
    assert np.array_equal(np.moveaxis(out, alphabet_axis, -1).reshape(10, 4), x)


# OneHot() encodes with one_hot_dna. OneHot(alphabet=list("ACGT")) encodes with one_hot,
# because a list never equals the tuple DNA.letters.
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


@pytest.mark.parametrize("name, value", [("alphabet", "ACGT"), ("neutral_alphabet", "N"), ("neutral_alphabet", "NRY")])
def test_OneHot_alphabet_str(name, value):
    expected = f"{name} must be a list or tuple of strings, not a str: "
    expected += f"pass a list of letters, e.g. list({value!r}) or [{value!r}]"
    with pytest.raises(TypeError, match=f"^{re.escape(expected)}$"):
        OneHot(**{name: value})


@pytest.mark.parametrize("name, value", [("alphabet", ["A", "C", "G", 1]), ("neutral_alphabet", [None])])
def test_OneHot_alphabet_not_strings(name, value):
    with pytest.raises(TypeError, match=f"^{name} must be a list or tuple of strings"):
        OneHot(**{name: value})


def test_OneHot_letters_of_different_lengths():
    with pytest.raises(ValueError, match="^all letters of alphabet and neutral_alphabet must have the same length"):
        OneHot(neutral_alphabet=["NRY"])("ACGT")


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
        "SplitSplicingSeq",
    }
