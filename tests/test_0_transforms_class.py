import pytest

import kipoiseq2.transforms
from kipoiseq2 import Encoder
from kipoiseq2.transforms.transforms import Compose, DummyAxis, SplitSplicingSeq, SwapAxes


# --------------------------------------------
def test_compose():
    c = Compose([Encoder().one_hot])
    print(str(c))
    assert c("ACGT").shape == (4, 4)
    assert Compose([Encoder().one_hot, DummyAxis(0), SwapAxes(1, 2)])("ACGTN").shape == (1, 4, 5)


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
        "SplitSplicingSeq",
    }
