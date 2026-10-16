import pytest

import kipoiseq2.transforms
from kipoiseq2 import Encoder
from kipoiseq2.dataclasses import Interval
from kipoiseq2.transforms.transforms import Compose, DummyAxis, ResizeInterval, SplitSplicingSeq, SwapAxes


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
    interval = Interval("chr2", 10, 20, strand="-", name="intname")
    assert ResizeInterval(4)(interval) == interval.resize(4)
    for width in [3, 4, 13]:
        for anchor in ["start", "center", "end"]:
            for use_strand in [True, False]:
                expected = interval.resize(width, anchor=anchor, use_strand=use_strand)
                assert ResizeInterval(width, anchor=anchor, use_strand=use_strand)(interval) == expected


@pytest.mark.parametrize(
    "args, kwargs, error, match",
    [
        ((-1,), {}, ValueError, "^width must be at least 0, got -1$"),
        ((4.0,), {}, TypeError, "^width must be an int, not float$"),
        ((4,), {"anchor": "middle"}, ValueError, "^anchor must be "),
        # anchor is keyword-only
        ((4, "start"), {}, TypeError, "positional argument"),
    ],
)
def test_ResizeInterval_invalid(args, kwargs, error, match):
    # ResizeInterval checks its arguments before it gets an interval
    with pytest.raises(error, match=match):
        ResizeInterval(*args, **kwargs)


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
