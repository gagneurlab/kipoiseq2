import pytest

import kipoiseq2.transforms
from kipoiseq2 import Encoder
from kipoiseq2.dataclasses import Interval
from kipoiseq2.transforms.transforms import Compose, DummyAxis, ResizeInterval, SwapAxes


# --------------------------------------------
def test_compose():
    c = Compose([Encoder().one_hot])
    print(str(c))
    assert c("ACGT").shape == (4, 4)
    assert Compose([Encoder().one_hot, DummyAxis(0), SwapAxes(1, 2)])("ACGTN").shape == (1, 4, 5)


def test_ResizeInterval():
    interval = Interval("chr2", 10, 20, strand="-", attrs={"name": "intname"})
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
    }
