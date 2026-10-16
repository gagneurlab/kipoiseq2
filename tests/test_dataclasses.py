"""Test kipoiseq2.dataclasses

Tests to perform:

# Both
- make sure the immutable objects are really immutable
- str() and from_str
- =, hashable

# Interval
- validation of the interval
- access to all the attributes
- shift, swapt_strand, trim, etc
"""

import copy
import dataclasses
import pickle
import re
from types import SimpleNamespace

import numpy as np
import pytest

from kipoiseq2.dataclasses import Interval, Variant


def test_variant():
    v = Variant("chr1", 10, "C", "T")

    assert v.start == 9
    assert v.chrom == "chr1"
    assert v.pos == 10
    assert v.ref == "C"
    assert v.alt == "T"
    assert isinstance(v.info, dict)
    assert len(v.info) == 0
    assert v.qual == 0
    assert v.filter == "PASS"
    v.info["test"] = 10
    assert v.info["test"] == 10
    assert isinstance(str(v), str)

    # make sure the original got unchangd
    v2 = v.copy()
    v.info["test"] = 20
    assert v2.info["test"] == 10
    v.__repr__()

    # __str__, from_str
    assert v == Variant.from_str(str(v))

    # hash test
    assert isinstance(hash(v), int)
    assert hash(v) == hash(Variant.from_str(str(v)))

    # fixed arguments
    with pytest.raises(AttributeError):
        v.chrom = "asd"
    with pytest.raises(AttributeError):
        v.pos = 10
    with pytest.raises(AttributeError):
        v.ref = "asd"
    with pytest.raises(AttributeError):
        v.alt = "asd"

    # non-fixed arguments
    v.id = "asd"
    v.qual = 10
    v.filter = "asd"
    v.source = 2

    assert isinstance(Variant("chr1", "10", "C", "T").pos, int)


@pytest.mark.parametrize("other", ["chr1:10:C>T", None, 1, Interval("chr1", 9, 10)])
def test_variant_eq_other_types(other):
    v = Variant("chr1", 10, "C", "T")
    assert not v == other
    assert v != other


def test_variant_eq_same_attributes():
    # objects with the same attributes, e.g. a kipoiseq Variant, compare by attributes
    v = Variant("chr1", 10, "C", "T")
    assert v == SimpleNamespace(chrom="chr1", pos=10, ref="C", alt="T")
    assert v != SimpleNamespace(chrom="chr1", pos=10, ref="C", alt="G")


def test_interval():
    interval = Interval("chr1", 10, 20, strand="-")
    interval.__repr__()

    assert interval.start == 10
    assert interval.end == 20
    assert interval.chrom == "chr1"
    assert interval.name == ""
    assert interval.score == 0
    assert interval.attrs == {}
    assert isinstance(str(interval), str)
    assert interval.neg_strand

    assert interval.width == 10

    # __str__, from_str
    assert interval == Interval.from_str(str(interval))

    # hash test
    assert isinstance(hash(interval), int)
    assert hash(interval) == hash(Interval.from_str(str(interval)))

    # the fields are read-only
    for field_name in ["chrom", "start", "end", "strand", "name", "score", "attrs", "width"]:
        with pytest.raises(AttributeError):
            setattr(interval, field_name, None)
    assert interval.strand == "-"

    assert interval.unstrand().strand == "."

    i2 = interval.shift(10, use_strand=False)

    # original unchanged
    assert interval.start == 10
    assert interval.end == 20

    assert i2.start == 20
    assert i2.end == 30

    i2 = interval.shift(10)  # use_strand = True by default
    assert i2.start == 0
    assert i2.end == 10

    assert not interval.shift(20, use_strand=True).is_valid()

    i2 = interval.shift(15, use_strand=True).truncate()
    assert i2.start == 0
    assert i2.end == 5

    # the end is exclusive, so an interval may end at the chromosome end
    assert interval.is_valid(chrom_len=20)
    assert not interval.is_valid(chrom_len=19)
    assert interval.truncate(chrom_len=20) == interval
    i2 = interval.truncate(chrom_len=15)
    assert i2.start == 10
    assert i2.end == 15

    assert interval.center() == 15

    # resize
    i2 = interval.resize(11)
    assert i2.start == 10 and i2.end == 21

    i2 = interval.resize(12)
    assert i2.start == 9 and i2.end == 21

    i2 = interval.resize(9)
    assert i2.start == 11 and i2.end == 20

    i2 = interval.swap_strand()
    assert interval.strand == "-"
    assert i2.strand == "+"
    assert i2.strand == "+"
    assert i2.width == 10
    i2 = i2.resize(11)
    assert i2.start == 9 and i2.end == 20

    assert interval.center(use_strand=True) == 15
    assert interval.center(use_strand=False) == 15

    i2 = interval.swap_strand()
    assert i2.strand == "+"

    i3 = i2.resize(11).shift(1)
    assert i3.center(use_strand=True) == 16
    assert i3.center(use_strand=False) == 16

    i3 = i2.resize(11).shift(1)
    assert i3.center(use_strand=True) == 16
    assert i3.center(use_strand=False) == 16

    i2 = interval.slice(1, 10)
    assert i2.start == 10 and i2.end == 19

    i2 = interval.slice(1, 10, use_strand=False)
    assert i2.start == 11 and i2.end == 20


@pytest.mark.parametrize("strand", ["+", "."])
def test_interval_slice(strand):
    interval = Interval("chr1", 10, 20, strand=strand)
    assert interval.slice(0, 3) == Interval("chr1", 10, 13, strand=strand)
    assert interval.slice(7, 10) == Interval("chr1", 17, 20, strand=strand)
    assert interval.slice(0, 10) == interval


@pytest.mark.parametrize(
    "i, j, error, match",
    [
        (0, 0, ValueError, re.escape("i and j must fulfill 0 <= i < j <= width, got i=0, j=0 and width=10")),
        (5, 3, ValueError, "^i and j must fulfill"),
        (-1, 5, ValueError, "^i and j must fulfill"),
        # trim(0, 20) extended the interval past its end
        (0, 20, ValueError, "^i and j must fulfill"),
        (0.0, 5, TypeError, "^i must be an int, not float$"),
        (0, None, TypeError, "^j must be an int, not NoneType$"),
    ],
)
def test_interval_slice_invalid(i, j, error, match):
    with pytest.raises(error, match=match):
        Interval("chr1", 10, 20).slice(i, j)


def test_interval_slice_use_strand_is_keyword_only():
    with pytest.raises(TypeError, match="positional argument"):
        Interval("chr1", 10, 20).slice(0, 5, False)


@pytest.mark.parametrize("other", ["chr1:10-20:-", None, 1, Variant("chr1", 11, "C", "T")])
def test_interval_eq_other_types(other):
    interval = Interval("chr1", 10, 20, strand="-")
    assert not interval == other
    assert interval != other


def test_interval_eq_only_intervals():
    # an object with the same attributes, e.g. a kipoiseq Interval, is not equal
    interval = Interval("chr1", 10, 20, strand="-")
    assert interval != SimpleNamespace(chrom="chr1", start=10, end=20, strand="-", name="", score=0, attrs={})


def test_interval_eq_all_fields():
    interval = Interval("chr1", 10, 20, strand="-", name="a", score=1, attrs={"gene": "A"})
    assert interval == Interval("chr1", 10, 20, strand="-", name="a", score=1.0, attrs={"gene": "A"})
    assert interval != Interval("chr1", 10, 20, strand="-", name="b", score=1, attrs={"gene": "A"})
    assert interval != Interval("chr1", 10, 20, strand="-", name="a", score=2, attrs={"gene": "A"})
    assert interval != Interval("chr1", 10, 20, strand="-", name="a", score=1, attrs={"gene": "B"})
    # the hash leaves out attrs, so attrs may hold unhashable values
    other = Interval("chr1", 10, 20, strand="-", name="a", score=1, attrs={"gene": ["B"]})
    assert hash(interval) == hash(other)
    assert len({interval, other}) == 2


@pytest.mark.parametrize(
    "args, kwargs, error, match",
    [
        ((1, 10, 20), {}, TypeError, "^chrom must be a str, not int$"),
        (("chr1", 10.0, 20), {}, TypeError, "^start must be an int, not float$"),
        (("chr1", "10", 20), {}, TypeError, "^start must be an int, not str$"),
        (("chr1", True, 20), {}, TypeError, "^start must be an int, not bool$"),
        (("chr1", 10, None), {}, TypeError, "^end must be an int, not NoneType$"),
        (("chr1", 20, 10), {}, ValueError, "^start must not be greater than end, got start=20, end=10$"),
        (("chr1", 10, 20), {"strand": "x"}, ValueError, re.escape("""strand must be "+", "-" or ".", got 'x'""")),
        (("chr1", 10, 20), {"strand": None}, TypeError, "^strand must be a str, not NoneType$"),
        (("chr1", 10, 20), {"name": 1}, TypeError, "^name must be a str, not int$"),
        (("chr1", 10, 20), {"score": "1"}, TypeError, "^score must be a number, not str$"),
        (("chr1", 10, 20), {"score": float("nan")}, ValueError, "^score must not be NaN$"),
        (("chr1", 10, 20), {"attrs": [("gene", "A")]}, TypeError, "^attrs must be a mapping, not list$"),
        # strand, name, score and attrs are keyword-only
        (("chr1", 10, 20, "-"), {}, TypeError, "positional argument"),
    ],
)
def test_interval_invalid(args, kwargs, error, match):
    with pytest.raises(error, match=match):
        Interval(*args, **kwargs)


def test_interval_numpy_integers():
    interval = Interval("chr1", np.int64(10), np.int32(20))
    assert type(interval.start) is int and type(interval.end) is int
    assert interval == Interval("chr1", 10, 20)


def test_interval_attrs_are_a_read_only_copy():
    attrs = {"gene": "A"}
    interval = Interval("chr1", 10, 20, attrs=attrs)
    attrs["gene"] = "B"
    assert interval.attrs == {"gene": "A"}
    with pytest.raises(TypeError):
        interval.attrs["gene"] = "C"


def test_interval_pickle_and_copy():
    interval = Interval("chr1", 10, 20, strand="-", name="a", score=1.5, attrs={"gene": "A"})
    copies = [pickle.loads(pickle.dumps(interval, protocol)) for protocol in range(pickle.HIGHEST_PROTOCOL + 1)]
    for i2 in [*copies, copy.copy(interval), copy.deepcopy(interval)]:
        assert i2 == interval
        assert repr(i2) == repr(interval)
        with pytest.raises(TypeError):
            i2.attrs["gene"] = "B"


def test_interval_asdict():
    interval = Interval("chr1", 10, 20, strand="-", name="a", score=1.5, attrs={"gene": "A"})
    fields = dataclasses.asdict(interval)
    assert fields == {
        "chrom": "chr1",
        "start": 10,
        "end": 20,
        "strand": "-",
        "name": "a",
        "score": 1.5,
        "attrs": {"gene": "A"},
    }
    assert dataclasses.astuple(interval)[:3] == ("chr1", 10, 20)


@pytest.mark.parametrize(
    "interval",
    [
        Interval("chr1", -5, 10, strand="-"),
        Interval("chr1", -10, -5),
        Interval("HLA-A*01:01:01:01", 0, 10, strand="+"),
    ],
)
def test_interval_from_str(interval):
    assert Interval.from_str(str(interval)) == interval


@pytest.mark.parametrize("s", ["chr1:10-20", "chr1:10:20:+", "chr1:a-20:+", "chr1:10-20-30:+"])
def test_interval_from_str_invalid(s):
    with pytest.raises(ValueError, match="^s must have the format"):
        Interval.from_str(s)


def test_interval_methods_raise_for_start_greater_than_end():
    with pytest.raises(ValueError, match="^start must not be greater than end"):
        Interval("chr1", 10, 20).slop(upstream=-6, downstream=-6)
    # the interval lies after the chromosome end
    with pytest.raises(ValueError, match="^start must not be greater than end"):
        Interval("chr1", 25, 30).truncate(20)


def test_interval_resize_use_strand():
    assert Interval("chr1", 0, 11, strand="+").resize(4) == Interval("chr1", 4, 8, strand="+")
    assert Interval("chr1", 0, 11, strand="-").resize(4) == Interval("chr1", 3, 7, strand="-")
    assert Interval("chr1", 0, 11, strand=".").resize(4) == Interval("chr1", 4, 8, strand=".")


@pytest.mark.parametrize("strand", ["+", "-", "."])
def test_interval_resize_without_use_strand(strand):
    i2 = Interval("chr1", 0, 11, strand=strand).resize(4, use_strand=False)
    assert i2 == Interval("chr1", 4, 8, strand=strand)

    # all strands give the result of an unstranded interval
    for start, end in [(0, 11), (0, 10), (3, 8)]:
        for width in [1, 4, 5, 12, 13]:
            expected = Interval("chr1", start, end).resize(width)
            i2 = Interval("chr1", start, end, strand=strand).resize(width, use_strand=False)
            assert (i2.start, i2.end, i2.strand) == (expected.start, expected.end, strand)


@pytest.mark.parametrize(
    "strand, anchor, use_strand, shrunk, grown",
    [
        ("+", "start", True, (10, 14), (10, 23)),
        ("+", "end", True, (16, 20), (7, 20)),
        (".", "start", True, (10, 14), (10, 23)),
        (".", "end", True, (16, 20), (7, 20)),
        # the start of a "-" interval is its genomic end
        ("-", "start", True, (16, 20), (7, 20)),
        ("-", "end", True, (10, 14), (10, 23)),
        ("-", "start", False, (10, 14), (10, 23)),
        ("-", "end", False, (16, 20), (7, 20)),
    ],
)
def test_interval_resize_anchor(strand, anchor, use_strand, shrunk, grown):
    interval = Interval("chr1", 10, 20, strand=strand)
    assert interval.resize(4, anchor=anchor, use_strand=use_strand) == Interval("chr1", *shrunk, strand=strand)
    assert interval.resize(13, anchor=anchor, use_strand=use_strand) == Interval("chr1", *grown, strand=strand)


def test_interval_resize_invalid():
    interval = Interval("chr1", 10, 20)
    assert interval.resize(0) == Interval("chr1", 15, 15)
    with pytest.raises(ValueError, match="^width must be at least 0, got -2$"):
        interval.resize(-2)
    with pytest.raises(TypeError, match="^width must be an int, not NoneType$"):
        interval.resize(None)
    with pytest.raises(TypeError, match="^width must be an int, not float$"):
        interval.resize(4.0)
    with pytest.raises(ValueError, match=re.escape("""anchor must be "start", "center" or "end", got 'middle'""")):
        interval.resize(4, anchor="middle")
    # anchor and use_strand are keyword-only
    with pytest.raises(TypeError, match="positional argument"):
        interval.resize(4, "start")


def test_interval_center_use_strand():
    # a center in the middle of a base is rounded towards the 3' end
    assert Interval("chr1", 0, 11, strand="+").center() == 6
    assert Interval("chr1", 0, 11, strand="-").center() == 5
    assert Interval("chr1", 0, 11, strand=".").center() == 6
    assert Interval("chr1", 0, 10, strand="-").center() == 5


@pytest.mark.parametrize("strand", ["+", "-", "."])
def test_interval_center_without_use_strand(strand):
    assert Interval("chr1", 0, 11, strand=strand).center(use_strand=False) == 6

    # all strands give the center of an unstranded interval
    for start, end in [(0, 11), (0, 10), (3, 8), (-3, 0)]:
        expected = Interval("chr1", start, end).center()
        assert Interval("chr1", start, end, strand=strand).center(use_strand=False) == expected


def test_interval_slop_use_strand():
    # upstream and downstream follow the strand
    assert Interval("chr1", 10, 20, strand="+").slop(upstream=2, downstream=5) == Interval("chr1", 8, 25, strand="+")
    assert Interval("chr1", 10, 20, strand="-").slop(upstream=2, downstream=5) == Interval("chr1", 5, 22, strand="-")
    assert Interval("chr1", 10, 20, strand=".").slop(upstream=2, downstream=5) == Interval("chr1", 8, 25, strand=".")


@pytest.mark.parametrize("strand", ["+", "-", "."])
def test_interval_slop_without_use_strand(strand):
    i2 = Interval("chr1", 10, 20, strand=strand).slop(upstream=2, downstream=5, use_strand=False)
    assert i2 == Interval("chr1", 8, 25, strand=strand)


@pytest.mark.parametrize(
    "method",
    [
        lambda i: i.slice(1, 5),
        lambda i: i.truncate(chrom_len=15),
        lambda i: i.resize(4),
        lambda i: i.shift(1),
        lambda i: i.slop(upstream=1),
        lambda i: i.unstrand(),
        lambda i: i.swap_strand(),
    ],
    ids=["slice", "truncate", "resize", "shift", "slop", "unstrand", "swap_strand"],
)
def test_interval_methods_keep_name_score_and_attrs(method):
    interval = Interval("chr1", 10, 20, strand="-", name="a", score=1.5, attrs={"gene": "A"})
    i2 = method(interval)
    assert i2 != interval
    assert (i2.chrom, i2.name, i2.score, i2.attrs) == ("chr1", "a", 1.5, {"gene": "A"})
