import pytest

from kipoiseq2.dataclasses import Interval
from kipoiseq2.transforms.functional import fixed_len, pad, resize_interval, translate, trim
from kipoiseq2.transforms.transforms import ResizeInterval


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


def test_resize_interval_center_use_strand():
    assert resize_interval(Interval("chr1", 0, 11, strand="+"), 4) == Interval("chr1", 4, 8, strand="+")
    assert resize_interval(Interval("chr1", 0, 11, strand="-"), 4) == Interval("chr1", 3, 7, strand="-")
    assert resize_interval(Interval("chr1", 0, 11, strand="."), 4) == Interval("chr1", 4, 8, strand=".")


@pytest.mark.parametrize("strand", ["+", "-", "."])
def test_resize_interval_center_is_interval_resize(strand):
    for start, end in [(0, 11), (0, 10), (3, 8)]:
        for width in [1, 4, 5, 12, 13]:
            interval = Interval("chr1", start, end, strand=strand)
            assert resize_interval(interval, width) == interval.resize(width)
            assert ResizeInterval(width)(interval) == interval.resize(width)


@pytest.mark.parametrize("strand", ["+", "-", "."])
def test_resize_interval_start_end_are_genomic(strand):
    # "start" keeps the genomic start and "end" keeps the genomic end, also on "-" intervals
    interval = Interval("chr1", 10, 20, strand=strand)
    assert resize_interval(interval, 4, "start") == Interval("chr1", 10, 14, strand=strand)
    assert resize_interval(interval, 4, "end") == Interval("chr1", 16, 20, strand=strand)
    assert resize_interval(interval, 13, "start") == Interval("chr1", 10, 23, strand=strand)
    assert resize_interval(interval, 13, "end") == Interval("chr1", 7, 20, strand=strand)
