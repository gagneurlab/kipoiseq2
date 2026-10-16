from kipoiseq2.utils import parse_alphabet


def test_parse_alphabet():
    assert parse_alphabet(["A", "C"]) == ["A", "C"]
    assert parse_alphabet("AC") == ["A", "C"]
