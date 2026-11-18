import pickle
import random
import re

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from kipoiseq2 import DNA, Alphabet, Encoder, Tokenizer
from kipoiseq2.encoders import _series

CODONS = Alphabet(["ACG", "TGA"], ambiguous={"NNN": ["ACG", "TGA"]})
"""An alphabet of letters with three characters."""

NON_ASCII = Alphabet(DNA.letters, ambiguous={**DNA.ambiguous, "é": DNA.letters})
""":data:`DNA` with the extra code ``é`` for all letters, which turns off the translate table."""

ENCODERS = [
    Tokenizer(),
    Tokenizer(dtype=np.int64),
    # the token 255 is the sentinel of the translate table, so tokenize takes the tokens from an array
    Tokenizer(token_map={"A": 7, "C": 8, "G": 255, "T": 10, **dict.fromkeys(DNA.ambiguous, 11)}),
    Tokenizer(dtype=">i4"),
    Tokenizer(NON_ASCII),
    Tokenizer(CODONS, dtype=np.int16),
    Encoder(),
    Encoder(ambiguous="fractional", dtype=np.float32),
    Encoder(dtype=bool),
    Encoder(ambiguous="fractional", dtype=np.float16),
    Encoder(ambiguous="fractional", dtype=">f4"),
    Encoder(ambiguous="ones"),
    Encoder(ambiguous="ones", dtype=bool),
    Encoder(NON_ASCII),
    Encoder(NON_ASCII, ambiguous="ones"),
    Encoder(CODONS, ambiguous="fractional", dtype=np.float64),
]
"""Encoders with the translate table and with the slow path, in several dtypes."""

LENGTHS = {
    "same lengths": [5, 5, 5],
    "different lengths": [5, 0, 3, 1, 8],
    "one sequence": [7],
    "empty sequences": [0, 0],
}
"""The numbers of letters of the sequences of the test Series."""


def _encode(encoder, seq):
    return encoder.tokenize(seq) if isinstance(encoder, Tokenizer) else encoder.one_hot(seq)


def _encode_series(encoder, seqs):
    return encoder.tokenize_series(seqs) if isinstance(encoder, Tokenizer) else encoder.one_hot_series(seqs)


def _seqs(encoder, lengths):
    """Return random sequences of the letters of ``encoder`` with ``lengths`` letters each."""
    rng = random.Random(0)
    letters = [*encoder.alphabet.letters, *encoder.alphabet.ambiguous]
    return ["".join(rng.choices(letters, k=length)) for length in lengths]


def _polars_dtype(encoder, lengths):
    """Return the polars dtype of the encoded sequences with ``lengths`` letters."""
    inner = pl.Series(np.empty(0, dtype=encoder.dtype.newbyteorder("="))).dtype
    if isinstance(encoder, Encoder):
        n_columns = len(encoder.alphabet.letters)
        if lengths and len(set(lengths)) == 1:
            return pl.Array(inner, (lengths[0], n_columns))
        return pl.List(pl.Array(inner, n_columns))
    if lengths and len(set(lengths)) == 1:
        return pl.Array(inner, lengths[0])
    return pl.List(inner)


def _check(encoder, series, out):
    """Check that ``out`` holds the arrays of the per-sequence method for the sequences of ``series``."""
    expected = [_encode(encoder, seq) for seq in series.to_list()]
    assert out.name == series.name
    assert out.dtype == _polars_dtype(encoder, [len(x) for x in expected])
    assert out.to_list() == [x.tolist() for x in expected]


@pytest.mark.parametrize("lengths", LENGTHS.values(), ids=LENGTHS.keys())
@pytest.mark.parametrize("encoder", ENCODERS, ids=repr)
def test_series_equals_the_method_per_sequence(encoder, lengths):
    series = pl.Series("seq", _seqs(encoder, lengths))
    _check(encoder, series, _encode_series(encoder, series))


@pytest.mark.parametrize("encoder", [Tokenizer(), Encoder(ambiguous="fractional", dtype=np.float32)], ids=repr)
def test_series_array_column_to_numpy(encoder):
    series = pl.Series("seq", ["ACGTN", "NNACG", "TTTTT"])
    x = _encode_series(encoder, series).to_numpy()
    np.testing.assert_array_equal(x, np.stack([_encode(encoder, seq) for seq in series]), strict=True)


@pytest.mark.parametrize("encoder", [Tokenizer(), Tokenizer(dtype=np.int64), Encoder(), Encoder(dtype=np.float32)])
def test_series_array_column_to_numpy_without_copy(encoder):
    out = _encode_series(encoder, pl.Series(["ACGTN", "NNACG"]))
    # each call returns a view of the same polars buffer
    assert np.shares_memory(out.to_numpy(), out.to_numpy())


@pytest.mark.parametrize(
    "values, offsets",
    [
        (np.arange(12, dtype=np.uint8), [0, 6, 12]),
        (np.arange(12, dtype=np.uint8), [0, 4, 12]),
        (np.arange(24, dtype=np.float32).reshape(6, 4), [0, 3, 6]),
        (np.arange(24, dtype=np.float32).reshape(6, 4), [0, 2, 6]),
    ],
    ids=["Array of tokens", "List of tokens", "Array of one-hot rows", "List of one-hot rows"],
)
def test_series_keeps_the_values_without_a_copy(values, offsets):
    arrow = _series("seq", values, np.array(offsets, dtype=np.int64)).to_arrow()
    while pa.types.is_nested(arrow.type):
        arrow = arrow.values
    assert arrow.buffers()[1].address == values.ctypes.data


@pytest.mark.parametrize("encoder", [Tokenizer(), Tokenizer(CODONS), Encoder(), Encoder(CODONS)], ids=repr)
def test_series_of_several_chunks_or_a_slice(encoder):
    seqs = _seqs(encoder, [5, 0, 3, 1, 8, 2])
    chunks = pl.concat([pl.Series("seq", seqs[:2]), pl.Series("seq", seqs[2:])], rechunk=False)
    assert chunks.n_chunks() == 2
    _check(encoder, chunks, _encode_series(encoder, chunks))
    sliced = pl.Series("seq", seqs).slice(1, 4)
    _check(encoder, sliced, _encode_series(encoder, sliced))


def _sliced_arrow(seqs):
    """Return the Arrow array of ``seqs`` as a slice of a longer array, whose other strings have unknown letters."""
    return pa.array(["XXX", *seqs, "UU"], type=pa.large_string()).slice(1, len(seqs))


@pytest.mark.parametrize("encoder", [Tokenizer(), Encoder()], ids=repr)
def test_series_from_a_sliced_arrow_array(monkeypatch, encoder):
    # polars gives offset 0 for a sliced Series, but an Arrow array can start at an offset in its buffers
    series = pl.Series("seq", ["ACGTN", "", "GGA", "NNNNNNN", "T", "AC"])
    monkeypatch.setattr(pl.Series, "to_arrow", lambda self, *args, **kwargs: _sliced_arrow(self.to_list()))
    assert series.to_arrow().offset == 1
    # the unknown letters outside of the slice must not raise an error
    _check(encoder, series, _encode_series(encoder, series))


@pytest.mark.parametrize(
    "encoder, dtype",
    [
        (Tokenizer(), pl.List(pl.UInt8)),
        (Tokenizer(dtype=np.int64), pl.List(pl.Int64)),
        (Tokenizer(CODONS, dtype=np.int64), pl.List(pl.Int64)),
        (Encoder(), pl.List(pl.Array(pl.UInt8, 4))),
        (Encoder(ambiguous="fractional", dtype=np.float32), pl.List(pl.Array(pl.Float32, 4))),
        (Encoder(CODONS), pl.List(pl.Array(pl.UInt8, 2))),
    ],
    ids=repr,
)
def test_series_empty(encoder, dtype):
    # an empty Series has no length for an Array column, so it gives an empty List column
    out = _encode_series(encoder, pl.Series("seq", [], dtype=pl.String))
    assert out.name == "seq"
    assert out.dtype == dtype
    assert out.is_empty()


@pytest.mark.parametrize("encoder", [Tokenizer(), Tokenizer(NON_ASCII), Encoder(), Encoder(NON_ASCII)], ids=repr)
@pytest.mark.parametrize(
    "seqs, row",
    [
        (["ACGT", "ACUT", "AXGT"], 1),
        (["ACGT", "", "acgt"], 2),
        (["ñA", "A"], 0),
        (["ACGT", "AC\xffT"], 1),
    ],
)
def test_series_unknown_letters(encoder, seqs, row):
    # the error is the one of the method per sequence for the first sequence with an unknown letter
    with pytest.raises(ValueError) as expected:
        _encode(encoder, seqs[row])
    with pytest.raises(ValueError) as excinfo:
        _encode_series(encoder, pl.Series(seqs))
    assert excinfo.value.args == expected.value.args
    assert excinfo.value.__notes__ == [f"in the row {row} of seqs"]


@pytest.mark.parametrize("encoder", [Tokenizer(CODONS), Encoder(CODONS)], ids=repr)
def test_series_length_not_a_multiple_of_the_letter_length(encoder):
    with pytest.raises(ValueError, match=r"multiple of the letter length 3, got len\(seq\)=4") as excinfo:
        _encode_series(encoder, pl.Series(["ACGTGA", "ACGT"]))
    assert excinfo.value.__notes__ == ["in the row 1 of seqs"]


@pytest.mark.parametrize("encoder", [Tokenizer(), Encoder(), Tokenizer(CODONS)], ids=repr)
@pytest.mark.parametrize(
    "seqs, match",
    [
        (["ACGT"], "seqs must be a polars Series, not list"),
        ("ACGT", "seqs must be a polars Series, not str"),
        (pl.Series(["ACGT"]).to_frame(), "seqs must be a polars Series, not DataFrame"),
        (pl.Series([1, 2]), "seqs must be a polars Series of strings, got the dtype Int64"),
        (pl.Series([b"ACGT"]), "seqs must be a polars Series of strings, got the dtype Binary"),
        (pl.Series(["ACGT"], dtype=pl.Categorical), "seqs must be a polars Series of strings, got the dtype Categ"),
    ],
)
def test_series_not_a_series_of_strings(encoder, seqs, match):
    with pytest.raises(TypeError, match=f"^{re.escape(match)}"):
        _encode_series(encoder, seqs)


@pytest.mark.parametrize("encoder", [Tokenizer(), Encoder(), Tokenizer(CODONS)], ids=repr)
def test_series_nulls(encoder):
    with pytest.raises(ValueError, match=r"^seqs must not contain nulls, but the row 1 is null$"):
        _encode_series(encoder, pl.Series(["ACG", None, None]))


@pytest.mark.skipif(np.dtype(np.longdouble).itemsize <= 8, reason="np.longdouble is float64 on this platform")
def test_one_hot_series_dtype_without_polars_type():
    with pytest.raises(TypeError, match=f"^polars has no type for the dtype {np.dtype(np.longdouble)}$"):
        Encoder(dtype=np.longdouble).one_hot_series(pl.Series(["ACGT"]))


def test_series_after_pickle():
    series = pl.Series("seq", ["ACGTN", "AC"])
    token_map = {"A": 7, "C": 8, "G": 9, "T": 10, **dict.fromkeys(DNA.ambiguous, 11)}
    for encoder in [Tokenizer(token_map=token_map), Encoder(dtype=np.float32)]:
        copied = pickle.loads(pickle.dumps(encoder))
        assert _encode_series(copied, series).to_list() == _encode_series(encoder, series).to_list()
