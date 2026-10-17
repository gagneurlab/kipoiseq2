import copy
import pickle
import re
import string

import numpy as np
import pytest

import kipoiseq2
from kipoiseq2 import AMINO_ACIDS, DNA, RNA, Alphabet, Encoder, Tokenizer

CODONS = Alphabet(["ACG", "TGA"], ambiguous={"NNN": ["ACG", "TGA"]})
"""An alphabet of letters with three characters."""

STRICT_DNA = Alphabet(list("ACGT"), ambiguous={"N": list("ACGT")})
"""DNA without the IUPAC ambiguity codes other than ``N``, e.g. for a vocabulary that lacks them."""

DNA_TOKENS = {"A": 7, "C": 8, "G": 9, "T": 10, **dict.fromkeys(DNA.ambiguous, 11)}
"""A token map for :data:`DNA` with other tokens than the default ones. The codes share the token 11."""

IUPAC_CODES = "NRYSWKMBDHV"
"""``N`` and the other IUPAC ambiguity codes, the codes of :data:`DNA` and :data:`RNA`."""

IUPAC_BASES = {
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
"""The DNA bases of each IUPAC ambiguity code."""

RULES = [("zeros", np.uint8), ("fractional", np.float32), ("ones", np.uint8)]
"""The rules for the rows of the ambiguity codes, each with a dtype that it accepts."""


def _non_ascii(alphabet):
    """Return ``alphabet`` with the extra code ``é`` for all letters.

    It turns off the translate table of the lookup and the byte lookup of :meth:`Encoder.decode`.
    """
    return Alphabet(alphabet.letters, ambiguous={**alphabet.ambiguous, "é": alphabet.letters})


def _code_row(letters, code_letters, ambiguous):
    """Return the row of a code of ``code_letters`` by the rule ``ambiguous``, as float64."""
    value = {"zeros": 0, "fractional": 1 / len(code_letters), "ones": 1}[ambiguous]
    return [value if letter in code_letters else 0 for letter in letters]


def _upper_hint(name, upper):
    return (
        f". The lookup is case-sensitive, but {name} has the uppercase forms of {upper}: "
        "uppercase the sequence with seq.upper(), or read it with FastaStringExtractor, which uppercases by default"
    )


# Tokenizer


def test_tokenize():
    tokenizer = Tokenizer()
    assert np.array_equal(tokenizer.tokenize("ACGTTA"), [0, 1, 2, 3, 3, 0])
    assert tokenizer.tokenize("ACGT").dtype == np.uint8
    assert tokenizer.tokenize("").shape == (0,)
    assert tokenizer.tokenize("").dtype == np.uint8
    # the codes follow the letters: N is the token 4, and R is the token 5
    assert np.array_equal(tokenizer.tokenize("TGTNR"), [3, 2, 3, 4, 5])
    assert tokenizer == Tokenizer(DNA, token_map=None, dtype=np.uint8)


def test_tokenize_code_tokens_are_out_of_range_of_a_table_per_letter():
    tokens = Tokenizer().tokenize("ACGTN")
    with pytest.raises(IndexError):
        np.eye(4)[tokens]
    # a table with a row for each letter and code takes the tokens as they are
    assert np.array_equal(np.eye(15, 4)[tokens], [*np.eye(4), [0] * 4])


@pytest.mark.parametrize("alphabet", [DNA, STRICT_DNA, RNA, AMINO_ACIDS, CODONS])
@pytest.mark.parametrize("ambiguous, dtype", RULES)
def test_tokens_are_the_rows_of_one_hot(alphabet, ambiguous, dtype):
    # token t is row t of the table of the encoder: the letters, then the codes
    code_rows = [_code_row(alphabet.letters, letters, ambiguous) for letters in alphabet.ambiguous.values()]
    rows = np.vstack([np.eye(len(alphabet.letters)), code_rows]).astype(dtype)
    seq = "".join((*alphabet.letters, *alphabet.ambiguous) * 2)
    expected = Encoder(alphabet, ambiguous=ambiguous, dtype=dtype).one_hot(seq)
    np.testing.assert_array_equal(rows[Tokenizer(alphabet).tokenize(seq)], expected, strict=True)


def test_tokenize_letters_with_several_characters():
    assert np.array_equal(Tokenizer(CODONS).tokenize("ACGTGATGANNN"), [0, 1, 1, 2])
    assert np.array_equal(Tokenizer(Alphabet(["ACG", "TGA"])).tokenize("ACGTGATGA"), [0, 1, 1])
    assert Tokenizer(CODONS).tokenize("").shape == (0,)
    with pytest.raises(ValueError, match=re.escape("['TGA']")):
        Tokenizer(Alphabet(["ACG", "CCC"], ambiguous={"NNN": ["ACG", "CCC"]})).tokenize("ACGTGATGA")


def test_tokenize_amino_acids():
    # N is asparagine, and the codes X, B and Z follow the 20 amino acids
    asparagine = AMINO_ACIDS.letters.index("N")
    assert np.array_equal(Tokenizer(AMINO_ACIDS).tokenize("NXBZ"), [asparagine, 20, 21, 22])


@pytest.mark.parametrize("alphabet", [DNA, RNA])
def test_tokenize_iupac_codes(alphabet):
    # each IUPAC ambiguity code has a token of its own after the letters, in the order of the codes
    seq = "".join(alphabet.letters) + IUPAC_CODES
    assert np.array_equal(Tokenizer(alphabet).tokenize(seq), range(15))


def test_tokenize_dtype():
    assert Tokenizer(dtype=np.int64).tokenize("ACGTN").dtype == np.int64
    assert np.array_equal(Tokenizer(dtype=np.int8).tokenize("ACGTN"), [0, 1, 2, 3, 4])
    assert Tokenizer(dtype="int16").dtype == np.int16
    assert Tokenizer(dtype=">u2").tokenize("ACGTN").dtype == ">u2"
    # the tokens 0 to 255 fit uint8
    letters = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    assert Tokenizer(Alphabet(letters[:254], ambiguous={"ZZ": letters[:2]})).tokenize("ZZ")[0] == 254
    assert Tokenizer(Alphabet(letters[:255], ambiguous={"ZZ": letters[:2]})).tokenize("ZZ")[0] == 255
    expected = "the tokens of dtype uint8 must be from 0 to 255, but 'ZZ' has 256"
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        Tokenizer(Alphabet(letters[:256], ambiguous={"ZZ": letters[:2]}))
    tokenizer = Tokenizer(Alphabet(letters[:256], ambiguous={"ZZ": letters[:2]}), dtype=np.int16)
    assert tokenizer.tokenize("ZZ")[0] == 256


@pytest.mark.parametrize("dtype", [np.float32, bool, "U1", np.complex64])
def test_tokenizer_dtype_not_an_integer_type(dtype):
    with pytest.raises(TypeError, match="^dtype must be an integer type, got "):
        Tokenizer(dtype=dtype)


def test_tokenizer_token_map():
    tokenizer = Tokenizer(DNA, token_map=DNA_TOKENS)
    assert np.array_equal(tokenizer.tokenize("ACGTNR"), [7, 8, 9, 10, 11, 11])
    assert tokenizer.tokenize("ACGTN").dtype == np.uint8
    # the attribute holds the tokens in the order of the letters and the codes
    assert list(Tokenizer(DNA, token_map=dict(reversed(DNA_TOKENS.items()))).token_map) == [*"ACGT", *IUPAC_CODES]
    with pytest.raises(TypeError):
        tokenizer.token_map["A"] = 0


def test_tokenizer_token_map_shared_tokens():
    # several letters and codes can share a token, e.g. a vocabulary without a token for each IUPAC code
    token_map = {"A": 1, "C": 2, "G": 3, "T": 4, **dict.fromkeys(DNA.ambiguous, 0)}
    assert np.array_equal(Tokenizer(DNA, token_map=token_map).tokenize("ACGTNRY"), [1, 2, 3, 4, 0, 0, 0])


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int64])
def test_tokenizer_token_map_large_tokens(dtype):
    highest = int(np.iinfo(dtype).max)
    token_map = {"A": 0, "C": 1, "G": highest, "T": 3, **dict.fromkeys(DNA.ambiguous, highest - 1)}
    tokens = Tokenizer(DNA, token_map=token_map, dtype=dtype).tokenize("ACGTN")
    assert tokens.dtype == dtype
    assert tokens.tolist() == [0, 1, highest, 3, highest - 1]


@pytest.mark.parametrize(
    "token_map, error, match",
    [
        (
            {"A": 0},
            ValueError,
            "the keys of token_map must be the letters and the ambiguity codes of alphabet, "
            f"but token_map lacks {['C', 'G', 'T', *IUPAC_CODES]}",
        ),
        # a token map for the strict alphabet without the IUPAC ambiguity codes other than N
        (
            {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4},
            ValueError,
            "but token_map lacks ['R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']",
        ),
        ({**DNA_TOKENS, "U": 4}, ValueError, "but token_map has the other keys ['U']"),
        (
            {"n" if letter == "N" else letter: token for letter, token in DNA_TOKENS.items()},
            ValueError,
            "but token_map lacks ['N'] and has the other keys ['n']",
        ),
        ({**DNA_TOKENS, "N": -1}, ValueError, "the tokens of dtype uint8 must be from 0 to 255, but 'N' has -1"),
        ({**DNA_TOKENS, "N": 256}, ValueError, "the tokens of dtype uint8 must be from 0 to 255, but 'N' has 256"),
        ({**DNA_TOKENS, "N": 4.0}, TypeError, "the token of 'N' in token_map must be an int, not float"),
        ({**DNA_TOKENS, "N": True}, TypeError, "the token of 'N' in token_map must be an int, not bool"),
        ({**DNA_TOKENS, "N": "4"}, TypeError, "the token of 'N' in token_map must be an int, not str"),
        ([("A", 0)], TypeError, "token_map must be a mapping from letters to tokens or None, not list"),
    ],
)
def test_tokenizer_token_map_invalid(token_map, error, match):
    with pytest.raises(error, match=re.escape(match) + "$"):
        Tokenizer(DNA, token_map=token_map)


def test_tokenizer_token_map_numpy_ints():
    token_map = {letter: np.int64(token) for letter, token in DNA_TOKENS.items()}
    assert np.array_equal(Tokenizer(token_map=token_map).tokenize("ACGTN"), [7, 8, 9, 10, 11])


def test_tokenizer_keyword_only():
    with pytest.raises(TypeError):
        Tokenizer(DNA, DNA_TOKENS)


@pytest.mark.parametrize("token_map", [None, DNA_TOKENS, {**DNA_TOKENS, "G": 255}])
@pytest.mark.parametrize("dtype", [np.uint8, np.int64])
@pytest.mark.parametrize("seq", ["", "ACGTN", "NTGCAAAN", "ACGT" + IUPAC_CODES])
def test_tokenize_translate_table_equals_dict_lookup(seq, dtype, token_map):
    fast = Tokenizer(DNA, token_map=token_map, dtype=dtype)
    slow_map = None if token_map is None else {**token_map, "é": 0}
    slow = Tokenizer(_non_ascii(DNA), token_map=slow_map, dtype=dtype)
    np.testing.assert_array_equal(fast.tokenize(seq), slow.tokenize(seq), strict=True)


@pytest.mark.parametrize("seq", ["ACGTX", "acgtx", "ACGTU", "ACGTr", "ACGT\xff"])
def test_tokenize_translate_table_equals_dict_lookup_unknown_letters(seq):
    with pytest.raises(ValueError) as excinfo:
        Tokenizer(_non_ascii(DNA)).tokenize(seq)
    with pytest.raises(ValueError, match=f"^{re.escape(str(excinfo.value))}$"):
        Tokenizer().tokenize(seq)


@pytest.mark.parametrize(
    "tokenizer",
    [Tokenizer(), Tokenizer(dtype=np.int64), Tokenizer(token_map={**DNA_TOKENS, "G": 255}), Tokenizer(CODONS)],
)
def test_tokenize_returns_a_writable_array(tokenizer):
    # torch.from_numpy warns for read-only arrays
    seq = "ACGTGANNN" if tokenizer.alphabet == CODONS else "ACGTN"
    assert tokenizer.tokenize(seq).flags.writeable


def test_tokenize_is_case_sensitive():
    # an alphabet in lowercase encodes lowercase letters only
    lowercase = Alphabet(list("acgt"), ambiguous={"n": list("acgt")})
    assert np.array_equal(Tokenizer(lowercase).tokenize("acgtn"), [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match=re.escape("not in the alphabet: ['A']") + "$"):
        Tokenizer(lowercase).tokenize("Acgt")
    # "A" and "a" can be different letters
    assert np.array_equal(Tokenizer(Alphabet(list("AaCc"))).tokenize("AaCc"), [0, 1, 2, 3])
    # a letter with several characters matches only in the same case
    expected = "not in the alphabet: ['ACg', 'nnN']" + _upper_hint("the alphabet", ["ACg", "nnN"])
    with pytest.raises(ValueError, match=re.escape(expected) + "$"):
        Tokenizer(CODONS).tokenize("ACgTGAnnN")


def test_tokenize_non_ascii_letters():
    alphabet = Alphabet(["é", "A"], ambiguous={"ñ": ["é", "A"]})
    assert np.array_equal(Tokenizer(alphabet).tokenize("éAñé"), [0, 1, 2, 0])
    # "É" is not ASCII, so the error has no hint
    with pytest.raises(ValueError, match=re.escape("not in the alphabet: ['É']") + "$"):
        Tokenizer(alphabet).tokenize("éÉ")


def test_tokenize_big_alphabet():
    letters = [f"{i:03}" for i in range(300)]
    tokens = Tokenizer(Alphabet(letters, ambiguous={"NNN": letters}), dtype=np.int16).tokenize("000299NNN")
    assert tokens.dtype == np.int16
    assert tokens.tolist() == [0, 299, 300]


# Encoder


def test_one_hot():
    encoder = Encoder()
    x = encoder.one_hot("ACGTTNR")
    assert x.dtype == np.uint8
    assert np.array_equal(x, [*np.eye(4)[[0, 1, 2, 3, 3]], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert encoder.one_hot("").shape == (0, 4)
    assert encoder.one_hot("").dtype == np.uint8
    assert encoder == Encoder(DNA, ambiguous="zeros", dtype=np.uint8)
    assert encoder.ambiguous == "zeros"


def test_one_hot_fractional():
    encoder = Encoder(DNA, ambiguous="fractional", dtype=np.float32)
    assert encoder.ambiguous == "fractional"
    x = encoder.one_hot("ACGTNRB")
    assert x.dtype == np.float32
    # N as in one_hot_dna of kipoiseq and kipoiseq2 0.1, R as half A and half G, and B as a third of C, G and T
    third = np.float32(1 / 3)
    np.testing.assert_array_equal(x, [*np.eye(4), [0.25] * 4, [0.5, 0, 0.5, 0], [0, third, third, third]])


def test_one_hot_ones():
    encoder = Encoder(DNA, ambiguous="ones")
    assert encoder.ambiguous == "ones"
    x = encoder.one_hot("ACGTNRB")
    assert x.dtype == np.uint8
    # N as 1 in every column, R as A and G, and B as C, G and T
    assert np.array_equal(x, [*np.eye(4), [1, 1, 1, 1], [1, 0, 1, 0], [0, 1, 1, 1]])


@pytest.mark.parametrize("alphabet", [DNA, RNA, AMINO_ACIDS])
@pytest.mark.parametrize("dtype", [np.uint8, np.int8, bool, np.float32])
def test_ones(alphabet, dtype):
    # a code of k letters gives 1 in each of its k columns, in any dtype
    encoder = Encoder(alphabet, ambiguous="ones", dtype=dtype)
    seq = "".join((*alphabet.letters, *alphabet.ambiguous))
    code_rows = [_code_row(alphabet.letters, letters, "ones") for letters in alphabet.ambiguous.values()]
    x = encoder.one_hot(seq)
    np.testing.assert_array_equal(x, np.array([*np.eye(len(alphabet.letters)), *code_rows], dtype=dtype), strict=True)
    # each code has its own row, so decode gives back every code, also the first one
    assert encoder.decode(x) == seq
    assert encoder.decode(x[::-1]) == seq[::-1]
    # the row of zeros and the row of "fractional" of the first code are not the row of a code
    n_columns = len(alphabet.letters)
    message = "the rows [1, 2] of x are neither the row of a letter nor the row of an ambiguity code"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        encoder.decode(np.vstack([np.eye(n_columns)[:1], np.zeros(n_columns), np.full(n_columns, 1 / n_columns)]))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, ">f4"])
def test_one_hot_fractional_dtype(dtype):
    x = Encoder(ambiguous="fractional", dtype=dtype).one_hot("ACGTNH")
    assert x.dtype == dtype
    expected = np.array([*np.eye(4), [0.25] * 4, [1 / 3, 1 / 3, 0, 1 / 3]], dtype=dtype)
    np.testing.assert_array_equal(x, expected, strict=True)


# the arrays keep a byte order that is not the native one
@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int64, np.float16, np.float32, np.float64, bool, ">f4", ">u2"])
def test_one_hot_dtype(dtype):
    x = Encoder(dtype=dtype).one_hot("ACGTN")
    assert x.dtype == dtype
    assert np.array_equal(x, [*np.eye(4), [0] * 4])


def test_one_hot_alphabets():
    assert np.array_equal(Encoder(RNA).one_hot("ACGUN"), [*np.eye(4), [0] * 4])
    with pytest.raises(ValueError, match=re.escape("['T']")):
        Encoder(RNA).one_hot("ACGT")


@pytest.mark.parametrize("ambiguous, dtype", RULES)
def test_one_hot_amino_acids(ambiguous, dtype):
    # 20 columns. N is asparagine, X is any amino acid, B is D or N, and Z is E or Q.
    x = Encoder(AMINO_ACIDS, ambiguous=ambiguous, dtype=dtype).one_hot("NXBZ")
    assert x.shape == (4, 20)
    column = {letter: i for i, letter in enumerate(AMINO_ACIDS.letters)}
    assert np.array_equal(x[0], np.eye(20)[column["N"]])
    if ambiguous == "zeros":
        assert not x[1:].any()
    else:
        np.testing.assert_array_equal(x[1], np.full(20, 1 / 20 if ambiguous == "fractional" else 1, dtype=dtype))
        assert x[2].nonzero()[0].tolist() == sorted([column["D"], column["N"]])
        assert x[3].nonzero()[0].tolist() == sorted([column["E"], column["Q"]])
        assert x[2].max() == x[3].max() == (1 / 2 if ambiguous == "fractional" else 1)


@pytest.mark.parametrize("alphabet", [DNA, RNA])
def test_one_hot_iupac_codes(alphabet):
    seq = "".join(alphabet.letters) + IUPAC_CODES
    # "zeros" gives a row of zeros for every code
    assert np.array_equal(Encoder(alphabet).one_hot(seq), [*np.eye(4), *[[0] * 4] * 11])
    # "fractional" gives 1 / k in the columns of the k bases of the code
    letters = "".join(alphabet.letters)
    bases = {code: bases.replace("T", letters[3]) for code, bases in IUPAC_BASES.items()}
    expected = [*np.eye(4), *[_code_row(alphabet.letters, bases[code], "fractional") for code in IUPAC_CODES]]
    x = Encoder(alphabet, ambiguous="fractional", dtype=np.float64).one_hot(seq)
    np.testing.assert_array_equal(x, expected)
    # "ones" gives 1 in the columns of the k bases of the code
    expected = [*np.eye(4), *[_code_row(alphabet.letters, bases[code], "ones") for code in IUPAC_CODES]]
    np.testing.assert_array_equal(Encoder(alphabet, ambiguous="ones").one_hot(seq), expected)


def test_one_hot_letters_with_several_characters():
    x = Encoder(CODONS, ambiguous="fractional", dtype=np.float32).one_hot("ACGTGANNN")
    assert np.array_equal(x, [[1, 0], [0, 1], [0.5, 0.5]])
    assert Encoder(CODONS).one_hot("").shape == (0, 2)


@pytest.mark.parametrize("ambiguous, dtype", RULES)
def test_one_hot_alphabet_without_codes(ambiguous, dtype):
    alphabet = Alphabet(DNA.letters)
    assert np.array_equal(Encoder(alphabet, ambiguous=ambiguous, dtype=dtype).one_hot("ACGT"), np.eye(4))
    with pytest.raises(ValueError, match=re.escape("['N']")):
        Encoder(alphabet, ambiguous=ambiguous, dtype=dtype).one_hot("ACGTN")
    assert np.array_equal(Encoder(Alphabet(["A"]), ambiguous=ambiguous, dtype=dtype).one_hot("AA"), [[1], [1]])


def test_one_hot_codes_with_the_same_letters():
    # e.g. N and * for any base, as in the alphabet of MMSplice
    alphabet = Alphabet(DNA.letters, ambiguous={"N": DNA.letters, "*": DNA.letters})
    encoder = Encoder(alphabet, ambiguous="fractional", dtype=np.float32)
    assert np.array_equal(encoder.one_hot("AN*"), [[1, 0, 0, 0], [0.25] * 4, [0.25] * 4])
    # both codes have the same row, so decode gives the first one
    assert encoder.decode(encoder.one_hot("AN*")) == "ANN"
    assert np.array_equal(Tokenizer(alphabet).tokenize("AN*"), [0, 4, 5])


@pytest.mark.parametrize("ambiguous, dtype", RULES)
def test_one_hot_big_alphabet(ambiguous, dtype):
    # 301 rows, so the internal tokens do not fit uint8
    letters = [f"{i:03}" for i in range(300)]
    encoder = Encoder(Alphabet(letters, ambiguous={"NNN": letters}), ambiguous=ambiguous, dtype=dtype)
    x = encoder.one_hot("000299NNN")
    assert x.shape == (3, 300)
    assert x[0, 0] == x[1, 299] == 1
    assert x[:2].sum() == 2
    value = {"zeros": 0, "fractional": 1 / 300, "ones": 1}[ambiguous]
    assert np.array_equal(x[2], np.full(300, value, dtype=dtype))
    assert encoder.decode(x) == "000299NNN"


@pytest.mark.parametrize("ambiguous, dtype", [*RULES, ("zeros", bool), ("fractional", np.float16), ("ones", bool)])
@pytest.mark.parametrize("seq", ["", "ACGTN", "NTGCAAAN", "ACGT" + IUPAC_CODES])
def test_one_hot_translate_table_equals_dict_lookup(seq, ambiguous, dtype):
    fast = Encoder(DNA, ambiguous=ambiguous, dtype=dtype)
    slow = Encoder(_non_ascii(DNA), ambiguous=ambiguous, dtype=dtype)
    np.testing.assert_array_equal(fast.one_hot(seq), slow.one_hot(seq), strict=True)


def test_one_hot_returns_a_writable_array():
    assert Encoder().one_hot("ACGTN").flags.writeable
    assert Encoder(ambiguous="fractional", dtype=np.float32).one_hot("ACGTN").flags.writeable


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
def test_one_hot_axes(dummy_axis, swap, shape):
    x = Encoder().one_hot("ACGTNACGTA")
    out = x if dummy_axis is None else np.expand_dims(x, dummy_axis)
    out = out if swap is None else np.swapaxes(out, *swap)
    assert out.shape == shape
    # the alphabet axis holds the one-hot vectors
    alphabet_axis = shape.index(4)
    assert np.array_equal(np.moveaxis(out, alphabet_axis, -1).reshape(10, 4), x)


@pytest.mark.parametrize("ambiguous", ["Zeros", "fraction", "one", "", None, 0.25, 1, True, ["zeros"]])
def test_encoder_ambiguous_invalid(ambiguous):
    expected = f'ambiguous must be "zeros", "fractional" or "ones", got {ambiguous!r}'
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        Encoder(ambiguous=ambiguous, dtype=np.float32)


@pytest.mark.parametrize("dtype", [np.uint8, np.int64, bool, ">u2"])
def test_encoder_fractional_needs_a_floating_point_dtype(dtype):
    expected = f'ambiguous="fractional" needs a floating-point dtype, e.g. dtype=np.float32, got {np.dtype(dtype)}'
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        Encoder(ambiguous="fractional", dtype=dtype)
    # also without codes, so the rule alone sets the valid dtypes
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        Encoder(Alphabet(DNA.letters), ambiguous="fractional", dtype=dtype)


@pytest.mark.parametrize("dtype", ["U1", object, np.complex64, "datetime64[s]"])
def test_encoder_dtype_not_a_number_type(dtype):
    with pytest.raises(TypeError, match="^dtype must be bool, an integer type or a floating-point type, got "):
        Encoder(dtype=dtype)


@pytest.mark.parametrize("make", [Encoder, Tokenizer])
def test_dtype_unknown_to_numpy(make):
    with pytest.raises(TypeError, match="^dtype must be a numpy dtype, got 'foo'$"):
        make(dtype="foo")


@pytest.mark.parametrize("make", [Encoder, Tokenizer])
def test_dtype_none(make):
    # numpy reads None as float64, so the encoders reject it instead of returning float64 silently
    with pytest.raises(TypeError, match="^dtype must be a numpy dtype, got None$"):
        make(dtype=None)


def test_encoder_keyword_only():
    with pytest.raises(TypeError):
        Encoder(DNA, "zeros")


# decode


@pytest.mark.parametrize(
    "encoder, seq",
    [
        (Encoder(), "ACGTNACGTN"),
        (Encoder(ambiguous="fractional", dtype=np.float32), "ACGTNACGTN"),
        (Encoder(ambiguous="fractional", dtype=np.float32), "ACGT" + IUPAC_CODES),
        (Encoder(ambiguous="fractional", dtype=np.float16), "ACGT" + IUPAC_CODES[::-1]),
        (Encoder(ambiguous="ones"), "ACGTNACGTN"),
        (Encoder(ambiguous="ones"), "ACGT" + IUPAC_CODES),
        (Encoder(ambiguous="ones", dtype=bool), "ACGT" + IUPAC_CODES[::-1]),
        (Encoder(dtype=bool), "ACGTNACGTN"),
        (Encoder(AMINO_ACIDS), "MNX"),
        (Encoder(AMINO_ACIDS, ambiguous="fractional", dtype=np.float64), "MNXBZ"),
        (Encoder(AMINO_ACIDS, ambiguous="ones", dtype=np.int8), "XMNXBZ"),
        (Encoder(RNA, ambiguous="fractional", dtype=np.float32), "ACGU" + IUPAC_CODES),
        (Encoder(RNA, ambiguous="ones", dtype=np.float32), "ACGU" + IUPAC_CODES),
        (Encoder(CODONS), "ACGTGANNN"),
        (Encoder(CODONS, ambiguous="fractional", dtype=np.float32), "ACGTGANNN"),
        (Encoder(CODONS, ambiguous="ones"), "NNNACGTGANNN"),
        (Encoder(Alphabet(DNA.letters)), "ACGT"),
        (Encoder(Alphabet(["é", "A"], ambiguous={"ñ": ["é", "A"]})), "éAñé"),
        (Encoder(_non_ascii(DNA), ambiguous="fractional", dtype=np.float32), "ACGT" + IUPAC_CODES),
        (Encoder(_non_ascii(DNA), ambiguous="ones"), "ACGT" + IUPAC_CODES),
        (Encoder(), ""),
        (Encoder(ambiguous="fractional", dtype=np.float32), ""),
        (Encoder(ambiguous="ones"), ""),
    ],
)
def test_decode(encoder, seq):
    assert encoder.decode(encoder.one_hot(seq)) == seq


@pytest.mark.parametrize("alphabet", [DNA, RNA])
def test_decode_iupac_codes(alphabet):
    # with "zeros", all codes have the row of zeros, which becomes the first code N
    encoder = Encoder(alphabet)
    assert encoder.decode(encoder.one_hot("A" + IUPAC_CODES)) == "A" + "N" * 11
    # with "fractional" and "ones", each code has its own row
    encoder = Encoder(alphabet, ambiguous="fractional", dtype=np.float32)
    assert encoder.decode(encoder.one_hot("A" + IUPAC_CODES)) == "A" + IUPAC_CODES
    encoder = Encoder(alphabet, ambiguous="ones")
    assert encoder.decode(encoder.one_hot("A" + IUPAC_CODES)) == "A" + IUPAC_CODES


@pytest.mark.parametrize(
    "alphabet, seq",
    [
        (DNA, "NTGCAAAN"),
        (DNA, "ANRY"),
        (DNA, "ACGT" + IUPAC_CODES),
        (Alphabet(DNA.letters), "TGCA"),
        (AMINO_ACIDS, "MNXBZ"),
        (DNA, ""),
    ],
)
@pytest.mark.parametrize("ambiguous, dtype", RULES)
def test_decode_byte_lookup_equals_join(alphabet, seq, ambiguous, dtype):
    # the extra code é turns off the byte lookup, so decode joins the letters and codes
    fast = Encoder(alphabet, ambiguous=ambiguous, dtype=dtype)
    slow = Encoder(_non_ascii(alphabet), ambiguous=ambiguous, dtype=dtype)
    x = fast.one_hot(seq)
    assert fast.decode(x) == slow.decode(x)


@pytest.mark.parametrize("ambiguous, dtype", RULES)
def test_decode_any_memory_layout(ambiguous, dtype):
    # decode copies the columns of x, so views and Fortran order give the same letters
    encoder = Encoder(ambiguous=ambiguous, dtype=dtype)
    x = encoder.one_hot("ACGTNACGTN")
    assert encoder.decode(np.asfortranarray(x)) == "ACGTNACGTN"
    assert encoder.decode(x[::-1]) == "NTGCANTGCA"
    assert encoder.decode(x[::2]) == "AGNCT"


def test_decode_other_dtypes():
    # decode compares the values, so the dtype of x can differ from the dtype of the encoder
    x = np.array([[0, 0, 1, 0], [0.25, 0.25, 0.25, 0.25], [0.5, 0, 0.5, 0]])
    assert Encoder(ambiguous="fractional", dtype=np.float32).decode(x) == "GNR"
    assert Encoder().decode(np.array([[False, True, False, False], [False] * 4])) == "CN"
    assert Encoder(ambiguous="ones").decode(np.array([[True, False, True, False], [True] * 4])) == "RN"
    assert Encoder(ambiguous="ones", dtype=bool).decode(np.array([[1.0, 0, 1, 0], [1, 1, 1, 1]])) == "RN"
    # the float32 values of one_hot keep their value in float64
    x = Encoder(ambiguous="fractional", dtype=np.float32).one_hot("ACGTB")
    assert Encoder(ambiguous="fractional", dtype=np.float32).decode(x.astype(np.float64)) == "ACGTB"


@pytest.mark.parametrize(
    "encoder, x, error, match",
    [
        (Encoder(), np.eye(4).tolist(), TypeError, "^x must be a numpy array of numbers, got "),
        (Encoder(), np.array([["A", "C", "G", "T"]]), TypeError, "^x must be a numpy array of numbers, got "),
        (Encoder(), np.zeros((2, 4, 4)), ValueError, re.escape("x must have the shape (length, 4), got (2, 4, 4)")),
        (Encoder(AMINO_ACIDS), np.eye(4), ValueError, re.escape("x must have the shape (length, 20), got (4, 4)")),
        (Encoder(), np.zeros(4), ValueError, re.escape("x must have the shape (length, 4), got (4,)")),
        # soft values, as in model predictions
        (
            Encoder(),
            np.array([[0.1, 0.7, 0.1, 0.1], [1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0.5]]),
            ValueError,
            re.escape("the rows [0, 2, 3] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        (
            Encoder(ambiguous="fractional", dtype=np.float32),
            np.array([[0.1, 0.7, 0.1, 0.1], [0.5, 0, 0, 0], [0.5, 0.5, 0.5, 0], [0.25, 0.25, 0.5, 0], [1, 0, 0, 0]]),
            ValueError,
            re.escape("the rows [0, 1, 2, 3] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        # the row of N with the other rule
        (
            Encoder(),
            np.full((1, 4), 0.25),
            ValueError,
            re.escape("the rows [0] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        (
            Encoder(ambiguous="fractional", dtype=np.float32),
            np.zeros((1, 4)),
            ValueError,
            re.escape("the rows [0] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        # decode compares the values exactly, and 1/3 in float64 differs from 1/3 in float32
        (
            Encoder(ambiguous="fractional", dtype=np.float32),
            np.array([[0, 1 / 3, 1 / 3, 1 / 3]]),
            ValueError,
            re.escape("the rows [0] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        # the rows of zeros and of "fractional" with "ones", and values other than 0 and 1
        (
            Encoder(ambiguous="ones"),
            np.array([[0, 0, 0, 0], [0.25] * 4, [0.5, 0, 0.5, 0], [1, 1, 1, 1], [2, 0, 0, 0], [1, 0, 1, 0.5]]),
            ValueError,
            re.escape("the rows [0, 1, 2, 4, 5] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        # 1 in the columns of letters that no code stands for
        (
            Encoder(STRICT_DNA, ambiguous="ones"),
            np.array([[1, 1, 1, 1], [1, 0, 1, 0]]),
            ValueError,
            re.escape("the rows [1] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        (
            Encoder(AMINO_ACIDS, ambiguous="ones"),
            np.array([np.ones(20), np.eye(20)[0] + np.eye(20)[1]]),
            ValueError,
            re.escape("the rows [1] of x are neither the row of a letter nor the row of an ambiguity code"),
        ),
        (
            Encoder(Alphabet(DNA.letters)),
            np.zeros((1, 4)),
            ValueError,
            re.escape("the rows [0] of x are not the row of a letter"),
        ),
        (
            Encoder(Alphabet(DNA.letters), ambiguous="ones"),
            np.ones((1, 4)),
            ValueError,
            re.escape("the rows [0] of x are not the row of a letter"),
        ),
        (
            Encoder(Alphabet(DNA.letters), ambiguous="fractional", dtype=np.float32),
            np.full((1, 4), 0.25),
            ValueError,
            re.escape("the rows [0] of x are not the row of a letter"),
        ),
        (
            Encoder(),
            np.full((12, 4), np.nan),
            ValueError,
            re.escape("the rows [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] of x are neither"),
        ),
        (
            Encoder(ambiguous="fractional", dtype=np.float32),
            np.full((12, 4), np.nan),
            ValueError,
            re.escape("the rows [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] of x are neither"),
        ),
        (
            Encoder(ambiguous="ones"),
            np.full((12, 4), np.nan),
            ValueError,
            re.escape("the rows [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] of x are neither"),
        ),
    ],
)
def test_decode_invalid(encoder, x, error, match):
    with pytest.raises(error, match=match):
        encoder.decode(x)


# errors for unknown letters


@pytest.mark.parametrize("make", [Tokenizer, Encoder])
@pytest.mark.parametrize(
    "seq, unknown",
    [
        ("ACGTX", ["X"]),
        ("AC GT-", [" ", "-"]),
        ("AC\x00\x7f", ["\x00", "\x7f"]),
        # byte 255 in Latin-1
        ("AC\xff", ["\xff"]),
        ("ACGé☃", ["é", "☃"]),
    ],
)
def test_unknown_letters(make, seq, unknown):
    expected = f"seq contains letters that are not in the alphabet: {sorted(unknown)}"
    encode = make().tokenize if make is Tokenizer else make().one_hot
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$") as excinfo:
        encode(seq)
    assert not isinstance(excinfo.value, UnicodeError)


@pytest.mark.parametrize("make", [Tokenizer, Encoder])
@pytest.mark.parametrize("seq, unknown", [("acgtn", "acgnt"), ("AcGtN", "ct"), ("ACGTn", "n"), ("ACGTr", "r")])
def test_unknown_letters_hint_at_upper(make, seq, unknown):
    # soft-masked repeats are lowercase, so the error hints at seq.upper()
    encode = make().tokenize if make is Tokenizer else make().one_hot
    expected = f"seq contains letters that are not in the alphabet: {sorted(unknown)}"
    expected += _upper_hint("the alphabet", sorted(unknown))
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        encode(seq)
    assert len(encode(seq.upper())) == len(seq)


def test_unknown_letters_hint_at_upper_only_for_ascii():
    # str.upper maps the non-ASCII "ſ" and "ı" to "S" and "I", but the lookup does not call it
    tokenizer = Tokenizer(Alphabet(["S", "I"]))
    assert np.array_equal(tokenizer.tokenize("SI"), [0, 1])
    with pytest.raises(ValueError, match=re.escape("not in the alphabet: ['ı', 'ſ']") + "$"):
        tokenizer.tokenize("ſı")
    # the uppercase form of "x" is not in the alphabet
    with pytest.raises(ValueError, match=re.escape("not in the alphabet: ['x']") + "$"):
        Tokenizer().tokenize("ACGTx")


@pytest.mark.parametrize("make", [Tokenizer, Encoder])
@pytest.mark.parametrize(
    "alphabet, seq, unknown",
    [
        # U is not a DNA base, and X and - are no IUPAC ambiguity codes
        (DNA, "ACGTUX-", ["-", "U", "X"]),
        (RNA, "ACGUT", ["T"]),
        # the strict alphabet has N as its only code
        (STRICT_DNA, "ACGTNRY", ["R", "Y"]),
        (Alphabet(DNA.letters), "ACGTN", ["N"]),
    ],
)
def test_unknown_letters_without_iupac_hint(make, alphabet, seq, unknown):
    # the message names the unknown letters, without a hint at an alphabet with the IUPAC ambiguity codes
    encode = make(alphabet).tokenize if make is Tokenizer else make(alphabet).one_hot
    expected = f"seq contains letters that are not in the alphabet: {unknown}"
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        encode(seq)


def test_unknown_letters_hint_at_upper_with_other_unknown_letters():
    # the hint names only the letters whose uppercase form is in the alphabet
    expected = "seq contains letters that are not in the alphabet: ['U', 'a']"
    expected += _upper_hint("the alphabet", ["a"])
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        Encoder().one_hot("aCGTU")


@pytest.mark.parametrize("make", [Tokenizer, Encoder])
def test_seq_not_a_str(make):
    encode = make().tokenize if make is Tokenizer else make().one_hot
    with pytest.raises(TypeError, match="^seq must be a str, not list$"):
        encode(["A", "C"])
    with pytest.raises(TypeError, match="^seq must be a str, not bytes$"):
        encode(b"AC")


@pytest.mark.parametrize("make", [Tokenizer, Encoder])
def test_seq_length_not_a_multiple_of_the_letter_length(make):
    encode = make(CODONS).tokenize if make is Tokenizer else make(CODONS).one_hot
    expected = "len(seq) must be a multiple of the letter length 3, got len(seq)=8"
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        encode("ACGTGANN")


@pytest.mark.parametrize("make", [Tokenizer, Encoder])
@pytest.mark.parametrize("alphabet", ["ACGT", list("ACGT"), ("A", "C", "G", "T"), None])
def test_alphabet_not_an_alphabet(make, alphabet):
    with pytest.raises(TypeError, match=f"^alphabet must be an Alphabet, not {type(alphabet).__name__}, e.g. "):
        make(alphabet)


# both classes

ENCODERS = [
    Encoder(),
    Encoder(DNA, ambiguous="fractional", dtype=np.float32),
    Encoder(AMINO_ACIDS, ambiguous="fractional", dtype=np.float16),
    Encoder(CODONS, ambiguous="fractional", dtype=np.float64),
    Encoder(DNA, ambiguous="ones", dtype=bool),
    Tokenizer(),
    Tokenizer(DNA, token_map=DNA_TOKENS),
    Tokenizer(AMINO_ACIDS),
    Tokenizer(CODONS, dtype=np.int64),
]


def _encode(encoder, seq):
    return encoder.tokenize(seq) if isinstance(encoder, Tokenizer) else encoder.one_hot(seq)


def _seq(encoder):
    return "".join((*encoder.alphabet.letters, *encoder.alphabet.ambiguous) * 3)


@pytest.mark.parametrize("encoder", ENCODERS, ids=repr)
def test_pickle(encoder):
    # the workers of a data loader receive the encoder as a pickle
    for copied in [pickle.loads(pickle.dumps(encoder)), copy.copy(encoder), copy.deepcopy(encoder)]:
        assert copied == encoder
        assert hash(copied) == hash(encoder)
        np.testing.assert_array_equal(_encode(copied, _seq(encoder)), _encode(encoder, _seq(encoder)), strict=True)


@pytest.mark.parametrize("encoder", ENCODERS, ids=repr)
def test_frozen(encoder):
    with pytest.raises(AttributeError):
        encoder.alphabet = RNA
    with pytest.raises(AttributeError):
        encoder.dtype = np.dtype(np.int64)
    with pytest.raises(AttributeError):
        encoder.other = 1


def test_encoder_ambiguous_is_read_only():
    encoder = Encoder(ambiguous="fractional", dtype=np.float32)
    with pytest.raises(AttributeError):
        encoder.ambiguous = "zeros"


def test_safe_as_default_argument():
    def one_hot(seq, encoder=Encoder()):
        return encoder.one_hot(seq)

    def tokenize(seq, tokenizer=Tokenizer()):
        return tokenizer.tokenize(seq)

    # changing a result does not change the lookup of the next call
    one_hot("ACGTN")[:] = 7
    tokenize("ACGTN")[:] = 7
    assert np.array_equal(one_hot("ACGTN"), [*np.eye(4), [0] * 4])
    assert np.array_equal(tokenize("ACGTN"), [0, 1, 2, 3, 4])


def test_eq_and_hash():
    assert Encoder() == Encoder(DNA, ambiguous="zeros", dtype="uint8")
    assert hash(Encoder()) == hash(Encoder(DNA, ambiguous="zeros", dtype="uint8"))
    assert Encoder(dtype=np.float32) != Encoder(ambiguous="fractional", dtype=np.float32)
    assert Encoder() != Encoder(ambiguous="ones")
    assert Encoder() != Encoder(dtype=np.int8)
    assert Encoder() != Encoder(STRICT_DNA)
    assert Encoder() != Tokenizer()
    same_dna = Alphabet(list("ACGT"), ambiguous={code: list(bases) for code, bases in IUPAC_BASES.items()})
    assert Encoder() == Encoder(same_dna)
    assert Tokenizer() == Tokenizer(same_dna)
    assert hash(Tokenizer()) == hash(Tokenizer(same_dna))
    # the order of the codes sets their tokens
    reordered_dna = Alphabet(DNA.letters, ambiguous=dict(reversed(DNA.ambiguous.items())))
    assert Tokenizer() != Tokenizer(reordered_dna)
    reordered = dict(reversed(DNA_TOKENS.items()))
    assert Tokenizer(token_map=DNA_TOKENS) == Tokenizer(token_map=reordered)
    assert hash(Tokenizer(token_map=DNA_TOKENS)) == hash(Tokenizer(token_map=reordered))
    assert Tokenizer(token_map=DNA_TOKENS) != Tokenizer(token_map={**DNA_TOKENS, "N": 12})
    # the attributes are the configuration as given, so the default tokens as a token_map are another configuration
    default_tokens = {letter: token for token, letter in enumerate([*"ACGT", *IUPAC_CODES])}
    assert Tokenizer(token_map=default_tokens) != Tokenizer()
    assert np.array_equal(Tokenizer(token_map=default_tokens).tokenize("ACGTNR"), Tokenizer().tokenize("ACGTNR"))
    assert len({Encoder(), Encoder(), Tokenizer(), Tokenizer()}) == 2


def test_repr():
    assert repr(Encoder()) == f"Encoder(alphabet={DNA!r}, ambiguous='zeros', dtype=dtype('uint8'))"
    expected = f"Encoder(alphabet={DNA!r}, ambiguous='fractional', dtype=dtype('float32'))"
    assert repr(Encoder(ambiguous="fractional", dtype=np.float32)) == expected
    assert repr(Tokenizer()) == f"Tokenizer(alphabet={DNA!r}, token_map=None, dtype=dtype('uint8'))"
    expected = f"Tokenizer(alphabet={DNA!r}, token_map={DNA_TOKENS!r}, dtype=dtype('int64'))"
    assert repr(Tokenizer(token_map=DNA_TOKENS, dtype=np.int64)) == expected


def test_custom_vectors_with_the_default_tokens():
    # a fifth column for the ambiguity codes, as in the README
    rows = np.zeros((15, 5), dtype=np.float32)
    rows[:4, :4] = np.eye(4)
    rows[4:, 4] = 1
    x = rows[Tokenizer().tokenize("ACGTNR")]
    assert x.dtype == np.float32
    assert np.array_equal(
        x, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], *[[0, 0, 0, 0, 1]] * 2]
    )


def test_package_exports_the_encoders():
    assert kipoiseq2.Encoder is Encoder
    assert kipoiseq2.Tokenizer is Tokenizer
    assert kipoiseq2.encoders.__all__ == ["Encoder", "Tokenizer"]
