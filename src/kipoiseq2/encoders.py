"""Encode sequences as one-hot arrays or integer tokens.

:class:`Encoder` one-hot encodes sequences, and :class:`Tokenizer` converts them to integer tokens.
Each holds the complete configuration of its output: the alphabet, the rule for the codes or the tokens, and the dtype.
The constructor checks the configuration and builds the lookup once, so the methods only apply it.
Both classes are frozen, hashable and picklable.
So they are safe as default arguments and in the worker processes of a data loader.

The lookup is case-sensitive: ``"a"`` and ``"A"`` are different letters.
So a lowercase letter, e.g. of a soft-masked repeat, is unknown unless the alphabet contains it.
Uppercase the sequences at the source instead, e.g. with ``seq.upper()``,
or read them with ``FastaStringExtractor``, which uppercases by default.
An unknown letter raises a ValueError that names it.

Example:

>>> from kipoiseq2 import Encoder, Tokenizer
>>> Encoder().one_hot("ACGTN")
array([[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 1],
       [0, 0, 0, 0]], dtype=uint8)
>>> Tokenizer().tokenize("ACGTNR")
array([0, 1, 2, 3, 4, 5], dtype=uint8)
"""

import functools
from collections.abc import Container, Mapping
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import DTypeLike, NDArray

from kipoiseq2.alphabets import DNA, Alphabet
from kipoiseq2.dataclasses import _check_int

__all__ = ["Encoder", "Tokenizer"]

_SENTINEL = 255
"""The value of the translate table for the bytes that are not letters."""

_AmbiguousRule = Literal["zeros", "fractional", "ones"]
"""Type of the rule of :class:`Encoder` for the rows of the ambiguity codes."""
_AMBIGUOUS_RULES = get_args(_AmbiguousRule)
"""The values of ``_AmbiguousRule``, for the checks at runtime."""


def _check_alphabet(alphabet: object) -> None:
    """Raise a TypeError if ``alphabet`` is not an :class:`~kipoiseq2.alphabets.Alphabet`."""
    if not isinstance(alphabet, Alphabet):
        raise TypeError(
            f"alphabet must be an Alphabet, not {type(alphabet).__name__}, "
            'e.g. kipoiseq2.DNA or Alphabet(["A", "C", "G", "T"], ambiguous={"N": ["A", "C", "G", "T"]})'
        )


def _letters_and_codes(alphabet: Alphabet) -> tuple[str, ...]:
    """Return the letters of ``alphabet``, then its ambiguity codes.

    The index of each item is its default token in :class:`Tokenizer` and its row in the table of :class:`Encoder`.
    """
    return (*alphabet.letters, *alphabet.ambiguous)


def _as_dtype(dtype: DTypeLike) -> np.dtype[Any]:
    """Return ``dtype`` as a numpy dtype, or raise a TypeError if it is None or if numpy does not know it."""
    # numpy reads None as float64, so a dtype=None would silently give float64 arrays
    if dtype is None:
        raise TypeError("dtype must be a numpy dtype, got None")
    try:
        return np.dtype(dtype)
    except TypeError:
        raise TypeError(f"dtype must be a numpy dtype, got {dtype!r}") from None


def _unknown_letters_error(unknown: set[str], known: Container[str], kind: str, name: str) -> ValueError:
    """Return the ValueError for the letters or codons of seq that are not in ``known``.

    The message hints at ``seq.upper()`` if ``known`` contains the uppercase form of an unknown ASCII item.

    :param unknown: the unknown letters or codons of seq
    :param known: the known letters or codons
    :param kind: the word for the items in the message, e.g. ``"letters"``
    :param name: the name of ``known`` in the message, e.g. ``"the alphabet"``
    """
    message = f"seq contains {kind} that are not in {name}: {sorted(unknown)}"
    # str.upper maps some non-ASCII letters to ASCII letters, e.g. "ſ" to "S", so only ASCII items get the hint
    upper = sorted(item for item in unknown if item.isascii() and item.upper() != item and item.upper() in known)
    if upper:
        message += (
            f". The lookup is case-sensitive, but {name} has the uppercase forms of {upper}: "
            "uppercase the sequence with seq.upper(), or read it with FastaStringExtractor, which uppercases by default"
        )
    return ValueError(message)


class Tokenizer:
    """Convert sequences to integer tokens.

    By default, the letter ``alphabet.letters[i]`` becomes the token ``i``.
    Each ambiguity code of ``alphabet.ambiguous`` becomes a token of its own, from ``len(alphabet.letters)`` on,
    in the order of the codes.
    For :data:`~kipoiseq2.alphabets.DNA`, ``A``, ``C``, ``G`` and ``T`` become 0 to 3, ``N`` becomes 4,
    ``R`` becomes 5, and ``V`` becomes 14.
    So the token of a letter is its column in :meth:`Encoder.one_hot`,
    and the token of a letter or code is its row in the table of rows of :class:`Encoder`.
    An embedding with ``len(alphabet.letters) + len(alphabet.ambiguous)`` rows takes the default tokens as they are.
    A table with one row per letter raises an IndexError for the token of a code, e.g. ``np.eye(4)[tokens]``.
    ``token_map`` sets other tokens, e.g. the vocabulary of a pretrained model.

    The lookup is case-sensitive, so a lowercase letter is unknown unless ``alphabet`` contains it.

    :param alphabet: the letters and the ambiguity codes of the sequences.
        For letters with several characters, e.g. codons, a sequence is split into letters of that length.
    :param token_map: the token of each letter and code, or None for the default tokens.
        Its keys must be the letters and the codes of ``alphabet``, e.g. 15 keys for :data:`~kipoiseq2.alphabets.DNA`.
        Its tokens are ints from 0 to the largest value of ``dtype``, and several keys can share a token.
    :param dtype: integer numpy dtype of the tokens
    :raises TypeError: if ``alphabet`` is not an :class:`~kipoiseq2.alphabets.Alphabet`,
        if ``token_map`` is not a mapping or None, if a token of ``token_map`` is not an int,
        or if ``dtype`` is None or not an integer type
    :raises ValueError: if the keys of ``token_map`` are not the letters and the codes of ``alphabet``,
        or if ``dtype`` cannot hold a token, e.g. a negative token or a token above 255 in uint8

    Example:

    >>> Tokenizer().tokenize("ACGTNR")
    array([0, 1, 2, 3, 4, 5], dtype=uint8)
    >>> Tokenizer(token_map={"A": 7, "C": 8, "G": 9, "T": 10, **dict.fromkeys(DNA.ambiguous, 11)}).tokenize("ACGTNR")
    array([ 7,  8,  9, 10, 11, 11], dtype=uint8)
    >>> strict = Alphabet(["A", "C", "G", "T"], ambiguous={"N": ["A", "C", "G", "T"]})
    >>> Tokenizer(strict, token_map={"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}).tokenize("ACGTN")
    array([ 7,  8,  9, 10, 11], dtype=uint8)
    """

    __slots__ = ("_alphabet", "_token_map", "_dtype", "_letter_length", "_tokens", "_table", "_ids")

    def __init__(
        self, alphabet: Alphabet = DNA, *, token_map: Mapping[str, int] | None = None, dtype: DTypeLike = np.uint8
    ) -> None:
        _check_alphabet(alphabet)
        dtype = _as_dtype(dtype)
        if dtype.kind not in "iu":
            raise TypeError(f"dtype must be an integer type, got {dtype}")
        # the lookup treats the codes as letters
        letters = _letters_and_codes(alphabet)
        if token_map is None:
            tokens = {letter: i for i, letter in enumerate(letters)}
        elif isinstance(token_map, Mapping):
            missing = [letter for letter in letters if letter not in token_map]
            extra = [key for key in token_map if key not in letters]
            if missing or extra:
                problems = [f"lacks {missing}"] if missing else []
                problems += [f"has the other keys {extra}"] if extra else []
                raise ValueError(
                    "the keys of token_map must be the letters and the ambiguity codes of alphabet, "
                    f"but token_map {' and '.join(problems)}"
                )
            for letter, token in token_map.items():
                _check_int(token, f"the token of {letter!r} in token_map")
            tokens = {letter: int(token_map[letter]) for letter in letters}
        else:
            raise TypeError(
                f"token_map must be a mapping from letters to tokens or None, not {type(token_map).__name__}"
            )
        highest = int(np.iinfo(dtype).max)
        for letter, token in tokens.items():
            if not 0 <= token <= highest:
                raise ValueError(f"the tokens of dtype {dtype} must be from 0 to {highest}, but {letter!r} has {token}")

        # For letters of single ASCII characters, tokenize translates the bytes of seq with a table of 256 bytes.
        # There are at most 128 such letters, so their row indices stay below the sentinel 255.
        table: bytes | None = None
        ids: NDArray[np.integer[Any]] | None = None
        if len(letters[0]) == 1 and all(letter.isascii() for letter in letters):
            row_tokens = np.array([tokens[letter] for letter in letters], dtype=dtype)
            # uint8 tokens below the sentinel are rows of their own, so tokenize needs no take
            direct = dtype == np.uint8 and int(row_tokens.max()) < _SENTINEL
            translation = bytearray([_SENTINEL]) * 256
            for row, letter in enumerate(letters):
                translation[ord(letter)] = int(row_tokens[row]) if direct else row
            table = bytes(translation)
            ids = None if direct else row_tokens

        self._alphabet = alphabet
        self._token_map = None if token_map is None else MappingProxyType(tokens)
        self._dtype = dtype
        self._letter_length = len(letters[0])
        # the slow path looks up each letter in this dict
        self._tokens = tokens
        self._table = table
        self._ids = ids

    @property
    def alphabet(self) -> Alphabet:
        """The letters and the ambiguity codes of the sequences."""
        return self._alphabet

    @property
    def token_map(self) -> Mapping[str, int] | None:
        """The token of each letter and code, or None for the default tokens.

        Its keys are in the order of the letters and the codes.
        """
        return self._token_map

    @property
    def dtype(self) -> np.dtype[Any]:
        """Integer numpy dtype of the tokens."""
        return self._dtype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tokenizer):
            return NotImplemented
        return (self._alphabet, self._token_map, self._dtype) == (other._alphabet, other._token_map, other._dtype)

    def __hash__(self) -> int:
        token_map = None if self._token_map is None else tuple(self._token_map.items())
        return hash((self._alphabet, token_map, self._dtype))

    def __repr__(self) -> str:
        token_map = None if self._token_map is None else dict(self._token_map)
        return f"Tokenizer(alphabet={self._alphabet!r}, token_map={token_map!r}, dtype={self._dtype!r})"

    def __reduce__(self) -> tuple[functools.partial["Tokenizer"], tuple[Alphabet]]:
        # a mappingproxy cannot be pickled, so the tokenizer is built again from its arguments
        token_map = None if self._token_map is None else dict(self._token_map)
        return functools.partial(Tokenizer, token_map=token_map, dtype=self._dtype), (self._alphabet,)

    def tokenize(self, seq: str) -> NDArray[np.integer[Any]]:
        """Convert a sequence to tokens.

        :param seq: the sequence
        :returns: array of :attr:`dtype` and of shape ``(len(seq) // letter length,)``
        :raises TypeError: if ``seq`` is not a str
        :raises ValueError: if ``len(seq)`` is not a multiple of the letter length,
            or if ``seq`` contains a letter that is not in :attr:`alphabet`

        Example:

        >>> Tokenizer(Alphabet(["ACG", "TGA"], ambiguous={"NNN": ["ACG", "TGA"]})).tokenize("ACGNNN")
        array([0, 2], dtype=uint8)
        """
        if not isinstance(seq, str):
            raise TypeError(f"seq must be a str, not {type(seq).__name__}")
        n = self._letter_length
        if len(seq) % n != 0:
            raise ValueError(f"len(seq) must be a multiple of the letter length {n}, got len(seq)={len(seq)}")
        if self._table is not None:
            try:
                # bytearray.translate returns a bytearray, so the array of frombuffer is writable
                rows = bytearray(seq, "ascii").translate(self._table)
            except UnicodeEncodeError:
                raise self._unknown_letters_error(seq) from None
            if rows.find(_SENTINEL) >= 0:
                raise self._unknown_letters_error(seq)
            tokens = np.frombuffer(rows, dtype=np.uint8)
            return tokens if self._ids is None else self._ids.take(tokens)
        try:
            return np.array([self._tokens[seq[i : i + n]] for i in range(0, len(seq), n)], dtype=self._dtype)
        except KeyError:
            raise self._unknown_letters_error(seq) from None

    def _unknown_letters_error(self, seq: str) -> ValueError:
        """Return the ValueError that names the letters of ``seq`` that are not in :attr:`alphabet`."""
        n = self._letter_length
        letters = set(seq) if n == 1 else {seq[i : i + n] for i in range(0, len(seq), n)}
        unknown = {letter for letter in letters if letter not in self._tokens}
        return _unknown_letters_error(unknown, self._tokens, "letters", "the alphabet")


class Encoder:
    """One-hot encode sequences.

    A letter of ``alphabet.letters`` becomes a row with 1 in its column and 0 in the other columns.
    An ambiguity code of ``alphabet.ambiguous`` becomes a row by the rule ``ambiguous``:

    - ``"zeros"`` gives 0 in every column.
    - ``"fractional"`` gives ``1 / k`` in the columns of the ``k`` letters of the code, and 0 in the other columns.
      For :data:`~kipoiseq2.alphabets.DNA`, ``N`` gives 0.25 in every column,
      ``R`` gives ``[0.5, 0, 0.5, 0]``, and ``B`` gives ``[0, 1/3, 1/3, 1/3]``.
    - ``"ones"`` gives 1 in the columns of the letters of the code, and 0 in the other columns.
      For :data:`~kipoiseq2.alphabets.DNA`, ``N`` gives 1 in every column, and ``R`` gives ``[1, 0, 1, 0]``.
      ``"ones"`` keeps the letters of each code in integer dtypes.
      Its rows do not sum to 1, so a model trained on one-hot input sees ``N`` as a strong signal.

    The encoder holds a table with the rows of the letters, then the row of each code, in the order of ``alphabet``.
    The index of a row is the default token of :class:`Tokenizer`,
    so ``one_hot(seq)`` returns the rows at the tokens of ``Tokenizer(alphabet).tokenize(seq)``.
    The lookup is case-sensitive, so a lowercase letter is unknown unless ``alphabet`` contains it.

    :param alphabet: the letters of the columns and the ambiguity codes.
        For letters with several characters, e.g. codons, a sequence is split into letters of that length.
    :param ambiguous: the rule for the rows of the ambiguity codes, ``"zeros"``, ``"fractional"`` or ``"ones"``
    :param dtype: numpy dtype of the arrays: bool, an integer type or a floating-point type.
        ``"fractional"`` needs a floating-point type.
    :raises TypeError: if ``alphabet`` is not an :class:`~kipoiseq2.alphabets.Alphabet`,
        or if ``dtype`` is None or not bool, an integer type or a floating-point type
    :raises ValueError: if ``ambiguous`` is not ``"zeros"``, ``"fractional"`` or ``"ones"``,
        or if ``ambiguous`` is ``"fractional"`` and ``dtype`` is not a floating-point type

    Example:

    >>> Encoder(ambiguous="fractional", dtype=np.float32).one_hot("ACNR")
    array([[1.  , 0.  , 0.  , 0.  ],
           [0.  , 1.  , 0.  , 0.  ],
           [0.25, 0.25, 0.25, 0.25],
           [0.5 , 0.  , 0.5 , 0.  ]], dtype=float32)
    >>> Encoder(ambiguous="ones").one_hot("ACNR")
    array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [1, 1, 1, 1],
           [1, 0, 1, 0]], dtype=uint8)
    """

    __slots__ = ("_alphabet", "_ambiguous", "_dtype", "_tokenizer", "_rows", "_letter_bytes")

    def __init__(
        self, alphabet: Alphabet = DNA, *, ambiguous: _AmbiguousRule = "zeros", dtype: DTypeLike = np.uint8
    ) -> None:
        _check_alphabet(alphabet)
        if ambiguous not in _AMBIGUOUS_RULES:
            raise ValueError(f'ambiguous must be "zeros", "fractional" or "ones", got {ambiguous!r}')
        dtype = _as_dtype(dtype)
        if dtype.kind not in "biuf":
            raise TypeError(f"dtype must be bool, an integer type or a floating-point type, got {dtype}")
        if ambiguous == "fractional" and dtype.kind != "f":
            raise ValueError(f'ambiguous="fractional" needs a floating-point dtype, e.g. dtype=np.float32, got {dtype}')
        n_columns = len(alphabet.letters)
        code_rows = np.zeros((len(alphabet.ambiguous), n_columns), dtype=dtype)
        if ambiguous != "zeros":
            column = {letter: i for i, letter in enumerate(alphabet.letters)}
            for row, letters in zip(code_rows, alphabet.ambiguous.values()):
                row[[column[letter] for letter in letters]] = 1 / len(letters) if ambiguous == "fractional" else 1
        # without dtype, vstack returns the native byte order for a dtype such as ">f4"
        rows = np.vstack([np.eye(n_columns, dtype=dtype), code_rows], dtype=dtype)
        # For letters of single ASCII characters, decode looks up the byte of each token in this array,
        # which is faster than a join of the letters and codes.
        letters = _letters_and_codes(alphabet)
        letter_bytes = (
            np.frombuffer("".join(letters).encode("ascii"), dtype=np.uint8)
            if len(letters[0]) == 1 and all(letter.isascii() for letter in letters)
            else None
        )

        self._alphabet = alphabet
        self._ambiguous = ambiguous
        self._dtype = dtype
        # the default tokens are the row indices: the letters, then the codes
        self._tokenizer = Tokenizer(alphabet, dtype=np.uint8 if len(rows) <= 256 else np.intp)
        self._rows = rows
        self._letter_bytes = letter_bytes

    @property
    def alphabet(self) -> Alphabet:
        """The letters of the columns and the ambiguity codes."""
        return self._alphabet

    @property
    def ambiguous(self) -> _AmbiguousRule:
        """The rule for the rows of the ambiguity codes, ``"zeros"``, ``"fractional"`` or ``"ones"``."""
        return self._ambiguous

    @property
    def dtype(self) -> np.dtype[Any]:
        """Numpy dtype of the arrays."""
        return self._dtype

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Encoder):
            return NotImplemented
        return (self._alphabet, self._ambiguous, self._dtype) == (other._alphabet, other._ambiguous, other._dtype)

    def __hash__(self) -> int:
        return hash((self._alphabet, self._ambiguous, self._dtype))

    def __repr__(self) -> str:
        return f"Encoder(alphabet={self._alphabet!r}, ambiguous={self._ambiguous!r}, dtype={self._dtype!r})"

    def __reduce__(self) -> tuple[functools.partial["Encoder"], tuple[Alphabet]]:
        # the encoder is built again from its arguments, which builds its table of rows and its tokenizer again
        return functools.partial(Encoder, ambiguous=self._ambiguous, dtype=self._dtype), (self._alphabet,)

    def one_hot(self, seq: str) -> NDArray[Any]:
        """One-hot encode a sequence.

        :param seq: the sequence
        :returns: array of :attr:`dtype` and of shape ``(len(seq) // letter length, len(alphabet.letters))``
        :raises TypeError: if ``seq`` is not a str
        :raises ValueError: if ``len(seq)`` is not a multiple of the letter length,
            or if ``seq`` contains a letter that is not in :attr:`alphabet`

        Example:

        >>> Encoder().one_hot("ACGTN")
        array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0]], dtype=uint8)
        """
        return self._rows.take(self._tokenizer.tokenize(seq), axis=0)

    def decode(self, x: NDArray[Any]) -> str:
        """Convert a one-hot encoded array back to a sequence.

        It inverts :meth:`one_hot`: each row becomes the letter or the ambiguity code whose row it equals.
        If several codes have this row, the row becomes the first of them.
        With ``"zeros"``, all codes have the row of zeros, so this row becomes the first code,
        e.g. ``N`` for :data:`~kipoiseq2.alphabets.DNA`.
        With ``"fractional"`` and ``"ones"``, each code has its own row, unless two codes stand for the same letters.
        So ``encoder.decode(encoder.one_hot(seq))`` equals ``seq`` with ``"fractional"`` and ``"ones"``,
        and with ``"zeros"`` if ``seq`` has no code other than the first one.

        :param x: array of numbers of shape ``(length, len(alphabet.letters))``, e.g. of :meth:`one_hot`
        :returns: the sequence of ``length`` letters and codes
        :raises TypeError: if ``x`` is not a numpy array of numbers
        :raises ValueError: if ``x`` has another shape,
            or if a row of ``x`` is neither the row of a letter nor the row of an ambiguity code

        Example:

        >>> Encoder().decode(np.array([[0, 0, 1, 0], [0, 0, 0, 0]]))
        'GN'
        >>> Encoder(ambiguous="fractional", dtype=np.float32).decode(np.array([[0.5, 0, 0.5, 0], [0.25] * 4]))
        'RN'
        >>> Encoder(ambiguous="ones").decode(np.array([[1, 0, 1, 0], [1, 1, 1, 1]]))
        'RN'
        """
        if not isinstance(x, np.ndarray) or x.dtype.kind not in "biuf":
            raise TypeError(f"x must be a numpy array of numbers, got {x!r}")
        n_columns = len(self._alphabet.letters)
        if x.ndim != 2 or x.shape[1] != n_columns:
            raise ValueError(f"x must have the shape (length, {n_columns}), got {x.shape}")
        # numpy reduces an axis of a few values slowly, so the checks loop over the columns
        columns = np.ascontiguousarray(x.T)
        counter = np.min_scalar_type(n_columns)
        n_ones = np.zeros(len(x), dtype=counter)
        n_zeros = np.zeros(len(x), dtype=counter)
        # tokens counts the columns before the first 1. In the row of a letter, this count is the column of the 1.
        # In a row without a 1, it is n_columns, the token of the first code.
        tokens = np.zeros(len(x), dtype=self._tokenizer.dtype)
        seen_one = np.zeros(len(x), dtype=bool)
        for column in columns:
            ones = column == 1
            n_ones += ones
            n_zeros += column == 0
            seen_one |= ones
            tokens += ~seen_one
        is_letter = (n_ones == 1) & (n_zeros == n_columns - 1)
        if self._alphabet.ambiguous and self._ambiguous != "zeros":
            # a code has at least 2 letters, so its row is never the row of a letter
            rest = np.flatnonzero(~is_letter)
            rest_columns = columns.take(rest, axis=1)
            found = np.zeros(len(rest), dtype=bool)
            for token in range(n_columns, len(self._rows)):
                if found.all():
                    break
                # the first code with this row wins
                match = ~found
                for column, value in zip(rest_columns, self._rows[token]):
                    match &= column == value
                tokens[rest[match]] = token
                found |= match
            invalid = rest[~found]
        else:
            # with "zeros", all codes have the row of zeros, whose token is n_columns, the token of the first code
            invalid = np.flatnonzero(~(is_letter | (n_zeros == n_columns)) if self._alphabet.ambiguous else ~is_letter)
        if invalid.size:
            kind = (
                "neither the row of a letter nor the row of an ambiguity code"
                if self._alphabet.ambiguous
                else "not the row of a letter"
            )
            raise ValueError(f"the rows {invalid[:10].tolist()} of x are {kind}")
        if self._letter_bytes is not None:
            return self._letter_bytes.take(tokens).tobytes().decode("ascii")
        letters = _letters_and_codes(self._alphabet)
        return "".join([letters[token] for token in tokens.tolist()])
