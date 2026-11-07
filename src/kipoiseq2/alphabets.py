"""Alphabets for the one-hot arrays and the tokens of :mod:`kipoiseq2.encoders`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType


def _as_letters(letters: object, name: str) -> tuple[str, ...]:
    """Return ``letters`` as a tuple.

    A str is ambiguous: ``"NRY"`` can mean three letters or one letter of three characters.

    :param letters: the value of the argument ``name``
    :param name: the name of the argument in the error message, e.g. ``"letters"``
    :raises TypeError: if ``letters`` is not a list or tuple, or if it contains an element that is not a str
    """
    if isinstance(letters, str):
        raise TypeError(
            f"{name} must be a list or tuple of strings, not a str: "
            f"pass a list of letters, e.g. list({letters!r}) or [{letters!r}]"
        )
    if not isinstance(letters, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of strings, not {type(letters).__name__}")
    for letter in letters:
        if not isinstance(letter, str):
            raise TypeError(f"{name} must be a list or tuple of strings, but contains {letter!r}")
    return tuple(letters)


def _repeated(letters: tuple[str, ...]) -> list[str]:
    """Return the letters that occur more than once in ``letters``, sorted."""
    return sorted({letter for letter in letters if letters.count(letter) > 1})


@dataclass(frozen=True, init=False, repr=False)
class Alphabet:
    """The letters of a sequence, the order of their columns and tokens, and the ambiguity codes.

    All letters and codes have the same length.
    For letters with several characters, e.g. codons, a sequence is split into letters of that length.
    Letters are case-sensitive, so ``"A"`` and ``"a"`` are different letters.

    An ambiguity code stands for one of several letters, e.g. ``R`` for ``A`` or ``G``.
    A code that stands for all letters, e.g. ``N``, is neutral.
    :class:`~kipoiseq2.encoders.Encoder` encodes a code as a row of zeros, with fractions of its letters,
    or with 1 in the column of each of its letters.
    :class:`~kipoiseq2.encoders.Tokenizer` gives each code its own token after the tokens of the letters.

    Two alphabets are equal if they have the same letters in the same order,
    and the same codes with the same letters in the same order.

    :param letters: the letters in the order of the one-hot columns and the tokens, e.g. ``["A", "C", "G", "T"]``
    :param ambiguous: the letters that each ambiguity code stands for, e.g. ``{"N": ["A", "C", "G", "T"]}``.
        The order of the codes sets the order of their tokens.
    :raises TypeError: if ``letters`` is not a list or tuple of strings,
        if ``ambiguous`` is not a mapping with str keys,
        or if a value of ``ambiguous`` is not a list or tuple of strings
    :raises ValueError: if ``letters`` is empty, if a letter is empty, if the letters differ in length,
        if a letter repeats, if a code is a letter, if a code differs in length from the letters,
        or if a value of ``ambiguous`` has fewer than 2 letters, a letter that is not in ``letters``
        or a letter twice

    Example:

    >>> strict = Alphabet(["A", "C", "G", "T"], ambiguous={"N": ["A", "C", "G", "T"]})
    >>> strict.ambiguous
    mappingproxy({'N': ('A', 'C', 'G', 'T')})
    >>> DNA.ambiguous["R"]
    ('A', 'G')
    """

    letters: tuple[str, ...]
    """The letters in the order of the one-hot columns and the tokens."""

    ambiguous: Mapping[str, tuple[str, ...]] = field(compare=False)
    """The letters of each ambiguity code, in the order of :attr:`letters`. The codes keep the order of the argument."""

    # a dict compares equal in any order, but the order of the codes sets their tokens, so eq and hash use this tuple
    _codes: tuple[tuple[str, tuple[str, ...]], ...] = field(init=False)

    def __init__(
        self, letters: list[str] | tuple[str, ...], ambiguous: Mapping[str, Sequence[str]] = MappingProxyType({})
    ) -> None:
        letters_tuple = _as_letters(letters, "letters")
        if not isinstance(ambiguous, Mapping):
            raise TypeError(f"ambiguous must be a mapping from codes to letters, not {type(ambiguous).__name__}")
        for code in ambiguous:
            if not isinstance(code, str):
                raise TypeError(f"the codes of ambiguous must be strings, but ambiguous has the key {code!r}")
        if not letters_tuple:
            raise ValueError("letters must not be empty")
        length = len(letters_tuple[0])
        if length == 0 or any(len(letter) != length for letter in letters_tuple):
            raise ValueError(f"letters must contain non-empty strings of the same length, got {letters_tuple!r}")
        repeated = _repeated(letters_tuple)
        if repeated:
            raise ValueError(f"letters must not repeat a letter, but repeat {repeated}")
        codes = {}
        for code, value in ambiguous.items():
            name = f"ambiguous[{code!r}]"
            value_tuple = _as_letters(value, name)
            if code in letters_tuple:
                raise ValueError(f"the codes of ambiguous must not be letters, but {code!r} is a letter")
            if len(code) != length:
                raise ValueError(f"the codes of ambiguous must have the length {length} of the letters, got {code!r}")
            unknown = sorted(set(value_tuple) - set(letters_tuple))
            if unknown:
                raise ValueError(f"the letters of {name} must be in letters, but {unknown} are not")
            repeated = _repeated(value_tuple)
            if repeated:
                raise ValueError(f"{name} must not repeat a letter, but repeats {repeated}")
            if len(value_tuple) < 2:
                raise ValueError(f"{name} must have at least 2 letters, got {value_tuple!r}")
            # in the order of the letters, so equal sets of letters compare equal
            codes[code] = tuple(letter for letter in letters_tuple if letter in value_tuple)
        object.__setattr__(self, "letters", letters_tuple)
        object.__setattr__(self, "ambiguous", MappingProxyType(codes))
        object.__setattr__(self, "_codes", tuple(codes.items()))

    def __repr__(self) -> str:
        return f"Alphabet(letters={self.letters!r}, ambiguous={dict(self.ambiguous)!r})"

    def __reduce__(self) -> tuple[type["Alphabet"], tuple[tuple[str, ...], dict[str, tuple[str, ...]]]]:
        # a mappingproxy cannot be pickled, so the alphabet is built again from its arguments
        return Alphabet, (self.letters, dict(self.ambiguous))


DNA = Alphabet(
    ["A", "C", "G", "T"],
    ambiguous={
        "N": ["A", "C", "G", "T"],
        "R": ["A", "G"],
        "Y": ["C", "T"],
        "S": ["C", "G"],
        "W": ["A", "T"],
        "K": ["G", "T"],
        "M": ["A", "C"],
        "B": ["C", "G", "T"],
        "D": ["A", "G", "T"],
        "H": ["A", "C", "T"],
        "V": ["A", "C", "G"],
    },
)
"""DNA bases, with ``N`` and the other IUPAC ambiguity codes.

Each code stands for a set of bases, e.g. ``R`` for ``A`` or ``G``, and ``N`` for any base.
``N`` comes first, so :meth:`Encoder.decode <kipoiseq2.encoders.Encoder.decode>` decodes a row of zeros as ``N``.
GRCh37 and GRCh38 contain a few of these codes besides ``N``.
For a vocabulary without them, e.g. the one of HyenaDNA, use
``Alphabet(["A", "C", "G", "T"], ambiguous={"N": ["A", "C", "G", "T"]})``.
"""

RNA = Alphabet(
    ["A", "C", "G", "U"],
    ambiguous={
        "N": ["A", "C", "G", "U"],
        "R": ["A", "G"],
        "Y": ["C", "U"],
        "S": ["C", "G"],
        "W": ["A", "U"],
        "K": ["G", "U"],
        "M": ["A", "C"],
        "B": ["C", "G", "U"],
        "D": ["A", "G", "U"],
        "H": ["A", "C", "U"],
        "V": ["A", "C", "G"],
    },
)
"""RNA bases, with ``N`` and the other IUPAC ambiguity codes, as in :data:`DNA` with ``U`` for ``T``."""

AMINO_ACIDS = Alphabet(
    ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"],
    ambiguous={
        "X": ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"],
        "B": ["D", "N"],
        "Z": ["E", "Q"],
    },
)
"""The 20 standard amino acids, with the ambiguity codes ``X`` for any amino acid, ``B`` for ``D`` or ``N``,
and ``Z`` for ``E`` or ``Q``.

So the one-hot arrays have 20 columns, and ``N`` is asparagine.
"""
