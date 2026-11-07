import string
from collections.abc import Mapping

from kipoiseq2.dataclasses import _Anchor, _check_resize, _resize_bounds

# IUPAC complement, the same table as pyfaidx.complement. The other characters
# below 256 are deleted, so a length change reveals them.
_COMPLEMENT_FROM = "ACTGNactgnYRWSKMDVHBXyrwskmdvhbx"
_COMPLEMENT_TO = "TGACNtgacnRYWSMKHBDVXrywsmkhbdvx"
_COMPLEMENT = str.maketrans(
    _COMPLEMENT_FROM, _COMPLEMENT_TO, "".join(chr(c) for c in range(256) if chr(c) not in _COMPLEMENT_FROM)
)


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence with IUPAC codes.

    :param seq: DNA sequence. Upper and lower case are kept.
    :returns: The reverse complement of ``seq``.
    :raises ValueError: if ``seq`` contains a character without IUPAC complement, e.g. ``U`` or ``-``.

    Example:

    >>> reverse_complement("ACGTNacgtn")
    'nacgtNACGT'
    """
    comp = seq.translate(_COMPLEMENT)
    # the table keeps the characters above 255, so isascii catches them
    if not seq.isascii() or len(comp) != len(seq):
        invalid = sorted(set(seq) - set(_COMPLEMENT_FROM))
        raise ValueError("Sequence contains non-DNA characters: {}".format(invalid))
    return comp[::-1]


# sequence trimming


def _check_seq(seq: str, length: int, anchor: str) -> None:
    """Raise a TypeError or ValueError for invalid ``seq``, ``length`` or ``anchor`` of :func:`fixed_len`."""
    if not isinstance(seq, str):
        raise TypeError(f"seq must be a str, not {type(seq).__name__}")
    _check_resize(length, anchor, width_name="length")


def _check_value(value: str) -> None:
    """Raise a TypeError or ValueError for an invalid ``value`` of :func:`pad`."""
    if not isinstance(value, str):
        raise TypeError(f"value must be a str, not {type(value).__name__}")
    if not value:
        raise ValueError("value must not be empty")


def pad(seq: str, length: int, *, anchor: _Anchor = "center", value: str = "N") -> str:
    """Pad the sequence to ``length``.

    It pads like :meth:`Interval.resize <kipoiseq2.dataclasses.Interval.resize>` grows
    an interval from 0 to ``len(seq)`` on the "+" strand.

    :param seq: the sequence to pad
    :param length: the length of the padded sequence, at least ``len(seq)``
    :param anchor: the part of ``seq`` to keep in place: ``"start"``, ``"center"`` or ``"end"``
    :param value: the letters to pad with. If ``value`` has several letters,
        it is repeated and cut to the pad length on each side,
        e.g. ``pad("A", 4, anchor="end", value="NM")`` returns ``"NMNA"``.
    :returns: the padded sequence
    :raises TypeError: if ``seq`` or ``value`` is not a str, or if ``length`` is not an int
    :raises ValueError: if ``length`` is less than ``len(seq)``, if ``anchor`` is not ``"start"``, ``"center"`` or ``"end"``,
        or if ``value`` is empty

    Example:

    >>> pad("ACGT", 7)
    'NNACGTN'
    """
    _check_seq(seq, length, anchor)
    _check_value(value)
    if length < len(seq):
        raise ValueError(f"length must be at least len(seq)={len(seq)}, got {length}")
    start, end = _resize_bounds(0, len(seq), length, anchor, reverse=False)
    # repeat value often enough and cut it to the pad length on each side
    padding = value * (max(-start, end - len(seq)) // len(value) + 1)
    return padding[:-start] + seq + padding[: end - len(seq)]


def trim(seq: str, length: int, *, anchor: _Anchor = "center") -> str:
    """Trim the sequence to ``length``.

    It keeps the letters that :meth:`Interval.resize <kipoiseq2.dataclasses.Interval.resize>` keeps
    of an interval from 0 to ``len(seq)`` on the "+" strand.

    :param seq: the sequence to trim
    :param length: the length of the trimmed sequence, at most ``len(seq)``
    :param anchor: the part of ``seq`` to keep: ``"start"``, ``"center"`` or ``"end"``
    :returns: the trimmed sequence
    :raises TypeError: if ``seq`` is not a str, or if ``length`` is not an int
    :raises ValueError: if ``length`` is negative or greater than ``len(seq)``,
        or if ``anchor`` is not ``"start"``, ``"center"`` or ``"end"``

    Example:

    >>> trim("ACGTACG", 4)
    'GTAC'
    """
    _check_seq(seq, length, anchor)
    if length > len(seq):
        raise ValueError(f"length must be at most len(seq)={len(seq)}, got {length}")
    start, end = _resize_bounds(0, len(seq), length, anchor, reverse=False)
    return seq[start:end]


def fixed_len(seq: str, length: int, *, anchor: _Anchor = "center", value: str = "N") -> str:
    """Pad or trim the sequence to ``length``, with :func:`pad` or :func:`trim`.

    :param seq: the sequence to pad or trim
    :param length: the length of the returned sequence
    :param anchor: the part of ``seq`` to keep in place: ``"start"``, ``"center"`` or ``"end"``
    :param value: the letters to pad with, see :func:`pad`
    :returns: the sequence of length ``length``
    :raises TypeError: if ``seq`` or ``value`` is not a str, or if ``length`` is not an int
    :raises ValueError: if ``length`` is negative, if ``anchor`` is not ``"start"``, ``"center"`` or ``"end"``,
        or if ``value`` is empty

    Example:

    >>> sequence = 'CTTACTCAGA'
    >>> fixed_len(sequence, 10, anchor="start")
    'CTTACTCAGA'
    >>> fixed_len(sequence, 4, anchor="center")
    'ACTC'

    >>> sequence = 'TCTTTA'
    >>> fixed_len(sequence, 10, anchor="start")
    'TCTTTANNNN'
    >>> fixed_len(sequence, 10, anchor="end")
    'NNNNTCTTTA'
    >>> fixed_len(sequence, 4, anchor="center")
    'CTTT'
    """
    _check_seq(seq, length, anchor)
    _check_value(value)
    if length > len(seq):
        return pad(seq, length, anchor=anchor, value=value)
    return trim(seq, length, anchor=anchor)


TRANSLATION_TABLE = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "_",
    "TAG": "_",
    "TGC": "C",
    "TGT": "C",
    "TGA": "_",
    "TGG": "W",
}


# NCBI genetic code 2, the vertebrate mitochondrial code
_VERTEBRATE_MITOCHONDRIAL_TABLE = {**TRANSLATION_TABLE, "AGA": "_", "AGG": "_", "ATA": "M", "TGA": "W"}

# NCBI genetic code id -> codon table. Table 11 differs from table 1 only in its start codons.
_GENETIC_CODES = {1: TRANSLATION_TABLE, 2: _VERTEBRATE_MITOCHONDRIAL_TABLE, 11: TRANSLATION_TABLE}

_TRANSL_EXCEPT_SYMBOLS = frozenset(string.ascii_uppercase + "_")


def translate(seq: str, transl_table: int = 1, transl_except: Mapping[int, str] | None = None) -> str:
    """Translate a DNA sequence into amino acids with an NCBI genetic code and annotated translation exceptions.

    Stop codons become ``_``, and translation continues after them.

    kipoiseq2 has no CDS model, so for the sequence of a variant the caller moves the exceptions.
    Keep an exception only at a reference codon that the variant leaves unchanged and in frame.
    Downstream of an in-frame indel, add ``(len(alt) - len(ref)) // 3`` to the exception positions.
    Downstream of a frameshift, drop the exceptions.
    Then a new TGA elsewhere translates as a stop.

    :param seq: DNA sequence in upper case, with a length that is a multiple of 3.
        A partial last codon is allowed if ``transl_except`` covers it,
        e.g. a mitochondrial stop that poly(A) completes.
    :param transl_table: NCBI genetic code id, as in cdot ``translation.transl_table`` and in GFF.
        1 and 11 use ``TRANSLATION_TABLE``.
        Table 11 differs from table 1 only in its start codons, so pass a non-AUG start as ``{1: "M"}``.
        2 uses the vertebrate mitochondrial code, where AGA and AGG are stops, ATA is M and TGA is W.
    :param transl_except: map from a 1-based codon number to a one-letter amino acid,
        as in cdot ``transl_except`` and HGVS p.,
        e.g. ``{8: "U"}`` for selenocysteine, ``{1: "M"}`` for a non-AUG start or ``{n: "_"}`` for a stop.
        The exception sets the amino acid of its codon, so that codon is not looked up in the table.
    :returns: Amino acid sequence with one letter per codon
    :raises ValueError: if ``transl_table`` is not 1, 2 or 11,
        if a position in ``transl_except`` is not an int from 1 to the number of codons,
        if an amino acid in ``transl_except`` is not an upper-case ASCII letter or ``_``,
        or if the length of ``seq`` is not a multiple of 3 and ``transl_except`` does not cover the partial last codon
    :raises KeyError: if a codon without an exception is not in the table, e.g. because it contains N
    """
    if transl_table not in _GENETIC_CODES:
        raise ValueError(f"transl_table must be one of {sorted(_GENETIC_CODES)}, got {transl_table!r}")
    table = _GENETIC_CODES[transl_table]

    transl_except = transl_except or {}
    n_codons = (len(seq) + 2) // 3
    for position, amino_acid in transl_except.items():
        if not isinstance(position, int) or not 1 <= position <= n_codons:
            raise ValueError(f"transl_except position must be an int from 1 to {n_codons}, got {position!r}")
        if not isinstance(amino_acid, str) or amino_acid not in _TRANSL_EXCEPT_SYMBOLS:
            raise ValueError(f"transl_except amino acid must be one upper-case ASCII letter or '_', got {amino_acid!r}")

    if len(seq) % 3 != 0 and n_codons not in transl_except:
        raise ValueError(f"len(seq) % 3 != 0 and transl_except does not cover the partial last codon {n_codons}")

    codons = (seq[i : i + 3] for i in range(0, len(seq), 3))
    return "".join(
        transl_except[number] if number in transl_except else table[codon]
        for number, codon in enumerate(codons, start=1)
    )
