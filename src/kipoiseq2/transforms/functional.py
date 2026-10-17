import string
from collections.abc import Mapping


def rc_dna(seq):
    """
    Reverse complement the DNA sequence

    >>> assert rc_dna("TATCG") == "CGATA"
    >>> assert rc_dna("tatcg") == "cgata"
    """
    rc_hash = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "a": "t",
        "t": "a",
        "c": "g",
        "g": "c",
    }
    return "".join([rc_hash[s] for s in reversed(seq)])


def rc_rna(seq):
    """
    Reverse complement the RNA sequence

    >>> assert rc_rna("UAUCG") == "CGAUA"
    """
    rc_hash = {
        "A": "U",
        "U": "A",
        "C": "G",
        "G": "C",
        "a": "u",
        "u": "a",
        "c": "g",
        "g": "c",
    }
    return "".join([rc_hash[s] for s in reversed(seq)])


# sequence trimming


def pad(seq, length, value="N", anchor="center"):
    """Pad a sequence to a fixed length

    :param seq: str or list of chars; Sequence to pad
    :param length: int; Final length of the sequence. Must be at least ``len(seq)``.
    :param value: Neutral element to pad the sequence with. Can be ``str`` or ``list``, of the same type as ``seq``.
        If ``value`` has several elements, it is repeated and cut to the pad length on each side,
        e.g. ``pad("A", 4, value="NM", anchor="end")`` returns ``"NMNA"``.
    :param anchor: character; 'start', 'end' or 'center'. To which end to anchor the sequence when padding.
    :returns: Sequence of length ``length``, of the same type as ``seq``
    """
    seq_len = len(seq)
    assert length >= seq_len
    if anchor == "end":
        n_left = length - seq_len
        n_right = 0
    elif anchor == "start":
        n_right = length - seq_len
        n_left = 0
    elif anchor == "center":
        n_left = (length - seq_len) // 2 + (length - seq_len) % 2
        n_right = (length - seq_len) // 2
    else:
        raise ValueError("anchor can be of: end, start or center")

    # repeat value often enough and cut it to the pad length on each side
    padding = value * (max(n_left, n_right) // len(value) + 1)

    return padding[:n_left] + seq + padding[:n_right]


def trim(seq, length, anchor="center"):
    seq_len = len(seq)

    assert length <= seq_len
    if anchor == "end":
        return seq[-length:]
    elif anchor == "start":
        return seq[0:length]
    elif anchor == "center":
        dl = seq_len - length
        n_left = dl // 2 + dl % 2
        n_right = seq_len - dl // 2
        return seq[n_left:n_right]
    else:
        raise ValueError("anchor can be of: end, start or center")


def fixed_len(seq, length, anchor="center", value="N"):
    """Pad or trim a sequence to a fixed length. Procedure:

    1. Pad the sequence with N's or any other string or list element (``value``) if it is shorter than ``length``
    2. Trim the sequence if it is longer than ``length``

    .. note::
       See also: https://keras.io/preprocessing/sequence/
       Applicable also for lists of characters

    :param seq: str or list of chars; Sequence to pad or trim
    :param length: int; Final length of the sequence.
    :param anchor: character; 'start', 'end' or 'center'. To which end to anchor the sequence when trimming/padding.
        See examples below.
    :param value: Neutral element to pad the sequence with. Can be ``str`` or ``list``, of the same type as ``seq``.
        If ``value`` has several elements, it is repeated and cut to the pad length on each side (see :func:`pad`).
    :returns: Sequence of length ``length``, of the same type as ``seq``

    Example:

    >>> sequence = 'CTTACTCAGA'
    >>> fixed_len(sequence, 10, anchor="start", value="N")
    'CTTACTCAGA'
    >>> fixed_len(sequence, 10, anchor="end", value="N")
    'CTTACTCAGA'
    >>> fixed_len(sequence, 4, anchor="center", value="N")
    'ACTC'

    >>> sequence = 'TCTTTA'
    >>> fixed_len(sequence, 10, anchor="start", value="N")
    'TCTTTANNNN'
    >>> fixed_len(sequence, 10, anchor="end", value="N")
    'NNNNTCTTTA'
    >>> fixed_len(sequence, 4, anchor="center", value="N")
    'CTTT'
    """
    # neutral element type checking
    assert isinstance(value, list) or isinstance(value, str)
    assert isinstance(value, type(seq))
    assert isinstance(length, int)

    # pad and subset
    if len(seq) < length:
        return pad(seq, length, value=value, anchor=anchor)
    elif len(seq) > length:
        return trim(seq, length, anchor=anchor)
    else:
        return seq


def resize_interval(interval, width, anchor="center"):
    """Resize the Interval. Returns new Interval instance with correct length.

    :param interval: an :class:`~kipoiseq2.dataclasses.Interval`. It is not modified.
    :param width: desired width of the output interval
    :param anchor: str; which part of the sequence should be anchored. Choices: 'start', 'center', or 'end'
    """
    if anchor == "start":
        start = interval.start
        end = interval.start + width
    elif anchor == "end":
        start = interval.end - width
        end = interval.end
    elif anchor == "center":
        center = int((interval.start + interval.end) / 2)
        half_len = int(width / 2)
        start = center - half_len
        end = center + half_len + width % 2
    else:
        raise Exception("Interval resizing anchor point can only be 'start', 'end' or 'center'")

    return interval.slop(upstream=interval.start - start, downstream=end - interval.end)


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
