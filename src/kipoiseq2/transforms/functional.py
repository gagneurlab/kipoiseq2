import string
from typing import Any, Mapping, Sequence

import numpy as np

from kipoiseq2.utils import DNA

# sequence -> array


def _get_alphabet_dict(alphabet):
    return {letter: i for i, letter in enumerate(alphabet)}


def _get_index_dict(alphabet):
    return {i: letter for i, letter in enumerate(alphabet)}


def one_hot2token(arr):
    return arr.argmax(axis=2)


def one_hot2string(arr, alphabet=DNA):
    """Convert a one-hot encoded array back to string"""
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(alphabet)

    return ["".join([indexToLetter[x] for x in row]) for row in tokens]


def rc_dna(seq):
    """
    Reverse complement the DNA sequence
    >>> assert rc_seq("TATCG") == "CGATA"
    >>> assert rc_seq("tatcg") == "cgata"
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
    >>> assert rc_seq("TATCG") == "CGATA"
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


def tokenize(seq, alphabet=DNA, neutral_alphabet=["N"]):
    """Convert sequence to integers

    # Arguments
       seq: Sequence to encode
       alphabet: Alphabet to use
       neutral_alphabet: Neutral alphabet -> assign those values to -1

    # Returns
       List of length `len(seq)` with integers from `-1` to `len(alphabet) - 1`
    """
    # Req: all alphabets have the same length
    if isinstance(neutral_alphabet, str):
        neutral_alphabet = [neutral_alphabet]

    nchar = len(alphabet[0])
    for letter in (*alphabet, *neutral_alphabet):
        assert len(letter) == nchar
    assert len(seq) % nchar == 0  # since we are using striding

    alphabet_dict = _get_alphabet_dict(alphabet)
    for letter in neutral_alphabet:
        alphabet_dict[letter] = -1
    # current performance bottleneck
    return np.array([alphabet_dict[seq[(i * nchar) : ((i + 1) * nchar)]] for i in range(len(seq) // nchar)])


def token2one_hot(tokens, alphabet_size=4, neutral_value=0.25, dtype=None):
    """
    Note: everything out of the alphabet is transformed into `np.zeros(alphabet_size)`
    """
    arr = np.zeros((len(tokens), alphabet_size), dtype=dtype)

    tokens_range = np.arange(len(tokens), dtype=int)
    arr[tokens_range[tokens >= 0], tokens[tokens >= 0]] = 1
    if neutral_value != 0:
        arr[tokens_range[tokens < 0], :] = neutral_value
    return arr


def one_hot(seq, alphabet=DNA, neutral_alphabet=["N"], neutral_value=0.25, dtype=None):
    if not isinstance(seq, str):
        raise ValueError("seq needs to be a string")
    return token2one_hot(tokenize(seq, alphabet, neutral_alphabet), len(alphabet), neutral_value, dtype=dtype)


# Reference: https://github.com/deepmind/deepmind-research/blob/fa8c9be4bb0cfd0b8492203eb2a9f31ef995633c/enformer/enformer.py#L306-L318
def one_hot_dna(
    seq: str, alphabet: Sequence[str] = DNA, neutral_alphabet: str = "N", neutral_value: Any = 0.25, dtype=np.float32
) -> np.ndarray:
    """One-hot encode sequence."""
    if not isinstance(seq, str):
        raise ValueError("sequence needs to be a string")

    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8("".join(alphabet))] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8("".join(neutral_alphabet))] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(seq)]


# sequence trimming


def pad(seq, length, value="N", anchor="center"):
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

    # normalize for the length
    n_left = n_left // len(value)
    n_right = n_right // len(value)

    return value * n_left + seq + value * n_right


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
    """Pad and/or trim a list of sequences to have common length. Procedure:

        1. Pad the sequence with N's or any other string or list element (`value`)
        2. Subset the sequence

    # Note
        See also: https://keras.io/preprocessing/sequence/
        Aplicable also for lists of characters

    # Arguments
        sequence_vec: list of chars or lists
            List of sequences that can have various lengths
        value: Neutral element to pad the sequence with. Can be `str` or `list`.
        length: int or None; Final lenght of sequences.
             If None, length is set to the longest sequence length.
        anchor: character; 'start', 'end' or 'center'
            To which end to anchor the sequences when triming/padding. See examples bellow.

    # Returns
        List of sequences of the same class as sequence_vec

    # Example

        ```python
            >>> sequence = 'CTTACTCAGA'
            >>> pad_sequence(sequence, 10, anchor="start", value="N")
            'CTTACTCAGA'
            >>> pad_sequence(sequence, 10, anchor="end", value="N")
            'CTTACTCAGA'
            >>> pad_sequences(sequence, 4, anchor="center", value="N")
            'ACTC'

            >>> sequence = 'TCTTTA'
            >>> pad_sequence(sequence, 10, anchor="start", value="N")
            'TCTTTANNNN'
            >>> pad_sequence(sequence, 10, anchor="end", value="N")
            'NNNNTCTTTA'
            >>> pad_sequences(sequence, 4, anchor="center", value="N")
            'CTTT'
        ```
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

    Arguments:
        interval: an `Interval`. It is not modified.
        width: desired width of the output interval
        anchor (str): which part of the sequence should be anchored. Choices: 'start', 'center', or 'end'
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

    Stop codons become `_`, and translation continues after them.

    kipoiseq2 has no CDS model, so for the sequence of a variant the caller moves the exceptions.
    Keep an exception only at a reference codon that the variant leaves unchanged and in frame.
    Downstream of an in-frame indel, add `(len(alt) - len(ref)) // 3` to the exception positions.
    Downstream of a frameshift, drop the exceptions.
    Then a new TGA elsewhere translates as a stop.

    # Arguments
        seq: DNA sequence in upper case, with a length that is a multiple of 3.
            A partial last codon is allowed if `transl_except` covers it,
            e.g. a mitochondrial stop that poly(A) completes.
        transl_table: NCBI genetic code id, as in cdot `translation.transl_table` and in GFF.
            1 and 11 use `TRANSLATION_TABLE`.
            Table 11 differs from table 1 only in its start codons, so pass a non-AUG start as `{1: "M"}`.
            2 uses the vertebrate mitochondrial code, where AGA and AGG are stops, ATA is M and TGA is W.
        transl_except: map from a 1-based codon number to a one-letter amino acid, as in cdot `transl_except` and HGVS p.,
            e.g. `{8: "U"}` for selenocysteine, `{1: "M"}` for a non-AUG start or `{n: "_"}` for a stop.
            The exception sets the amino acid of its codon, so that codon is not looked up in the table.

    # Returns
        Amino acid sequence with one letter per codon

    # Raises
        ValueError: if `transl_table` is not 1, 2 or 11,
            if a position in `transl_except` is not an int from 1 to the number of codons,
            if an amino acid in `transl_except` is not an upper-case ASCII letter or `_`,
            or if the length of `seq` is not a multiple of 3 and `transl_except` does not cover the partial last codon
        KeyError: if a codon without an exception is not in the table, e.g. because it contains N
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
