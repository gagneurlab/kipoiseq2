# kipoiseq2

[![CI](https://github.com/gagneurlab/kipoiseq2/actions/workflows/ci.yml/badge.svg)](https://github.com/gagneurlab/kipoiseq2/actions/workflows/ci.yml)

Sequence extractors and transforms for DNA sequence-based models.
kipoiseq2 extracts reference and variant sequences from FASTA and VCF files and encodes them for model input, e.g. as one-hot arrays.

kipoiseq2 is the successor of [kipoiseq](https://github.com/kipoi/kipoiseq) without the Kipoi model zoo dataloaders (`kipoiseq.dataloaders`) and without the kipoi dependencies.
Its sequence extractors, transforms, `Interval` and `Variant` are based on those of kipoiseq and are imported from `kipoiseq2` instead of `kipoiseq`, so both packages can be installed side by side.
kipoiseq2 reads VCF files as polars tables with polars-bio instead of cyvcf2.
[Migrating from kipoiseq](https://github.com/gagneurlab/kipoiseq2/blob/main/MIGRATING.md) lists the differences.

## Installation

Requires Python >= 3.12.

```bash
pip install kipoiseq2           # FASTA extractors and transforms (numpy, pyfaidx)
pip install 'kipoiseq2[ranges]' # + scan_vcf_variants, scan_vcf_genotypes and the variant matchers (polars, polars-bio)
```

kipoiseq2 does not depend on pandas or pyranges.

## Getting started

```python
from kipoiseq2 import Encoder, Interval, Variant
from kipoiseq2.extractors import FastaStringExtractor, VariantSeqExtractor

interval = Interval("chr1", 10, 20, strand="+")

# reference sequence, uppercase by default because the encoders are case-sensitive
fasta = FastaStringExtractor("genome.fa", use_strand=True)
seq = fasta.extract(interval)  # 10 bp string
encoded = Encoder().one_hot(seq)  # uint8 array of shape (10, 4)

# sequence with variants applied, anchored at the interval start
variants = [Variant("chr1", 15, "A", "T")]
# for the variants of a VCF file in each interval, see MultiVariantsMatcher below
alt_seq = VariantSeqExtractor(fasta, use_strand=True).extract(interval, variants, anchor_pos=10)
```

`VariantSeqExtractor` reads the reference sequence from `fasta`, or from any object with the members of `SequenceExtractor`.
Close the FASTA file with `fasta.close()`, or open it in a `with` block.
With `anchor_pos`, the sequence keeps the width of the interval: a deletion pulls in reference bases at the interval end away from `anchor_pos`, and an insertion pushes bases out there.
Without `anchor_pos`, indels change the length of the sequence.

`Interval` coordinates are 0-based and half-open, as in BED: `Interval("chr1", 10, 20)` covers the 1-based positions 11 to 20.
`Interval` is immutable: methods such as `resize`, `shift` and `slop` return a new interval.
`Variant` is immutable too, and holds only `chrom`, `pos`, `ref` and `alt`. The other VCF columns stay in the frame of `scan_vcf_variants`.
`Variant.start` is `pos - 1` and `Variant.end` is `start + len(ref)`.

### Encoding sequences

`Encoder` one-hot encodes sequences, and `Tokenizer` converts them to integer tokens.
Each holds the complete configuration of its output and checks it once, so create it once and reuse it.
Both are frozen and picklable, e.g. for the worker processes of a data loader.
They take an `Alphabet`: the presets `DNA` (the default), `RNA` and `AMINO_ACIDS`, or your own.
An `Alphabet` holds the letters of the columns and the letters that each ambiguity code stands for, e.g. `R` for A or G.

```python
import numpy as np
from kipoiseq2 import AMINO_ACIDS, DNA, Encoder, Tokenizer

Encoder().one_hot("ACGTNR")  # uint8 array of shape (6, 4), N and R are rows of zeros
encoder = Encoder(DNA, ambiguous="fractional", dtype=np.float32)
encoder.one_hot("ACGTNR")  # float32 array of shape (6, 4), N is 0.25 in every column, R is [0.5, 0, 0.5, 0]
encoder.decode(encoder.one_hot("ACGTNR"))  # "ACGTNR"
Encoder(DNA, ambiguous="ones").one_hot("NR")  # uint8 array of shape (2, 4), N is [1, 1, 1, 1], R is [1, 0, 1, 0]
Encoder(AMINO_ACIDS).one_hot("MNX")  # array of shape (3, 20), X is a row of zeros
Tokenizer().tokenize("ACGTNR")  # array([0, 1, 2, 3, 4, 5], dtype=uint8), the codes follow the letters
```

The encoders are case-sensitive, and a letter that is not in the alphabet raises a ValueError that names it.
So with `DNA`, a lowercase letter of a soft-masked repeat raises this ValueError.
Sequences must be uppercase at the source.
`FastaStringExtractor` uppercases them by default, so soft-masked repeats encode like other bases.
`scan_vcf_variants` and `scan_vcf_genotypes` uppercase REF and ALT.
For sequences from other sources, call `seq.upper()` first.
`DNA` and `RNA` have `N` and the other IUPAC ambiguity codes, e.g. `R` for A or G.
GRCh37 and GRCh38 contain a few of them besides `N`.
`Encoder` encodes the codes as rows of zeros by default.
With `ambiguous="fractional"` and a float dtype, a code of k letters gives 1/k in the columns of its letters.
With `ambiguous="ones"`, a code gives 1 in the columns of its letters, in any dtype.
Its rows do not sum to 1, so a model trained on one-hot input sees `N` as a strong signal.

`Tokenizer` gives each code a token of its own after the letters, e.g. 4 for `N` and 5 for `R`.
Its `token_map` sets other tokens, e.g. the vocabulary of a pretrained model.
It needs a key for each letter and code of the alphabet, so 15 keys for `DNA`.
For a vocabulary without the IUPAC ambiguity codes, e.g. the one of HyenaDNA,
use the strict alphabet `Alphabet(["A", "C", "G", "T"], ambiguous={"N": ["A", "C", "G", "T"]})`.
For other vectors, index a table with one row per token, e.g. to add a fifth column for the codes:

```python
# one row per default token: A, C, G and T, then the 11 codes of DNA
rows = np.zeros((15, 5), dtype=np.float32)
rows[:4, :4] = np.eye(4)
rows[4:, 4] = 1
rows[Tokenizer().tokenize("ACGTNR")]  # float32 array of shape (6, 5)
```

### Reading VCF files as tables

`scan_vcf_variants` reads a VCF file with polars-bio as a polars LazyFrame, with one row per ALT allele.
Its coordinates follow `Variant`: `start` is `pos - 1` and `end` is `start + len(ref)`.
It keeps only the ALT alleles that are sequences of the bases A, C, G, T and N.
VCF bases are case-insensitive, so `ref` and `alt` are uppercase, and the case-sensitive encoders take them as they are.
`sequence_alts_only=False` keeps the other ALT alleles too, e.g. `*` and symbolic ALT alleles such as `<DEL>`.
Use it only for tables, because `VariantSeqExtractor` inserts symbolic ALT alleles as text.

```python
import polars as pl
from kipoiseq2.extractors import scan_vcf_variants

# chrom, start, end, pos, id, ref, alt, allele_idx, qual, filter and the INFO field AF
variants = scan_vcf_variants("variants.vcf.gz", info_fields=["AF"])
# AF holds one value per ALT allele of the record, and allele_idx picks the one of this row
common = variants.filter(pl.col("AF").list.get(pl.col("allele_idx") - 1) > 0.01).collect()
```

`scan_vcf_genotypes` adds one row per sample, with the FORMAT fields of the sample.
By default it keeps only the samples that carry the ALT allele of the row: a sample with GT 0/2 carries the second ALT allele, but not the first.

```python
from kipoiseq2.extractors import scan_vcf_genotypes

# the columns of scan_vcf_variants, then sample, GT, GQ and carrier
carriers = scan_vcf_genotypes("variants.vcf.gz", format_fields=["GT", "GQ"])
carriers.sink_csv("carriers.csv")
```

### Matching variants with intervals

`SingleVariantMatcher` finds every interval-variant pair with one [polars-bio](https://biodatageeks.org/polars-bio/) overlap join.
A variant matches an interval if `variant.start < interval.end` and `interval.start < variant.end`.
Give the variants either as `vcf_file`, the path of a VCF file that `scan_vcf_variants` reads, or as `variants`.
`variants` takes a polars DataFrame or LazyFrame with the columns `chrom`, `pos`, `ref` and `alt`, or Variant objects.

```python
import polars as pl
from kipoiseq2.extractors import SingleVariantMatcher

# chrom, start, end (0-based, half-open), optional strand, and any attribute columns
exons = pl.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200], "strand": ["+"], "exon_id": ["e1"]})
matcher = SingleVariantMatcher("variants.vcf.gz", intervals=exons, interval_attrs=["exon_id"])

# all pairs as one polars DataFrame: interval columns, interval_idx, variant_* columns, variant_idx
pairs = matcher.pairs()

# for a large VCF, write the pairs to a file, or process them in batches of variant_batch_size variants
matcher.scan_pairs().sink_parquet("pairs.parquet")
for batch in matcher.iter_batches():
    ...

# or iterate (Interval, Variant) pairs; interval.attrs holds exon_id
for interval, variant in matcher:
    ...
```

`scan_pairs()` streams the VCF through the overlap join.
`iter_batches()` and iteration run one overlap join per batch of `variant_batch_size` variants (default 10000).
With a VCF file, the peak memory of `scan_pairs().sink_parquet(...)` and `iter_batches()` does not grow with the size of the VCF.
`pairs()` holds all pairs in memory.
The pairs come in VCF order, and per variant in the order of the intervals.
`MultiVariantsMatcher` yields each interval with an iterator over its variants, also if the interval has no variants.
polars-bio prints a progress bar on stderr for each collect, so `iter_batches()` prints one per batch.
Set the environment variable `TQDM_DISABLE=1` to turn it off.

More examples:
- The tests in [tests/](tests/) show the usage of every extractor and transform.
- API docs: <https://gagneurlab.github.io/kipoiseq2/>

## Migrating from kipoiseq

kipoiseq2 changes the API of kipoiseq, and [MIGRATING.md](https://github.com/gagneurlab/kipoiseq2/blob/main/MIGRATING.md) lists the changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
