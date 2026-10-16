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
pip install 'kipoiseq2[ranges]' # + scan_vcf_variants, scan_vcf_genotypes, scan_variant_pairs and iter_variant_seqs (polars, polars-bio)
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
# for the variants of a VCF file in each interval, see scan_variant_pairs below
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

`scan_variant_pairs` finds every interval-variant pair with one [polars-bio](https://biodatageeks.org/polars-bio/) overlap join.
A variant matches an interval if `variant.start < interval.end` and `interval.start < variant.end`.
The intervals are a polars DataFrame with the columns `chrom`, `start` and `end`, an optional `strand` and any other columns.
The variants are a polars DataFrame or LazyFrame with the columns `chrom`, `pos`, `ref` and `alt`, e.g. from `scan_vcf_variants`.

```python
import polars as pl
from kipoiseq2 import Interval, Variant
from kipoiseq2.extractors import scan_variant_pairs, scan_vcf_variants

# chrom, start, end (0-based, half-open), optional strand, and any other columns
exons = pl.DataFrame(
    {"chrom": ["chr1"], "start": [100], "end": [200], "strand": ["+"], "exon_id": ["e1"], "source": ["HAVANA"]}
)
# the pairs keep all columns, so select those that the pairs should carry
exons = exons.select("chrom", "start", "end", "strand", "exon_id")
# a LazyFrame with the interval columns, interval_idx, the variant_* columns and variant_idx
pairs = scan_variant_pairs(exons, scan_vcf_variants("variants.vcf.gz"))

# all pairs in memory, sorted by variant and then by interval
pairs.collect().sort("variant_idx", "interval_idx")

# for a large VCF, write the pairs to a file, or process them in batches
pairs.sink_parquet("pairs.parquet")
for batch in pairs.collect_batches(chunk_size=10_000):
    ...

# Interval and Variant objects, e.g. for VariantSeqExtractor
for r in pairs.collect().iter_rows(named=True):
    interval = Interval(r["chrom"], r["start"], r["end"], strand=r["strand"])
    variant = Variant(r["chrom"], r["variant_pos"], r["variant_ref"], r["variant_alt"])
```

`pl.DataFrame(intervals)` turns a list of Interval objects into an intervals frame, and `pl.DataFrame(variants)` does the same for Variant objects.
`scan_variant_pairs` holds the intervals in memory and streams the variants through the overlap join.
With a VCF file, the peak memory of `sink_parquet` does not grow with the size of the VCF.
The pairs come in no fixed order, so sort them if the order matters.
A null strand in the intervals frame means `"."`, as a missing strand column.
polars-bio prints a progress bar on stderr for each collect.
Set the environment variable `TQDM_DISABLE=1` to turn it off.

### Batches of reference and alternative sequences

`iter_variant_seqs` applies the variants of the pairs to the reference, and yields the sequences as polars DataFrame batches.
By default, each pair gives one row, and the sequences keep the width of the interval around its center.
A batch holds the keys `interval_idx` and `variant_idx`, the interval, the sequences `ref_seq` and `alt_seq`, and the encoded sequences `ref` and `alt`.
The default encoding `Tokenizer()` gives uint8 tokens.
An `Encoder` as `encoding` gives one-hot arrays in the dtype of the encoder.

```python
import polars as pl
from kipoiseq2.extractors import FastaStringExtractor, iter_variant_seqs, scan_variant_pairs, scan_vcf_variants

variants = scan_vcf_variants("variants.vcf.gz").collect()
# a 1 kb window around each variant, in the order of the variants
windows = variants.select("chrom", start=pl.col("start") - 500, end=pl.col("start") + 500)
# keep each variant in its own window only
pairs = scan_variant_pairs(windows, variants).filter(pl.col("interval_idx") == pl.col("variant_idx"))

with FastaStringExtractor("genome.fa", use_strand=False) as fasta:
    for batch in iter_variant_seqs(pairs, fasta, use_strand=True, pad_with_n=True, batch_size=64):
        ref = batch["ref"].to_numpy()  # uint8 array of shape (64, 1000), without a copy
        alt = batch["alt"].to_numpy()
```

A variant within 500 bases of a chromosome end gets a window that runs past the chromosome.
`pad_with_n=True` fills the bases outside the chromosome with N.

`group_by` sets the rows: the pairs with equal values in these columns give one row, with all their variants applied.
`group_by=["interval_idx"]` applies all variants of an interval together.
Variants that overlap within a row raise a ValueError, as in `VariantSeqExtractor`.
Each pair applies its ALT allele. So for the genotypes of `scan_vcf_genotypes`, keep the default `carriers_only=True`: a False or a null in the column `variant_carrier` raises a ValueError.
`encoding=Encoder()` gives uint8 one-hot arrays of shape `(rows, width, 4)`, and `encoding=None` gives only the sequences.
`anchor="start"` or `"end"` keeps the 5' or the 3' end of the interval, and `anchor=None` lets indels change the length of `alt_seq`.
Sequences of different lengths give polars List columns instead of Array columns.

More examples:
- The tests in [tests/](tests/) show the usage of every extractor and transform.
- API docs: <https://gagneurlab.github.io/kipoiseq2/>

## Migrating from kipoiseq

kipoiseq2 changes the API of kipoiseq, and [MIGRATING.md](https://github.com/gagneurlab/kipoiseq2/blob/main/MIGRATING.md) lists the changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
