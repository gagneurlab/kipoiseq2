# kipoiseq2

[![CI](https://github.com/gagneurlab/kipoiseq2/actions/workflows/ci.yml/badge.svg)](https://github.com/gagneurlab/kipoiseq2/actions/workflows/ci.yml)

Sequence extractors and transforms for DNA sequence-based models.
kipoiseq2 extracts reference and variant sequences from FASTA and VCF files and encodes them for model input, e.g. as one-hot arrays.

kipoiseq2 is the successor of [kipoiseq](https://github.com/kipoi/kipoiseq) without the Kipoi model zoo dataloaders (`kipoiseq.dataloaders`) and without the kipoi dependencies.
Its sequence extractors, transforms, `Interval` and `Variant` are those of kipoiseq, imported from `kipoiseq2` instead of `kipoiseq`, so both packages can be installed side by side.
kipoiseq2 reads VCF files as polars tables with polars-bio instead of cyvcf2, see [Migrating from kipoiseq](https://github.com/gagneurlab/kipoiseq2/blob/main/MIGRATING.md).

## Installation

Requires Python >= 3.12.

```bash
pip install kipoiseq2           # FASTA extractors and transforms (numpy, pyfaidx)
pip install 'kipoiseq2[ranges]' # + scan_vcf_variants, scan_vcf_genotypes and the variant matchers (polars, polars-bio)
```

kipoiseq2 does not depend on pandas or pyranges.

## Getting started

```python
from kipoiseq2 import Interval, Variant
from kipoiseq2.extractors import FastaStringExtractor, VariantSeqExtractor
from kipoiseq2.transforms.functional import one_hot_dna

interval = Interval("chr1", 10, 20, strand="+")

# reference sequence
fasta = FastaStringExtractor("genome.fa", use_strand=True)
seq = fasta.extract(interval)  # 10 bp string
one_hot = one_hot_dna(seq)  # array of shape (10, 4)

# sequence with variants applied, anchored at the interval start
variants = [Variant("chr1", 15, "A", "T")]
# for the variants of a VCF file in each interval, see MultiVariantsMatcher below
alt_seq = VariantSeqExtractor("genome.fa").extract(interval, variants, anchor=10)
```

`Interval` coordinates are 0-based and half-open, as in BED: `Interval("chr1", 10, 20)` covers the 1-based positions 11 to 20.
`Variant.start` is `pos - 1` and `Variant.end` is `start + len(ref)`.

### Reading VCF files as tables

`scan_vcf_variants` reads a VCF file with polars-bio as a polars LazyFrame, with one row per ALT allele.
Its coordinates follow `Variant`: `start` is `pos - 1` and `end` is `start + len(ref)`.

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
- API docs: the docstrings in [src/kipoiseq2/extractors](src/kipoiseq2/extractors) and [src/kipoiseq2/transforms](src/kipoiseq2/transforms) (functional and class-based).

## Migrating from kipoiseq

kipoiseq2 changes the API of kipoiseq, and [MIGRATING.md](https://github.com/gagneurlab/kipoiseq2/blob/main/MIGRATING.md) lists the changes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
