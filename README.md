# kipoiseq2

[![CI](https://github.com/kipoi/kipoiseq2/actions/workflows/ci.yml/badge.svg)](https://github.com/kipoi/kipoiseq2/actions/workflows/ci.yml)

Sequence extractors and transforms for DNA sequence-based models.
kipoiseq2 extracts reference and variant sequences from FASTA and VCF files and encodes them for model input, e.g. as one-hot arrays.

kipoiseq2 is the successor of [kipoiseq](https://github.com/kipoi/kipoiseq) without the Kipoi model zoo dataloaders (`kipoiseq.dataloaders`) and without the kipoi dependencies.
Its sequence and VCF extractors, transforms, `Interval` and `Variant` are those of kipoiseq, imported from `kipoiseq2` instead of `kipoiseq`, so both packages can be installed side by side.

## Installation

Requires Python >= 3.12.

```bash
pip install kipoiseq2               # FASTA extractors and transforms (numpy, pyfaidx)
pip install 'kipoiseq2[vcf]'        # + MultiSampleVCF and the VCF-based extractors (cyvcf2)
pip install 'kipoiseq2[ranges]'     # + scan_vcf_variants, scan_vcf_genotypes and the variant matchers (polars, polars-bio)
pip install 'kipoiseq2[vcf,ranges]' # everything
```

kipoiseq2 does not depend on pandas or pyranges.

## Getting started

```python
from kipoiseq2 import Interval, Variant
from kipoiseq2.extractors import FastaStringExtractor, MultiSampleVCF, VariantSeqExtractor
from kipoiseq2.transforms.functional import one_hot_dna

interval = Interval("chr1", 10, 20, strand="+")

# reference sequence
fasta = FastaStringExtractor("genome.fa", use_strand=True)
seq = fasta.extract(interval)  # 10 bp string
one_hot = one_hot_dna(seq)  # array of shape (10, 4)

# sequence with variants applied, anchored at the interval start
variants = [Variant("chr1", 15, "A", "T")]
# or all variants of a VCF file in the interval: MultiSampleVCF("variants.vcf.gz").fetch_variants(interval)
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

# chrom, start, end, pos, ref, alt, allele_idx and the INFO field AF
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

```python
import polars as pl
from kipoiseq2.extractors import SingleVariantMatcher

# chrom, start, end (0-based, half-open), optional strand, and any attribute columns
exons = pl.DataFrame({"chrom": ["chr1"], "start": [100], "end": [200], "strand": ["+"], "exon_id": ["e1"]})
matcher = SingleVariantMatcher("variants.vcf.gz", intervals=exons, interval_attrs=["exon_id"])

# all pairs as one polars DataFrame: interval columns, interval_idx, variant_* columns, variant_idx
pairs = matcher.pairs()

# or iterate (Interval, Variant) pairs; interval.attrs holds exon_id, variant.source the cyvcf2 record
for interval, variant in SingleVariantMatcher("variants.vcf.gz", intervals=exons, interval_attrs=["exon_id"]):
    ...
```

Iteration runs one overlap join per batch of `variant_batch_size` variants (default 10000), so the Variant objects of a large VCF are not all held in memory.
The pairs come in VCF order, and per variant in the order of the intervals.

More examples:
- The tests in [tests/](tests/) show the usage of every extractor and transform.
- API docs: the docstrings in [src/kipoiseq2/extractors](src/kipoiseq2/extractors) and [src/kipoiseq2/transforms](src/kipoiseq2/transforms) (functional and class-based).

## Migrating from kipoiseq

Replace `kipoiseq` with `kipoiseq2` in imports and dependencies.
The Kipoi dataloaders (`SeqIntervalDl`, `StringSeqIntervalDl`, `AnchoredGTFDl`, `MMSpliceDl` and the protein and UTR dataloaders) have no replacement in kipoiseq2; keep using `kipoiseq` for them.
kipoiseq2 does not install `kipoi`, `kipoi-utils`, `kipoi-conda` or `gffutils`, so declare them yourself if you import them.

kipoiseq2 also drops these parts of kipoiseq:
- the GTF, protein and UTR extractors (`kipoiseq.extractors.gtf`, `protein` and `multi_interval`) and `VariantCombinator`
- `Interval.from_pybedtools` and `Interval.to_pybedtools`
- the `progress` argument of `MultiSampleVCF.query_variants`, `MultiSampleVCF.query_all` and `VariantIntervalQueryable`

The matchers use polars instead of pyranges:
- `SingleVariantMatcher` and `MultiVariantsMatcher` take the intervals as `intervals`, a polars DataFrame with the columns `chrom`, `start`, `end` and optionally `strand`, instead of `pranges`. `intervals` also takes a sequence of Interval objects.
  For a PyRanges object `pr`, pass `intervals=pl.from_pandas(pr.df).rename({"Chromosome": "chrom", "Start": "start", "End": "end", "Strand": "strand"})`.
- `gtf_path` and `bed_path` are gone. Read the file yourself, e.g. with polars-bio, and pass the frame as `intervals`.
- Arguments after `variant_fetcher` are keyword-only.
- `SingleVariantMatcher` yields the pairs in VCF order instead of grouped by chromosome and strand.
  With a sequence of Interval objects as `intervals`, it yields these objects, so they keep their name and attrs.
- `SingleVariantMatcher.iter_pyranges` and `iter_rows` are replaced by `pairs()` and `iter_batches()`, which return polars DataFrames.
- `variants_to_pyranges`, `intervals_to_pyranges` and `pyranges_to_intervals` are replaced by `variants_to_polars` and `intervals_to_polars`; `PyrangesVariantFetcher` is now `VariantListFetcher`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
