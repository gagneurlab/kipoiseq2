# kipoiseq2

[![CI](https://github.com/kipoi/kipoiseq2/actions/workflows/ci.yml/badge.svg)](https://github.com/kipoi/kipoiseq2/actions/workflows/ci.yml)

Sequence extractors and transforms for DNA sequence-based models.
kipoiseq2 extracts reference and variant sequences from FASTA, VCF and GTF files and encodes them for model input, e.g. as one-hot arrays.

kipoiseq2 is the successor of [kipoiseq](https://github.com/kipoi/kipoiseq) without the Kipoi model zoo dataloaders (`kipoiseq.dataloaders`) and without the kipoi dependencies.
Its extractors, transforms, `Interval` and `Variant` are those of kipoiseq, imported from `kipoiseq2` instead of `kipoiseq`, so both packages can be installed side by side.

## Installation

Requires Python >= 3.10.

```bash
pip install kipoiseq2
```

Optional dependencies:
- `cyvcf2` for the VCF-based extractors (`MultiSampleVCF` and everything that reads a VCF file)
- `pybedtools` for `Interval.from_pybedtools` and `Interval.to_pybedtools`

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

More examples:
- The extractors notebook [notebooks/getting-started-with-VariantExtractors.ipynb](notebooks/getting-started-with-VariantExtractors.ipynb).
- The tests in [tests/](tests/) show the usage of every extractor and transform.
- API docs: the docstrings in [kipoiseq2/extractors](kipoiseq2/extractors) and [kipoiseq2/transforms](kipoiseq2/transforms) (functional and class-based).

## Migrating from kipoiseq

Replace `kipoiseq` with `kipoiseq2` in imports and dependencies.
The Kipoi dataloaders (`SeqIntervalDl`, `StringSeqIntervalDl`, `AnchoredGTFDl`, `MMSpliceDl` and the protein and UTR dataloaders) have no replacement in kipoiseq2; keep using `kipoiseq` for them.
kipoiseq2 does not install `kipoi`, `kipoi-utils`, `kipoi-conda` or `gffutils`, so declare them yourself if you import them.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
