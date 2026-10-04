# Migrating from kipoiseq

Replace `kipoiseq` with `kipoiseq2` in imports and dependencies.
The Kipoi dataloaders (`SeqIntervalDl`, `StringSeqIntervalDl`, `AnchoredGTFDl`, `MMSpliceDl` and the protein and UTR dataloaders) have no replacement in kipoiseq2; keep using `kipoiseq` for them.
kipoiseq2 does not install `kipoi`, `kipoi-utils`, `kipoi-conda` or `gffutils`, so declare them yourself if you import them.

kipoiseq2 also drops these parts of kipoiseq:
- the GTF, protein and UTR extractors (`kipoiseq.extractors.gtf`, `protein` and `multi_interval`) and `VariantCombinator`
- `Interval.from_pybedtools` and `Interval.to_pybedtools`
- `translate(seq, hg38=True)` and `TRANSLATION_TABLE_FOR_HG38`, because they read every TGA as selenocysteine.
  Pass the annotated selenocysteine codons instead, e.g. `translate(seq, transl_except={8: "U"})`.

kipoiseq2 reads VCF files with polars-bio instead of cyvcf2, so the `vcf` extra is gone and the `ranges` extra covers VCF reading:
- `MultiSampleVCF` becomes `scan_vcf_variants`, with one row per ALT allele, or `scan_vcf_genotypes`, with one row per ALT allele and sample.
  Both return a polars LazyFrame instead of Variant objects.
- `query_all().filter(lambda ...)` and `FilterVariantQuery` become a polars filter on `scan_vcf_variants(path)`.
  For example, `filter(lambda v: v.qual > 10)` becomes `.filter(pl.col("qual") > 10)`, and `FilterVariantQuery()` becomes `.filter(pl.col("filter") == "PASS")`.
  polars-bio gives a missing QUAL (`.`) as null and a missing FILTER (`.`) as an empty string.
- `fetch_variants`, `query_variants`, `get_variant` and `get_variants` become `SingleVariantMatcher` or `MultiVariantsMatcher`, or a filter or join on `scan_vcf_variants(path)`.
- `get_samples`, `has_variant` and `to_sample_csv` become `scan_vcf_genotypes(path, format_fields=[...])`, and `.sink_csv(path)` writes the CSV.
  The carrier check works per ALT allele, not per record: a sample with GT 0/2 carries the second ALT allele only.
- `to_vcf` becomes `polars_bio.sink_vcf`, e.g. `pb.sink_vcf(pb.scan_vcf(path).filter(pl.col("qual") > 10), out_path)`.
- `VariantIntervalQueryable` and the query classes of `kipoiseq.extractors.vcf_query` are gone: filter the polars frames instead.
- `SingleVariantVCFSeqExtractor` and `SingleSeqVCFSeqExtractor` are gone: match the variants with a matcher, and apply them with `VariantSeqExtractor`.
- `Variant.from_cyvcf` and `Variant.from_cyvcf_and_given_alt` are gone. `Variant.source` stays as a slot for any source object.

The matchers use polars instead of pyranges:
- `SingleVariantMatcher` and `MultiVariantsMatcher` take the intervals as `intervals`, a polars DataFrame with the columns `chrom`, `start`, `end` and optionally `strand`, instead of `pranges`. `intervals` also takes a sequence of Interval objects.
  For a PyRanges object `pr`, pass `intervals=pl.from_pandas(pr.df).rename({"Chromosome": "chrom", "Start": "start", "End": "end", "Strand": "strand"})`.
- The matcher attribute `pr` becomes `intervals`: a polars DataFrame with the columns `chrom`, `start`, `end`, `strand` and the `interval_attrs` columns, instead of a PyRanges object.
  In `MultiVariantsMatcher`, `intervals` was a list of Interval objects and is now this DataFrame too.
- `gtf_path` and `bed_path` are gone. Read the file yourself, e.g. with polars-bio, and pass the frame as `intervals`.
- `variant_fetcher` and `VariantFetcher` are gone. Pass the path of a VCF file as `vcf_file`, or a polars frame or Variant objects as `variants`.
  Arguments after `variants` are keyword-only.
- `SingleVariantMatcher` yields the pairs in VCF order instead of grouped by chromosome and strand.
  With a sequence of Interval objects as `intervals`, it yields these objects, so they keep their name and attrs.
- With a VCF file or a polars frame as the variant source, the matchers yield new Variant objects with chrom, pos, ref and alt, and without `source`.
  For the genotypes of the samples, use `scan_vcf_genotypes`.
- `SingleVariantMatcher.iter_pyranges` and `iter_rows` are replaced by `pairs()`, `scan_pairs()` and `iter_batches()`.
  `pairs()` returns all pairs as a polars DataFrame, `scan_pairs()` as a LazyFrame, and `iter_batches()` yields one DataFrame per batch of variants.
- `MultiVariantsMatcher` finds the variants of all intervals with one overlap join instead of one VCF query per interval.
- `variants_to_pyranges`, `intervals_to_pyranges` and `pyranges_to_intervals` are replaced by `variants_to_polars` and `intervals_to_polars`.
  `PyrangesVariantFetcher` is gone: pass the variants as `variants`.
