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
  Both keep only the ALT alleles that are sequences of the bases A, C, G, T and N, in any case.
  Unlike `MultiSampleVCF`, they keep the ALT alleles that contain N, and they drop symbolic ALT alleles such as `<DEL>`.
  They also drop a missing ALT (`.`), which `MultiSampleVCF` gave as an empty alt.
  Both give `ref` and `alt` in uppercase, because VCF bases are case-insensitive. `MultiSampleVCF` kept the case of the file.
- `query_all().filter(lambda ...)` and `FilterVariantQuery` become a polars filter on `scan_vcf_variants(path)`.
  For example, `filter(lambda v: v.qual > 10)` becomes `.filter(pl.col("qual") > 10)`, and `FilterVariantQuery()` becomes `.filter(pl.col("filter") == "PASS")`.
  polars-bio gives a missing QUAL (`.`) as null and a missing FILTER (`.`) as an empty string.
- `fetch_variants`, `query_variants`, `get_variant` and `get_variants` become `scan_variant_pairs`, or a filter or join on `scan_vcf_variants(path)`.
- `get_samples`, `has_variant` and `to_sample_csv` become `scan_vcf_genotypes(path, format_fields=[...])`, and `.sink_csv(path)` writes the CSV.
  The carrier check works per ALT allele, not per record: a sample with GT 0/2 carries the second ALT allele only.
- `to_vcf` becomes `polars_bio.sink_vcf`, e.g. `pb.sink_vcf(pb.scan_vcf(path).filter(pl.col("qual") > 10), out_path)`.
- `VariantIntervalQueryable` and the query classes of `kipoiseq.extractors.vcf_query` are gone: filter the polars frames instead.
- `SingleVariantVCFSeqExtractor` and `SingleSeqVCFSeqExtractor` are gone: match the variants with `scan_variant_pairs`, and apply them with `iter_variant_seqs`, which yields batches of reference and alternative sequences.
  `VariantSeqExtractor` applies the variants of one interval.
- `Variant.from_cyvcf` and `Variant.from_cyvcf_and_given_alt` are gone.
  A `Variant` has only `chrom`, `pos`, `ref` and `alt`, so `id`, `qual`, `filter`, `info`, `source` and `copy()` are gone.
  The VCF columns ID, QUAL, FILTER and INFO stay in the frame of `scan_vcf_variants`.

`SingleVariantMatcher` and `MultiVariantsMatcher` become one function, `scan_variant_pairs(intervals, variants)`, which uses polars instead of pyranges:
- `scan_variant_pairs` returns the interval-variant pairs as a polars LazyFrame, with the interval columns, `interval_idx`, the `variant_` columns and `variant_idx`.
  The matchers were iterable and yielded Interval and Variant objects. Build these objects from the rows of the pairs instead:
  ```python
  pairs = scan_variant_pairs(intervals, scan_vcf_variants(path)).collect().sort("variant_idx", "interval_idx")
  for r in pairs.iter_rows(named=True):
      interval = Interval(r["chrom"], r["start"], r["end"], strand=r["strand"])
      variant = Variant(r["chrom"], r["variant_pos"], r["variant_ref"], r["variant_alt"])
  ```
  The sort gives the pairs in VCF order, and per variant in the order of the intervals. `SingleVariantMatcher` grouped them by chromosome and strand.
  For the variants of each interval, as `MultiVariantsMatcher` gave them, group the pairs by `interval_idx`. An interval without variants has no pairs.
  `iter_variant_seqs(pairs, reference, use_strand=True, group_by=["interval_idx"])` applies the variants of each interval together.
- `scan_variant_pairs` takes the intervals as `intervals`, a polars DataFrame with the columns `chrom`, `start`, `end`, optionally `strand` and any other columns, instead of `pranges`.
  For a PyRanges object `pr`, pass `intervals=pl.from_pandas(pr.df).rename({"Chromosome": "chrom", "Start": "start", "End": "end", "Strand": "strand"})`.
  For a list of Interval objects, pass `pl.DataFrame(intervals)`.
- The matcher attribute `pr` is gone, because `scan_variant_pairs` returns only the pairs.
- `gtf_path` and `bed_path` are gone. Read the file yourself, e.g. with polars-bio, and pass the frame as `intervals`.
- `variant_fetcher`, `VariantFetcher` and `vcf_file` are gone. Pass a polars frame as `variants`, e.g. `scan_vcf_variants(path)`.
  For a list of Variant objects, pass `pl.DataFrame(variants)`.
- `interval_attrs` is gone, because the pairs keep all columns of `intervals`.
  The pairs also keep all columns of `variants`, with the prefix `variant_`, e.g. `variant_qual` and `variant_filter` of `scan_vcf_variants`.
  For the genotypes of the samples, use `scan_vcf_genotypes`.
- `SingleVariantMatcher.iter_pyranges` and `iter_rows` are replaced by the LazyFrame of `scan_variant_pairs`.
  Collect it with `collect()`, write it with `sink_parquet(path)`, or process it in batches with `collect_batches(chunk_size=n)`.
- `scan_variant_pairs` finds the variants of all intervals with one overlap join. `MultiVariantsMatcher` ran one VCF query per interval.
- `variants_to_pyranges`, `intervals_to_pyranges`, `pyranges_to_intervals` and `PyrangesVariantFetcher` are gone.
  Build the frames from objects with `pl.DataFrame(variants)` and `pl.DataFrame(intervals)`.

## Breaking changes in 0.2

- The classes `Encoder` and `Tokenizer` of the new module `kipoiseq2.encoders` replace the encoding functions and transforms.
  `Encoder(alphabet, *, ambiguous="zeros", dtype=np.uint8)` one-hot encodes a sequence with its method `one_hot(seq)`, and its method `decode(x)` converts the array back.
  `Tokenizer(alphabet, *, token_map=None, dtype=np.uint8)` converts a sequence to integer tokens with its method `tokenize(seq)`.
  Each instance holds the complete configuration of its output, and its constructor checks it once, so create it once and reuse it.
  The package `kipoiseq2` exports both classes, `Alphabet` and the presets, e.g. `from kipoiseq2 import DNA, Encoder`.
- `Encoder` returns uint8 and encodes the ambiguity codes, e.g. `N`, as rows of zeros by default.
  In 0.1, the neutral letters were 0.25 in every column, `one_hot_dna` returned float32, and `one_hot` and `OneHot` returned float64.
  `Encoder(DNA, ambiguous="fractional", dtype=np.float32)` gives the old result of `one_hot_dna` for `N`, and `dtype=np.float64` gives the one of `one_hot` and `OneHot`.
  `"fractional"` gives a code of k letters 1/k in the columns of its letters, so `N` is 0.25 in every column and `R` is `[0.5, 0, 0.5, 0]`.
  `"fractional"` needs a floating-point dtype, so `Encoder(ambiguous="fractional")` with the default uint8 raises a ValueError.
  `"ones"` gives a code 1 in the columns of its letters, in any dtype, so `N` is 1 in every column and `R` is `[1, 0, 1, 0]`.
  `Encoder` and `Tokenizer` raise a TypeError for `dtype=None`, the default of `one_hot` and `OneHot`, which numpy reads as float64.
  Pass `dtype=np.float64` instead.
- The encoders take the letters as an `Alphabet` from the new module `kipoiseq2.alphabets`, instead of `alphabet` and `neutral_alphabet`.
  An `Alphabet` holds the letters of the columns and the letters that each ambiguity code stands for, e.g. `R` for A or G.
- `neutral_alphabet` and `neutral_value` are gone. A neutral letter of 0.1 stood for all letters, so it becomes a code for all letters.
  `alphabet, neutral_alphabet` becomes `Alphabet(alphabet, ambiguous=dict.fromkeys(neutral_alphabet, alphabet))`.
  For example, `alphabet=list("ACGT"), neutral_alphabet=["N"]` becomes `Alphabet(list("ACGT"), ambiguous={"N": list("ACGT")})`.
  `neutral_value=0` becomes `Encoder(alphabet, ambiguous="zeros")`, the default.
  `neutral_value=0.25` becomes `Encoder(alphabet, ambiguous="fractional")` with a float dtype.
  `neutral_value=1` becomes `Encoder(alphabet, ambiguous="ones")`.
  For a code of all n letters, `"fractional"` gives 1/n in every column. So it matches `neutral_value=0.25` only for 4 letters, e.g. for DNA.
  Other neutral values have no replacement. For other vectors, index a table of rows with the tokens of `Tokenizer`.
- `Alphabet` takes the letters only as a list or tuple of strings and raises a TypeError for a str.
  A str was ambiguous: `one_hot_dna` read `neutral_alphabet="NRY"` as three letters, but the others read it as one letter of three characters.
- `Alphabet` raises a ValueError if two letters are equal or differ in length, or if a code is a letter. Letters are case-sensitive, so `list("ACGTacgt")` holds eight letters.
  In 0.1, letters of different lengths raised an AssertionError, and a letter in both `alphabet` and `neutral_alphabet` was neutral.
- `kipoiseq2.utils.DNA`, `RNA`, `AMINO_ACIDS` and `alphabets` are gone. Use the `Alphabet` presets `DNA`, `RNA` and `AMINO_ACIDS` instead.
  Their letters are `DNA.letters` and so on.
  `DNA` and `RNA` also have `N` and the other IUPAC ambiguity codes, and `AMINO_ACIDS` has the codes `X`, `B` and `Z`.
  `AMINO_ACIDS.letters` holds only the 20 standard amino acids, so the arrays of `Encoder(AMINO_ACIDS)` have 20 columns.
  In 0.1, `AMINO_ACIDS` also held `B` and `Z` as letters, so its one-hot arrays had 22 columns.
  `Encoder(AMINO_ACIDS)` encodes `N` as asparagine. In 0.1, `one_hot(seq, AMINO_ACIDS)` kept the default `neutral_alphabet=["N"]` and encoded `N` as neutral.
- The encoders raise a ValueError that names the letters of the sequence that are not in the alphabet.
  In kipoiseq and kipoiseq2 0.1, `one_hot_dna` encoded lowercase and unknown letters as rows of zeros.
  `OneHot` did the same with the default `alphabet`, `neutral_alphabet` and `neutral_value`.
  Otherwise they raised a KeyError, as did `one_hot` and `tokenize`.
  In 0.2, `DNA` and `RNA` have the IUPAC ambiguity codes, so `Encoder` encodes them by its rule `ambiguous`.
  With the default `"zeros"`, they are still rows of zeros.
  With `"fractional"`, they are no longer rows of zeros, but fractions of their bases, e.g. `[0.5, 0, 0.5, 0]` for `R`.
  With `"ones"`, they are 1 in the columns of their bases, e.g. `[1, 0, 1, 0]` for `R`.
  All other unknown letters raise the ValueError.
- The lookup is case-sensitive, so a lowercase letter, e.g. of a soft-masked repeat, also raises this ValueError.
  Uppercase the sequence at the source, e.g. with `seq.upper()`, or read it with `FastaStringExtractor`, which uppercases by default.
  If the alphabet has the uppercase form of a lowercase letter, the message points to both.
- The encoders raise a ValueError instead of an AssertionError if the length of `seq` is not a multiple of the letter length.
  They raise a TypeError instead of a ValueError if `seq` is not a str.
- `one_hot(seq, alphabet, neutral_alphabet, neutral_value=0.25, dtype)` becomes `Encoder(Alphabet(alphabet, ambiguous=dict.fromkeys(neutral_alphabet, alphabet)), ambiguous="fractional", dtype=dtype).one_hot(seq)`.
  For example, `one_hot(seq)` becomes `Encoder(Alphabet(DNA.letters, ambiguous={"N": DNA.letters}), ambiguous="fractional", dtype=np.float64).one_hot(seq)`.
  `Encoder(DNA, ambiguous="fractional", dtype=np.float64)` gives the same result, and it also encodes the other IUPAC ambiguity codes.
- `one_hot_dna(seq)` becomes `Encoder(DNA, ambiguous="fractional", dtype=np.float32).one_hot(seq)`.
  For the IUPAC ambiguity codes other than `N`, it gives fractions of their bases, e.g. `[0.5, 0, 0.5, 0]` for `R`, where `one_hot_dna` gave rows of zeros.
  `one_hot_dna(seq, neutral_alphabet="NRY")` becomes `Encoder(Alphabet(DNA.letters, ambiguous=dict.fromkeys("NRY", DNA.letters)), ambiguous="fractional", dtype=np.float32).one_hot(seq)`.
- `OneHot(...)(seq)` becomes `Encoder(...).one_hot(seq)`. Where a callable is needed, pass the method `Encoder(...).one_hot`.
- `tokenize(seq, alphabet, neutral_alphabet)` becomes `Tokenizer(Alphabet(alphabet, ambiguous=dict.fromkeys(neutral_alphabet, alphabet))).tokenize(seq)`.
  It returns uint8 instead of int64 by default. Pass `dtype=np.int64` for int64.
- `Tokenizer` gives each ambiguity code a token of its own after the letters, instead of `-1` for every neutral letter.
  For `DNA`, `N` is `4`, `R` is `5` and `V` is `14`.
  With `-1`, numpy indexing took the last row, so `np.eye(4)[tokens]` silently encoded `N` as `T`.
  Now indexing a table with one row per letter raises an IndexError, and an embedding with `len(alphabet.letters) + len(alphabet.ambiguous)` rows takes the tokens as they are.
  `token_map` sets other tokens, e.g. one token for all codes. Tokens are non-negative, so `-1` is not possible.
- A `token_map` must have a key for each letter and code of its alphabet, so 15 keys for `DNA`.
  Several letters and codes can share a token, e.g. `Tokenizer(DNA, token_map={"A": 7, "C": 8, "G": 9, "T": 10, **dict.fromkeys(DNA.ambiguous, 11)})`.
  For a vocabulary without the IUPAC ambiguity codes, e.g. the one of HyenaDNA, use the strict alphabet `Alphabet(["A", "C", "G", "T"], ambiguous={"N": ["A", "C", "G", "T"]})`.
  Its `token_map` has the 5 keys A, C, G, T and N, and the other IUPAC ambiguity codes raise the ValueError.
- `token2one_hot(tokens, alphabet_size)` is gone. One-hot encode the sequence with `Encoder.one_hot` instead.
  For the tokens of `Tokenizer`, index a table with one row per token, e.g. `np.vstack([np.eye(4), np.full((11, 4), 0.25)])[tokens]` for `DNA`.
- `one_hot2token(arr)` is gone. It returned `arr.argmax(axis=2)`, so use that instead.
  For the tokens of `Tokenizer`, with the token of each code for the rows of the codes, use `tokenizer.tokenize(encoder.decode(x))`.
- `one_hot2string(arr)` becomes `[encoder.decode(x) for x in arr]`, with the encoder of the arrays, e.g. `Encoder(DNA, ambiguous="fractional", dtype=np.float32)` for the arrays of `one_hot_dna`.
  `one_hot2string` took an array of shape `(n_seqs, seq_len, n_columns)` and returned a list of strings, but `decode` takes the array of one sequence and returns a str.
  `decode` maps the row of an ambiguity code to that code, e.g. `N`, instead of the letter of the first column, e.g. `A`.
  With `"zeros"`, all codes share the row of zeros, so this row becomes the first code, e.g. `N`.
  It compares the rows exactly, so the rule `ambiguous` of the encoder must be the one of the array.
  It raises a ValueError for a row that is neither the row of a letter nor the row of an ambiguity code, e.g. for soft values of model predictions.
  For these, take the letter of the largest value, e.g. `"".join(np.array(DNA.letters)[x.argmax(axis=1)])`.
- `ReorderedOneHot` is gone. Reorder the axes of the `Encoder.one_hot` array with numpy instead.
  `Encoder().one_hot(seq)` returns an array of shape `(len(seq), 4)`, `np.expand_dims(x, axis)` inserts an axis of length 1, and `np.swapaxes(x, axis1, axis2)` swaps two axes.
  For example, `ReorderedOneHot(alphabet_axis=0, dummy_axis=1)`, with the shape `(4, 1, len(seq))`, becomes `np.swapaxes(np.expand_dims(Encoder().one_hot(seq), 1), 0, 2)`.
  `ReorderedOneHot.get_output_shape` has no replacement.
- `Interval` is immutable. Its fields cannot be reassigned, and `copy()` is gone.
  To change a field, use `dataclasses.replace`, e.g. `replace(interval, strand="+")`.
- `Interval` takes `strand` only as a keyword.
  In 0.1, `Interval("chr1", 0, 10, "-")` set `name` to `"-"` and kept the strand `"."`.
- `Interval.name`, `Interval.score` and `Interval.attrs` are gone, so an `Interval` has only `chrom`, `start`, `end` and `strand`.
  Keep the other fields in the columns of a polars frame, e.g. in the `intervals` of `scan_variant_pairs`.
- `Interval` checks its fields. It raises a TypeError if `start` or `end` is not an int, e.g. `10.0`.
  It raises a ValueError if `start` is greater than `end`, or if `strand` is not `"+"`, `"-"` or `"."`.
  Methods that would return such an interval raise the same ValueError, e.g. `slop` that removes more bases than the width, or `truncate` of an interval that starts after `chrom_len`.
- `Interval.width` is a property: use `interval.width` instead of `interval.width()` and `len(interval)`.
  The aliases `chr` and `stop` are gone: use `chrom` and `end`.
- `Interval.unstrand`, `swap_strand`, `neg_strand` and `is_valid` are gone.
  `interval.unstrand()` becomes `dataclasses.replace(interval, strand=".")`, and `interval.neg_strand` becomes `interval.strand == "-"`.
  `interval.swap_strand()` becomes `replace(interval, strand={"+": "-", "-": "+", ".": "."}[interval.strand])`.
  `interval.is_valid(chrom_len)` becomes `0 <= interval.start and interval.end <= chrom_len`.
- `Interval.truncate` requires `chrom_len`, and `chrom_len` must be an int, not a float such as `math.inf`.
  `interval.truncate()` without `chrom_len` becomes `replace(interval, start=max(interval.start, 0))`.
- An `Interval` is equal only to an `Interval` with the same fields.
  In 0.1, `==` compared `chrom`, `start`, `end` and `strand` also with other objects that have these attributes.
- `use_strand` of `Interval.center` and `Interval.shift` is keyword-only, e.g. `interval.shift(10, use_strand=False)` instead of `interval.shift(10, False)`.
  The parameter `x` of `shift` is renamed to `offset`. `Interval.slop` takes `upstream`, `downstream` and `use_strand` only as keywords.
- `Interval.center(use_strand=False)` returns the center of the unstranded interval, so it rounds a center in the middle of a base up instead of down.
  On all strands, the result grows by 1 for intervals with an odd `start + end`.
  `Interval.center()` with the default `use_strand=True` is unchanged.
- `resize_interval` is gone. Use `Interval.resize`, which takes an `anchor` now.
  `resize_interval(interval, width, anchor)` becomes `interval.resize(width, anchor=anchor, use_strand=False)` for `anchor="start"` or `"end"`, and `interval.resize(width)` for `anchor="center"`.
  With `anchor="center"`, the result on "+" and "." intervals moves by one base if `interval.width` and `width` differ in parity: towards `end` for an even `width`, and towards `start` for an odd `width`.
  For example, `Interval("chr1", 0, 11, strand="+")` resized to width 4 gives 4-8 instead of 3-7.
  "-" intervals give the same result as before.
  This holds for `start + end >= 0`. For a negative `start + end`, the old code rounded the center towards zero, so more results change.
- `Interval.resize(width, *, anchor="center", use_strand=True)` keeps the start, center or end of the interval.
  `anchor="start"` keeps the 5' end, which is `end` on "-" intervals, unless `use_strand=False`.
  `use_strand` is keyword-only. A negative `width` raises a ValueError, and `width=None` a TypeError.
  In 0.1, a negative `width` gave an interval with `start` greater than `end`, and `width=None` a copy.
- `Interval.slop` follows the strand: on "-" intervals, `upstream` now moves `end` and `downstream` moves `start`.
  "+" and "." intervals are unchanged. For the old behavior, pass `use_strand=False`.
- `Interval.trim(i, j)` becomes `Interval.slice(i, j)`, because `trim` of `kipoiseq2.transforms` trims a sequence to a length.
  `use_strand` is keyword-only. `slice` raises a ValueError unless `0 <= i < j <= interval.width`.
  In 0.1, `trim` raised an AssertionError if `j <= i`, and returned a longer interval if `j > interval.width`.
- `pad`, `trim` and `fixed_len` take only a str as `seq` and as `value`. In 0.1, they also took lists.
  `anchor` and `value` are keyword-only: `pad(seq, length, value, anchor)` becomes `pad(seq, length, anchor=anchor, value=value)`,
  and `fixed_len(seq, length, anchor, value)` becomes `fixed_len(seq, length, anchor=anchor, value=value)`.
- They raise a TypeError if `length` is not an int, and a ValueError if `length` is out of range or if `value` is empty.
  In 0.1, a `length` out of range raised an AssertionError, except a negative `length` in `trim` and `fixed_len`, which returned a wrong slice.
  An empty `value` raised a ZeroDivisionError.
- With `anchor="center"`, they keep the letters that `Interval.resize` keeps of an interval from 0 to `len(seq)`, so the result can move by one letter.
  For example, `trim(seq, 7)` of a sequence of 10 letters returns `seq[1:8]` instead of `seq[2:9]`, and `pad("A", 4)` returns `"NANN"` instead of `"NNAN"`.
- `trim(seq, 0, anchor="end")` returns `""` instead of `seq`.
- `FastaStringExtractor` and `VariantSeqExtractor` require `use_strand` as a keyword argument.
  In 0.1, `FastaStringExtractor` defaulted to `use_strand=False` and `VariantSeqExtractor` to `use_strand=True`.
  To keep the old behavior, pass `use_strand=False` to `FastaStringExtractor` and `use_strand=True` to `VariantSeqExtractor`.
- `FastaStringExtractor.extract` and `VariantSeqExtractor.extract` raise a TypeError for unknown arguments.
  In 0.1, they ignored them, so a typo like `fixed_length=False` had no effect.
- `VariantSeqExtractor(reference, *, use_strand)` takes the reference extractor as its first argument, instead of `fasta_file` or `reference_sequence`.
  `VariantSeqExtractor(fasta_file=path, use_strand=True)` becomes `VariantSeqExtractor(FastaStringExtractor(path, use_strand=False), use_strand=True)`,
  and `VariantSeqExtractor(reference_sequence=extractor, use_strand=True)` becomes `VariantSeqExtractor(extractor, use_strand=True)`.
  It raises a TypeError if `reference` has no method `extract` or no attribute `chrom_sizes`.
- `BaseExtractor` is gone. A reference extractor needs no base class: any object with the members of the protocol `SequenceExtractor` fits.
  These are `extract(interval)` and `chrom_sizes`, a mapping of each chromosome name to its length.
- `VariantSeqExtractor.ref_seq_extractor` and its deprecated alias `fasta` are gone. Keep the reference extractor that you pass instead.
- `FastaStringExtractor` is a context manager, which closes the FASTA file at the end of a `with` block.
  `VariantSeqExtractor` opens no file, so it has no `close()`. You close the reference that you pass yourself.
- `FastaStringExtractor` raises a ValueError for a FASTA file that is compressed with gzip instead of bgzip, and an ImportError for a bgzip file if Biopython is missing.
  In 0.1, pyfaidx raised an ImportError about Biopython for a gzip file whose name ends in `.gz`, and a UnicodeDecodeError for other gzip files.
- `force_upper` of `FastaStringExtractor` is now `uppercase=True` by default, so the extracted sequences are uppercase.
  `Encoder` and `Tokenizer` are case-sensitive, so they take these sequences as they are.
  Pass `uppercase=False` to keep the lowercase letters of soft-masked repeats. `uppercase` is keyword-only.
- `FastaStringExtractor.extract` raises a ValueError for an interval that starts before 0 or ends after the end of its chromosome, and for a chromosome that is not in the FASTA file.
  In 0.1, an interval past the chromosome end gave a shorter sequence, and the other cases raised a pyfaidx `FetchError`.
  It raises a TypeError if `interval` is not an `Interval`.
- `FastaStringExtractor.chrom_sizes` maps each chromosome to its length.
  The pyfaidx object `FastaStringExtractor.fasta` is private: use `chrom_sizes[chrom]` instead of `len(extractor.fasta[chrom])`, and `chrom_sizes.keys()` for the chromosome names.
- `VariantSeqExtractor.extract` no longer takes `chrom_len`. It reads the length of the chromosome from `reference.chrom_sizes`.
  If the sequence needs bases before 0 or after the end of the chromosome, it fills them with N for `pad_with_n=True` and raises a ValueError otherwise.
  In 0.1, without `chrom_len`, the reference returned fewer bases at the chromosome end. For an anchor outside the chromosome, the N were at the wrong end.
- `VariantSeqExtractor.extract` checks the variants. It raises a ValueError for a variant on another chromosome than `interval`, for a variant outside the chromosome, and for a variant whose REF differs from the reference, ignoring case.
  In 0.1, it replaced the reference bases at the position of such a variant with its ALT.
  It raises a TypeError if `interval` is not an `Interval` or a variant is not a `Variant`.
- `VariantSeqExtractor.extract` raises a ValueError for two variants that overlap, i.e. whose REF share a base, e.g. two ALT alleles of one VCF record.
  In 0.1, it applied both: two SNVs at one base gave two bases, in the order of the variants.
- `VariantSeqExtractor.extract(interval, variants, *, anchor_pos=None, pad_with_n=False)` takes its options only as keywords.
  `anchor` and `fixed_len` become `anchor_pos`: `anchor=a` becomes `anchor_pos=a`, and `anchor=a, fixed_len=False` becomes `anchor_pos=None`.
  `is_padding` is renamed to `pad_with_n`.
- `VariantSeqExtractor.extract` raises a ValueError unless `interval.start <= anchor_pos <= interval.end`.
  In 0.1, it moved an anchor outside the interval to the nearest end of the interval.
- `FastaStringExtractor.extract` and `VariantSeqExtractor.extract` no longer take `use_strand`.
  Create the extractor with the `use_strand` you need instead.
- `rc_dna(seq)` becomes `reverse_complement(seq)` from `kipoiseq2.transforms`.
  `reverse_complement` also takes the IUPAC codes, e.g. N, and raises ValueError instead of KeyError for other characters.
- `rc_rna(seq)` is gone. For an RNA sequence in upper case, use `reverse_complement(seq.replace("U", "T")).replace("T", "U")`.
- `kipoiseq2.utils.parse_dtype` is gone, because it called `eval()` on strings, and only `ReorderedOneHot` used it.
  For a dtype name, use `np.dtype(name)`, e.g. `np.dtype("float32")`.
- `SplitSplicingSeq` is gone, because only the MMSplice dataloader used it. Keep using `kipoiseq` for it.
- The module `kipoiseq2.utils` is gone. Its `DNA`, `RNA` and `AMINO_ACIDS` moved to `kipoiseq2.alphabets` in 0.2.
  Its other helpers have these replacements:
  - `to_scalar(x)` becomes `x.item()` for a numpy scalar.
  - `batch_iter(items, n)` becomes `itertools.batched(items, n)`, which yields tuples instead of lists.
  - `parse_alphabet(alphabet)` becomes `list(alphabet)`. For `Encoder` and `Tokenizer`, pass `Alphabet(list(alphabet))` from `kipoiseq2.alphabets`.
- `Variant` is immutable and has only the fields `chrom`, `pos`, `ref` and `alt`.
  `id`, `qual`, `filter`, `info`, `source` and `copy()` are gone, so `Variant("chr1", 10, "C", "T", id="rs1")` raises a TypeError.
  The VCF columns ID, QUAL, FILTER and INFO stay in the frame of `scan_vcf_variants`, e.g. `scan_vcf_variants(path).select("chrom", "pos", "id", "qual", "filter")`.
- `Variant` raises a TypeError for arguments of the wrong type instead of converting them, and a ValueError for a `pos` below 1 or an empty `ref`.
  In 0.1, `Variant("chr1", 10.7, "C", "T")` gave `pos=10`, and `Variant("chr1", "10", "C", "T")` gave `pos=10` too.
- A `Variant` is equal only to a `Variant` with the same fields.
  In 0.1, it was also equal to any object with the same `chrom`, `pos`, `ref` and `alt`, e.g. a kipoiseq `Variant`.
- `scan_vcf_variants` takes only the options `info_fields` and `sequence_alts_only`, as keyword arguments.
  It no longer passes other keyword arguments to `polars_bio.scan_vcf`.
  For the FORMAT fields of the samples, use `scan_vcf_genotypes(path, samples=..., format_fields=..., carriers_only=False)` instead of `scan_vcf_variants(path, samples=..., format_fields=...)`.
  `use_zero_based` is gone, because it did not change the output.
- `scan_vcf_genotypes` takes its options only as keyword arguments, and `info_fields` and `sequence_alts_only` replace the keyword arguments of `polars_bio.scan_vcf`.
  `scan_vcf_genotypes(path, ["S1"])` becomes `scan_vcf_genotypes(path, samples=["S1"])`.
- `scan_vcf_variants` and `scan_vcf_genotypes` raise a TypeError if `info_fields`, `format_fields` or `samples` is a str, or if it is not a sequence of strings, e.g. a numpy array.
  `info_fields="AF"` becomes `info_fields=["AF"]`.
- `SingleVariantMatcher`, `MultiVariantsMatcher` and `BaseVariantMatcher` become one function, `scan_variant_pairs(intervals, variants)`.
  It returns the pairs as a polars LazyFrame, with the columns of `pairs()`, in no fixed order.
  `SingleVariantMatcher(path, intervals=exons).pairs()` becomes `scan_variant_pairs(exons, scan_vcf_variants(path)).collect().sort("variant_idx", "interval_idx")`, and `scan_pairs()` becomes `scan_variant_pairs(...)`.
- `vcf_file` is gone: pass `scan_vcf_variants(path)` as `variants` instead.
- `scan_variant_pairs` takes only polars frames: `intervals` is a DataFrame, and `variants` is a DataFrame or LazyFrame.
  For lists of Interval or Variant objects, pass `pl.DataFrame(intervals)` or `pl.DataFrame(variants)`.
- The matchers were iterable, and the pairs are no longer Interval and Variant objects. To build these objects, loop over the rows of the pairs, as shown in the kipoiseq section above.
  For the variants of each interval, as `MultiVariantsMatcher` gave them, group the pairs by `interval_idx`.
- `variant_batch_size` and `iter_batches(batch_size=n)` become `scan_variant_pairs(...).collect_batches(chunk_size=n)`.
  A batch holds `n` pairs instead of the pairs of `n` variants, so the pairs of one variant can be in two batches.
- `intervals_to_polars`, `variants_to_polars` and `overlap_variants` are gone.
  `pl.DataFrame(intervals)` and `pl.DataFrame(variants)` replace the converters, and `scan_variant_pairs(intervals, variants).collect().sort("variant_idx", "interval_idx")` replaces `overlap_variants`.
- `interval_attrs` is gone, because the pairs keep all columns of the `intervals` DataFrame.
  To keep fewer columns, select them first: `interval_attrs=["exon_id"]` becomes `scan_variant_pairs(exons.select("chrom", "start", "end", "strand", "exon_id"), variants)`.
  In 0.1, the pairs had only `chrom`, `start`, `end`, `strand` and the `interval_attrs` columns.
- `scan_variant_pairs` raises a ValueError if `intervals` has a column named `interval_idx` or starting with `variant_`, or if `variants` has a column named `idx` or `variant_idx`, because the pairs use these names.
- A null strand in `intervals` gives the strand `"."` in the pairs, as a missing strand column.
- With `scan_vcf_variants(path)` as `variants`, the pairs keep all its columns, e.g. `variant_qual` and `variant_filter`.
  `vcf_file` kept only the coordinates, `variant_ref`, `variant_alt` and `variant_allele_idx`.
- `translate` takes `transl_table` and `transl_except` only as keyword arguments: `translate(seq, 2)` becomes `translate(seq, transl_table=2)`.
- `translate` raises a ValueError that names the codons that are not in the genetic code, e.g. `ATN`, instead of a KeyError.
  The lookup stays case-sensitive, so a lowercase codon such as `atg` raises this ValueError too, and the message hints at `seq.upper()`.
  It raises a TypeError instead of a ValueError if a position in `transl_except` is not an int or an amino acid is not a str.
- `TRANSLATION_TABLE` is read-only. In 0.1, it was a dict, and changing it changed the result of `translate` with `transl_table` 1 and 11.
  For a table that you can change, use `dict(TRANSLATION_TABLE)`.
- The transform classes `Compose`, `DummyAxis`, `SwapAxes` and `ResizeInterval` are gone, because they only wrapped functions for the configs of the Kipoi dataloaders.
  They have these replacements:
  - `DummyAxis(axis)(x)` becomes `np.expand_dims(x, axis)`.
  - `SwapAxes(axis1, axis2)(x)` becomes `np.swapaxes(x, axis1, axis2)`.
  - `ResizeInterval(width, anchor)(interval)` becomes `interval.resize(width, anchor=anchor)`.
    For the 0.1 behavior of the anchors `"start"` and `"end"`, pass `use_strand=False`, as for `resize_interval`.
  - `Compose([f, g])` becomes a function that calls the steps in order, e.g. `lambda x: g(f(x))`.
- The functions and constants of `kipoiseq2.transforms.functional` moved to the module `kipoiseq2.transforms`, and `kipoiseq2.transforms.functional` is gone.
  For example, `from kipoiseq2.transforms.functional import translate` becomes `from kipoiseq2.transforms import translate`.
