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
  With a sequence of Interval objects as `intervals`, it yields these objects, so they keep their attrs.
- With a VCF file or a polars frame as the variant source, the matchers yield new Variant objects with chrom, pos, ref and alt, and without `source`.
  For the genotypes of the samples, use `scan_vcf_genotypes`.
- `SingleVariantMatcher.iter_pyranges` and `iter_rows` are replaced by `pairs()`, `scan_pairs()` and `iter_batches()`.
  `pairs()` returns all pairs as a polars DataFrame, `scan_pairs()` as a LazyFrame, and `iter_batches()` yields one DataFrame per batch of variants.
- `MultiVariantsMatcher` finds the variants of all intervals with one overlap join instead of one VCF query per interval.
- `variants_to_pyranges`, `intervals_to_pyranges` and `pyranges_to_intervals` are replaced by `variants_to_polars` and `intervals_to_polars`.
  `PyrangesVariantFetcher` is gone: pass the variants as `variants`.

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
  Uppercase the sequence at the source, e.g. with `seq.upper()` or `FastaStringExtractor(force_upper=True)`.
  If the alphabet has the uppercase form of a lowercase letter, the message points to both.
- The encoders raise a ValueError instead of an AssertionError if the length of `seq` is not a multiple of the letter length.
  They raise a TypeError instead of a ValueError if `seq` is not a str.
- `one_hot(seq, alphabet, neutral_alphabet, neutral_value=0.25, dtype)` becomes `Encoder(Alphabet(alphabet, ambiguous=dict.fromkeys(neutral_alphabet, alphabet)), ambiguous="fractional", dtype=dtype).one_hot(seq)`.
  For example, `one_hot(seq)` becomes `Encoder(Alphabet(DNA.letters, ambiguous={"N": DNA.letters}), ambiguous="fractional", dtype=np.float64).one_hot(seq)`.
  `Encoder(DNA, ambiguous="fractional", dtype=np.float64)` gives the same result, and it also encodes the other IUPAC ambiguity codes.
- `one_hot_dna(seq)` becomes `Encoder(DNA, ambiguous="fractional", dtype=np.float32).one_hot(seq)`.
  For the IUPAC ambiguity codes other than `N`, it gives fractions of their bases, e.g. `[0.5, 0, 0.5, 0]` for `R`, where `one_hot_dna` gave rows of zeros.
  `one_hot_dna(seq, neutral_alphabet="NRY")` becomes `Encoder(Alphabet(DNA.letters, ambiguous=dict.fromkeys("NRY", DNA.letters)), ambiguous="fractional", dtype=np.float32).one_hot(seq)`.
- `OneHot(...)(seq)` becomes `Encoder(...).one_hot(seq)`. In a `Compose`, pass the method, e.g. `Compose([Encoder().one_hot, DummyAxis(0)])`.
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
- `Interval` is immutable. Its fields cannot be reassigned, `attrs` is a read-only copy of the given mapping, and `copy()` is gone.
  To change a field, use `dataclasses.replace`, e.g. `replace(interval, attrs={**interval.attrs, "gene_id": "g1"})`.
- `Interval` takes `strand` and `attrs` only as keywords.
  In 0.1, `Interval("chr1", 0, 10, "-")` set `name` to `"-"` and kept the strand `"."`.
- `Interval.name` and `Interval.score` are gone. Keep them in `attrs`, e.g. `Interval("chr1", 0, 10, attrs={"name": "a", "score": 1.5})`.
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
- Two intervals are equal if all their fields are equal, including `attrs`. The hash leaves out `attrs`.
  In 0.1, `==` compared only `chrom`, `start`, `end` and `strand`, also with other objects that have these attributes.
  To compare the locations, compare `(i.chrom, i.start, i.end, i.strand)`.
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
- `ResizeInterval(width, *, anchor="center", use_strand=True)` calls `Interval.resize`, so the anchors `"start"` and `"end"` follow the strand.
  For the 0.1 behavior of these anchors, pass `use_strand=False`.
  `anchor` is keyword-only. `ResizeInterval` checks `width` and `anchor` when it is created.
  In 0.1, it raised an Exception for an invalid `anchor` at the first call, and a negative `width` gave intervals with `start` greater than `end`.
- `Interval.slop` follows the strand: on "-" intervals, `upstream` now moves `end` and `downstream` moves `start`.
  "+" and "." intervals are unchanged. For the old behavior, pass `use_strand=False`.
- `Interval.trim(i, j)` becomes `Interval.slice(i, j)`, because `trim` of `kipoiseq2.transforms.functional` trims a sequence to a length.
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
  `force_upper` is keyword-only too: `FastaStringExtractor(path, True, True)` becomes `FastaStringExtractor(path, use_strand=True, force_upper=True)`.
- `FastaStringExtractor.extract` and `VariantSeqExtractor.extract` raise a TypeError for unknown arguments.
  In 0.1, they ignored them, so a typo like `fixed_length=False` had no effect.
- `FastaStringExtractor` raises a ValueError for a FASTA file that is compressed with gzip instead of bgzip, and an ImportError for a bgzip file if Biopython is missing.
  In 0.1, pyfaidx raised an ImportError about Biopython for a gzip file whose name ends in `.gz`, and a UnicodeDecodeError for other gzip files.
