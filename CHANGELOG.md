# Changelog

## [0.1.1](https://github.com/gagneurlab/kipoiseq2/compare/v0.1.0...v0.1.1) (2026-10-15)


### Bug Fixes

* **dataclasses:** ignore the strand in Interval.resize(use_strand=False) ([#6](https://github.com/gagneurlab/kipoiseq2/issues/6)) ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))
* **dataclasses:** return NotImplemented when comparing with other types ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))
* **extractors:** do not raise in FastaStringExtractor.close after a ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))
* **extractors:** enforce BaseExtractor.extract as an abstract method ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))
* name the right arguments in error messages ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))
* **transforms:** correct the splice site warnings of SplitSplicingSeq ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))
* **transforms:** export only the transform classes from ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))
* **transforms:** pad to the requested length when value has several ([3ec2672](https://github.com/gagneurlab/kipoiseq2/commit/3ec26728f32314c740248f9ac88d8c4a2da70161))


### Documentation

* build and publish API docs with Sphinx ([#7](https://github.com/gagneurlab/kipoiseq2/issues/7)) ([c24903f](https://github.com/gagneurlab/kipoiseq2/commit/c24903fd121f2e4b00241c1624578ed728d66678))

## 0.1.0 (2026-10-04)


### Features

* **extractors:** add scan_vcf_genotypes for per-sample carriers ([e4aa23f](https://github.com/gagneurlab/kipoiseq2/commit/e4aa23f5148833ad4223cf95708e37ef03542caf))
* **extractors:** keep id, qual and filter in scan_vcf_variants ([e5ba15e](https://github.com/gagneurlab/kipoiseq2/commit/e5ba15e118cea5451fafb3648daca796d35660c5))
* **extractors:** match variant tables in the matchers ([0cb5de5](https://github.com/gagneurlab/kipoiseq2/commit/0cb5de5a6abebe598044a4d5baf34bd44eda7de5))
* **extractors:** match variants with a polars-bio overlap join ([91c02a4](https://github.com/gagneurlab/kipoiseq2/commit/91c02a4030ba44f373acd503056ec2c9d4981a1c))
* **extractors:** read VCF files as polars tables with polars-bio ([841e6a3](https://github.com/gagneurlab/kipoiseq2/commit/841e6a3aa46d62928534547aa98f160eebf73b3f))
* kipoiseq2, kipoiseq without the Kipoi dataloaders and kipoi dependencies ([a012369](https://github.com/gagneurlab/kipoiseq2/commit/a012369e3ef96fdf28ef2f3b6eee19cae8e4391a))
* **transforms:** translate with the NCBI genetic code and annotated translation exceptions ([bcf8bf6](https://github.com/gagneurlab/kipoiseq2/commit/bcf8bf6100acfae52c7b832ef8184e8bd1207212))


### Bug Fixes

* allow intervals that end at the chromosome end ([e9f5de1](https://github.com/gagneurlab/kipoiseq2/commit/e9f5de143faccde84146b4b86887645225082cf1))
* **extractors:** accept a generator of Interval objects in the matchers ([6d4c8ff](https://github.com/gagneurlab/kipoiseq2/commit/6d4c8ff24efdd623a1265eb6c580801523751bd3))
* **extractors:** keep the fixed length when the anchor is at the interval start ([3ac59db](https://github.com/gagneurlab/kipoiseq2/commit/3ac59dbcfd807af09e29bd61b217030dcf3f668a))
* **extractors:** keep the interval table public as intervals ([cbc896d](https://github.com/gagneurlab/kipoiseq2/commit/cbc896d38b51815d9b83de23aae2520f2fe11d6f))
* **extractors:** mark the scan_vcf_variants output as 0-based ([0865e03](https://github.com/gagneurlab/kipoiseq2/commit/0865e031414e7083a95d9f5c77efab5845f26831))
* **transforms:** build a new Interval in resize_interval ([cf039b6](https://github.com/gagneurlab/kipoiseq2/commit/cf039b6099e4150a159f072d2525ad69ffa46cbb))


### Performance Improvements

* **extractors:** join with the variants as the streaming side ([dcce7f4](https://github.com/gagneurlab/kipoiseq2/commit/dcce7f4c0b9cdc20eed0aa6a9f11479353c40d2f))


### Documentation

* drop protein from the package description ([ec46061](https://github.com/gagneurlab/kipoiseq2/commit/ec46061bd95d2de441f568b543bd7e8c686f7997))
* move the migration guide from kipoiseq to MIGRATING.md ([468bdfc](https://github.com/gagneurlab/kipoiseq2/commit/468bdfc4bdaf256d57f27595acfbe474cba88828))
* say how to turn off the polars-bio progress bars ([617855e](https://github.com/gagneurlab/kipoiseq2/commit/617855ea0d03e2ca8aa4f73025fc48a5ba6feb3d))
* say main instead of master ([45319cb](https://github.com/gagneurlab/kipoiseq2/commit/45319cb8cbb8b5d22a270634509cca2dab2435f0))
