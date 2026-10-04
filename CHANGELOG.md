# Changelog

## 0.1.0 (2026-10-04)


### Features

* **extractors:** add scan_vcf_genotypes for per-sample carriers ([e4aa23f](https://github.com/gagneurlab/kipoiseq2/commit/e4aa23f5148833ad4223cf95708e37ef03542caf))
* **extractors:** keep id, qual and filter in scan_vcf_variants ([e5ba15e](https://github.com/gagneurlab/kipoiseq2/commit/e5ba15e118cea5451fafb3648daca796d35660c5))
* **extractors:** match variant tables in the matchers ([0cb5de5](https://github.com/gagneurlab/kipoiseq2/commit/0cb5de5a6abebe598044a4d5baf34bd44eda7de5))
* **extractors:** match variants with a polars-bio overlap join ([91c02a4](https://github.com/gagneurlab/kipoiseq2/commit/91c02a4030ba44f373acd503056ec2c9d4981a1c))
* **extractors:** read VCF files as polars tables with polars-bio ([841e6a3](https://github.com/gagneurlab/kipoiseq2/commit/841e6a3aa46d62928534547aa98f160eebf73b3f))
* kipoiseq2, kipoiseq without the Kipoi dataloaders and kipoi dependencies ([36c521f](https://github.com/gagneurlab/kipoiseq2/commit/36c521f7d335bd3e72a6f94d45a6500da57aa415))
* kipoiseq2, kipoiseq without the Kipoi dataloaders and kipoi dependencies ([a012369](https://github.com/gagneurlab/kipoiseq2/commit/a012369e3ef96fdf28ef2f3b6eee19cae8e4391a))
* **transforms:** translate with the NCBI genetic code and annotated translation exceptions ([bcf8bf6](https://github.com/gagneurlab/kipoiseq2/commit/bcf8bf6100acfae52c7b832ef8184e8bd1207212))


### Bug Fixes

* allow intervals that end at the chromosome end ([e9f5de1](https://github.com/gagneurlab/kipoiseq2/commit/e9f5de143faccde84146b4b86887645225082cf1))
* **extractors:** accept a generator of Interval objects in the matchers ([6d4c8ff](https://github.com/gagneurlab/kipoiseq2/commit/6d4c8ff24efdd623a1265eb6c580801523751bd3))
* **extractors:** infer UTRs from the CDS with pandas 2 and later ([a012369](https://github.com/gagneurlab/kipoiseq2/commit/a012369e3ef96fdf28ef2f3b6eee19cae8e4391a))
* **extractors:** keep the fixed length when the anchor is at the interval start ([3ac59db](https://github.com/gagneurlab/kipoiseq2/commit/3ac59dbcfd807af09e29bd61b217030dcf3f668a))
* **extractors:** keep the interval in VariantIntervalQueryable.filter ([7d113ab](https://github.com/gagneurlab/kipoiseq2/commit/7d113ab5b0c82ccdf4f8923a27492eaeb10d2587))
* **extractors:** keep the interval table public as intervals ([cbc896d](https://github.com/gagneurlab/kipoiseq2/commit/cbc896d38b51815d9b83de23aae2520f2fe11d6f))
* **extractors:** mark the scan_vcf_variants output as 0-based ([0865e03](https://github.com/gagneurlab/kipoiseq2/commit/0865e031414e7083a95d9f5c77efab5845f26831))
* **transforms:** build a new Interval in resize_interval ([cf039b6](https://github.com/gagneurlab/kipoiseq2/commit/cf039b6099e4150a159f072d2525ad69ffa46cbb))
* **vcf:** name the vcf extra when cyvcf2 is missing in to_vcf ([775c4d6](https://github.com/gagneurlab/kipoiseq2/commit/775c4d6b5f135790b86aae510035ae5690345645))
* **vcf:** use variant_gap when get_variants builds the regions ([0afa1d4](https://github.com/gagneurlab/kipoiseq2/commit/0afa1d4742c6f05611edadc80ec7a6410c774554))


### Performance Improvements

* **extractors:** join with the variants as the streaming side ([dcce7f4](https://github.com/gagneurlab/kipoiseq2/commit/dcce7f4c0b9cdc20eed0aa6a9f11479353c40d2f))


### Documentation

* drop protein from the package description ([ec46061](https://github.com/gagneurlab/kipoiseq2/commit/ec46061bd95d2de441f568b543bd7e8c686f7997))
* move the migration guide from kipoiseq to MIGRATING.md ([44df5b4](https://github.com/gagneurlab/kipoiseq2/commit/44df5b4fb542822309d22f2559c90afc2cdc2d41))
* move the migration guide from kipoiseq to MIGRATING.md ([468bdfc](https://github.com/gagneurlab/kipoiseq2/commit/468bdfc4bdaf256d57f27595acfbe474cba88828))
* say how to turn off the polars-bio progress bars ([617855e](https://github.com/gagneurlab/kipoiseq2/commit/617855ea0d03e2ca8aa4f73025fc48a5ba6feb3d))
* say main instead of master ([45319cb](https://github.com/gagneurlab/kipoiseq2/commit/45319cb8cbb8b5d22a270634509cca2dab2435f0))
