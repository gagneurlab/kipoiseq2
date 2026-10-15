"""Import polars and polars-bio, the optional dependencies of the `ranges` extra."""


def _import_polars():
    try:
        import polars as pl
        import polars_bio as pb
    except ImportError as e:
        raise ImportError(
            "VCF reading and variant matching need the `ranges` extra: pip install 'kipoiseq2[ranges]'"
        ) from e
    return pl, pb
