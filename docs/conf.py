"""Sphinx configuration for the kipoiseq2 docs.

Build the docs with ``uv run tox -e docs``.
"""

import re
from importlib.metadata import version as _version

project = "kipoiseq2"
author = "Kipoi team"
copyright = "2018, Kipoi team"
release = _version("kipoiseq2")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

exclude_patterns = ["_build"]

# autodoc
# class docstrings document the constructor parameters, so skip __init__ docstrings
autoclass_content = "class"
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "polars": ("https://docs.pola.rs/api/python/stable", None),
}

# MyST renders README.md, MIGRATING.md and CHANGELOG.md
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# HTML
html_theme = "furo"
html_title = f"kipoiseq2 {release}"

# The included Markdown files link to repo paths such as tests/ and CONTRIBUTING.md.
# The docs do not contain these paths, so the links point to them on GitHub.
_REPO_BLOB_URL = "https://github.com/gagneurlab/kipoiseq2/blob/main/"
_RELATIVE_LINK = re.compile(r"\]\((?![a-z]+:|#)")


def _link_repo_paths(app, relative_path, parent_docname, content):
    content[0] = _RELATIVE_LINK.sub("](" + _REPO_BLOB_URL, content[0])


def setup(app):
    app.connect("include-read", _link_repo_paths)
