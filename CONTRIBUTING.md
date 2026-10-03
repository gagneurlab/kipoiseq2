# Contributing to `kipoiseq2`

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

## Types of contributions

### Report bugs

Report bugs at <https://github.com/kipoi/kipoiseq2/issues>.

If you are reporting a bug, please include:

-   Your operating system name and version.
-   Any details about your local setup that might be helpful in troubleshooting.
-   Detailed steps to reproduce the bug.

### Fix bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write documentation

kipoiseq2 could always use more documentation, whether as docstrings, in the README, or even on the web in blog posts, articles, and such.

### Submit feedback

The best way to send feedback is to file an issue at <https://github.com/kipoi/kipoiseq2/issues>.

If you are proposing a feature:

-   Explain in detail how it would work.
-   Keep the scope as narrow as possible, to make it easier to implement.
-   Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Workflow

- make an issue for the thing you want to implement
- create the corresponding branch
- develop
- write unit tests in tests/
- write documentation in markdown (see other functions for example)
- push the changes
- make a pull request with a [Conventional Commits](https://www.conventionalcommits.org/) title (see [Releases](#releases))
- once the pull request is merged, the issue will be closed

## Development environment

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

### Initial setup

```bash
git clone git@github.com:kipoi/kipoiseq2.git
cd kipoiseq2

# Create the conda env (provides Python + uv)
micromamba env create -f environment-dev.yml
micromamba activate kipoiseq2

# Install all dependency groups (runtime + dev + test + lint) into a uv-managed venv
uv sync --all-groups
```

All subsequent commands assume the env is activated and `uv` is on `PATH`.

## Running tasks

The project standardises tasks through [`tox`](https://tox.wiki/) with the `tox-uv` runner. Run any environment with:

```bash
uv run tox -e <env>
```

Available environments (defined in `pyproject.toml`):

| Env             | Purpose                                  |
|-----------------|------------------------------------------|
| `format-check`  | `ruff format --check .`                  |
| `lints`         | `ruff check .`                           |
| `typecheck`     | `mypy src/kipoiseq2`                     |
| `py3.12`        | Run pytest under Python 3.12             |
| `py3.13`        | Run pytest under Python 3.13             |
| `py3.14`        | Run pytest under Python 3.14             |

Run the full matrix CI runs with:

```bash
uv run tox
```

### Quick commands

```bash
# Format code
uv run ruff format .

# Lint with autofix
uv run ruff check --fix .

# Run tests directly (single Python)
uv run pytest

# Run tests in parallel on 4 cores
uv run pytest -n 4

# Run a single test
uv run pytest tests/test_1_extractors.py::test_name -x
```

## Pull request guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring.
3.  The pull request should pass CI on all supported Python versions (3.12 to 3.14, see `.github/workflows/ci.yml`).

## Releases

Versioning and tagging are automated by [release-please](https://github.com/googleapis/release-please) (`.github/workflows/release-please.yml`). Publishing is handled by `.github/workflows/publish.yml`:

- release-please watches commits on `master` and opens/maintains a release PR that bumps `pyproject.toml` and `src/kipoiseq2/__init__.py` and updates `CHANGELOG.md`.
- Merging the release PR cuts a `vX.Y.Z` tag and a GitHub release.
- `release-please.yml` then starts `publish.yml` with `workflow_dispatch` for the new tag. `publish.yml` builds sdist + wheel and uploads to PyPI via trusted publishing (OIDC). It can also be triggered by hand from the Actions tab.
- `release-please.yml` does not call `publish.yml` via `workflow_call`, because PyPI then rejects the upload's attestation: the attestation names `release-please.yml`, but the trusted publisher on PyPI is `publish.yml`.

Pull requests are squash-merged, so the pull request title becomes the commit message on `master`. Use [Conventional Commits](https://www.conventionalcommits.org/) for it so release-please can pick the next version (`fix:` → patch, `feat:` → minor, `feat!:` / `BREAKING CHANGE:` → major).
