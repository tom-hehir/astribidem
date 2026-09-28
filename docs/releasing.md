# Development and releases

## Local checks

Use Python 3.11 or newer and a project environment:

```sh
uv sync --locked --extra dev
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run --all-files
uv run --no-sync pytest
uv build
uv run --no-sync twine check --strict dist/*
```

Update `uv.lock` with `uv lock` when changing dependencies. It fixes the
development tools; consumers resolve the ranges in `pyproject.toml`. CI tests
both current dependencies and `tests/minimum-requirements.txt` against the
built wheel. Keep that file aligned with the supported dependency floors.
DuckDB 1.5 is required for the Arrow reader used by sorted segment resolution.

CI runs on pull requests, pushes to `main`, and manual dispatch. It includes
pre-commit, a wheel built from the source distribution, strict metadata checks,
Python 3.11–3.14 on Linux, Python 3.13 on macOS and Windows, and separate base
and `large` installation checks without test dependencies. Coverage XML is
retained as a workflow artifact; no external coverage service is required.

## One-time Trusted Publisher registration

Create a pending publisher for the new project in each registry you will use.
The first successful upload creates the project. Register these exact values:

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project name | `astribidem` | `astribidem` |
| GitHub owner | `tom-hehir` | `tom-hehir` |
| Repository | `astribidem` | `astribidem` |
| Workflow filename | `release.yml` | `release.yml` |
| GitHub environment | `pypi` | `testpypi` |

The production and test registries have separate accounts and registrations.
Use GitHub environments with those names; restrict their deployment policy to
release tags (`v*`). No PyPI API token or repository secret is needed. See
[PyPI's pending-publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

The source repository is currently private. Package uploads expose the source
included in the wheel and sdist. Make the linked repository documentation and
issue tracker public before announcing a public release, or provide public
alternatives and update the package links. A public GitHub repository is not a
technical prerequisite for a PyPI upload.

## Rehearse on TestPyPI

1. Merge the release preparation and confirm CI is green.
2. Check `pyproject.toml`, `CHANGELOG.md` and documentation for the intended
   version. The initial version is `0.0.0a0`.
3. Create and push the immutable tag `v0.0.0a0` at the tested commit.
4. Manually run **Release** from that tag. A manual run uploads only to
   TestPyPI. Selecting a branch whose name does not match the version fails
   the tag check.
5. Install the TestPyPI distribution in a fresh environment and exercise a
   spatial and UID example. Install dependencies from normal PyPI first;
   then fetch only astribidem from TestPyPI:

```sh
uv venv /tmp/astribidem-release-check
uv pip install --python /tmp/astribidem-release-check numpy scipy pyarrow 'duckdb>=1.5'
uv pip install --python /tmp/astribidem-release-check --no-deps \
  --index-url https://test.pypi.org/simple/ 'astribidem==0.0.0a0'
uv run --no-project --python /tmp/astribidem-release-check python -c \
  'import astribidem; print(astribidem.__version__)'
```

## Publish to PyPI

Publish a GitHub release for `v0.0.0a0`, marking it as a prerelease. The
`release: published` event handles both prereleases and final releases.
Creating a tag or a draft GitHub release alone does not upload to PyPI.

The release workflow checks that the tag exactly equals `v` plus the package
version, builds once, and runs the same pre-commit, test and installation jobs
as CI. Only when all checks pass does a separate job download and publish the
tested wheel and sdist. That job alone receives `id-token: write`. It does not
check out or execute repository source and does not rebuild distributions.

Verify a fresh `pip install --pre astribidem` after publishing. Release numbers
and uploaded filenames cannot be reused; make another alpha version for a
corrected release rather than moving a tag. Publishing `astribidem-hf` later
requires replacing its private Git dependency with a compatible released
astribidem version.
