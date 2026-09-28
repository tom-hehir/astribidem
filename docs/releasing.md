# Development and releases

## Local checks

Use Python 3.14 to match CI and create a project environment:

```sh
uv sync --locked --extra dev --python 3.14
uv run --no-sync pre-commit install
uv run --no-sync pre-commit run --all-files
uv run --no-sync pytest
uv build
uv run --no-sync twine check --strict dist/*
```

Update `uv.lock` with `uv lock` when changing dependencies. It fixes the
development tools; consumers resolve the ranges in `pyproject.toml`. CI tests
the built wheel with current compatible dependencies on Ubuntu and Python 3.14.
The package still accepts Python 3.11+, but CI only exercises Python 3.14.
DuckDB 1.5 is required for the Arrow reader used by sorted segment resolution.

CI runs on pull requests, pushes to `main`, and manual dispatch. Its three jobs
are defined in separate reusable workflow files:

- `pre-commit.yml` runs formatting and static checks.
- `build.yml` builds the source distribution and a wheel from it, checks package
  metadata and README rendering, and uploads the distributions.
- `test.yml` checks isolated base and `large` installations without test
  dependencies, then runs the full suite against the built wheel and retains
  coverage XML as a workflow artifact.

All three use Ubuntu and Python 3.14. `ci.yml` connects them, running pre-commit
alongside the build and starting tests once the distributions are available.
`release.yml` calls that same pipeline before publishing; the three reusable
workflows do not run independently on repository events.

External Actions are pinned to exact commit SHAs; comments record their version
labels. To update an Action, resolve the intended release in its official
repository, update the SHA and comment together, and verify CI on the PR.

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

Ensure the package's documentation and issue-tracker links are publicly
accessible before announcing a public release.

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
corrected release rather than moving a tag.
