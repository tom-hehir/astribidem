"""Reject a release tag that differs from the distribution's declared version."""

import sys
import tomllib
from pathlib import Path


def main():
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    expected = f"v{project['version']}"
    if tag and tag != expected:
        raise SystemExit(f"Release tag {tag!r} must equal {expected!r}")
    print(f"Building {project['name']} {project['version']}")


if __name__ == "__main__":
    main()
