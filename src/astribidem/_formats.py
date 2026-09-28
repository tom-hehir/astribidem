"""Versioned persistence contracts, independent of the package release number."""

from importlib.metadata import version

__version__ = version("astribidem")
FORMAT_VERSION = 1


def file_metadata(kind: str) -> dict:
    return {
        "format": f"astribidem.{kind}",
        "format_version": FORMAT_VERSION,
        "astribidem_version": __version__,
    }


def check_file_metadata(metadata: dict, kind: str) -> None:
    expected = f"astribidem.{kind}"
    if (
        metadata.get("format") != expected
        or type(metadata.get("format_version")) is not int
        or metadata["format_version"] != FORMAT_VERSION
    ):
        raise ValueError(
            f"unsupported {expected} format/version: "
            f"{metadata.get('format')!r}/{metadata.get('format_version')!r}; "
            "regenerate unversioned development files, or use a compatible release"
        )


def index_metadata() -> dict[bytes, bytes]:
    return {
        b"astribidem.index_format_version": str(FORMAT_VERSION).encode(),
        b"astribidem.version": __version__.encode(),
    }


def check_index_metadata(metadata: dict[bytes, bytes]) -> None:
    if metadata.get(b"astribidem.index_format_version") != str(FORMAT_VERSION).encode():
        raise ValueError(
            "unsupported astribidem index format version; regenerate unversioned "
            "development indexes, or use a compatible release"
        )
