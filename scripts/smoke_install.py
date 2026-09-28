"""Exercise the wheel's base/large dependency boundaries without test extras."""

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

import astribidem
from astribidem import (
    DegenerateCrossmatchConfig,
    build_edges,
    crossmatch,
    match_uids,
    resolve_to_file,
    rows_to_ids,
    survey_coords_from_arrays,
    write_edges,
)


def main():
    extra = sys.argv[1]
    assert extra in ("base", "large")
    assert "site-packages" in astribidem.__file__, astribidem.__file__
    assert importlib.util.find_spec("astropy") is None
    assert (importlib.util.find_spec("duckdb") is not None) == (extra == "large")
    surveys = [
        survey_coords_from_arrays("a", [10.0, 20.0], [0.0, 0.0]),
        survey_coords_from_arrays("b", [10.0001, 30.0], [0.0, 0.0]),
    ]
    settings = {"radius_arcsec": 1.0, "dedupe_radius_arcsec": {"a": 0.0, "b": 0.0}}
    mode = DegenerateCrossmatchConfig(surveys=["a", "b"])
    rows = crossmatch(surveys, mode=mode, **settings)
    assert rows_to_ids(rows, {"a": [101, 102]})["a/id"].to_pylist() == [101]
    assert match_uids({"a": [1, 2], "b": [2, 3]}, join="outer").num_rows == 3
    with TemporaryDirectory() as directory:
        root = Path(directory)
        write_edges(build_edges(surveys, **settings), root / "edges")
        resolve_to_file(
            root / "edges", mode, root / "index.parquet", sort=extra == "large"
        )
        assert pq.read_table(root / "index.parquet").equals(rows, check_metadata=True)
    print(f"{extra} installation passed ({astribidem.__version__})")


if __name__ == "__main__":
    main()
