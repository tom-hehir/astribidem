"""Resolve saved edges segment by segment into one index file.

``resolve_to_file`` resolves each segment of a saved edge directory on its own,
optionally in parallel processes, and writes the index as one Parquet file.
Segments never share a connected group of rows, so the entities are exactly
those of resolving every segment together. With ``sort=True`` (the default)
the per-segment indexes are sorted into the global entity order: first
present survey, then that survey's row. The result then equals
``resolve(read_edges(directory), mode).table``, metadata included.

The sort runs in DuckDB, the optional ``large`` dependency, which spills to
disk beyond ``memory_limit``. DuckDB could write the sorted index to Parquet
itself, which is faster, but it would choose the column types and drop the
index metadata; its rows instead stream back a batch at a time and are written
here with exactly the resolved schema, trading some speed for that control.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from astro_crossmatch.api import resolve
from astro_crossmatch.edge_files import read_segment, segment_names
from astro_crossmatch.modes import CrossmatchModeConfig

_ROW_BITS = 40  # row numbers below 2**40; survey positions fill the bits above
_BATCH_ROWS = 65_536


def _resolve_segment(arguments) -> None:
    edges_directory, name, mode, path = arguments
    table = resolve(read_segment(edges_directory, name), mode).table
    pq.write_table(table, path)


def _is_count(key: str) -> bool:
    """Per-segment counts that add up; ``n_rows`` is each survey's total."""
    return key != "n_rows" and key.startswith(("n_", "rows_in_"))


def _merged(values: list, key: str | None = None):
    """Add counts across segments; every other value must agree."""
    first = values[0]
    if isinstance(first, dict):
        return {k: _merged([v[k] for v in values], k) for k in first}
    if key is not None and _is_count(key):
        return sum(values)
    if any(value != first for value in values):
        raise ValueError(f"segments disagree on {key!r}: {first!r}")
    return first


def _merged_metadata(metadatas: list[dict[bytes, bytes]]) -> dict[bytes, bytes]:
    merged = dict(metadatas[0])
    for key in (b"astro_crossmatch.dedupe", b"astro_crossmatch.summary"):
        combined = _merged([json.loads(metadata[key]) for metadata in metadatas])
        merged[key] = json.dumps(
            combined, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    return merged


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sorted_batches(paths, schema, work: Path, memory_limit, threads):
    """The rows of every per-segment index, in entity order, from DuckDB."""
    try:
        import duckdb
    except ImportError as exc:
        raise ImportError(
            "sorting segments needs DuckDB: install astro-crossmatch[large]"
        ) from exc
    rows = [_quote(n) for n in schema.names if n.endswith("/row_index")]
    # The entity order: first present survey, then that survey's row.
    key = " ".join(
        f"WHEN {row} IS NOT NULL THEN ({position}::BIGINT << {_ROW_BITS}) + {row}"
        for position, row in enumerate(rows)
    )
    files = ", ".join(_literal(path) for path in paths)
    connection = duckdb.connect()
    try:
        connection.execute("SET enable_progress_bar = false")
        connection.execute(f"SET temp_directory = {_literal(work / 'spill')}")
        if memory_limit is not None:
            connection.execute(f"SET memory_limit = {_literal(memory_limit)}")
        if threads is not None:
            connection.execute(f"SET threads = {int(threads)}")
        result = connection.execute(
            f"SELECT * FROM read_parquet([{files}]) ORDER BY CASE {key} END"
        ).to_arrow_reader(_BATCH_ROWS)
        for batch in result:
            yield pa.Table.from_batches([batch]).cast(schema)
    except duckdb.OutOfMemoryException as exc:
        raise MemoryError(
            f"DuckDB ran out of memory within memory_limit={memory_limit!r}: "
            "raise memory_limit or lower threads"
        ) from exc
    finally:
        connection.close()


def resolve_to_file(
    edges_directory: str | Path,
    mode: CrossmatchModeConfig,
    index_path: str | Path,
    *,
    processes: int = 1,
    sort: bool = True,
    memory_limit: str | None = None,
    threads: int | None = None,
) -> None:
    """Resolve every segment of saved edges and write one index file.

    ``processes`` resolves segments in parallel. With ``sort=False`` the index
    lists each segment's entities in segment order instead of sorting them,
    and DuckDB is not needed. ``memory_limit`` (such as ``"4GB"``) and
    ``threads`` set DuckDB's limit and threads for the sort; when the limit is
    too small for the threads, the sort fails with ``MemoryError``.
    """
    edges_directory = Path(edges_directory)
    index_path = Path(index_path)
    names = segment_names(edges_directory)
    work = index_path.with_name(f".{index_path.name}.segments")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        paths = [work / f"{name}.parquet" for name in names]
        tasks = [
            (str(edges_directory), name, mode, str(path))
            for name, path in zip(names, paths)
        ]
        if processes > 1:
            with ProcessPoolExecutor(processes) as pool:
                list(pool.map(_resolve_segment, tasks))
        else:
            for task in tasks:
                _resolve_segment(task)
        metadatas = [pq.read_schema(path).metadata for path in paths]
        schema = pq.read_schema(paths[0]).with_metadata(_merged_metadata(metadatas))
        with pq.ParquetWriter(index_path, schema) as writer:
            writer.write_table(schema.empty_table())
            if sort:
                tables = _sorted_batches(paths, schema, work, memory_limit, threads)
            else:
                tables = (
                    pa.Table.from_batches([batch]).cast(schema)
                    for path in paths
                    for batch in pq.ParquetFile(path).iter_batches(_BATCH_ROWS)
                )
            for table in tables:
                writer.write_table(table)
    finally:
        shutil.rmtree(work, ignore_errors=True)
